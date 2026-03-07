"""Pure-Python flat (brute-force) vector backend — zero external dependencies.

Strategy: **Binary Quantization prescore → exact rerank**.

Tier 1 — BQ Hamming (``int.__xor__`` + ``int.bit_count``):
    Every float32 vector is compressed to a single Python ``int`` where each
    bit encodes the *sign* of one component (1 = positive, 0 = negative).
    Hamming distance between two BQ codes = ``(a ^ b).bit_count()``.
    Both ``__xor__`` and ``bit_count`` execute in C (``mpn_xor`` + POPCNT
    on the internal digit array) — **no Python eval-loop**.

    The entire prescore pipeline is built from C-level iterators::

        map(xor, repeat(q_bin), bins)   →   map(bit_count, ...)
        →   zip(dists, range(n))   →   heapq.nsmallest(oversample, ...)

    ``heapq`` consumes the ``zip`` iterator; ``zip.__next__`` chains through
    ``map.__next__`` calls — all via ``tp_iternext`` C-slots, never touching
    the Python eval loop.

Tier 2 — Exact rerank:
    The top ``k * OVERSAMPLE`` candidates are rescored with exact L2 distance.
    Two paths, auto-detected at import time:

    * **BLAS path** (fastest): ``cblas_sdot`` loaded via ``ctypes`` from the
      system BLAS (macOS Accelerate, libopenblas, libblas).  Zero-copy
      ``from_buffer`` on ``array.array('f')``.  ~1-2 µs per vector.

    * **math.dist path** (fallback): stdlib ``math.dist`` on ``array.array``.
      ~50 µs per vector, but only runs on ~100-200 candidates so still fast.

Memory layout (100 K × 1024 dims):
    ``_bins``   — ``list[int]``           — BQ codes, ~16 MB
    ``_vecs``   — ``list[array.array]``   — full float32 vectors, ~410 MB
    ``_ids``    — ``array.array('I')``    — doc IDs, ~0.4 MB

Thread-safety: single-writer / multi-reader via atomic reference swap (GIL).
Persistence: vectors live in ``documents.embedding`` — no extra tables.
"""

from __future__ import annotations

import array
import ctypes
import ctypes.util
import heapq
import logging
import math
import sqlite3
from itertools import repeat
from operator import xor
from typing import Any, NamedTuple

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# BLAS auto-detection  (optional, pure stdlib)
# ------------------------------------------------------------------

_cblas_sdot = None  # type: ignore[assignment]


def _try_load_blas():
    """Try to find and load ``cblas_sdot`` from a system BLAS library.

    Returns the callable, or ``None`` if nothing usable is found.
    """
    _c_int = ctypes.c_int
    _c_float = ctypes.c_float
    _POINTER = ctypes.POINTER

    candidates = ["cblas", "blas", "openblas", "Accelerate"]
    for name in candidates:
        path = ctypes.util.find_library(name)
        if not path:
            continue
        try:
            lib = ctypes.CDLL(path)
            fn = lib.cblas_sdot
            fn.restype = _c_float
            fn.argtypes = [
                _c_int, _POINTER(_c_float), _c_int,
                _POINTER(_c_float), _c_int,
            ]
            # Smoke test: dot([1],[1]) should be 1.0
            _One = _c_float * 1
            a, b = _One(1.0), _One(1.0)
            if abs(fn(1, a, 1, b, 1) - 1.0) < 0.01:
                logger.debug("BLAS loaded from %s (%s)", name, path)
                return fn
        except (OSError, AttributeError):
            continue
    return None


try:
    _cblas_sdot = _try_load_blas()
except Exception:
    _cblas_sdot = None


def blas_loaded() -> bool:
    """Return *True* if a system BLAS library was detected at import time."""
    return _cblas_sdot is not None


# ------------------------------------------------------------------
# Popcount helper  (Python 3.10+: int.bit_count, else fallback)
# ------------------------------------------------------------------

_popcount = getattr(int, "bit_count", None)
if _popcount is None:
    def _popcount(x: int) -> int:  # type: ignore[assignment]
        return bin(x).count("1")

# ------------------------------------------------------------------
# Binary Quantization
# ------------------------------------------------------------------

def _quantize_list(vec: list[float] | array.array) -> int:
    """Quantize a float vector to a single Python ``int`` (BQ code).

    Each component → 1 bit: positive → 1, else → 0.
    """
    code = 0
    for i, x in enumerate(vec):
        if x > 0:
            code |= 1 << i
    return code


# ------------------------------------------------------------------
# BLAS-based L2² distance (zero-copy, pre-normed)
# ------------------------------------------------------------------

def _make_blas_dist_prenorm(ndim: int):
    """Optimised L2² for pre-normed query: caches ||q||², reuses ||v||².

    Returns ``(prepare_fn, dist_fn)`` or ``None``.

    * ``prepare_fn(q)`` — call once, returns (q_ptr, qq).
    * ``dist_fn(q_ptr, qq, v_ptr, vv)`` — returns L2².
    """
    if _cblas_sdot is None:
        return None

    CFloatArr = ctypes.c_float * ndim

    def prepare(q: array.array):
        pq = CFloatArr.from_buffer(q)
        qq = float(_cblas_sdot(ndim, pq, 1, pq, 1))
        return pq, qq

    def dist(pq, qq: float, v: array.array, vv: float) -> float:
        pv = CFloatArr.from_buffer(v)
        dot = float(_cblas_sdot(ndim, pq, 1, pv, 1))
        return qq + vv - 2.0 * dot

    return prepare, dist


# ===================================================================
# SqliteFlatBackend
# ===================================================================

# Oversample factor for BQ prescore (k * factor = number of candidates).
# Industry standard for binary quantization is 20-30x.  On real embeddings
# (semantically structured sign patterns) 20x gives 95%+ recall.
# Random vectors are pathological for BQ — low recall is expected there.
_BQ_OVERSAMPLE = 20


class _Snapshot(NamedTuple):
    """Immutable bundle of parallel index arrays.

    A single ``self._snap = new_snap`` assignment is atomic under the GIL,
    guaranteeing readers never see a mix of old and new arrays.
    """
    bins: list[int]                # BQ codes
    vecs: list[array.array]        # full float32 vectors
    ids: array.array               # doc IDs (uint32)
    norms_sq: list[float]          # precomputed ||v||²
    id_set: set[int]               # O(1) membership check
    id_pos: dict[int, int]         # key → index for O(1) upsert
    count: int


def _make_empty_snap() -> _Snapshot:
    """Create a fresh empty snapshot — avoid sharing mutable containers."""
    return _Snapshot(
        bins=[], vecs=[], ids=array.array("I"),
        norms_sq=[], id_set=set(), id_pos={}, count=0,
    )


class SqliteFlatBackend:
    """Brute-force vector backend: BQ prescore + exact rerank.

    Pure Python, zero external dependencies.  Optionally uses system BLAS
    (via ``ctypes``) for SIMD-accelerated exact distance on rerank tier.

    Parameters
    ----------
    oversample : int
        BQ prescore returns ``k * oversample`` candidates for reranking.
        Higher = better recall, slower.  Default 20.
    """

    def __init__(self, *, oversample: int = _BQ_OVERSAMPLE) -> None:
        if oversample < 1:
            raise ValueError(f"oversample must be >= 1, got {oversample}")
        self._oversample = oversample
        self._ndim: int | None = None
        self._writer_conn: sqlite3.Connection | None = None

        # In-memory index — single atomic snapshot for thread safety.
        self._snap: _Snapshot = _make_empty_snap()
        self._initialized: bool = False
        # Write lock — serializes add()/remove() to prevent snapshot overwrite race
        self._write_lock = __import__("threading").Lock()

        # BLAS rerank function (resolved in initialize when ndim is known)
        self._blas_prenorm = None

    # ------------------------------------------------------------------
    # VectorBackend protocol
    # ------------------------------------------------------------------

    def set_writer_conn(self, conn: sqlite3.Connection) -> None:
        self._writer_conn = conn

    def initialize(self, db_path: str, ndim: int) -> None:
        self._ndim = ndim

        # Try to set up BLAS for this dimensionality
        self._blas_prenorm = _make_blas_dist_prenorm(ndim)

        blas_status = "BLAS" if self._blas_prenorm else "math.dist"
        logger.info(
            "SqliteFlatBackend initialized: ndim=%d, rerank=%s",
            ndim, blas_status,
        )

        # Load existing vectors from documents.embedding
        conn = self._writer_conn
        if conn is not None:
            try:
                rows = conn.execute(
                    "SELECT id, embedding FROM documents "
                    "WHERE vec_indexed = 1 AND embedding IS NOT NULL "
                    "ORDER BY id"
                ).fetchall()
            except sqlite3.OperationalError:
                rows = []

            if rows:
                bins: list[int] = []
                vecs: list[array.array] = []
                ids = array.array("I")
                norms_sq: list[float] = []
                id_set: set[int] = set()
                id_pos: dict[int, int] = {}
                skipped = 0
                expected_blob_size = ndim * 4
                for doc_id, blob in rows:
                    if len(blob) != expected_blob_size:
                        logger.warning(
                            "Skipping doc %d: blob size %d != expected %d",
                            doc_id, len(blob), expected_blob_size,
                        )
                        skipped += 1
                        continue
                    va = array.array("f")
                    va.frombytes(blob)
                    if not all(math.isfinite(x) for x in va):
                        skipped += 1
                        continue
                    idx = len(bins)
                    bins.append(_quantize_list(va))
                    vecs.append(va)
                    ids.append(doc_id)
                    norms_sq.append(sum(x * x for x in va))
                    id_set.add(doc_id)
                    id_pos[doc_id] = idx
                if skipped:
                    logger.warning(
                        "Skipped %d vectors with NaN/Inf during initialize",
                        skipped,
                    )
                # Publish under write lock to prevent race with add()
                with self._write_lock:
                    self._snap = _Snapshot(
                        bins, vecs, ids, norms_sq, id_set, id_pos, len(bins),
                    )
                logger.info(
                    "Loaded %d vectors from DB into flat cache",
                    self._snap.count,
                )

        self._initialized = True

    def add(self, keys: list[int], vectors: list[list[float]]) -> None:
        with self._write_lock:
            self._add_unlocked(keys, vectors)

    def _add_unlocked(self, keys: list[int], vectors: list[list[float]]) -> None:
        snap = self._snap
        # Build new lists (copy-on-write for thread safety)
        new_bins = list(snap.bins)
        new_vecs = list(snap.vecs)
        new_ids = array.array("I", snap.ids)
        new_norms_sq = list(snap.norms_sq)
        new_id_set = set(snap.id_set)
        new_id_pos = dict(snap.id_pos)

        changed = False
        for key, vec in zip(keys, vectors):
            va = array.array("f", vec)
            # Dimension check — wrong-dim vector corrupts index / crashes BLAS
            if self._ndim is not None and len(va) != self._ndim:
                logger.warning(
                    "Skipping vector for key %d: dimension %d != expected %d",
                    key, len(va), self._ndim,
                )
                continue
            # Guard against NaN/Inf — one bad vector should not corrupt
            # the entire index silently.
            if not all(math.isfinite(x) for x in va):
                logger.warning(
                    "Skipping vector for key %d: contains NaN or Inf", key,
                )
                continue
            bq = _quantize_list(va)
            nsq = sum(x * x for x in va)
            changed = True

            if key in new_id_set:
                # Upsert: O(1) lookup via id_pos
                i = new_id_pos[key]
                new_bins[i] = bq
                new_vecs[i] = va
                new_norms_sq[i] = nsq
            else:
                new_id_pos[key] = len(new_bins)
                new_bins.append(bq)
                new_vecs.append(va)
                new_ids.append(key)
                new_norms_sq.append(nsq)
                new_id_set.add(key)

        if not changed:
            return
        # Atomic publish — single reference swap, readers see old or new
        self._snap = _Snapshot(
            new_bins, new_vecs, new_ids, new_norms_sq,
            new_id_set, new_id_pos, len(new_bins),
        )

    def remove(self, keys: list[int]) -> None:
        with self._write_lock:
            self._remove_unlocked(keys)

    def _remove_unlocked(self, keys: list[int]) -> None:
        snap = self._snap
        remove_set = set(keys)
        if not remove_set.intersection(snap.id_set):
            return

        new_bins: list[int] = []
        new_vecs: list[array.array] = []
        new_ids = array.array("I")
        new_norms_sq: list[float] = []
        new_id_set: set[int] = set()
        new_id_pos: dict[int, int] = {}

        for i, eid in enumerate(snap.ids):
            if eid not in remove_set:
                new_id_pos[eid] = len(new_bins)
                new_bins.append(snap.bins[i])
                new_vecs.append(snap.vecs[i])
                new_ids.append(eid)
                new_norms_sq.append(snap.norms_sq[i])
                new_id_set.add(eid)

        self._snap = _Snapshot(
            new_bins, new_vecs, new_ids, new_norms_sq,
            new_id_set, new_id_pos, len(new_bins),
        )

    def search(
        self,
        query: list[float],
        k: int,
        **kwargs: Any,
    ) -> list[tuple[int, float]]:
        """KNN search.  Returns ``(doc_id, L2 distance)`` ascending by distance.

        Distances are always Euclidean (``||q - v||``) regardless of whether
        BLAS or ``math.dist`` is used for computation.
        """
        # Capture snapshot — single atomic read, consistent view
        snap = self._snap
        n = snap.count

        if n == 0:
            return []

        q = array.array("f", query)
        # Guard against NaN/Inf in query — would poison all distances
        if not all(math.isfinite(x) for x in q):
            logger.warning("Query vector contains NaN or Inf — returning empty")
            return []
        k = min(k, n)
        if k == 0:
            return []
        oversample = min(n, k * self._oversample)

        # ── Small dataset or high oversample: exact brute-force ─────
        if oversample >= n:
            return self._exact_search(q, snap, k)

        # ── Tier 1: BQ Hamming — full C-level pipeline ────────────
        q_bin = _quantize_list(q)
        xors = map(xor, repeat(q_bin), snap.bins)       # C: int.__xor__
        dists = map(_popcount, xors)                     # C: int.bit_count
        pipe = zip(dists, range(n))                      # C: zip + range
        cands = heapq.nsmallest(oversample, pipe)        # C: _heapq

        # ── Tier 2: Exact rerank ──────────────────────────────────
        if self._blas_prenorm is not None:
            prepare, dist_fn = self._blas_prenorm
            pq, qq = prepare(q)
            exact = []
            for _, idx in cands:
                d2 = dist_fn(pq, qq, snap.vecs[idx], snap.norms_sq[idx])
                if not math.isfinite(d2):
                    continue  # corrupt stored vector — skip
                exact.append((max(0.0, d2), int(snap.ids[idx])))
            exact.sort()
            # Convert L2² → L2 for consistent metric
            return [(did, math.sqrt(d)) for d, did in exact[:k]]
        else:
            exact = sorted(
                (math.dist(q, snap.vecs[idx]), int(snap.ids[idx]))
                for _, idx in cands
            )
            return [(did, d) for d, did in exact[:k]]

    def _exact_search(
        self,
        q: array.array,
        snap: _Snapshot,
        k: int,
    ) -> list[tuple[int, float]]:
        """Full brute-force, no BQ tier.  Uses BLAS when available."""
        if self._blas_prenorm is not None:
            prepare, dist_fn = self._blas_prenorm
            pq, qq = prepare(q)

            def _blas_pairs():
                for i in range(snap.count):
                    d2 = dist_fn(pq, qq, snap.vecs[i], snap.norms_sq[i])
                    if math.isfinite(d2):
                        yield max(0.0, d2), int(snap.ids[i])

            results = heapq.nsmallest(k, _blas_pairs())
            # Convert L2² → L2 for consistent metric
            return [(int(did), math.sqrt(float(d))) for d, did in results]
        else:
            results = heapq.nsmallest(
                k,
                zip(map(math.dist, repeat(q), snap.vecs), snap.ids),
            )
            return [(int(did), float(d)) for d, did in results]

    def flush(self) -> None:
        """No-op — vectors are persisted in ``documents.embedding``."""

    def count(self) -> int:
        return self._snap.count

    def close(self) -> None:
        self._snap = _make_empty_snap()

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def blas_available(self) -> bool:
        """Whether CBLAS acceleration is active for reranking."""
        return self._blas_prenorm is not None

    @property
    def rerank_method(self) -> str:
        """Description of the active rerank method."""
        if self._blas_prenorm is not None:
            return "cblas_sdot (SIMD)"
        return "math.dist"

    # ------------------------------------------------------------------
    # Consistency / rebuild  (same contract as SqliteHnswBackend)
    # ------------------------------------------------------------------

    def verify_consistency(self, expected_count: int) -> bool:
        return self._snap.count == expected_count

    def reset(self, ndim: int | None = None) -> None:
        """Clear all in-memory state (for rebuild)."""
        self._snap = _make_empty_snap()
        if ndim is not None:
            self._ndim = ndim
            self._blas_prenorm = _make_blas_dist_prenorm(ndim)

    def rebuild_from_blobs(
        self,
        rows: list[tuple[int, bytes]],
        ndim: int,
    ) -> None:
        """Rebuild flat index from embedding BLOBs (one batch, OOM-safe)."""
        logger.info("Rebuilding flat index from %d blobs", len(rows))
        self.reset(ndim)

        bins: list[int] = []
        vecs: list[array.array] = []
        ids = array.array("I")
        norms_sq: list[float] = []
        id_set: set[int] = set()
        id_pos: dict[int, int] = {}

        skipped = 0
        expected_blob_size = ndim * 4
        for doc_id, blob in rows:
            if len(blob) != expected_blob_size:
                logger.warning(
                    "Skipping doc %d: blob size %d != expected %d",
                    doc_id, len(blob), expected_blob_size,
                )
                skipped += 1
                continue
            va = array.array("f")
            va.frombytes(blob)
            if not all(math.isfinite(x) for x in va):
                skipped += 1
                continue
            idx = len(bins)
            bins.append(_quantize_list(va))
            vecs.append(va)
            ids.append(doc_id)
            norms_sq.append(sum(x * x for x in va))
            id_set.add(doc_id)
            id_pos[doc_id] = idx

        if skipped:
            logger.warning(
                "Skipped %d vectors with NaN/Inf during rebuild", skipped,
            )
        self._snap = _Snapshot(
            bins, vecs, ids, norms_sq, id_set, id_pos, len(bins),
        )
        logger.info("Flat index rebuilt: %d vectors", self._snap.count)
