"""Pure-Python HNSW backend — single-file, zero external dependencies.

Graph stored inside SQLite as a binary BLOB (CSR format).
Vectors live in ``documents.embedding`` column.
Distance computed via ``math.dist`` + ``array.array`` (both C-speed).
"""

from __future__ import annotations

import array
import bisect
import heapq
import itertools
import logging
import math
import random
import struct
import sqlite3
import threading
from typing import Any

logger = logging.getLogger(__name__)

# CSR-serialisation constants
_MAGIC = b"HNSW"
_VERSION = 1
_EMPTY: array.array = array.array("I")


# ---------------------------------------------------------------------------
# CSR level — one layer of the HNSW graph
# ---------------------------------------------------------------------------

class _CsrLevel:
    """One level of the HNSW graph: CSR base + mutation overlay.

    Base arrays are immutable between flushes.  Mutations (new nodes,
    updated neighbor lists) go into ``_delta`` and are merged back on
    flush via :meth:`merge`.
    """

    __slots__ = ("node_ids", "offsets", "edges", "_delta")

    def __init__(self) -> None:
        self.node_ids: array.array = array.array("I")  # sorted
        self.offsets: array.array = array.array("I", [0])
        self.edges: array.array = array.array("I")
        self._delta: dict[int, list[int]] = {}

    # -- read --

    def neighbors(self, node_id: int) -> list[int] | array.array:
        if node_id in self._delta:
            return self._delta[node_id]
        pos = bisect.bisect_left(self.node_ids, node_id)
        if pos < len(self.node_ids) and self.node_ids[pos] == node_id:
            s = self.offsets[pos]
            e = self.offsets[pos + 1]
            return self.edges[s:e]
        return _EMPTY

    def has_node(self, node_id: int) -> bool:
        if node_id in self._delta:
            return True
        pos = bisect.bisect_left(self.node_ids, node_id)
        return pos < len(self.node_ids) and self.node_ids[pos] == node_id

    def all_node_ids(self) -> set[int]:
        ids = set(self.node_ids)
        ids.update(self._delta.keys())
        return ids

    # -- write --

    def set_neighbors(self, node_id: int, nbrs: list[int]) -> None:
        self._delta[node_id] = nbrs

    # -- maintenance --

    def merge(self) -> None:
        """Compact delta into the CSR arrays."""
        if not self._delta:
            return

        # Gather final neighbor list per node
        all_nodes: dict[int, list[int] | array.array] = {}
        for i in range(len(self.node_ids)):
            nid = self.node_ids[i]
            if nid not in self._delta:
                s = self.offsets[i]
                e = self.offsets[i + 1]
                all_nodes[nid] = self.edges[s:e]
        all_nodes.update(self._delta)

        sorted_ids = sorted(all_nodes)
        new_ids = array.array("I", sorted_ids)
        new_offsets = array.array("I")
        new_edges = array.array("I")
        for nid in sorted_ids:
            new_offsets.append(len(new_edges))
            new_edges.extend(all_nodes[nid])
        new_offsets.append(len(new_edges))

        self.node_ids = new_ids
        self.offsets = new_offsets
        self.edges = new_edges
        self._delta.clear()


# ---------------------------------------------------------------------------
# SqliteHnswBackend
# ---------------------------------------------------------------------------

class SqliteHnswBackend:
    """HNSW index stored entirely inside a single SQLite file.

    * **Graph** (~13 MB / 100K vectors @ M=16) lives in RAM, serialised
      as CSR binary in ``__sqfox_hnsw`` table.
    * **Vectors** stay in ``documents.embedding BLOB`` — fetched on
      demand via batched ``SELECT … WHERE id IN (…)``.
    * Distances via ``math.dist`` + ``array.array('f').frombytes(blob)``
      — both execute at C speed, zero numpy.
    """

    def __init__(
        self,
        M: int = 16,
        ef_construction: int = 200,
        ef_search: int = 64,
    ) -> None:
        self._M = M
        self._M0 = 2 * M
        self._ef_construction = ef_construction
        self._ef_search = ef_search
        self._mL = 1.0 / math.log(M) if M > 1 else 1.0

        self._ndim: int | None = None
        self._db_path: str | None = None
        self._writer_conn: sqlite3.Connection | None = None

        # graph state
        self._entry_point: int | None = None
        self._max_level: int = -1
        self._node_levels: dict[int, int] = {}
        self._graphs: list[_CsrLevel] = []
        self._deleted: set[int] = set()
        self._dirty: bool = False
        self._count: int = 0

        self._initialized: bool = False
        self._lock = threading.Lock()

        # vector cache (bounded, for insert/rebuild hot path)
        self._vec_cache: dict[int, array.array] = {}
        self._vec_cache_max: int = 10_000

    # ------------------------------------------------------------------
    # VectorBackend protocol
    # ------------------------------------------------------------------

    def set_writer_conn(self, conn: sqlite3.Connection) -> None:
        self._writer_conn = conn

    def initialize(self, db_path: str, ndim: int) -> None:
        self._db_path = db_path
        self._ndim = ndim

        conn = self._writer_conn
        if conn is None:
            self._initialized = True
            return

        conn.execute(
            "CREATE TABLE IF NOT EXISTS __sqfox_hnsw "
            "(key TEXT PRIMARY KEY, value BLOB)"
        )
        conn.commit()

        row = conn.execute(
            "SELECT value FROM __sqfox_hnsw WHERE key = 'graph'"
        ).fetchone()
        if row is not None and row[0]:
            try:
                self._deserialize(row[0])
            except Exception as exc:
                logger.warning(
                    "HNSW graph BLOB is corrupt, will rebuild on next "
                    "ingest: %s", exc,
                )
                # Leave graph empty — verify_consistency in engine will
                # detect the mismatch and trigger rebuild_from_blobs.

        self._initialized = True

    def add(self, keys: list[int], vectors: list[list[float]]) -> None:
        for key, vec in zip(keys, vectors):
            # If re-inserting a previously deleted key, un-delete it
            self._deleted.discard(key)
            va = array.array("f", vec)
            self._cache_put(key, va)
            self._insert(key, va)
        self._dirty = True

    def remove(self, keys: list[int]) -> None:
        with self._lock:
            for key in keys:
                if key in self._node_levels:
                    self._deleted.add(key)
                    self._count -= 1
            self._dirty = True

    def search(
        self,
        query: list[float],
        k: int,
        **kwargs: Any,
    ) -> list[tuple[int, float]]:
        conn: sqlite3.Connection | None = kwargs.get("conn")
        if conn is None:
            raise ValueError(
                "SqliteHnswBackend.search() requires conn= kwarg"
            )

        with self._lock:
            ep = self._entry_point
            ml = self._max_level
            cnt = self._count
        if ep is None or cnt == 0:
            return []

        q = array.array("f", query)

        # navigate upper layers (greedy, ef=1)
        current = ep
        current_vec = self._fetch_one(current, conn)
        if current_vec is None:
            return []
        current_dist = math.dist(q, current_vec)

        for level in range(ml, 0, -1):
            changed = True
            while changed:
                changed = False
                with self._lock:
                    nbrs = list(self._neighbors(current, level))
                alive = [n for n in nbrs if n not in self._deleted]
                if not alive:
                    break
                vecs = self._fetch_batch(alive, conn)
                for n, v in vecs.items():
                    d = math.dist(q, v)
                    if d < current_dist:
                        current_dist = d
                        current = n
                        changed = True

        # search layer 0 with ef_search
        results = self._search_layer(
            q, current, max(k, self._ef_search), 0, conn
        )
        return results[:k]

    def flush(self) -> None:
        if not self._dirty or self._writer_conn is None:
            return
        with self._lock:
            blob = self._serialize()  # merges CSR inside
            # Clean up deleted entries now that they're excluded from blob
            for nid in self._deleted:
                self._node_levels.pop(nid, None)
            self._count = len(self._node_levels)
            self._deleted.clear()
        self._writer_conn.execute(
            "INSERT OR REPLACE INTO __sqfox_hnsw (key, value) "
            "VALUES ('graph', ?)",
            (blob,),
        )
        self._writer_conn.commit()
        self._dirty = False

    def count(self) -> int:
        return self._count

    def close(self) -> None:
        if self._dirty:
            self.flush()
        self._vec_cache.clear()

    # ------------------------------------------------------------------
    # Consistency / rebuild
    # ------------------------------------------------------------------

    def verify_consistency(self, expected_count: int) -> bool:
        return self._count == expected_count

    def reset(self, ndim: int | None = None) -> None:
        """Clear all graph state (for rebuild).  Optionally update ndim."""
        with self._lock:
            self._graphs.clear()
            self._node_levels.clear()
            self._entry_point = None
            self._max_level = -1
            self._count = 0
            self._deleted.clear()
        if ndim is not None:
            self._ndim = ndim
        self._vec_cache.clear()
        self._dirty = True

    def rebuild_from_blobs(
        self,
        rows: list[tuple[int, bytes]],
        ndim: int,
    ) -> None:
        """Rebuild HNSW index from embedding BLOBs (batched, OOM-safe)."""
        logger.info("Rebuilding HNSW from %d blobs …", len(rows))
        self.reset(ndim)

        BATCH = 2000
        for i in range(0, len(rows), BATCH):
            batch = rows[i : i + BATCH]
            keys = [r[0] for r in batch]
            vecs = [list(struct.unpack(f"{ndim}f", r[1])) for r in batch]
            self.add(keys, vecs)

        self._vec_cache.clear()
        logger.info("HNSW rebuilt: %d vectors", self._count)

    # ------------------------------------------------------------------
    # core HNSW
    # ------------------------------------------------------------------

    def _random_level(self) -> int:
        r = random.random()
        if r == 0.0:
            r = 1e-18  # guard against log(0)
        return min(int(-math.log(r) * self._mL), 255)

    def _neighbors(self, node_id: int, level: int) -> list[int] | array.array:
        """Return neighbor list (caller must hold self._lock)."""
        if level >= len(self._graphs):
            return _EMPTY
        return self._graphs[level].neighbors(node_id)

    def _insert(self, key: int, vec: array.array) -> None:
        level = self._random_level()

        with self._lock:
            is_update = key in self._node_levels

            # ensure levels exist
            while len(self._graphs) <= level:
                self._graphs.append(_CsrLevel())

            self._node_levels[key] = level

            if self._entry_point is None:
                self._entry_point = key
                self._max_level = level
                for lv in range(level + 1):
                    self._graphs[lv].set_neighbors(key, [])
                if not is_update:
                    self._count += 1
                return

            ep = self._entry_point
            ml = self._max_level

        conn = self._writer_conn
        ep_vec = self._get_vector(ep, conn)
        if ep_vec is None:
            with self._lock:
                if not is_update:
                    self._count += 1
            return
        ep_dist = math.dist(vec, ep_vec)
        current, current_dist = ep, ep_dist

        # traverse upper layers above insertion level
        for lv in range(ml, level, -1):
            changed = True
            while changed:
                changed = False
                with self._lock:
                    nbrs = list(self._neighbors(current, lv))
                for n in nbrs:
                    if n in self._deleted:
                        continue
                    n_vec = self._get_vector(n, conn)
                    if n_vec is None:
                        continue
                    d = math.dist(vec, n_vec)
                    if d < current_dist:
                        current_dist = d
                        current = n
                        changed = True

        # insert from min(level, max_level) down to 0
        entry_for_layer = current
        for lv in range(min(level, ml), -1, -1):
            max_edges = self._M0 if lv == 0 else self._M

            candidates = self._search_layer(
                vec, entry_for_layer, self._ef_construction, lv, conn
            )
            selected = [c[0] for c in candidates[:max_edges]]

            with self._lock:
                self._graphs[lv].set_neighbors(key, selected)

            # bidirectional connections + pruning (lock per-neighbor)
            for nbr_id in selected:
                with self._lock:
                    nbr_nbrs = list(self._neighbors(nbr_id, lv))
                if key not in nbr_nbrs:
                    nbr_nbrs.append(key)
                    if len(nbr_nbrs) > max_edges:
                        nbr_nbrs = self._prune(
                            nbr_id, nbr_nbrs, max_edges, conn
                        )
                    with self._lock:
                        self._graphs[lv].set_neighbors(nbr_id, nbr_nbrs)

            if candidates:
                entry_for_layer = candidates[0][0]

        with self._lock:
            if level > self._max_level:
                # Register node on upper levels (empty neighbor lists)
                while len(self._graphs) <= level:
                    self._graphs.append(_CsrLevel())
                for lv in range(self._max_level + 1, level + 1):
                    self._graphs[lv].set_neighbors(key, [])
                self._max_level = level
                self._entry_point = key
            if not is_update:
                self._count += 1

    def _prune(
        self,
        node_id: int,
        neighbors: list[int],
        max_edges: int,
        conn: sqlite3.Connection | None,
    ) -> list[int]:
        """Keep closest max_edges neighbors of node_id."""
        node_vec = self._get_vector(node_id, conn)
        if node_vec is None:
            return neighbors[:max_edges]
        scored: list[tuple[float, int]] = []
        for n in neighbors:
            if n in self._deleted:
                continue
            n_vec = self._get_vector(n, conn)
            if n_vec is not None:
                scored.append((math.dist(node_vec, n_vec), n))
            else:
                scored.append((float("inf"), n))
        scored.sort()
        return [x[1] for x in scored[:max_edges]]

    def _search_layer(
        self,
        query: array.array,
        entry_point: int,
        ef: int,
        level: int,
        conn: sqlite3.Connection | None,
    ) -> list[tuple[int, float]]:
        """Beam search at a single HNSW level."""
        ep_vec = self._get_vector(entry_point, conn)
        if ep_vec is None:
            return []
        ep_dist = math.dist(query, ep_vec)

        # min-heap of candidates (closest first)
        candidates: list[tuple[float, int]] = [(ep_dist, entry_point)]
        # max-heap of results  (farthest first via negated distance)
        results: list[tuple[float, int]] = [(-ep_dist, entry_point)]
        visited: set[int] = {entry_point}

        while candidates:
            c_dist, c_id = heapq.heappop(candidates)
            f_dist = -results[0][0]
            if c_dist > f_dist:
                break

            with self._lock:
                nbrs = list(self._neighbors(c_id, level))

            unvisited = [
                n for n in nbrs
                if n not in visited and n not in self._deleted
            ]
            visited.update(unvisited)
            if not unvisited:
                continue

            vecs = self._fetch_batch(unvisited, conn)
            for n_id, n_vec in vecs.items():
                n_dist = math.dist(query, n_vec)
                f_dist = -results[0][0]

                if n_dist < f_dist or len(results) < ef:
                    heapq.heappush(candidates, (n_dist, n_id))
                    heapq.heappush(results, (-n_dist, n_id))
                    if len(results) > ef:
                        heapq.heappop(results)

        out = sorted((-d, nid) for d, nid in results)
        return [(nid, dist) for dist, nid in out]

    # ------------------------------------------------------------------
    # vector I/O (cache → SQLite)
    # ------------------------------------------------------------------

    def _cache_put(self, key: int, vec: array.array) -> None:
        if len(self._vec_cache) >= self._vec_cache_max:
            # Evict oldest quarter.  Snapshot only the keys we need
            # to delete — avoids allocating a full 10K-element list.
            drop_count = len(self._vec_cache) // 4
            victims = list(itertools.islice(self._vec_cache, drop_count))
            for k in victims:
                del self._vec_cache[k]
        self._vec_cache[key] = vec

    def _get_vector(
        self, node_id: int, conn: sqlite3.Connection | None
    ) -> array.array | None:
        """Get vector: cache first, then SQLite."""
        cached = self._vec_cache.get(node_id)
        if cached is not None:
            return cached
        if conn is None:
            return None
        return self._fetch_one(node_id, conn)

    def _fetch_one(
        self, node_id: int, conn: sqlite3.Connection
    ) -> array.array | None:
        row = conn.execute(
            "SELECT embedding FROM documents WHERE id = ?", (node_id,)
        ).fetchone()
        if row is None or row[0] is None:
            return None
        v = array.array("f")
        v.frombytes(row[0])
        self._cache_put(node_id, v)
        return v

    def _fetch_batch(
        self, ids: list[int], conn: sqlite3.Connection | None,
    ) -> dict[int, array.array]:
        if not ids or conn is None:
            return {}
        # split cached vs uncached
        result: dict[int, array.array] = {}
        need: list[int] = []
        for nid in ids:
            cached = self._vec_cache.get(nid)
            if cached is not None:
                result[nid] = cached
            else:
                need.append(nid)

        if need:
            ph = ",".join("?" * len(need))
            rows = conn.execute(
                f"SELECT id, embedding FROM documents "
                f"WHERE id IN ({ph})",
                need,
            ).fetchall()
            for rid, blob in rows:
                if blob:
                    v = array.array("f")
                    v.frombytes(blob)
                    self._cache_put(rid, v)
                    result[rid] = v
        return result

    # ------------------------------------------------------------------
    # CSR binary serialisation
    # ------------------------------------------------------------------

    def _serialize(self) -> bytes:
        """Pack graph into a compact binary blob (CSR format)."""
        parts: list[bytes] = []

        # --- node levels (compute first so we can write true count) ---
        active = sorted(
            (nid, lv)
            for nid, lv in self._node_levels.items()
            if nid not in self._deleted
        )

        # --- header ---
        # If entry point was deleted, pick the highest-level active node
        if (
            self._entry_point is not None
            and self._entry_point in self._deleted
        ):
            if active:
                best_nid, best_lv = max(active, key=lambda x: x[1])
                self._entry_point = best_nid
                self._max_level = best_lv
            else:
                self._entry_point = None
                self._max_level = -1

        entry = (
            self._entry_point if self._entry_point is not None else 0xFFFFFFFF
        )
        num_levels = max(self._max_level + 1, 0)
        active_count = len(active)
        parts.append(_MAGIC)
        parts.append(
            struct.pack(
                "<BHHIHHI",
                _VERSION,
                self._M,
                self._ef_construction,
                entry,
                num_levels,
                self._ndim or 0,
                active_count,
            )
        )
        parts.append(struct.pack("<I", len(active)))
        if active:
            parts.append(array.array("I", [n for n, _ in active]).tobytes())
            parts.append(array.array("B", [l for _, l in active]).tobytes())

        # --- per-level CSR ---
        for lv in range(num_levels):
            g = self._graphs[lv] if lv < len(self._graphs) else None
            if g is None:
                # empty level
                parts.append(struct.pack("<III", 0, 1, 0))
                parts.append(array.array("I", [0]).tobytes())
                continue

            g.merge()  # compact before writing

            # filter deleted
            lv_ids = array.array("I")
            lv_offsets = array.array("I")
            lv_edges = array.array("I")

            for nid in sorted(g.all_node_ids()):
                if nid in self._deleted:
                    continue
                lv_ids.append(nid)
                lv_offsets.append(len(lv_edges))
                for n in g.neighbors(nid):
                    if n not in self._deleted:
                        lv_edges.append(n)
            lv_offsets.append(len(lv_edges))

            parts.append(
                struct.pack("<III", len(lv_ids), len(lv_offsets), len(lv_edges))
            )
            parts.append(lv_ids.tobytes())
            parts.append(lv_offsets.tobytes())
            parts.append(lv_edges.tobytes())

        return b"".join(parts)

    def _deserialize(self, data: bytes) -> None:
        """Unpack binary CSR blob → in-memory graph."""
        pos = 0

        # --- magic ---
        if data[pos : pos + 4] != _MAGIC:
            raise ValueError("Invalid HNSW blob: bad magic")
        pos += 4

        # --- header ---
        fmt = "<BHHIHHI"
        sz = struct.calcsize(fmt)
        ver, M, ef_con, entry, n_levels, ndim, count = struct.unpack_from(
            fmt, data, pos
        )
        pos += sz
        if ver != _VERSION:
            raise ValueError(f"Unknown HNSW version {ver}")

        self._M = M
        self._M0 = 2 * M
        self._ef_construction = ef_con
        self._mL = 1.0 / math.log(M) if M > 1 else 1.0
        self._entry_point = entry if entry != 0xFFFFFFFF else None
        self._max_level = n_levels - 1
        self._ndim = ndim
        self._count = count

        # --- node levels ---
        (n_nodes,) = struct.unpack_from("<I", data, pos)
        pos += 4
        self._node_levels = {}
        if n_nodes:
            ids_a = array.array("I")
            ids_a.frombytes(data[pos : pos + n_nodes * 4])
            pos += n_nodes * 4
            lvs_a = array.array("B")
            lvs_a.frombytes(data[pos : pos + n_nodes])
            pos += n_nodes
            for nid, lv in zip(ids_a, lvs_a):
                self._node_levels[nid] = lv

        # --- per-level CSR ---
        self._graphs = []
        for _ in range(n_levels):
            g = _CsrLevel()
            n_ids, n_off, n_edg = struct.unpack_from("<III", data, pos)
            pos += 12

            if n_ids:
                g.node_ids = array.array("I")
                g.node_ids.frombytes(data[pos : pos + n_ids * 4])
            pos += n_ids * 4

            g.offsets = array.array("I")
            g.offsets.frombytes(data[pos : pos + n_off * 4])
            pos += n_off * 4

            if n_edg:
                g.edges = array.array("I")
                g.edges.frombytes(data[pos : pos + n_edg * 4])
            pos += n_edg * 4

            self._graphs.append(g)

        self._deleted.clear()
        self._dirty = False
