"""USearch HNSW backend — O(log N) approximate nearest neighbors with mmap."""

from __future__ import annotations

import logging
import math
import os
import struct as _struct
import threading
from pathlib import Path
from typing import Any

logger = logging.getLogger("sqfox.backends.usearch")


class USearchBackend:
    """VectorBackend using USearch HNSW index.

    Features:
      - O(log N) approximate nearest neighbor search
      - mmap readers via view() — minimal RAM on weak hardware
      - Quantization: f32, f16, i8, b1 via dtype param
      - Index file: app.db.usearch (alongside the .db)

    Thread-safety:
      - _writer_index: mutated only from writer thread
      - Readers: thread-local mmap views, refreshed by generation counter

    Args:
        metric:           Distance metric ('cos', 'ip', 'l2sq').
        dtype:            Storage type ('f32', 'f16', 'i8', 'b1').
        connectivity:     HNSW M parameter.
        expansion_add:    ef_construction.
        expansion_search: ef_search.
    """

    def __init__(
        self,
        *,
        metric: str = "cos",
        dtype: str = "f32",
        connectivity: int = 16,
        expansion_add: int = 128,
        expansion_search: int = 64,
    ) -> None:
        self._metric = metric
        self._dtype = dtype
        self._connectivity = connectivity
        self._expansion_add = expansion_add
        self._expansion_search = expansion_search

        self._ndim: int | None = None
        self._index_path: str | None = None
        self._writer_index = None  # usearch.index.Index
        self._reader_local = threading.local()
        self._generation = 0
        self._generation_lock = threading.Lock()
        self._writer_lock = threading.Lock()  # protects writer index for memory-mode reads
        self._initialized = False
        self._is_memory = False

    # -- lifecycle --

    def set_writer_conn(self, conn: Any) -> None:  # noqa: ARG002
        """Accept writer connection (unused — USearch manages its own files)."""

    def initialize(self, db_path: str, ndim: int) -> None:
        try:
            from usearch.index import Index
        except ImportError:
            raise ImportError(
                "USearch is required for the 'usearch' vector backend. "
                "Install with: pip install usearch"
            ) from None

        self._ndim = ndim
        self._is_memory = (
            db_path == ":memory:"
            or db_path.startswith("file::memory:")
            or db_path.startswith("file:sqfox_")
        )

        idx_kwargs = dict(
            ndim=ndim,
            metric=self._metric,
            dtype=self._dtype,
            connectivity=self._connectivity,
            expansion_add=self._expansion_add,
            expansion_search=self._expansion_search,
        )

        if self._is_memory:
            self._index_path = None
            self._writer_index = Index(**idx_kwargs)
        else:
            self._index_path = db_path + ".usearch"
            if os.path.exists(self._index_path):
                self._writer_index = Index(**idx_kwargs)
                self._writer_index.load(self._index_path)
                logger.info(
                    "Loaded USearch index %s (%d vectors)",
                    self._index_path, len(self._writer_index),
                )
            else:
                self._writer_index = Index(**idx_kwargs)
                logger.info(
                    "Created USearch index (ndim=%d, metric=%s, dtype=%s)",
                    ndim, self._metric, self._dtype,
                )

        self._initialized = True

    def close(self) -> None:
        if self._index_path and self._writer_index is not None:
            try:
                if len(self._writer_index) > 0:
                    self._writer_index.save(self._index_path)
            except Exception as exc:
                logger.warning("Failed to save USearch index on close: %s", exc)
        self._writer_index = None
        self._reader_local = threading.local()
        self._initialized = False

    # -- write path --

    def add(self, keys: list[int], vectors: list[list[float]]) -> None:
        if self._writer_index is None:
            raise RuntimeError("USearchBackend.add() called before initialize()")
        import numpy as np
        vecs_arr = np.array(vectors, dtype=np.float32)
        # Dimension validation
        if self._ndim is not None and vecs_arr.ndim == 2 and vecs_arr.shape[1] != self._ndim:
            raise ValueError(
                f"Vector dimension mismatch: expected {self._ndim}, "
                f"got {vecs_arr.shape[1]}"
            )
        # NaN/Inf guard
        if not np.all(np.isfinite(vecs_arr)):
            raise ValueError("Vectors contain NaN or Inf values")
        keys_arr = np.array(keys, dtype=np.int64)
        with self._writer_lock:
            self._writer_index.add(keys_arr, vecs_arr)

    def remove(self, keys: list[int]) -> None:
        if self._writer_index is None:
            raise RuntimeError("USearchBackend.remove() called before initialize()")
        import numpy as np
        keys_arr = np.array(keys, dtype=np.int64)
        with self._writer_lock:
            self._writer_index.remove(keys_arr)

    def flush(self) -> None:
        if self._writer_index is None or self._index_path is None:
            return
        self._writer_index.save(self._index_path)
        with self._generation_lock:
            self._generation += 1
        logger.debug(
            "Flushed USearch index to %s (gen=%d)",
            self._index_path, self._generation,
        )

    # -- read path --

    def search(
        self,
        query: list[float],
        k: int,
        **kwargs,
    ) -> list[tuple[int, float]]:
        reader = self._get_reader_index()
        if reader is None:
            # No mmap reader available — fall back to writer index.
            # Lock protects against concurrent add/remove on writer thread.
            with self._writer_lock:
                if self._writer_index is not None and len(self._writer_index) > 0:
                    return self._do_search(self._writer_index, query, k)
            return []
        return self._do_search(reader, query, k)

    def _do_search(self, index, query: list[float], k: int) -> list[tuple[int, float]]:
        import numpy as np
        q = np.array(query, dtype=np.float32)
        actual_k = min(k, len(index))
        if actual_k == 0:
            return []
        matches = index.search(q, actual_k)
        return [
            (int(matches.keys[i]), float(matches.distances[i]))
            for i in range(len(matches))
        ]

    def _get_reader_index(self):
        """Get or refresh the mmap reader for the current thread."""
        if self._index_path is None or not os.path.exists(self._index_path):
            return None

        reader = getattr(self._reader_local, "index", None)
        local_gen = getattr(self._reader_local, "generation", -1)

        with self._generation_lock:
            current_gen = self._generation

        if reader is not None and local_gen == current_gen:
            return reader

        try:
            from usearch.index import Index
            reader = Index.restore(self._index_path, view=True)
            self._reader_local.index = reader
            self._reader_local.generation = current_gen
            return reader
        except Exception as exc:
            logger.warning("Failed to open USearch mmap view: %s", exc)
            return None

    def count(self) -> int:
        if self._writer_index is None:
            return 0
        return len(self._writer_index)

    # -- crash recovery --

    def verify_consistency(self, expected_count: int) -> bool:
        """Check if index vector count matches SQLite vec_indexed count."""
        if self._writer_index is None:
            return expected_count == 0
        index_count = len(self._writer_index)
        if index_count != expected_count:
            logger.warning(
                "USearch count mismatch: index=%d, SQLite=%d",
                index_count, expected_count,
            )
            return False
        return True

    def reset(self, ndim: int | None = None) -> None:
        """Clear index state (for rebuild).  Optionally update ndim."""
        from usearch.index import Index

        effective_ndim = ndim if ndim is not None else self._ndim
        if effective_ndim is None:
            raise ValueError("ndim must be set before reset()")
        self._ndim = effective_ndim
        self._writer_index = Index(
            ndim=effective_ndim,
            metric=self._metric,
            dtype=self._dtype,
            connectivity=self._connectivity,
            expansion_add=self._expansion_add,
            expansion_search=self._expansion_search,
        )

    def rebuild_from_blobs(
        self,
        rows: list[tuple[int, bytes]],
        ndim: int,
    ) -> None:
        """Rebuild index from raw embedding blobs stored in SQLite.

        Processes rows in batches of 2000 to avoid OOM on large datasets.
        """
        import numpy as np

        logger.info("Rebuilding USearch index from %d blobs", len(rows))
        self.reset(ndim)

        expected_blob_size = ndim * 4
        BATCH = 2000
        for i in range(0, len(rows), BATCH):
            batch_rows = rows[i : i + BATCH]
            keys = []
            vecs = []
            for r in batch_rows:
                if len(r[1]) != expected_blob_size:
                    logger.warning(
                        "Skipping doc %d: blob size %d != expected %d",
                        r[0], len(r[1]), expected_blob_size,
                    )
                    continue
                keys.append(r[0])
                vecs.append(list(_struct.unpack(f"{ndim}f", r[1])))
            if keys:
                self._writer_index.add(
                    np.array(keys, dtype=np.int64),
                    np.array(vecs, dtype=np.float32),
                )

        if self._index_path:
            self._writer_index.save(self._index_path)
            with self._generation_lock:
                self._generation += 1

        logger.info("USearch index rebuilt: %d vectors", len(self._writer_index))
