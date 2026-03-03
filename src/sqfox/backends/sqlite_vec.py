"""sqlite-vec backend — brute-force O(N) vector search via SQLite virtual table."""

from __future__ import annotations

import logging
import sqlite3
import struct

logger = logging.getLogger("sqfox.backends.sqlite_vec")


class SqliteVecBackend:
    """VectorBackend wrapping the existing sqlite-vec (vec0) logic.

    Vectors live inside SQLite via the vec0 virtual table.  Search is
    brute-force O(N) but benefits from SQLite ACID guarantees and zero
    external files.

    Thread-safety:
      - add/remove use the writer connection (writer thread only).
      - search uses the caller's reader connection (passed as conn kwarg).
    """

    def __init__(self) -> None:
        self._ndim: int | None = None
        self._writer_conn: sqlite3.Connection | None = None
        self._initialized = False

    # -- lifecycle --

    def initialize(self, db_path: str, ndim: int) -> None:
        self._ndim = ndim
        self._initialized = True

    def set_writer_conn(self, conn: sqlite3.Connection) -> None:
        """Inject the writer connection (called by engine on start)."""
        self._writer_conn = conn

    def close(self) -> None:
        self._writer_conn = None
        self._initialized = False

    # -- write path --

    def add(self, keys: list[int], vectors: list[list[float]]) -> None:
        assert self._writer_conn is not None
        assert self._ndim is not None
        for key, vec in zip(keys, vectors):
            blob = struct.pack(f"{self._ndim}f", *vec)
            self._writer_conn.execute(
                "INSERT OR REPLACE INTO documents_vec(rowid, embedding) "
                "VALUES (?, ?)",
                (key, blob),
            )

    def remove(self, keys: list[int]) -> None:
        assert self._writer_conn is not None
        for key in keys:
            self._writer_conn.execute(
                "DELETE FROM documents_vec WHERE rowid = ?", (key,),
            )

    def flush(self) -> None:
        pass  # SQLite handles persistence

    # -- read path --

    def search(
        self,
        query: list[float],
        k: int,
        **kwargs,
    ) -> list[tuple[int, float]]:
        """KNN search via sqlite-vec.  Requires conn= kwarg."""
        conn: sqlite3.Connection | None = kwargs.get("conn")
        if conn is None:
            raise ValueError("SqliteVecBackend.search() requires conn= kwarg")
        assert self._ndim is not None
        blob = struct.pack(f"{self._ndim}f", *query)
        try:
            rows = conn.execute(
                "SELECT rowid, distance FROM documents_vec "
                "WHERE embedding MATCH ? AND k = ?",
                (blob, k),
            ).fetchall()
        except sqlite3.OperationalError as exc:
            logger.warning("sqlite-vec search failed: %s", exc)
            return []
        return [(row[0], row[1]) for row in rows]

    def count(self) -> int:
        if self._writer_conn is None:
            return 0
        try:
            row = self._writer_conn.execute(
                "SELECT COUNT(*) FROM documents_vec",
            ).fetchone()
            return row[0] if row else 0
        except sqlite3.OperationalError:
            return 0
