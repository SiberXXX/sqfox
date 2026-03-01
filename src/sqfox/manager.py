"""Multi-database manager for sqfox."""

from __future__ import annotations

import logging
import os
import threading
from pathlib import Path
from typing import Any

from .engine import SQFox
from .types import (
    ChunkerFn,
    EmbedFn,
    Priority,
    SchemaState,
    SearchResult,
)

logger = logging.getLogger("sqfox.manager")


class SQFoxManager:
    """Manages multiple SQFox databases in a single directory.

    Each database is a separate .db file, isolated with its own
    writer thread and WAL.  Databases are created lazily on first access.

    Usage::

        manager = SQFoxManager("./databases")
        manager.start()

        iot = manager["sensors"]
        rag = manager["knowledge"]

        iot.write("INSERT INTO ...", (...,))
        results = manager.search_all("query", embed_fn=fn)

        manager.stop()

    Or as a context manager::

        with SQFoxManager("./databases") as mgr:
            mgr["sensors"].write(...)
    """

    def __init__(
        self,
        base_dir: str | Path,
        *,
        db_ext: str = ".db",
        **default_kwargs: Any,
    ) -> None:
        """
        Args:
            base_dir:        Directory where .db files are stored.
            db_ext:          File extension for databases.
            **default_kwargs: Default kwargs passed to each SQFox instance
                              (max_queue_size, batch_size, etc.).
        """
        self._base_dir = Path(base_dir)
        self._db_ext = db_ext
        self._default_kwargs = default_kwargs
        self._databases: dict[str, SQFox] = {}
        self._db_lock = threading.Lock()
        self._started = False

    # ------------------------------------------------------------------
    # Database access
    # ------------------------------------------------------------------

    def get_or_create(self, name: str, **kwargs: Any) -> SQFox:
        """Get an existing database or create a new one.

        Thread-safe: concurrent calls for the same name return the same instance.

        Args:
            name:    Logical name (becomes filename: {name}{db_ext}).
            **kwargs: Override default SQFox kwargs for this database.

        Returns:
            Running SQFox instance.
        """
        with self._db_lock:
            if name in self._databases:
                return self._databases[name]

            # Ensure base directory exists
            self._base_dir.mkdir(parents=True, exist_ok=True)

            db_path = self._base_dir / f"{name}{self._db_ext}"
            merged_kwargs = {**self._default_kwargs, **kwargs}
            db = SQFox(db_path, **merged_kwargs)

            if self._started:
                db.start()

            self._databases[name] = db
            logger.info("Database '%s' opened at %s", name, db_path)
            return db

    def __getitem__(self, name: str) -> SQFox:
        """Shorthand: manager['sensors'] == manager.get_or_create('sensors')."""
        return self.get_or_create(name)

    def __contains__(self, name: str) -> bool:
        """Check if a database with this name is already open."""
        return name in self._databases

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start all existing databases."""
        with self._db_lock:
            self._started = True
            for name, db in self._databases.items():
                if not db.is_running:
                    db.start()
                    logger.info("Started database '%s'", name)

    def stop(self, timeout: float = 10.0) -> None:
        """Stop all databases gracefully."""
        with self._db_lock:
            for name, db in list(self._databases.items()):
                if db.is_running:
                    db.stop(timeout=timeout)
                    logger.info("Stopped database '%s'", name)
            self._databases.clear()
            self._started = False

    def __enter__(self) -> SQFoxManager:
        self.start()
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.stop()

    # ------------------------------------------------------------------
    # Cross-database operations
    # ------------------------------------------------------------------

    def search_all(
        self,
        query: str,
        *,
        embed_fn: EmbedFn | None = None,
        limit: int = 10,
        alpha: float | None = None,
    ) -> list[tuple[str, SearchResult]]:
        """Search across all databases and merge results.

        Returns:
            List of (db_name, SearchResult) tuples, sorted by score descending.
        """
        all_results: list[tuple[str, SearchResult]] = []

        with self._db_lock:
            snapshot = list(self._databases.items())

        for name, db in snapshot:
            if not db.is_running:
                continue
            # Skip databases without documents table
            try:
                row = db.fetch_one(
                    "SELECT 1 FROM sqlite_master WHERE type='table' AND name='documents'"
                )
                if row is None:
                    continue
            except Exception:
                continue
            try:
                results = db.search(
                    query, embed_fn=embed_fn, limit=limit, alpha=alpha,
                )
                for r in results:
                    all_results.append((name, r))
            except Exception as exc:
                logger.warning("Search failed in '%s': %s", name, exc)

        # Sort by score descending, take top `limit`
        all_results.sort(key=lambda x: x[1].score, reverse=True)
        return all_results[:limit]

    def ingest_to(
        self,
        name: str,
        content: str,
        *,
        metadata: dict[str, Any] | None = None,
        chunker: ChunkerFn | None = None,
        embed_fn: EmbedFn | None = None,
        priority: Priority = Priority.NORMAL,
        wait: bool = False,
    ) -> Any:
        """Ingest a document into a specific database.

        Creates the database if it doesn't exist.
        """
        db = self.get_or_create(name)
        return db.ingest(
            content,
            metadata=metadata,
            chunker=chunker,
            embed_fn=embed_fn,
            priority=priority,
            wait=wait,
        )

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def databases(self) -> dict[str, SQFox]:
        """All open databases {name: SQFox}."""
        return dict(self._databases)

    @property
    def names(self) -> list[str]:
        """Names of all open databases."""
        return list(self._databases.keys())

    def drop(self, name: str, *, delete_file: bool = False) -> None:
        """Stop and remove a database.

        Args:
            name:        Database name.
            delete_file: If True, also delete the .db file and WAL/SHM files.
        """
        with self._db_lock:
            if name not in self._databases:
                return
            db = self._databases.pop(name)

        db_path = db.path
        if db.is_running:
            db.stop()

        if delete_file:
            for suffix in ("", "-wal", "-shm"):
                p = Path(db_path + suffix)
                if p.exists():
                    try:
                        p.unlink()
                        logger.info("Deleted %s", p)
                    except OSError as exc:
                        logger.warning("Failed to delete %s: %s", p, exc)

        logger.info("Dropped database '%s'", name)
