"""Multi-database manager for sqfox."""

from __future__ import annotations

import logging
import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
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
        # Sanitize name to prevent path traversal
        if not re.match(r"^[a-zA-Z0-9_-]+$", name):
            raise ValueError(
                f"Invalid database name: {name!r}. "
                "Only alphanumeric characters, hyphens, and underscores are allowed."
            )

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
        with self._db_lock:
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
            dbs = list(self._databases.items())
            self._databases.clear()
            self._started = False
        for name, db in dbs:
            if db.is_running:
                db.stop(timeout=timeout)
                logger.info("Stopped database '%s'", name)

    def __enter__(self) -> SQFoxManager:
        self.start()
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.stop()

    def __del__(self) -> None:
        if self._started and self._databases:
            logger.warning(
                "SQFoxManager was garbage-collected while still running. "
                "Call stop() explicitly or use the context manager."
            )
            try:
                self.stop(timeout=2.0)
            except Exception:
                pass

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
        """Search across all databases in parallel and merge results.

        Returns:
            List of (db_name, SearchResult) tuples, sorted by score descending.
        """
        with self._db_lock:
            snapshot = list(self._databases.items())

        if not snapshot:
            return []

        def _search_one(name: str, db: SQFox) -> list[tuple[str, SearchResult]]:
            if not db.is_running:
                return []
            # Skip databases without documents table
            try:
                row = db.fetch_one(
                    "SELECT 1 FROM sqlite_master WHERE type='table' AND name='documents'"
                )
                if row is None:
                    return []
            except Exception:
                return []
            try:
                results = db.search(
                    query, embed_fn=embed_fn, limit=limit, alpha=alpha,
                )
                return [(name, r) for r in results]
            except Exception as exc:
                logger.warning("Search failed in '%s': %s", name, exc)
                return []

        all_results: list[tuple[str, SearchResult]] = []

        with ThreadPoolExecutor(max_workers=min(len(snapshot), 8) or 1) as pool:
            futures = {
                pool.submit(_search_one, name, db): name
                for name, db in snapshot
            }
            for future in as_completed(futures):
                all_results.extend(future.result())

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
        with self._db_lock:
            return dict(self._databases)

    @property
    def names(self) -> list[str]:
        with self._db_lock:
            return list(self._databases.keys())

    def drop(self, name: str, *, delete_file: bool = False) -> None:
        """Stop and remove a database.

        Args:
            name:        Database name.
            delete_file: If True, also delete the .db file and WAL/SHM files.
        """
        # Sanitize name to prevent path traversal
        if not re.match(r"^[a-zA-Z0-9_-]+$", name):
            raise ValueError(
                f"Invalid database name: {name!r}. "
                "Only alphanumeric characters, hyphens, and underscores are allowed."
            )

        with self._db_lock:
            if name not in self._databases:
                return
            db = self._databases.pop(name)

        db_path = db.path
        if db.is_running:
            db.stop()

        resolved = Path(db_path).resolve()
        base_resolved = self._base_dir.resolve()
        if base_resolved not in resolved.parents and resolved != base_resolved:
            raise ValueError("Database path escaped base directory")

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
