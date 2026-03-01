"""Thread-safe SQLite engine with single-writer, multi-reader architecture."""

from __future__ import annotations

import logging
import queue
import sqlite3
import threading
import time
from concurrent.futures import Future
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator

from .types import (
    ChunkerFn,
    EmbedFn,
    Embedder,
    EngineClosedError,
    Priority,
    QueueFullError,
    SchemaState,
    SearchResult,
    SQFoxError,
    WriteRequest,
    embed_for_documents,
)

logger = logging.getLogger("sqfox.engine")

# Sentinel type for poison pill
_STOP = object()


class SQFox:
    """Main sqfox engine: thread-safe SQLite with single writer + multi reader.

    Usage::

        db = SQFox("mydata.db")
        db.start()
        db.write("INSERT INTO t VALUES (?)", (42,))
        rows = db.fetch_all("SELECT * FROM t")
        db.stop()

    Or as a context manager::

        with SQFox("mydata.db") as db:
            db.write("INSERT INTO t VALUES (?)", (42,))
    """

    def __init__(
        self,
        path: str | Path,
        *,
        max_queue_size: int = 10_000,
        batch_size: int = 64,
        batch_time_ms: float = 50.0,
        cache_size_kb: int = 64_000,
        busy_timeout_ms: int = 5_000,
        enable_vec: bool = True,
        error_callback: Callable[[str, Exception], None] | None = None,
    ) -> None:
        self._path = str(path)
        self._max_queue_size = max_queue_size
        self._batch_size = batch_size
        self._batch_time_ms = batch_time_ms
        self._cache_size_kb = cache_size_kb
        self._busy_timeout_ms = busy_timeout_ms
        self._enable_vec = enable_vec
        self._error_callback = error_callback

        # Writer state
        self._writer_conn: sqlite3.Connection | None = None
        self._writer_thread: threading.Thread | None = None
        self._queue: queue.PriorityQueue[tuple[int, int, WriteRequest | object | None]] = (
            queue.PriorityQueue(maxsize=max_queue_size)
        )
        self._seq = 0
        self._seq_lock = threading.Lock()

        # Reader state: {thread_id: connection}
        self._local = threading.local()
        self._reader_connections: dict[int, sqlite3.Connection] = {}
        self._reader_lock = threading.Lock()

        # Lifecycle
        self._running = threading.Event()
        self._stopped = False
        self._vec_available = False

        # Hooks
        self._on_startup_hooks: list[Callable[[SQFox], None]] = []

    # ------------------------------------------------------------------
    # Error notification
    # ------------------------------------------------------------------

    def _notify_error(self, sql: str, exc: Exception) -> None:
        """Notify about a write error via callback and logging."""
        logger.error("Write error [%s]: %s", sql[:80], exc)
        if self._error_callback is not None:
            try:
                self._error_callback(sql, exc)
            except Exception:
                logger.exception("error_callback raised an exception")

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    def _configure_connection(self, conn: sqlite3.Connection) -> None:
        """Apply PRAGMA tuning to a connection."""
        is_file = self._path != ":memory:" and not self._path.startswith("file::memory:")

        if is_file:
            conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute(f"PRAGMA busy_timeout={self._busy_timeout_ms}")
        conn.execute("PRAGMA temp_store=MEMORY")
        conn.execute(f"PRAGMA cache_size=-{self._cache_size_kb}")
        if is_file:
            conn.execute("PRAGMA mmap_size=268435456")  # 256 MB
        conn.execute("PRAGMA foreign_keys=ON")

    def _try_load_vec(self, conn: sqlite3.Connection) -> bool:
        """Attempt to load sqlite-vec extension. Returns True on success."""
        if not self._enable_vec:
            return False
        try:
            conn.enable_load_extension(True)
            import sqlite_vec
            sqlite_vec.load(conn)
            conn.enable_load_extension(False)
            return True
        except ImportError:
            logger.warning(
                "sqlite-vec not installed. Vector search disabled. "
                "Install with: pip install sqlite-vec\n"
                "Supported platforms: Linux x86-64/ARM64, "
                "macOS Intel/Apple Silicon, Windows x86-64.\n"
                "Alpine Linux (musl), 32-bit systems, and Windows ARM64 "
                "are NOT supported."
            )
            return False
        except AttributeError:
            logger.warning(
                "sqlite-vec installed but enable_load_extension() not "
                "available. This Python build does not support SQLite "
                "extensions. On macOS, use Homebrew Python: "
                "brew install python"
            )
            return False
        except sqlite3.OperationalError as exc:
            logger.warning("Failed to load sqlite-vec: %s", exc)
            return False

    def _create_writer_connection(self) -> sqlite3.Connection:
        """Create and configure the writer connection."""
        conn = sqlite3.connect(self._path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        self._configure_connection(conn)
        self._vec_available = self._try_load_vec(conn)
        return conn

    def _get_reader_connection(self) -> sqlite3.Connection:
        """Get or create a reader connection for the current thread."""
        if self._stopped:
            raise EngineClosedError("Engine is stopped, cannot create reader connection")

        conn = getattr(self._local, "conn", None)
        if conn is not None:
            return conn

        if self._stopped:
            raise EngineClosedError("Engine is stopped, cannot create reader connection")

        conn = sqlite3.connect(self._path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        self._configure_connection(conn)
        conn.execute("PRAGMA query_only=ON")

        if self._vec_available:
            self._try_load_vec(conn)

        tid = threading.get_ident()
        with self._reader_lock:
            self._reader_connections[tid] = conn
            # Periodically prune connections from dead threads (every 50 connections)
            if len(self._reader_connections) % 50 == 0:
                alive_ids = {t.ident for t in threading.enumerate()}
                dead_ids = [k for k in self._reader_connections if k not in alive_ids]
                for dead_id in dead_ids:
                    try:
                        self._reader_connections.pop(dead_id).close()
                    except Exception:
                        pass

        self._local.conn = conn
        return conn

    # ------------------------------------------------------------------
    # Writer thread
    # ------------------------------------------------------------------

    def _next_seq(self) -> int:
        """Thread-safe monotonic sequence counter."""
        with self._seq_lock:
            self._seq += 1
            return self._seq

    def _writer_loop(self) -> None:
        """Main loop for the writer thread. Processes WriteRequests from the queue."""
        assert self._writer_conn is not None

        try:
            while self._running.is_set() or not self._queue.empty():
                batch: list[WriteRequest] = []
                deadline = time.monotonic() + (self._batch_time_ms / 1000.0)

                # Drain up to batch_size items, waiting up to batch_time_ms
                while len(batch) < self._batch_size:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        break
                    try:
                        item = self._queue.get(timeout=max(remaining, 0.001))
                        _priority, _seq, request = item
                        if request is _STOP:
                            # Poison pill: process remaining batch, then exit
                            if batch:
                                self._execute_batch(batch)
                            return
                        assert isinstance(request, WriteRequest)
                        batch.append(request)
                    except queue.Empty:
                        break

                if batch:
                    self._execute_batch(batch)

        except Exception as exc:
            logger.error("Writer thread crashed: %s", exc, exc_info=True)
            # Drain remaining items and set exceptions on their futures
            while not self._queue.empty():
                try:
                    _, _, request = self._queue.get_nowait()
                    if isinstance(request, WriteRequest):
                        crash_exc = SQFoxError(f"Writer thread crashed: {exc}")
                        self._notify_error(request.sql, crash_exc)
                        if request.future is not None:
                            request.future.set_exception(crash_exc)
                except queue.Empty:
                    break
        finally:
            self._running.clear()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the writer thread and run startup hooks."""
        if self._stopped:
            raise EngineClosedError("Engine has been stopped and cannot be restarted")
        if self._running.is_set():
            return  # Already running

        self._writer_conn = self._create_writer_connection()
        self._running.set()
        self._writer_thread = threading.Thread(
            target=self._writer_loop,
            name="sqfox-writer",
            daemon=False,
        )
        self._writer_thread.start()

        # Run startup hooks
        for hook in self._on_startup_hooks:
            try:
                hook(self)
            except Exception as exc:
                logger.error("Startup hook failed: %s", exc, exc_info=True)
                self.stop()
                raise

    def stop(self, timeout: float = 10.0) -> None:
        """Gracefully shut down: drain queue, close all connections."""
        if not self._running.is_set() and self._writer_thread is None:
            self._stopped = True
            return

        # Signal writer to stop — set stopped first to reject new writes
        self._stopped = True
        self._running.clear()

        # Send poison pill with highest priority
        try:
            self._queue.put_nowait((Priority.HIGH, 0, _STOP))
        except queue.Full:
            logger.warning("Queue full during shutdown, forcing stop")

        # Wait for writer thread
        if self._writer_thread is not None:
            self._writer_thread.join(timeout=timeout)
            if self._writer_thread.is_alive():
                logger.warning(
                    "Writer thread did not stop within %.1fs, "
                    "deferring connection close", timeout
                )
                # Do NOT close connection while writer is still using it
                self._writer_thread = None
            else:
                self._writer_thread = None
                # Safe to close — writer thread has exited
                if self._writer_conn is not None:
                    try:
                        self._writer_conn.close()
                    except Exception:
                        pass
                    self._writer_conn = None
        else:
            if self._writer_conn is not None:
                try:
                    self._writer_conn.close()
                except Exception:
                    pass
                self._writer_conn = None

        # Drain any orphaned items (from write() calls that raced with stop())
        while not self._queue.empty():
            try:
                _, _, request = self._queue.get_nowait()
                if isinstance(request, WriteRequest) and request.future is not None:
                    request.future.set_exception(
                        EngineClosedError("Engine stopped before request was processed")
                    )
            except queue.Empty:
                break

        # Close all reader connections
        with self._reader_lock:
            for conn in self._reader_connections.values():
                try:
                    conn.close()
                except Exception:
                    pass
            self._reader_connections.clear()

    def __enter__(self) -> SQFox:
        self.start()
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.stop()

    # ------------------------------------------------------------------
    # Public API — hooks
    # ------------------------------------------------------------------

    def on_startup(self, hook: Callable[[SQFox], None]) -> Callable[[SQFox], None]:
        """Decorator to register a startup hook.

        Hooks receive the SQFox instance and run after the writer thread starts.
        Use them to create tables, run migrations, etc.
        """
        self._on_startup_hooks.append(hook)
        return hook

    # ------------------------------------------------------------------
    # Public API — writes
    # ------------------------------------------------------------------

    def write(
        self,
        sql: str,
        params: tuple[Any, ...] | list[tuple[Any, ...]] = (),
        *,
        priority: Priority = Priority.NORMAL,
        wait: bool = False,
        many: bool = False,
    ) -> Future[Any] | None:
        """Submit a write request to the writer queue.

        Args:
            sql:      SQL statement.
            params:   Bind parameters.
            priority: Queue priority.
            wait:     If True, block until the write completes and return
                      the result.  Raises any exception from the writer.
            many:     If True, use executemany.

        Returns:
            Future if wait=False, else the direct result (lastrowid).

        Raises:
            EngineClosedError: If engine is stopped.
            QueueFullError:    If queue is at max capacity.
        """
        if self._stopped or not self._running.is_set():
            raise EngineClosedError("Engine is not running")

        future: Future[Any] = Future()
        request = WriteRequest(
            sql=sql,
            params=params,
            priority=priority,
            future=future,
            many=many,
        )

        seq = self._next_seq()
        try:
            self._queue.put_nowait((priority, seq, request))
        except queue.Full:
            raise QueueFullError(
                f"Write queue is full ({self._max_queue_size} items)"
            ) from None

        if wait:
            return future.result()
        return future

    def execute_on_writer(
        self,
        fn: Callable[[sqlite3.Connection], Any],
        *,
        priority: Priority = Priority.HIGH,
        wait: bool = True,
    ) -> Any:
        """Execute a callable on the writer connection.

        Useful for schema migrations and operations that need
        direct connection access (e.g., loading extensions).

        The callable receives the writer's sqlite3.Connection.
        """
        if self._stopped or not self._running.is_set():
            raise EngineClosedError("Engine is not running")

        future: Future[Any] = Future()

        # We wrap the callable into a special WriteRequest with a sentinel SQL
        # and handle it in _execute_batch by detecting the callable
        request = WriteRequest(
            sql="__EXEC_FN__",
            params=fn,  # type: ignore[arg-type]
            priority=priority,
            future=future,
        )

        seq = self._next_seq()
        try:
            self._queue.put_nowait((priority, seq, request))
        except queue.Full:
            raise QueueFullError(
                f"Write queue is full ({self._max_queue_size} items)"
            ) from None

        if wait:
            return future.result()
        return future

    def _execute_batch(self, batch: list[WriteRequest]) -> None:
        """Execute a batch of write requests in a single transaction."""
        assert self._writer_conn is not None

        # Separate callable requests from SQL requests
        fn_requests = [r for r in batch if r.sql == "__EXEC_FN__"]
        sql_requests = [r for r in batch if r.sql != "__EXEC_FN__"]

        # Execute callable requests (each in its own transaction)
        for req in fn_requests:
            try:
                fn = req.params
                assert callable(fn)
                result = fn(self._writer_conn)
                if req.future is not None:
                    req.future.set_result(result)
            except Exception as exc:
                # Rollback any uncommitted work from the callable
                try:
                    self._writer_conn.rollback()
                except Exception:
                    pass
                self._notify_error(req.sql, exc)
                if req.future is not None:
                    req.future.set_exception(exc)

        # Execute SQL requests in a batch transaction
        if not sql_requests:
            return

        try:
            self._writer_conn.execute("BEGIN IMMEDIATE")
        except Exception as exc:
            logger.error("Failed to begin transaction: %s", exc)
            for req in sql_requests:
                self._notify_error(req.sql, exc)
                if req.future is not None:
                    req.future.set_exception(exc)
            return

        # Collect results — only resolve futures AFTER successful commit
        results: list[tuple[WriteRequest, Any]] = []

        for i, req in enumerate(sql_requests):
            try:
                if req.many:
                    cursor = self._writer_conn.executemany(req.sql, req.params)
                else:
                    cursor = self._writer_conn.execute(req.sql, req.params)
                results.append((req, cursor.lastrowid))
            except Exception as exc:
                try:
                    self._writer_conn.rollback()
                except Exception:
                    pass
                # Failed request + all remaining get exceptions
                self._notify_error(req.sql, exc)
                if req.future is not None:
                    req.future.set_exception(exc)
                for already_done_req, _ in results:
                    rollback_exc = SQFoxError(
                        f"Batch rolled back due to error in later statement: {exc}"
                    )
                    if already_done_req.future is not None:
                        already_done_req.future.set_exception(rollback_exc)
                for remaining in sql_requests[i + 1:]:
                    abort_exc = SQFoxError(f"Batch aborted due to prior error: {exc}")
                    self._notify_error(remaining.sql, abort_exc)
                    if remaining.future is not None:
                        remaining.future.set_exception(abort_exc)
                return

        try:
            self._writer_conn.commit()
        except Exception as exc:
            logger.error("Transaction commit failed: %s", exc)
            try:
                self._writer_conn.rollback()
            except Exception:
                pass
            for req in sql_requests:
                self._notify_error(req.sql, exc)
                if req.future is not None and not req.future.done():
                    req.future.set_exception(exc)
            return

        # Commit succeeded — now resolve all futures
        for req, lastrowid in results:
            if req.future is not None:
                req.future.set_result(lastrowid)

    # ------------------------------------------------------------------
    # Public API — reads
    # ------------------------------------------------------------------

    def fetch_one(
        self,
        sql: str,
        params: tuple[Any, ...] = (),
    ) -> sqlite3.Row | None:
        """Execute a read query and return the first row."""
        conn = self._get_reader_connection()
        cursor = conn.execute(sql, params)
        return cursor.fetchone()

    def fetch_all(
        self,
        sql: str,
        params: tuple[Any, ...] = (),
    ) -> list[sqlite3.Row]:
        """Execute a read query and return all rows."""
        conn = self._get_reader_connection()
        cursor = conn.execute(sql, params)
        return cursor.fetchall()

    @contextmanager
    def reader(self) -> Iterator[sqlite3.Connection]:
        """Context manager for direct reader connection access.

        Useful for complex queries or when you need cursor control.
        """
        yield self._get_reader_connection()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def vec_available(self) -> bool:
        """Whether sqlite-vec extension was loaded successfully."""
        return self._vec_available

    @property
    def is_running(self) -> bool:
        """Whether the engine is currently active."""
        return self._running.is_set()

    @property
    def queue_size(self) -> int:
        """Current number of pending write requests."""
        return self._queue.qsize()

    @property
    def path(self) -> str:
        """Database file path."""
        return self._path

    def diagnostics(self) -> dict[str, Any]:
        """Return diagnostic info about the engine and its capabilities.

        Useful for debugging platform-specific issues (e.g., sqlite-vec
        not loading on Alpine Linux).
        """
        import platform
        info: dict[str, Any] = {
            "sqfox_version": "0.1.0",
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "sqlite_version": sqlite3.sqlite_version,
            "path": self._path,
            "is_running": self.is_running,
            "vec_available": self._vec_available,
            "queue_size": self.queue_size,
            "max_queue_size": self._max_queue_size,
            "batch_size": self._batch_size,
            "batch_time_ms": self._batch_time_ms,
        }

        # Check optional deps
        try:
            import sqlite_vec
            info["sqlite_vec_version"] = getattr(sqlite_vec, "__version__", "installed")
        except ImportError:
            info["sqlite_vec_version"] = None

        try:
            import simplemma
            info["simplemma_version"] = getattr(simplemma, "__version__", "installed")
        except ImportError:
            info["simplemma_version"] = None

        try:
            import pymorphy3
            info["pymorphy3_version"] = getattr(pymorphy3, "__version__", "installed")
        except ImportError:
            info["pymorphy3_version"] = None

        return info

    # ------------------------------------------------------------------
    # High-level API — schema, ingest, search
    # ------------------------------------------------------------------

    def ensure_schema(
        self,
        target: SchemaState = SchemaState.BASE,
        *,
        vec_dimension: int | None = None,
    ) -> SchemaState:
        """Ensure schema is at least at the target state.

        Runs on the writer connection.  Blocks until complete.
        """
        from .schema import migrate_to

        def _do_migrate(conn: sqlite3.Connection) -> SchemaState:
            return migrate_to(conn, target, vec_dimension=vec_dimension)

        return self.execute_on_writer(_do_migrate, priority=Priority.HIGH, wait=True)

    def ingest(
        self,
        content: str,
        *,
        metadata: dict[str, Any] | None = None,
        chunker: ChunkerFn | None = None,
        embed_fn: EmbedFn | None = None,
        priority: Priority = Priority.NORMAL,
        wait: bool = False,
    ) -> Future[int] | int:
        """Ingest a document (optionally chunked and embedded).

        If chunker is provided, splits content into chunks and inserts each
        as a child document with chunk_of pointing to the parent.

        If embed_fn is provided, computes embeddings and stores in vec0.
        Also lemmatizes content for FTS if a tokenizer is available.

        Returns the parent document ID.
        """
        import json as _json
        import struct as _struct
        from .schema import (
            detect_state,
            migrate_to,
            validate_dimension,
        )
        from .tokenizer import lemmatize

        meta_str = _json.dumps(metadata) if metadata else "{}"

        def _do_ingest(conn: sqlite3.Connection) -> int:
            # Ensure at least BASE schema
            current = detect_state(conn)
            if current < SchemaState.BASE:
                migrate_to(conn, SchemaState.BASE)

            # Insert parent document
            cursor = conn.execute(
                "INSERT INTO documents (content, metadata) VALUES (?, ?)",
                (content, meta_str),
            )
            parent_id = cursor.lastrowid
            assert parent_id is not None

            # Determine chunks
            if chunker is not None:
                chunks = chunker(content)
                if not chunks:
                    chunks = [content]  # Fallback: empty chunker result
            else:
                chunks = [content]

            # Process chunks
            chunk_ids: list[int] = []
            for chunk_text in chunks:
                if chunker is not None:
                    # Insert chunk as child document
                    cur = conn.execute(
                        "INSERT INTO documents (content, metadata, chunk_of) "
                        "VALUES (?, ?, ?)",
                        (chunk_text, meta_str, parent_id),
                    )
                    chunk_id = cur.lastrowid
                    assert chunk_id is not None
                    chunk_ids.append(chunk_id)
                else:
                    chunk_ids.append(parent_id)

            # Lemmatize
            for cid, chunk_text in zip(chunk_ids, chunks):
                try:
                    lemmatized = lemmatize(chunk_text)
                    conn.execute(
                        "UPDATE documents SET content_lemmatized = ? WHERE id = ?",
                        (lemmatized, cid),
                    )
                except Exception as exc:
                    logger.warning("Lemmatization failed for doc %d: %s", cid, exc)

            # Embed if embed_fn provided
            if embed_fn is not None:
                embeddings = embed_for_documents(embed_fn, chunks)

                # Validate dimension
                if embeddings:
                    dim = len(embeddings[0])
                    validate_dimension(conn, dim, commit=False)

                    # Ensure INDEXED schema
                    current = detect_state(conn)
                    if current < SchemaState.INDEXED:
                        migrate_to(conn, SchemaState.INDEXED, vec_dimension=dim)

                    for cid, emb in zip(chunk_ids, embeddings):
                        vec_blob = _struct.pack(f"{len(emb)}f", *emb)
                        conn.execute(
                            "INSERT OR REPLACE INTO documents_vec(rowid, embedding) "
                            "VALUES (?, ?)",
                            (cid, vec_blob),
                        )
                        conn.execute(
                            "UPDATE documents SET vec_indexed = 1 WHERE id = ?",
                            (cid,),
                        )

            # Ensure FTS if SEARCHABLE or ENRICHED
            current = detect_state(conn)
            if current >= SchemaState.SEARCHABLE:
                for cid in chunk_ids:
                    row = conn.execute(
                        "SELECT content_lemmatized FROM documents WHERE id = ?",
                        (cid,),
                    ).fetchone()
                    if row and row[0] and current < SchemaState.ENRICHED:
                        # Manual FTS insert (no triggers)
                        conn.execute(
                            "INSERT INTO documents_fts(rowid, content_lemmatized) "
                            "VALUES (?, ?)",
                            (cid, row[0]),
                        )
                        conn.execute(
                            "UPDATE documents SET fts_indexed = 1 WHERE id = ?",
                            (cid,),
                        )

            conn.commit()
            return parent_id

        return self.execute_on_writer(
            _do_ingest, priority=priority, wait=wait
        )

    def search(
        self,
        query: str,
        *,
        embed_fn: EmbedFn | None = None,
        limit: int = 10,
        alpha: float | None = None,
    ) -> list[SearchResult]:
        """Search documents using hybrid FTS5 + vector search.

        If embed_fn is None, performs FTS-only search.
        """
        from .search import hybrid_search, fts_search
        from .tokenizer import lemmatize_query

        conn = self._get_reader_connection()

        if embed_fn is not None:
            return hybrid_search(
                conn,
                query,
                embed_fn,
                lemmatize_fn=lemmatize_query,
                limit=limit,
                alpha=alpha,
            )
        else:
            # FTS-only search
            query_lemmatized = lemmatize_query(query, None)
            try:
                fts_results = fts_search(conn, query_lemmatized, limit=limit)
            except Exception as exc:
                logger.warning("FTS search failed: %s", exc)
                return []

            if not fts_results:
                return []

            # Hydrate
            import json as _json
            doc_ids = [doc_id for doc_id, _ in fts_results]
            score_map = {doc_id: score for doc_id, score in fts_results}
            placeholders = ",".join(["?"] * len(doc_ids))
            rows = conn.execute(
                f"SELECT id, content, metadata, chunk_of FROM documents "
                f"WHERE id IN ({placeholders})",
                doc_ids,
            ).fetchall()
            doc_map = {row[0]: row for row in rows}

            results = []
            for doc_id, score in fts_results:
                if doc_id not in doc_map:
                    continue
                row = doc_map[doc_id]
                try:
                    meta = _json.loads(row[2]) if row[2] else {}
                except (ValueError, TypeError):
                    meta = {}
                results.append(SearchResult(
                    doc_id=row[0],
                    score=score,
                    text=row[1],
                    metadata=meta,
                    chunk_id=row[3],
                ))
            return results[:limit]
