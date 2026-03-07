"""Thread-safe SQLite engine with single-writer, multi-reader architecture."""

from __future__ import annotations

import atexit
import logging
import queue
import sqlite3
import struct
import sys
import threading
import time
from concurrent.futures import Future, InvalidStateError
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator

from .types import (
    ChunkerFn,
    EmbedFn,
    EngineClosedError,
    Priority,
    QueueFullError,
    RerankerFn,
    SchemaState,
    SearchResult,
    SQFoxError,
    WriteRequest,
    embed_for_documents,
)

logger = logging.getLogger("sqfox.engine")


def _get_version() -> str:
    """Get package version without circular import."""
    try:
        from importlib.metadata import version
        return version("sqfox")
    except Exception:
        return "unknown"


# Sentinel type for poison pill
_STOP = object()


def _safe_set_result(future: Future, value: Any) -> None:
    """Set result on a Future, ignoring cancelled futures."""
    try:
        future.set_result(value)
    except InvalidStateError:
        pass  # Future was cancelled — nothing to do


def _safe_set_exception(future: Future, exc: BaseException) -> None:
    """Set exception on a Future, ignoring cancelled futures."""
    try:
        future.set_exception(exc)
    except InvalidStateError:
        pass  # Future was cancelled — nothing to do


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
        cache_size_kb: int | str = "auto",
        mmap_size_mb: int | str = "auto",
        busy_timeout_ms: int = 5_000,
        vector_backend: str | Any | None = None,
        error_callback: Callable[[str, Exception], None] | None = None,
    ) -> None:
        self._path = str(path)
        # Shared-cache URI for :memory: databases so readers see writer's data
        if self._path == ":memory:" or self._path.startswith("file::memory:"):
            self._shared_mem_uri = f"file:sqfox_{id(self)}?mode=memory&cache=shared"
        else:
            self._shared_mem_uri = None
        self._max_queue_size = max_queue_size
        self._base_batch_size = max(1, batch_size)
        self._batch_size = max(1, batch_size)
        self._batch_time_ms = batch_time_ms
        # Raw values: may be AUTO sentinel.  Resolved in start().
        self._cache_size_kb_raw = cache_size_kb
        self._mmap_size_mb_raw = mmap_size_mb
        # Resolved values (defaults until start() runs detection)
        self._cache_size_kb: int = cache_size_kb if isinstance(cache_size_kb, int) else 64_000
        self._mmap_size_mb: int = mmap_size_mb if isinstance(mmap_size_mb, int) else 256
        self._busy_timeout_ms = busy_timeout_ms
        self._vector_backend_spec = vector_backend
        self._vector_backend = None  # resolved in start()
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
        self._stop_event = threading.Event()
        # Lock to prevent race between write() and stop()
        self._write_lock = threading.Lock()

        # Schema state cache — only mutated from writer thread,
        # so no lock needed.  Avoids redundant detect_state() queries
        # to sqlite_master on every ingest.
        self._schema_state_cache: SchemaState | None = None
        self._atexit_registered: bool = False
        self._stopped = threading.Event()

        # Auto-adaptive state
        self._env = None  # EnvironmentInfo, populated in start()
        self._reader_prune_threshold: int = 10
        self._ingest_counter: int = 0

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
        is_file = (
            self._path != ":memory:"
            and not self._path.startswith("file::memory:")
            and self._shared_mem_uri is None
        )

        if is_file:
            conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute(f"PRAGMA busy_timeout={self._busy_timeout_ms}")
        conn.execute("PRAGMA temp_store=MEMORY")
        conn.execute(f"PRAGMA cache_size=-{self._cache_size_kb}")
        if is_file and self._mmap_size_mb > 0:
            try:
                conn.execute(
                    f"PRAGMA mmap_size={self._mmap_size_mb * 1_048_576}"
                )
            except sqlite3.OperationalError:
                logger.debug("mmap_size PRAGMA failed, disabling mmap")
                self._mmap_size_mb = 0
        conn.execute("PRAGMA foreign_keys=ON")

    def _create_writer_connection(self) -> sqlite3.Connection:
        """Create and configure the writer connection."""
        if self._shared_mem_uri:
            conn = sqlite3.connect(self._shared_mem_uri, uri=True, check_same_thread=False)
        else:
            conn = sqlite3.connect(self._path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        self._configure_connection(conn)
        return conn

    def _get_reader_connection(self) -> sqlite3.Connection:
        """Get or create a reader connection for the current thread."""
        if self._stop_event.is_set():
            raise EngineClosedError("Engine is stopped, cannot create reader connection")

        conn = getattr(self._local, "conn", None)
        if conn is not None:
            return conn

        if self._shared_mem_uri:
            conn = sqlite3.connect(self._shared_mem_uri, uri=True, check_same_thread=False)
        else:
            conn = sqlite3.connect(self._path, check_same_thread=False)
        try:
            conn.row_factory = sqlite3.Row
            self._configure_connection(conn)
            if self._shared_mem_uri:
                conn.execute("PRAGMA read_uncommitted=ON")
            conn.execute("PRAGMA query_only=ON")
        except Exception:
            conn.close()
            raise

        tid = threading.get_ident()
        with self._reader_lock:
            # Check under lock: if stop() ran while we were creating
            # the connection, close it and reject.
            if self._stop_event.is_set():
                conn.close()
                raise EngineClosedError("Engine stopped during reader creation")

            # Close old connection if thread ID was reused
            old_conn = self._reader_connections.get(tid)
            if old_conn is not None:
                try:
                    old_conn.close()
                except Exception:
                    pass
            self._reader_connections[tid] = conn
            # Prune connections from dead threads when count exceeds threshold.
            # Threshold is auto-tuned: LOW=5, MEDIUM=10, HIGH=20.
            if len(self._reader_connections) > self._reader_prune_threshold:
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
        if self._writer_conn is None:
            raise SQFoxError("Writer connection is None — engine not started properly")
        current_batch: list[WriteRequest] = []

        try:
            while not self._stop_event.is_set() or not self._queue.empty():
                current_batch = []
                deadline = time.monotonic() + (self._batch_time_ms / 1000.0)

                # Drain up to batch_size items, waiting up to batch_time_ms
                while len(current_batch) < self._batch_size:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        break
                    try:
                        item = self._queue.get(timeout=max(remaining, 0.001))
                        _priority, _seq, request = item
                        if request is _STOP:
                            # Poison pill: process remaining batch, then exit
                            if current_batch:
                                self._execute_batch(current_batch)
                                current_batch = []
                            return
                        if not isinstance(request, WriteRequest):
                            raise SQFoxError(f"Unexpected item in queue: {type(request)}")
                        current_batch.append(request)
                    except queue.Empty:
                        break

                if current_batch:
                    self._execute_batch(current_batch)
                    current_batch = []

                    # --- Elastic batch: adapt to queue pressure ---
                    pending = self._queue.qsize()
                    if pending > self._batch_size * 2:
                        # Backlog growing — double batch to gulp faster
                        self._batch_size = min(
                            self._base_batch_size * 8, self._batch_size * 2
                        )
                    elif pending < self._base_batch_size // 2:
                        # Queue calm — shrink back for low latency
                        self._batch_size = self._base_batch_size

        except Exception as exc:
            logger.error("Writer thread crashed: %s", exc, exc_info=True)
            self._stop_event.set()
            # Fail futures from the batch that was being processed
            crash_exc = SQFoxError(f"Writer thread crashed: {exc}")
            for req in current_batch:
                if req.future is not None:
                    _safe_set_exception(req.future, crash_exc)
            # Drain remaining items and set exceptions on their futures
            while not self._queue.empty():
                try:
                    _, _, request = self._queue.get_nowait()
                    if isinstance(request, WriteRequest):
                        self._notify_error(request.sql, crash_exc)
                        if request.future is not None:
                            _safe_set_exception(request.future, crash_exc)
                except queue.Empty:
                    break
        finally:
            self._running.clear()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the writer thread and run startup hooks."""
        with self._write_lock:
            if self._stop_event.is_set():
                raise EngineClosedError("Engine has been stopped and cannot be restarted")
            if self._running.is_set():
                return  # Already running

            # --- Auto-adaptive detection (runs once) ---
            from ._auto import detect_environment, resolve_param, AUTO

            if self._env is None:
                self._env = detect_environment(self._path)

            self._cache_size_kb = resolve_param(
                self._cache_size_kb_raw, self._env.recommended_cache_size_kb,
            )
            self._mmap_size_mb = resolve_param(
                self._mmap_size_mb_raw, self._env.recommended_mmap_size_mb,
            )
            # Clamp invalid explicit values to safe minimums
            if self._cache_size_kb < 1:
                self._cache_size_kb = 1
            if self._mmap_size_mb < 0:
                self._mmap_size_mb = 0
            self._reader_prune_threshold = (
                self._env.recommended_reader_prune_threshold
            )

            self._writer_conn = self._create_writer_connection()

            try:
                # --- Auto-vacuum for NEW databases ---
                self._setup_auto_vacuum()

                # Resolve vector backend BEFORE writer thread starts
                # (avoids concurrent access to _writer_conn)
                if self._vector_backend_spec is not None:
                    from .backends.registry import get_backend
                    self._vector_backend = get_backend(self._vector_backend_spec)
                    if hasattr(self._vector_backend, "set_writer_conn"):
                        self._vector_backend.set_writer_conn(self._writer_conn)
                    # Early-initialize if dimension already known (restart scenario)
                    if not getattr(self._vector_backend, "_initialized", False):
                        from .schema import get_stored_dimension
                        ndim = get_stored_dimension(self._writer_conn)
                        if ndim is not None:
                            self._vector_backend.initialize(self._path, ndim)
                            self._startup_verify_backend(ndim)

                # FTS self-healing check (before writer thread starts)
                self._startup_fts_check()
            except Exception:
                # Clean up writer connection on any pre-thread failure
                try:
                    self._writer_conn.close()
                except Exception:
                    pass
                self._writer_conn = None
                self._vector_backend = None
                raise

            self._running.set()
            self._writer_thread = threading.Thread(
                target=self._writer_loop,
                name="sqfox-writer",
                daemon=False,
            )
            self._writer_thread.start()

            # Register atexit so process exit triggers graceful shutdown
            if not self._atexit_registered:
                atexit.register(self._atexit_stop)
                self._atexit_registered = True

        # Run startup hooks outside the lock
        for hook in self._on_startup_hooks:
            try:
                hook(self)
            except Exception as exc:
                logger.error("Startup hook failed: %s", exc, exc_info=True)
                self.stop()
                raise

    def _setup_auto_vacuum(self) -> None:
        """Enable incremental auto-vacuum on NEW (empty) databases."""
        if self._writer_conn is None or self._shared_mem_uri is not None:
            return  # skip for :memory:
        try:
            row = self._writer_conn.execute("PRAGMA auto_vacuum").fetchone()
            if row and row[0] == 0:  # NONE
                user_tables = self._writer_conn.execute(
                    "SELECT count(*) FROM sqlite_master "
                    "WHERE type='table' AND name NOT LIKE 'sqlite_%'"
                ).fetchone()
                if user_tables and user_tables[0] == 0:
                    self._writer_conn.execute(
                        "PRAGMA auto_vacuum=INCREMENTAL"
                    )
                    self._writer_conn.execute("VACUUM")
                    logger.info("Auto-vacuum set to INCREMENTAL on new database")
        except sqlite3.OperationalError as exc:
            logger.debug("Auto-vacuum setup skipped: %s", exc)

    def _startup_verify_backend(self, ndim: int) -> None:
        """Verify vector backend consistency at startup, rebuild if needed."""
        if not hasattr(self._vector_backend, "verify_consistency"):
            return
        count_row = self._writer_conn.execute(
            "SELECT COUNT(*) FROM documents WHERE vec_indexed = 1"
        ).fetchone()
        expected = count_row[0] if count_row else 0
        if expected > 0 and not self._vector_backend.verify_consistency(expected):
            logger.info(
                "Vector backend inconsistency: expected %d, rebuilding", expected,
            )
            _REBUILD_BATCH = 2000
            cursor = self._writer_conn.execute(
                "SELECT id, embedding FROM documents "
                "WHERE vec_indexed = 1 AND embedding IS NOT NULL ORDER BY id"
            )
            if hasattr(self._vector_backend, "reset"):
                self._vector_backend.reset(ndim)
            while True:
                batch = cursor.fetchmany(_REBUILD_BATCH)
                if not batch:
                    break
                keys = [r[0] for r in batch]
                vecs = [
                    list(struct.unpack(f"{ndim}f", r[1]))
                    for r in batch
                ]
                self._vector_backend.add(keys, vecs)
            self._vector_backend.flush()
            logger.info(
                "Vector backend rebuilt: %d vectors",
                self._vector_backend.count(),
            )

    def _auto_select_backend(self, conn: sqlite3.Connection) -> None:
        """Auto-select vector backend when user didn't specify one.

        Priority:
          1. Previously stored choice in _sqfox_meta (restart scenario)
          2. Platform / memory heuristic:
             - ANDROID_TERMUX   → flat  (phone RAM shared with OS+LLM, OOM risk)
             - LOW  (<1 GB RAM) → flat  (less RAM overhead, no graph)
             - MEDIUM / HIGH    → hnsw  (O(log N), scales to 100K+)

        Falls back to "hnsw" if stored backend is unavailable (e.g. usearch
        was uninstalled).
        """
        from ._auto import MemoryTier, PlatformClass
        from .backends.registry import get_backend
        from .schema import get_vector_backend_meta, set_vector_backend_meta

        stored = get_vector_backend_meta(conn)
        if stored:
            chosen = stored
        elif self._env is not None and (
            self._env.platform_class == PlatformClass.ANDROID_TERMUX
            or self._env.memory_tier == MemoryTier.LOW
        ):
            chosen = "flat"
        else:
            chosen = "hnsw"

        # Deduplicate: if chosen is already "hnsw", don't try it twice
        candidates = list(dict.fromkeys([chosen, "hnsw", "flat"]))
        for candidate in candidates:
            try:
                self._vector_backend = get_backend(candidate)
                chosen = candidate
                break
            except (ValueError, ImportError):
                logger.warning(
                    "Backend %r unavailable, trying next", candidate,
                )
        else:
            logger.error("No vector backend available — vector search disabled")
            self._vector_backend = None
            return

        if self._vector_backend is not None:
            if hasattr(self._vector_backend, "set_writer_conn"):
                self._vector_backend.set_writer_conn(conn)
            if not stored or stored != chosen:
                set_vector_backend_meta(conn, chosen)
                conn.commit()
                if stored and stored != chosen:
                    logger.info(
                        "Stored backend %r unavailable, switched to %r",
                        stored, chosen,
                    )
                else:
                    logger.info("Auto-selected vector backend: %s", chosen)

    def _startup_fts_check(self) -> None:
        """Lightweight FTS5 integrity check; auto-rebuild if corrupt."""
        if self._writer_conn is None:
            return
        try:
            from .schema import detect_state
            state = detect_state(self._writer_conn)
            if state < SchemaState.SEARCHABLE:
                return
            self._writer_conn.execute(
                "INSERT INTO documents_fts(documents_fts) "
                "VALUES('integrity-check')"
            )
            # Close the implicit transaction opened by the DML statement
            # so the writer thread can start fresh with BEGIN IMMEDIATE.
            self._writer_conn.commit()
        except (sqlite3.OperationalError, sqlite3.DatabaseError) as exc:
            # Rollback any implicit transaction from the failed integrity-check
            try:
                self._writer_conn.rollback()
            except Exception:
                pass
            logger.warning("FTS5 index may be corrupt: %s — rebuilding", exc)
            try:
                self._writer_conn.execute(
                    "INSERT INTO documents_fts(documents_fts) "
                    "VALUES('rebuild')"
                )
                self._writer_conn.commit()
                logger.info("FTS5 index rebuilt successfully")
            except (sqlite3.OperationalError, sqlite3.DatabaseError) as rebuild_exc:
                try:
                    self._writer_conn.rollback()
                except Exception:
                    pass
                logger.error(
                    "FTS5 rebuild failed: %s — full-text search may not work",
                    rebuild_exc,
                )

    def _atexit_stop(self) -> None:
        """Called by atexit — stop engine if still running."""
        if self._running.is_set():
            try:
                self.stop(timeout=5.0)
            except Exception:
                pass

    def stop(self, timeout: float = 10.0) -> None:
        """Gracefully shut down: drain queue, close all connections."""
        if self._stopped.is_set():
            return
        self._stopped.set()  # Atomic; guards concurrent stop() calls
        # Unregister atexit to avoid double-stop
        if self._atexit_registered:
            atexit.unregister(self._atexit_stop)
            self._atexit_registered = False

        with self._write_lock:
            if not self._running.is_set() and self._writer_thread is None:
                self._stop_event.set()
                return

            # Signal writer to stop — set stopped first to reject new writes
            self._stop_event.set()

            # Poison pill at LOW priority + sys.maxsize seq so it drains
            # after ALL pending requests (HIGH, NORMAL, LOW) in the queue
            try:
                self._queue.put_nowait((Priority.LOW, sys.maxsize, _STOP))
            except queue.Full:
                logger.warning("Queue full during shutdown, forcing stop")

        # Wait for writer thread
        writer_alive = False
        if self._writer_thread is not None:
            self._writer_thread.join(timeout=timeout)
            writer_alive = self._writer_thread.is_alive()
            if writer_alive:
                logger.warning(
                    "Writer thread did not stop within %.1fs, "
                    "deferring connection close", timeout
                )
            self._writer_thread = None

        if not writer_alive:
            # Only close backend/connection when writer is definitely gone.
            # Otherwise the writer thread would crash on a closed connection.

            # Close vector backend BEFORE closing writer_conn —
            # backend.close() may flush dirty state via _writer_conn.
            if self._vector_backend is not None:
                try:
                    self._vector_backend.close()
                except Exception as exc:
                    logger.warning("VectorBackend.close() failed: %s", exc)
                self._vector_backend = None

            # PRAGMA optimize: let SQLite refresh query planner stats
            # after heavy ingestion.  Cheap (~0-5 ms), only if meaningful.
            if self._writer_conn is not None and self._ingest_counter > 1000:
                try:
                    self._writer_conn.execute("PRAGMA optimize")
                except Exception:
                    pass

            # Now safe to close writer connection
            if self._writer_conn is not None:
                try:
                    self._writer_conn.close()
                except Exception:
                    pass
                self._writer_conn = None
        else:
            logger.warning(
                "Writer thread still alive — skipping connection close "
                "to avoid corruption. Resources will leak."
            )

        # Drain any orphaned items (from write() calls that raced with stop())
        while not self._queue.empty():
            try:
                _, _, request = self._queue.get_nowait()
                if isinstance(request, WriteRequest) and request.future is not None:
                    _safe_set_exception(
                        request.future,
                        EngineClosedError("Engine stopped before request was processed"),
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
            self._local = threading.local()

        self._schema_state_cache = None

    def __enter__(self) -> SQFox:
        self.start()
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.stop()

    def __del__(self) -> None:
        if self._running.is_set():
            logger.warning(
                "SQFox engine at %r was garbage-collected while still running. "
                "Call stop() explicitly or use the context manager.",
                self._path,
            )
            try:
                self.stop(timeout=2.0)
            except Exception:
                pass

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
    ) -> Future[Any] | Any:
        """Submit a write request to the writer queue.

        Args:
            sql:      SQL statement.
            params:   Bind parameters.
            priority: Queue priority.
            wait:     If True, block until the write completes and return
                      the result (lastrowid).  Raises any exception from
                      the writer.
            many:     If True, use executemany.

        Returns:
            Future[int] if wait=False, else the direct result (lastrowid).

        Raises:
            EngineClosedError: If engine is stopped.
            QueueFullError:    If queue is at max capacity.
        """
        future: Future[Any] = Future()
        request = WriteRequest(
            sql=sql,
            params=params,
            priority=priority,
            future=future,
            many=many,
        )

        with self._write_lock:
            if self._stop_event.is_set() or not self._running.is_set():
                raise EngineClosedError("Engine is not running")

            seq = self._next_seq()
            try:
                self._queue.put_nowait((priority, seq, request))
            except queue.Full:
                raise QueueFullError(
                    f"Write queue is full ({self._max_queue_size} items)"
                ) from None

        if wait:
            try:
                return future.result(timeout=300)
            except TimeoutError:
                raise SQFoxError(
                    "Write timed out after 300s — writer thread may be stuck"
                ) from None
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
        future: Future[Any] = Future()

        request = WriteRequest(
            sql="",
            priority=priority,
            future=future,
            fn=fn,
        )

        with self._write_lock:
            if self._stop_event.is_set() or not self._running.is_set():
                raise EngineClosedError("Engine is not running")

            seq = self._next_seq()
            try:
                self._queue.put_nowait((priority, seq, request))
            except queue.Full:
                raise QueueFullError(
                    f"Write queue is full ({self._max_queue_size} items)"
                ) from None

        if wait:
            try:
                return future.result(timeout=300)
            except TimeoutError:
                raise SQFoxError(
                    "Writer operation timed out after 300s — writer thread may be stuck"
                ) from None
        return future

    def _execute_batch(self, batch: list[WriteRequest]) -> None:
        """Execute a batch of write requests in a single transaction."""
        if self._writer_conn is None:
            raise SQFoxError("Writer connection is None — cannot execute batch")

        # Separate callable requests from SQL requests
        fn_requests = [r for r in batch if r.fn is not None]
        sql_requests = [r for r in batch if r.fn is None]

        # Execute callable requests (each in its own transaction)
        for req in fn_requests:
            try:
                result = req.fn(self._writer_conn)
                # Auto-commit if the callable left an open transaction.
                # This prevents poisoning the connection for subsequent
                # BEGIN IMMEDIATE in _execute_batch.
                if self._writer_conn.in_transaction:
                    self._writer_conn.commit()
                if req.future is not None:
                    _safe_set_result(req.future, result)
            except Exception as exc:
                # Rollback any uncommitted work from the callable
                try:
                    self._writer_conn.rollback()
                except Exception:
                    pass
                self._notify_error("<execute_on_writer>", exc)
                if req.future is not None:
                    _safe_set_exception(req.future, exc)

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
                    _safe_set_exception(req.future, exc)
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
                    _safe_set_exception(req.future, exc)
                for already_done_req, _ in results:
                    rollback_exc = SQFoxError(
                        f"Batch rolled back due to error in later statement: {exc}"
                    )
                    if already_done_req.future is not None:
                        _safe_set_exception(already_done_req.future, rollback_exc)
                for remaining in sql_requests[i + 1:]:
                    abort_exc = SQFoxError(f"Batch aborted due to prior error: {exc}")
                    self._notify_error(remaining.sql, abort_exc)
                    if remaining.future is not None:
                        _safe_set_exception(remaining.future, abort_exc)
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
                if req.future is not None:
                    _safe_set_exception(req.future, exc)
            return

        # Commit succeeded — now resolve all futures
        for req, lastrowid in results:
            if req.future is not None:
                _safe_set_result(req.future, lastrowid)

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
    def vector_backend_name(self) -> str | None:
        """Name of the active vector backend, or None."""
        if self._vector_backend is not None:
            name = type(self._vector_backend).__name__
            # Friendly names for known backends
            _friendly = {
                "SqliteHnswBackend": "hnsw",
                "SqliteFlatBackend": "flat",
                "USearchBackend": "usearch",
            }
            return _friendly.get(name, name)
        return None

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
        """Return diagnostic info about the engine and its capabilities."""
        import platform
        info: dict[str, Any] = {
            "sqfox_version": _get_version(),
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "sqlite_version": sqlite3.sqlite_version,
            "path": self._path,
            "is_running": self.is_running,
            "queue_size": self.queue_size,
            "max_queue_size": self._max_queue_size,
            "batch_size": self._base_batch_size,
            "batch_size_current": self._batch_size,
            "batch_time_ms": self._batch_time_ms,
            "mmap_size_mb": self._mmap_size_mb,
            "schema_state": self._schema_state_cache.name if self._schema_state_cache is not None else None,
            "vector_backend": self.vector_backend_name,
        }

        # Auto-adaptive environment info
        if self._env is not None:
            info["auto"] = {
                "total_ram_mb": self._env.total_ram_mb,
                "memory_tier": self._env.memory_tier.name,
                "cpu_count": self._env.cpu_count,
                "platform_class": self._env.platform_class.name,
                "is_sd_card": self._env.is_sd_card,
                "fts5_available": self._env.fts5_available,
                "resolved_cache_size_kb": self._cache_size_kb,
                "resolved_mmap_size_mb": self._mmap_size_mb,
            }

        # Check optional deps
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
    # Public API — backup
    # ------------------------------------------------------------------

    def backup(
        self,
        target: str | Path,
        *,
        pages: int = -1,
        progress: Callable[[int, int, int], None] | None = None,
    ) -> None:
        """Create a consistent online backup of the database.

        Uses SQLite's built-in backup API.  When an external vector
        index file exists (USearch backend), the entire operation runs
        on the writer thread so that no ingest can slip between the
        SQLite snapshot and the index file copy.  This guarantees the
        backup is a consistent point-in-time snapshot.

        For backends without external files (hnsw, flat)
        or when no backend is configured, the backup uses a reader
        connection (or writer for :memory:) as before.

        Args:
            target:   Path for the backup file.
            pages:    Pages per step (-1 = all at once, the default).
            progress: Optional callback ``(status, remaining, total)``.
        """
        if not self._running.is_set():
            raise EngineClosedError("Engine is not running, cannot backup")

        is_memory = self._shared_mem_uri is not None

        if not is_memory:
            # Guard against backing up to the same file
            src_path = Path(self._path).resolve()
            dst_path = Path(target).resolve()
            if src_path == dst_path:
                raise ValueError(
                    f"Backup target is the same as the source database: {dst_path}"
                )

        # Check if the backend has an external index file that needs
        # atomic copying alongside the SQLite database.
        has_external_index = (
            self._vector_backend is not None
            and hasattr(self._vector_backend, "_index_path")
            and self._vector_backend._index_path
        )

        if has_external_index:
            # Atomic backup: flush + SQLite backup + index copy all on
            # the writer thread.  No ingest can interleave because the
            # writer processes requests sequentially.
            self._backup_atomic(target, pages=pages, progress=progress)
        elif is_memory:
            dst = sqlite3.connect(str(target))
            try:
                self.execute_on_writer(
                    lambda conn: conn.backup(dst, pages=pages, progress=progress),
                    priority=Priority.HIGH,
                    wait=True,
                )
            finally:
                dst.close()
        else:
            dst = sqlite3.connect(str(target))
            try:
                conn = self._get_reader_connection()
                conn.backup(dst, pages=pages, progress=progress)
            finally:
                dst.close()

    def _backup_atomic(
        self,
        target: str | Path,
        *,
        pages: int = -1,
        progress: Callable[[int, int, int], None] | None = None,
    ) -> None:
        """Atomic backup: SQLite + external index in one writer op.

        Runs entirely on the writer thread so that the index file is
        guaranteed to match the SQLite snapshot — no ingest can happen
        between the two copies.
        """
        import shutil

        dst_idx = str(target) + ".usearch"

        def _do_backup(conn: sqlite3.Connection) -> None:
            # 1. Flush pending in-memory vectors to the index file
            self._vector_backend.flush()

            # 2. Copy the index file FIRST (while writer is quiesced)
            idx_copied = False
            src_idx = self._vector_backend._index_path
            if Path(src_idx).exists():
                shutil.copy2(src_idx, dst_idx)
                idx_copied = True

            # 3. SQLite backup from the writer connection itself —
            #    this sees exactly the same committed state as the
            #    index file we just copied.
            dst = sqlite3.connect(str(target))
            try:
                conn.backup(dst, pages=pages, progress=progress)
            except Exception:
                # SQLite backup failed — clean up the already-copied
                # index file to avoid leaving an orphaned partial backup.
                if idx_copied:
                    try:
                        Path(dst_idx).unlink(missing_ok=True)
                    except OSError:
                        pass
                raise
            finally:
                dst.close()

        self.execute_on_writer(
            _do_backup, priority=Priority.HIGH, wait=True,
        )

    # ------------------------------------------------------------------
    # Public API — maintenance
    # ------------------------------------------------------------------

    def vacuum(self, *, into: str | Path | None = None) -> None:
        """Reclaim disk space by rebuilding the database file.

        On long-running IoT deployments, frequent DELETEs leave empty
        pages inside the ``.db`` file.  ``vacuum()`` rewrites the file
        to remove them.

        Args:
            into: If provided, writes the compacted copy to a new file
                  (``VACUUM INTO``).  The original file is unchanged.
                  Requires SQLite 3.27+.

        Note:
            VACUUM on the main database needs roughly 2× the current
            file size in free disk space.  ``vacuum(into=...)`` avoids
            this by writing to a separate file.
        """
        if into is not None:
            self.execute_on_writer(
                lambda conn: conn.execute("VACUUM INTO ?", (str(into),)),
                priority=Priority.HIGH,
                wait=True,
            )
        else:
            self.execute_on_writer(
                lambda conn: conn.execute("VACUUM"),
                priority=Priority.HIGH,
                wait=True,
            )

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
        Invalidates and refreshes the schema state cache.

        If FTS5 is unavailable and target requires it (SEARCHABLE or
        ENRICHED), the migration proceeds as far as possible and logs a
        warning instead of crashing.
        """
        from .schema import migrate_to, detect_state

        def _do_migrate(conn: sqlite3.Connection) -> SchemaState:
            effective = target
            if effective >= SchemaState.SEARCHABLE:
                fts5_ok = (
                    self._env.fts5_available if self._env is not None
                    else True
                )
                if not fts5_ok:
                    logger.info(
                        "FTS5 not available — skipping SEARCHABLE/ENRICHED "
                        "migration, vector-only search active"
                    )
                    effective = SchemaState.INDEXED if vec_dimension else SchemaState.BASE
            result = migrate_to(conn, effective, vec_dimension=vec_dimension)
            self._schema_state_cache = detect_state(conn)
            return result

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

        Heavy computation (chunking, lemmatization, embedding) runs in the
        calling thread.  Only pure SQL writes are sent to the writer thread.

        Returns the parent document ID.
        """
        import json as _json
        import struct as _struct
        from .schema import (
            detect_state,
            ensure_embedding_column,
            migrate_to,
            validate_dimension,
        )
        from .tokenizer import lemmatize

        # Reject content that would pollute indexes
        if not content or not content.strip() or "\x00" in content:
            raise SQFoxError(
                "content must be a non-empty string without NUL bytes"
            )

        if metadata:
            try:
                meta_str = _json.dumps(metadata)
            except TypeError as exc:
                raise SQFoxError(f"metadata is not JSON-serializable: {exc}") from exc
        else:
            meta_str = "{}"

        # ----------------------------------------------------------
        # Phase 1: heavy computation in the CALLING thread
        # ----------------------------------------------------------

        # Chunking
        if chunker is not None:
            chunks = chunker(content)
            chunks = [c for c in chunks if c.strip()]
            if not chunks:
                chunks = [content]
        else:
            chunks = [content]

        # Lemmatization
        lemmatized_chunks: list[str | None] = []
        for chunk_text in chunks:
            try:
                lemmatized_chunks.append(lemmatize(chunk_text))
            except Exception as exc:
                logger.warning("Lemmatization failed: %s", exc)
                lemmatized_chunks.append(None)

        # Embedding
        embeddings: list[list[float]] | None = None
        vec_blobs: list[bytes] | None = None
        if embed_fn is not None:
            embeddings = embed_for_documents(embed_fn, chunks)
            if embeddings:
                if len(embeddings) != len(chunks):
                    raise SQFoxError(
                        f"embed_fn returned {len(embeddings)} embeddings for "
                        f"{len(chunks)} chunks — must return exactly one "
                        f"embedding per chunk"
                    )
                vec_blobs = [
                    _struct.pack(f"{len(emb)}f", *emb)
                    for emb in embeddings
                ]

        # ----------------------------------------------------------
        # Phase 2: pure SQL on the writer thread
        # ----------------------------------------------------------

        has_chunker = chunker is not None
        vec_dim = len(embeddings[0]) if embeddings else None

        def _do_migrate(conn: sqlite3.Connection) -> None:
            """Ensure schema is ready before ingest. Migrations commit
            internally (CREATE IF NOT EXISTS), so they are idempotent and
            complete before the ingest transaction starts."""
            current = self._schema_state_cache
            if current is None:
                current = detect_state(conn)

            if current < SchemaState.BASE:
                migrate_to(conn, SchemaState.BASE)
                current = SchemaState.BASE

            # Ensure embedding column exists (upgrade from older schema)
            ensure_embedding_column(conn)

            if vec_blobs is not None and vec_dim is not None:
                validate_dimension(conn, vec_dim, commit=True)

                # --- Auto-backend selection ---
                if self._vector_backend is None:
                    self._auto_select_backend(conn)

                if self._vector_backend is not None:
                    # Initialize backend if not yet done
                    if not getattr(self._vector_backend, "_initialized", False):
                        self._vector_backend.initialize(self._path, vec_dim)
                        # Crash recovery: verify consistency
                        if hasattr(self._vector_backend, "verify_consistency"):
                            count_row = conn.execute(
                                "SELECT COUNT(*) FROM documents WHERE vec_indexed = 1"
                            ).fetchone()
                            expected = count_row[0] if count_row else 0
                            if not self._vector_backend.verify_consistency(expected):
                                # Batched rebuild to avoid OOM on low-RAM devices
                                _REBUILD_BATCH = 2000
                                cursor = conn.execute(
                                    "SELECT id, embedding FROM documents "
                                    "WHERE vec_indexed = 1 AND embedding IS NOT NULL"
                                    " ORDER BY id"
                                )
                                # Reset backend state before batched insert
                                if hasattr(self._vector_backend, "reset"):
                                    self._vector_backend.reset(vec_dim)
                                elif hasattr(self._vector_backend, "rebuild_from_blobs"):
                                    self._vector_backend.rebuild_from_blobs(
                                        [], vec_dim,
                                    )
                                while True:
                                    batch = cursor.fetchmany(_REBUILD_BATCH)
                                    if not batch:
                                        break
                                    keys = [r[0] for r in batch]
                                    vecs = [
                                        list(_struct.unpack(
                                            f"{vec_dim}f", r[1]
                                        ))
                                        for r in batch
                                    ]
                                    self._vector_backend.add(keys, vecs)
                                self._vector_backend.flush()
                                logger.info(
                                    "Vector backend rebuilt during ingest: "
                                    "%d vectors",
                                    self._vector_backend.count(),
                                )

            # Always re-detect after potential migrations
            self._schema_state_cache = detect_state(conn)

        def _do_ingest(conn: sqlite3.Connection) -> int:
            # Phase 2a: ensure schema is migrated (commits separately)
            _do_migrate(conn)

            # Phase 2b: all data operations in an explicit atomic transaction
            conn.execute("BEGIN IMMEDIATE")
            try:
                # Insert parent document
                cursor = conn.execute(
                    "INSERT INTO documents (content, metadata) VALUES (?, ?)",
                    (content, meta_str),
                )
                parent_id = cursor.lastrowid
                if parent_id is None:
                    raise SQFoxError("INSERT returned None lastrowid — unexpected")

                # Insert chunks
                chunk_ids: list[int] = []
                for i, chunk_text in enumerate(chunks):
                    if has_chunker:
                        cur = conn.execute(
                            "INSERT INTO documents (content, metadata, chunk_of) "
                            "VALUES (?, ?, ?)",
                            (chunk_text, meta_str, parent_id),
                        )
                        chunk_id = cur.lastrowid
                        if chunk_id is None:
                            raise SQFoxError("INSERT chunk returned None lastrowid — unexpected")
                        chunk_ids.append(chunk_id)
                    else:
                        chunk_ids.append(parent_id)

                # Write lemmatized content
                for cid, lem in zip(chunk_ids, lemmatized_chunks):
                    if lem is not None:
                        conn.execute(
                            "UPDATE documents SET content_lemmatized = ? WHERE id = ?",
                            (lem, cid),
                        )

                # Write embeddings
                if vec_blobs is not None and vec_dim is not None:
                    # Always store raw blobs in documents (source of truth)
                    for cid, blob in zip(chunk_ids, vec_blobs):
                        conn.execute(
                            "UPDATE documents SET embedding = ?, vec_indexed = 1 "
                            "WHERE id = ?",
                            (blob, cid),
                        )

                    if self._vector_backend is not None:
                        self._vector_backend.add(chunk_ids, embeddings)

                # FTS sync for SEARCHABLE (no triggers) state
                current = self._schema_state_cache or SchemaState.EMPTY
                if current >= SchemaState.SEARCHABLE:
                    for cid in chunk_ids:
                        row = conn.execute(
                            "SELECT content_lemmatized FROM documents WHERE id = ?",
                            (cid,),
                        ).fetchone()
                        if row and row[0] and current < SchemaState.ENRICHED:
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
            except Exception:
                conn.rollback()
                raise

            # Flush external backend to disk after SQLite commit.
            # If flush fails (e.g. disk full), the graph is still in RAM
            # and documents are already committed — consistency is restored
            # on next flush or rebuild at restart.  Do NOT let a flush
            # failure kill the writer thread.
            if self._vector_backend is not None and vec_blobs is not None:
                try:
                    self._vector_backend.flush()
                except Exception as exc:
                    logger.warning(
                        "Vector backend flush failed (will retry on next "
                        "ingest or recover at restart): %s", exc
                    )

            # Incremental vacuum: reclaim ~100 pages every 100 ingests.
            # Takes <1ms, no effect if auto_vacuum is not INCREMENTAL.
            self._ingest_counter += 1
            if self._ingest_counter % 100 == 0:
                try:
                    conn.execute("PRAGMA incremental_vacuum(100)")
                except sqlite3.OperationalError:
                    pass

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
        reranker_fn: RerankerFn | None = None,
        rerank_top_n: int | None = None,
    ) -> list[SearchResult]:
        """Search documents using hybrid FTS5 + vector search.

        If embed_fn is None, performs FTS-only search.
        If reranker_fn is provided, top candidates are re-scored by a
        cross-encoder before returning the final ``limit`` results.
        """
        from .search import hybrid_search, fts_search
        from .tokenizer import lemmatize_query

        conn = self._get_reader_connection()

        if embed_fn is not None:
            try:
                return hybrid_search(
                    conn,
                    query,
                    embed_fn,
                    lemmatize_fn=lemmatize_query,
                    limit=limit,
                    alpha=alpha,
                    reranker_fn=reranker_fn,
                    rerank_top_n=rerank_top_n,
                    vector_backend=self._vector_backend,
                )
            except (sqlite3.OperationalError, SQFoxError) as exc:
                logger.warning("Hybrid search failed: %s", exc)
                return []
        else:
            # FTS-only search
            if reranker_fn is not None:
                logger.warning(
                    "reranker_fn provided without embed_fn; "
                    "reranker will not be applied to FTS-only results"
                )
            query_lemmatized = lemmatize_query(query, None)
            try:
                fts_results = fts_search(conn, query_lemmatized, limit=limit)
            except (sqlite3.OperationalError, SQFoxError) as exc:
                logger.warning("FTS search failed: %s", exc)
                return []

            if not fts_results:
                return []

            # Hydrate
            import json as _json
            doc_ids = tuple(doc_id for doc_id, _ in fts_results)
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
