"""Async facade for SQFox — safe for FastAPI / asyncio event loops.

Separates I/O-bound reads (default thread pool) from CPU-bound work
like embedding and reranking (dedicated limited pool) so that light
queries are never blocked by heavy ingest operations.

Usage with FastAPI::

    from contextlib import asynccontextmanager
    from fastapi import FastAPI
    from sqfox import AsyncSQFox

    db = AsyncSQFox("data.db", max_cpu_workers=2)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        async with db:
            yield

    app = FastAPI(lifespan=lifespan)

    @app.post("/ingest")
    async def ingest(text: str):
        doc_id = await db.ingest(text, embed_fn=my_embed)
        return {"id": doc_id}

    @app.get("/search")
    async def search(q: str):
        results = await db.search(q, embed_fn=my_embed)
        return results
"""

from __future__ import annotations

import asyncio
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable

from .engine import SQFox
from .types import (
    ChunkerFn,
    EmbedFn,
    EngineClosedError,
    Priority,
    RerankerFn,
    SearchResult,
    SchemaState,
)

__all__ = ["AsyncSQFox"]


class AsyncSQFox:
    """Async facade over :class:`SQFox`.

    Two thread pools keep I/O reads fast while CPU-heavy work
    (embedding, reranking, chunking) is rate-limited:

    * **I/O pool** — default ``asyncio`` executor (many threads).
      Used for ``fetch_one``, ``fetch_all``, FTS-only ``search``,
      ``backup``.
    * **CPU pool** — ``ThreadPoolExecutor(max_cpu_workers)``.
      Used for ``ingest`` and ``search`` with ``embed_fn`` /
      ``reranker_fn``.

    This prevents OOM on edge devices: even if 200 requests call
    ``ingest`` simultaneously, only ``max_cpu_workers`` embeddings
    are computed at once.

    Args:
        *args:           Forwarded to :class:`SQFox`.
        max_cpu_workers: Max threads for CPU-heavy operations.
        **kwargs:        Forwarded to :class:`SQFox`.
    """

    def __init__(
        self,
        *args: Any,
        max_cpu_workers: int = 2,
        **kwargs: Any,
    ) -> None:
        self._db = SQFox(*args, **kwargs)
        self._cpu_executor = ThreadPoolExecutor(
            max_workers=max_cpu_workers,
            thread_name_prefix="sqfox-cpu",
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def __aenter__(self) -> AsyncSQFox:
        self._db.start()
        return self

    async def __aexit__(self, *exc_info: Any) -> None:
        await asyncio.to_thread(self._db.stop)
        await asyncio.to_thread(self._cpu_executor.shutdown, True)

    def start(self) -> None:
        """Start the underlying sync engine (non-blocking)."""
        self._db.start()

    async def stop(self) -> None:
        """Gracefully stop the engine and shut down the CPU pool."""
        await asyncio.to_thread(self._db.stop)
        await asyncio.to_thread(self._cpu_executor.shutdown, True)

    # ------------------------------------------------------------------
    # Writes — near-instant queue put + optional await
    # ------------------------------------------------------------------

    async def write(
        self,
        sql: str,
        params: tuple[Any, ...] | list[tuple[Any, ...]] = (),
        *,
        priority: Priority = Priority.NORMAL,
        wait: bool = False,
        many: bool = False,
    ) -> Any:
        """Submit a write to the queue.

        The queue ``put`` is near-instant.  If ``wait=True``, awaits
        the writer thread's confirmation (commit).  If ``wait=False``,
        returns immediately (fire-and-forget).
        """
        sync_future = self._db.write(
            sql, params, priority=priority, wait=False, many=many,
        )
        if not wait:
            return None
        assert isinstance(sync_future, Future)
        return await asyncio.wrap_future(sync_future)

    async def execute_on_writer(
        self,
        fn: Callable[..., Any],
        *,
        priority: Priority = Priority.HIGH,
    ) -> Any:
        """Run a callable on the writer connection (awaits result)."""
        sync_future = self._db.execute_on_writer(
            fn, priority=priority, wait=False,
        )
        assert isinstance(sync_future, Future)
        return await asyncio.wrap_future(sync_future)

    # ------------------------------------------------------------------
    # Reads — default I/O pool (never blocked by CPU work)
    # ------------------------------------------------------------------

    async def fetch_one(self, sql: str, params: tuple[Any, ...] = ()) -> Any:
        """Execute a read query and return the first row."""
        return await asyncio.to_thread(self._db.fetch_one, sql, params)

    async def fetch_all(self, sql: str, params: tuple[Any, ...] = ()) -> list[Any]:
        """Execute a read query and return all rows."""
        return await asyncio.to_thread(self._db.fetch_all, sql, params)

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    async def ensure_schema(
        self,
        target: SchemaState = SchemaState.BASE,
        *,
        vec_dimension: int | None = None,
    ) -> SchemaState:
        """Ensure schema is at the target state (runs on writer)."""
        sync_future = self._db.execute_on_writer(
            lambda conn: __import__("sqfox.schema", fromlist=["migrate_to"]).migrate_to(
                conn, target, vec_dimension=vec_dimension
            ),
            priority=Priority.HIGH,
            wait=False,
        )
        assert isinstance(sync_future, Future)
        return await asyncio.wrap_future(sync_future)

    # ------------------------------------------------------------------
    # Heavy operations — dedicated CPU pool
    # ------------------------------------------------------------------

    async def ingest(
        self,
        content: str,
        *,
        metadata: dict[str, Any] | None = None,
        chunker: ChunkerFn | None = None,
        embed_fn: EmbedFn | None = None,
        priority: Priority = Priority.NORMAL,
    ) -> int:
        """Ingest a document (CPU pool for chunking/embedding).

        Always waits for completion and returns the parent document ID.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._cpu_executor,
            lambda: self._db.ingest(
                content,
                metadata=metadata,
                chunker=chunker,
                embed_fn=embed_fn,
                priority=priority,
                wait=True,
            ),
        )

    async def search(
        self,
        query: str,
        *,
        embed_fn: EmbedFn | None = None,
        limit: int = 10,
        alpha: float | None = None,
        reranker_fn: RerankerFn | None = None,
        rerank_top_n: int | None = None,
    ) -> list[SearchResult]:
        """Search documents.

        Routes to CPU pool if ``embed_fn`` or ``reranker_fn`` is
        provided, otherwise uses the fast I/O pool for FTS-only search.
        """
        kwargs: dict[str, Any] = dict(
            limit=limit, alpha=alpha,
            reranker_fn=reranker_fn, rerank_top_n=rerank_top_n,
        )
        if embed_fn is not None:
            kwargs["embed_fn"] = embed_fn

        if embed_fn is not None or reranker_fn is not None:
            # Heavy: embedding / reranking → CPU pool
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(
                self._cpu_executor,
                lambda: self._db.search(query, **kwargs),
            )

        # Light: FTS-only → default I/O pool
        return await asyncio.to_thread(self._db.search, query, **kwargs)

    # ------------------------------------------------------------------
    # Backup — I/O pool
    # ------------------------------------------------------------------

    async def backup(
        self,
        target: str | Path,
        *,
        pages: int = -1,
        progress: Callable[[int, int, int], None] | None = None,
    ) -> None:
        """Online backup (I/O pool, does not block CPU work)."""
        await asyncio.to_thread(
            self._db.backup, target, pages=pages, progress=progress,
        )

    # ------------------------------------------------------------------
    # Properties (sync, non-blocking)
    # ------------------------------------------------------------------

    @property
    def vec_available(self) -> bool:
        return self._db.vec_available

    @property
    def is_running(self) -> bool:
        return self._db.is_running

    @property
    def queue_size(self) -> int:
        return self._db.queue_size

    @property
    def path(self) -> str:
        return self._db.path

    def diagnostics(self) -> dict[str, Any]:
        return self._db.diagnostics()

    def on_startup(self, hook: Callable[[SQFox], None]) -> Callable[[SQFox], None]:
        """Register a startup hook (runs synchronously on start)."""
        return self._db.on_startup(hook)
