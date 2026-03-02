"""Shared types and protocols for sqfox."""

from __future__ import annotations

import enum
from concurrent.futures import Future
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol, runtime_checkable


# ---------------------------------------------------------------------------
# Plugin protocols
# ---------------------------------------------------------------------------

@runtime_checkable
class ChunkerFn(Protocol):
    """Protocol for text chunking callables.

    A chunker takes a single text string and returns a list of chunks.
    Each chunk is a substring suitable for embedding.
    """

    def __call__(self, text: str) -> list[str]: ...


@runtime_checkable
class EmbedFn(Protocol):
    """Protocol for simple embedding callables.

    An embedder takes a batch of text strings and returns a list of
    float vectors (one per input string).  All vectors MUST have the
    same dimension.  The dimension is locked on first call and
    mismatches raise DimensionError.
    """

    def __call__(self, texts: list[str]) -> list[list[float]]: ...


@runtime_checkable
class RerankerFn(Protocol):
    """Protocol for cross-encoder reranking callables.

    A reranker takes the original query and a list of candidate texts,
    and returns a relevance score for each text (higher = more relevant).
    """

    def __call__(self, query: str, texts: list[str]) -> list[float]: ...


@runtime_checkable
class Embedder(Protocol):
    """Protocol for instruction-aware embedding adapters.

    Models like Qwen3, E5, BGE produce different embeddings for
    documents vs queries.  Implement this protocol to let sqfox
    automatically call the right method.

    If you pass an object with embed_documents/embed_query to sqfox,
    it will use the appropriate method.  If you pass a plain callable,
    it will be used for both.
    """

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...
    def embed_query(self, text: str) -> list[float]: ...


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class Priority(enum.IntEnum):
    """Write request priority. Lower numeric value = higher priority."""

    HIGH = 0
    NORMAL = 1
    LOW = 2


class SchemaState(enum.IntEnum):
    """Database schema evolution states.

    EMPTY      - No sqfox tables exist.
    BASE       - Core documents + _sqfox_meta tables exist.
    INDEXED    - vec0 virtual table exists for vector search.
    SEARCHABLE - FTS5 virtual table exists for full-text search.
    ENRICHED   - Both vec0 and FTS5 exist with sync triggers.
    """

    EMPTY = 0
    BASE = 1
    INDEXED = 2
    SEARCHABLE = 3
    ENRICHED = 4


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class WriteRequest:
    """A write operation queued for the writer thread.

    Attributes:
        sql:      SQL statement(s) to execute.
        params:   Bind parameters.  Can be a tuple for single execution
                  or a list of tuples for executemany.
        priority: Queue priority (HIGH executes first).
        future:   Optional Future for the caller to await the result
                  or catch exceptions.
        many:     If True, use executemany instead of execute.
        fn:       Optional callable for ``execute_on_writer``.  When set,
                  ``sql`` and ``params`` are ignored — the callable receives
                  the writer's ``sqlite3.Connection`` directly.

    Note:
        This dataclass intentionally does NOT define ordering (__lt__ etc.).
        PriorityQueue comparisons are handled by the (priority, seq, request)
        tuple wrapper — the seq field breaks ties so WriteRequest.__lt__ is
        never reached.  If direct comparison is attempted it will raise
        TypeError, which is the desired behaviour (fail-fast).
    """

    sql: str
    params: tuple[Any, ...] | list[tuple[Any, ...]] = ()
    priority: Priority = Priority.NORMAL
    future: Future[Any] | None = None
    many: bool = False
    fn: Callable[..., Any] | None = None


@dataclass(frozen=True, slots=True)
class SearchResult:
    """A single result from hybrid search.

    Attributes:
        doc_id:   Primary key in the documents table.
        score:    Fused relevance score (higher = more relevant).
        text:     The original document text.
        metadata: Parsed metadata dict (from JSON column).
        chunk_id: Optional chunk identifier if document was chunked.
    """

    doc_id: int
    score: float
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    chunk_id: int | None = None


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class SQFoxError(Exception):
    """Base exception for all sqfox errors."""


class DimensionError(SQFoxError):
    """Raised when vector dimension mismatches the stored dimension."""

    def __init__(self, expected: int, got: int) -> None:
        self.expected = expected
        self.got = got
        super().__init__(
            f"Vector dimension mismatch: database expects {expected}, "
            f"got {got}"
        )


class SchemaError(SQFoxError):
    """Raised when schema migration fails or state is invalid."""


class EngineClosedError(SQFoxError):
    """Raised when operations are attempted on a stopped engine."""


class QueueFullError(SQFoxError):
    """Raised when the write queue is full and backpressure kicks in."""


# ---------------------------------------------------------------------------
# Embed dispatch helpers
# ---------------------------------------------------------------------------

def _is_embedder(obj: object) -> bool:
    """Check if object implements both Embedder methods."""
    return hasattr(obj, "embed_documents") and hasattr(obj, "embed_query")


def embed_for_documents(
    embed_fn: EmbedFn | Embedder,
    texts: list[str],
) -> list[list[float]]:
    """Call the right method for document embedding.

    - Embedder object (both methods present) → embed_documents()
    - Plain callable → __call__()
    - Half-implemented (only one method) → __call__() with warning
    """
    if _is_embedder(embed_fn):
        return embed_fn.embed_documents(texts)  # type: ignore[union-attr]
    if hasattr(embed_fn, "embed_documents") and not hasattr(embed_fn, "embed_query"):
        import logging
        logging.getLogger("sqfox.types").warning(
            "Object has embed_documents() but not embed_query(). "
            "Implement both methods for correct instruction-aware embedding."
        )
    if not callable(embed_fn):
        raise TypeError(
            f"embed_fn must be callable or implement Embedder protocol, got {type(embed_fn)}"
        )
    return embed_fn(texts)  # type: ignore[call-arg]


def embed_for_query(
    embed_fn: EmbedFn | Embedder,
    text: str,
) -> list[float]:
    """Call the right method for query embedding.

    - Embedder object (both methods present) → embed_query()
    - Plain callable → __call__([text])[0]
    - Half-implemented (only one method) → __call__([text])[0] with warning
    """
    if _is_embedder(embed_fn):
        return embed_fn.embed_query(text)  # type: ignore[union-attr]
    if hasattr(embed_fn, "embed_query") and not hasattr(embed_fn, "embed_documents"):
        import logging
        logging.getLogger("sqfox.types").warning(
            "Object has embed_query() but not embed_documents(). "
            "Implement both methods for correct instruction-aware embedding."
        )
    if not callable(embed_fn):
        raise TypeError(
            f"embed_fn must be callable or implement Embedder protocol, got {type(embed_fn)}"
        )
    return embed_fn([text])[0]  # type: ignore[call-arg]
