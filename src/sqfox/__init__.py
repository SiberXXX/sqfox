"""sqfox: Embedded SQLite micro-framework for hybrid search, IoT and RAG."""

from .chunkers import html_to_text, markdown_chunker, paragraph_chunker, recursive_chunker, sentence_chunker
from .async_engine import AsyncSQFox
from .backends.hnsw import SqliteHnswBackend
from .engine import SQFox
from .manager import SQFoxManager
from .types import (
    ChunkerFn,
    DimensionError,
    EmbedFn,
    Embedder,
    EngineClosedError,
    Priority,
    QueueFullError,
    RerankerFn,
    SchemaError,
    SchemaState,
    SearchResult,
    SQFoxError,
    VectorBackend,
    VectorBackendError,
    WriteRequest,
)

__all__ = [
    "AsyncSQFox",
    "SQFox",
    "SQFoxManager",
    "html_to_text",
    "paragraph_chunker",
    "sentence_chunker",
    "markdown_chunker",
    "recursive_chunker",
    "ChunkerFn",
    "DimensionError",
    "EmbedFn",
    "Embedder",
    "EngineClosedError",
    "Priority",
    "QueueFullError",
    "RerankerFn",
    "SchemaError",
    "SchemaState",
    "SearchResult",
    "SQFoxError",
    "VectorBackend",
    "VectorBackendError",
    "SqliteHnswBackend",
    "WriteRequest",
]

__version__ = "0.2.1"
