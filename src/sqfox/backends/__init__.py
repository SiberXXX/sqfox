"""Vector backend implementations."""

from .registry import get_backend
from .sqlite_vec import SqliteVecBackend
from .hnsw import SqliteHnswBackend

__all__ = ["SqliteVecBackend", "SqliteHnswBackend", "get_backend"]

# USearchBackend imported lazily via registry to avoid hard numpy/usearch dep
