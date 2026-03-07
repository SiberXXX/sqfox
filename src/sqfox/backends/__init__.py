"""Vector backend implementations."""

from .registry import get_backend
from .hnsw import SqliteHnswBackend
from .flat import SqliteFlatBackend

__all__ = ["SqliteHnswBackend", "SqliteFlatBackend", "get_backend"]

# USearchBackend imported lazily via registry to avoid hard numpy/usearch dep
