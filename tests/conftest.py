"""Shared test fixtures for sqfox test suite."""
import sys
import sqlite3
import pytest
from pathlib import Path

# Ensure src is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def _has_sqlite_vec() -> bool:
    """Check if sqlite-vec extension is available."""
    try:
        import sqlite_vec
        conn = sqlite3.connect(":memory:")
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.close()
        return True
    except (ImportError, AttributeError, Exception):
        return False


requires_sqlite_vec = pytest.mark.skipif(
    not _has_sqlite_vec(),
    reason="sqlite-vec not available",
)
