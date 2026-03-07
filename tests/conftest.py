"""Shared test fixtures for sqfox test suite."""
import sys
import pytest
from pathlib import Path

# Ensure src is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def _has_usearch() -> bool:
    """Check if usearch + numpy are available."""
    try:
        import usearch.index  # noqa: F401
        import numpy  # noqa: F401
        return True
    except ImportError:
        return False


requires_usearch = pytest.mark.skipif(
    not _has_usearch(),
    reason="usearch not available (pip install usearch numpy)",
)


def _has_sentence_transformers() -> bool:
    """Check if sentence-transformers is available."""
    try:
        import sentence_transformers  # noqa: F401
        return True
    except ImportError:
        return False


requires_sentence_transformers = pytest.mark.skipif(
    not _has_sentence_transformers(),
    reason="sentence-transformers not available (pip install sentence-transformers)",
)
