"""Backend resolution — maps string aliases to VectorBackend instances."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..types import VectorBackend

_ALIASES: dict[str, tuple[str, str]] = {
    "sqlite-vec": ("sqfox.backends.sqlite_vec", "SqliteVecBackend"),
    "sqlite_vec": ("sqfox.backends.sqlite_vec", "SqliteVecBackend"),
    "usearch": ("sqfox.backends.usearch", "USearchBackend"),
    "hnsw": ("sqfox.backends.hnsw", "SqliteHnswBackend"),
}


def get_backend(spec: str | VectorBackend | None) -> VectorBackend | None:
    """Resolve a backend specification to a VectorBackend instance.

    Args:
        spec: None, a string alias ("sqlite-vec", "usearch", "hnsw"),
              or an existing VectorBackend instance.

    Returns:
        A VectorBackend instance, or None.
    """
    if spec is None:
        return None

    if isinstance(spec, str):
        if spec not in _ALIASES:
            raise ValueError(
                f"Unknown vector backend: {spec!r}. "
                f"Available: {sorted(set(_ALIASES))}"
            )
        import importlib
        module_path, class_name = _ALIASES[spec]
        module = importlib.import_module(module_path)
        cls = getattr(module, class_name)
        return cls()

    # Assume it's already a VectorBackend instance
    return spec
