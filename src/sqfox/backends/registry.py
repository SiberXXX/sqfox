"""Backend resolution — maps string aliases to VectorBackend instances."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..types import VectorBackend

_ALIASES: dict[str, tuple[str, str]] = {
    "flat": ("sqfox.backends.flat", "SqliteFlatBackend"),
    "hnsw": ("sqfox.backends.hnsw", "SqliteHnswBackend"),
    "usearch": ("sqfox.backends.usearch", "USearchBackend"),
}


def get_backend(spec: str | VectorBackend | None) -> VectorBackend | None:
    """Resolve a backend specification to a VectorBackend instance.

    Args:
        spec: None, a string alias ("flat", "hnsw", "usearch"),
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

    # Validate it looks like a VectorBackend (has required methods)
    required = ("set_writer_conn", "initialize", "search", "add", "remove", "flush", "count", "close")
    missing = [m for m in required if not hasattr(spec, m)]
    if missing:
        raise TypeError(
            f"vector_backend must be None, a string alias, or a "
            f"VectorBackend instance — got {type(spec).__name__} "
            f"(missing: {', '.join(missing)})"
        )
    return spec
