"""
Trace adapters: the framework-neutral ingestion boundary of HTIR.

Importing this package registers all built-in adapters. Third-party packages
can add their own by (a) calling :func:`register_adapter` at import time, or
(b) declaring a ``htir.adapters`` entry point whose value is an import path to
a module that registers on import (loaded lazily by :func:`load_entry_point_adapters`).

Typical use::

    from htir import load_trace
    steps = load_trace("trace.json")            # auto-detect framework
    steps = load_trace(data, adapter="openai")  # or force one
"""

from __future__ import annotations

from htir.adapters.base import (
    CANONICAL_KEYS,
    TraceAdapter,
    available_adapters,
    canonical_step,
    coerce_json_args,
    detect_adapter,
    get_adapter,
    load_trace,
    read_source,
    register_adapter,
    tool_call,
)

# Importing each module runs its @register_adapter decorators.
from htir.adapters import (  # noqa: E402,F401  (imported for side effects)
    anthropic_messages,
    langchain,
    openai_messages,
    openinference,
    raw,
    terminal,
    turns,
)


def load_entry_point_adapters() -> list[str]:
    """
    Discover and register third-party adapters advertised under the
    ``htir.adapters`` entry-point group. Returns the names of newly loaded
    entry points. Safe to call more than once; failures are ignored so one
    broken plugin cannot break ingestion.
    """
    loaded: list[str] = []
    try:
        from importlib.metadata import entry_points
    except ImportError:  # pragma: no cover - Python < 3.8
        return loaded
    try:
        eps = entry_points(group="htir.adapters")
    except TypeError:  # pragma: no cover - older importlib API
        eps = entry_points().get("htir.adapters", [])  # type: ignore[attr-defined]
    for ep in eps:
        try:
            ep.load()  # module registers on import
            loaded.append(ep.name)
        except Exception:
            continue
    return loaded


__all__ = [
    "TraceAdapter",
    "register_adapter",
    "get_adapter",
    "available_adapters",
    "detect_adapter",
    "load_trace",
    "read_source",
    "canonical_step",
    "tool_call",
    "coerce_json_args",
    "CANONICAL_KEYS",
    "load_entry_point_adapters",
]
