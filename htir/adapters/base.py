"""
Trace adapters: turn a trace from *any* agent framework into the canonical
step list HTIR compiles.

This is the framework-neutral boundary of HTIR. An adapter's only job is to
map a source trace (OpenAI/Anthropic chat messages, a LangChain/LangGraph run,
OpenInference/OpenTelemetry spans, a bespoke JSON log, ...) into a list of
**canonical step dicts**, without deciding anything about verification.

Canonical step dict (all keys optional except that a step should carry at
least a ``request`` or a ``response`` or a ``tool_calls`` entry)::

    {
        "request":     str,   # what prompted this step (user/system/prior turn)
        "response":    str,   # the agent/tool/model output text for this step
        "tool_calls":  [ {name, arguments, arguments_text, result,
                          status, tool_call_id, raw}, ... ],
        "role_hint":   str,   # optional S_d operation-type hint (skips the LLM
                              # annotation pass -> fully offline compile)
        "status_hint": str,   # optional execution status hint
        "artifact_effects": [ {effect_category, affected_resource,
                               observed_change, supporting_evidence}, ... ],
        "metadata":    dict,  # anything else, preserved on TraceStep.raw_metadata
    }

Adapters are registered by name (see ``register_adapter``) so new frameworks
can be added -- including by third-party packages via the ``htir.adapters``
entry-point group -- without editing HTIR core. Use ``load_trace`` for the
common "give me a path or some data, figure out the framework" path.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

# Canonical step keys the compiler understands (used to split known keys from
# free-form metadata).
CANONICAL_KEYS = frozenset(
    {"request", "response", "tool_calls", "role_hint", "status_hint", "artifact_effects", "metadata"}
)


class TraceAdapter(ABC):
    """
    Base class for a trace adapter. Subclass, set ``name`` (and optionally
    ``aliases``/``priority``), implement ``detect`` and ``parse``, and register
    with ``@register_adapter``.
    """

    #: Primary registry name, e.g. ``"openai"``.
    name: str = ""
    #: Alternate names this adapter also answers to.
    aliases: tuple[str, ...] = ()
    #: Autodetection priority; higher is tried first. Generic/passthrough
    #: adapters should use a low value so specific formats win.
    priority: int = 0

    @abstractmethod
    def detect(self, data: Any) -> bool:
        """Return True if ``data`` looks like this adapter's source format."""

    @abstractmethod
    def parse(self, data: Any) -> list[dict[str, Any]]:
        """Map an in-memory source trace to canonical step dicts."""

    def load(self, path: str | Path) -> list[dict[str, Any]]:
        """Read a JSON/JSONL file at ``path`` and ``parse`` it."""
        return self.parse(read_source(path))


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_REGISTRY: dict[str, TraceAdapter] = {}


def register_adapter(adapter: type[TraceAdapter] | TraceAdapter) -> type[TraceAdapter] | TraceAdapter:
    """
    Register an adapter under its ``name`` and any ``aliases``. Usable as a
    class decorator (``@register_adapter``) or called with an instance/class.
    Returns its argument unchanged so it works as a decorator.
    """
    instance = adapter() if isinstance(adapter, type) else adapter
    if not instance.name:
        raise ValueError(f"{type(instance).__name__} must set a non-empty 'name'")
    for key in (instance.name, *instance.aliases):
        _REGISTRY[key] = instance
    return adapter


def get_adapter(name: str) -> TraceAdapter:
    """Look up a registered adapter by name or alias."""
    try:
        return _REGISTRY[name]
    except KeyError:
        raise KeyError(
            f"Unknown adapter '{name}'. Available: {', '.join(available_adapters())}"
        ) from None


def available_adapters() -> list[str]:
    """Sorted list of unique primary adapter names."""
    return sorted({a.name for a in _REGISTRY.values()})


def detect_adapter(data: Any) -> TraceAdapter:
    """
    Return the highest-priority registered adapter whose ``detect`` accepts
    ``data``. Raises ``ValueError`` if nothing matches.
    """
    for adapter in sorted(set(_REGISTRY.values()), key=lambda a: -a.priority):
        try:
            if adapter.detect(data):
                return adapter
        except Exception:
            # A misbehaving third-party adapter must not break detection.
            continue
    raise ValueError(
        "No registered adapter recognised this trace. Pass an explicit "
        f"adapter= (one of: {', '.join(available_adapters())})."
    )


# ---------------------------------------------------------------------------
# Convenience entry point
# ---------------------------------------------------------------------------

def read_source(source: str | Path) -> Any:
    """Read a ``.json`` (object/array) or ``.jsonl`` file into Python data."""
    text = Path(source).read_text(encoding="utf-8")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # JSON-lines fallback.
        return [json.loads(line) for line in text.splitlines() if line.strip()]


def load_trace(source: str | Path | list | dict, adapter: str = "auto") -> list[dict[str, Any]]:
    """
    Load a trace from any supported framework into canonical step dicts.

    ``source`` may be a path to a JSON/JSONL file or already-loaded data.
    ``adapter`` selects the adapter by name, or ``"auto"`` (the default) to
    autodetect from the data's shape.
    """
    data = read_source(source) if isinstance(source, (str, Path)) else source
    chosen = detect_adapter(data) if adapter == "auto" else get_adapter(adapter)
    return chosen.parse(data)


# ---------------------------------------------------------------------------
# Helpers shared by concrete adapters
# ---------------------------------------------------------------------------

def canonical_step(
    *,
    request: str = "",
    response: str = "",
    tool_calls: list[dict[str, Any]] | None = None,
    role_hint: str | None = None,
    status_hint: str | None = None,
    artifact_effects: list[dict[str, Any]] | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a canonical step dict, omitting empty optional keys."""
    step: dict[str, Any] = {"request": request, "response": response}
    if tool_calls:
        step["tool_calls"] = tool_calls
    if role_hint is not None:
        step["role_hint"] = role_hint
    if status_hint is not None:
        step["status_hint"] = status_hint
    if artifact_effects:
        step["artifact_effects"] = artifact_effects
    if metadata:
        step["metadata"] = metadata
    return step


def tool_call(
    name: str,
    *,
    arguments: dict[str, Any] | None = None,
    arguments_text: str = "",
    result: str = "",
    status: str = "unknown",
    tool_call_id: str = "",
    raw: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a canonical tool-call dict (matches ``htir.models.htir.ToolCall``)."""
    return {
        "name": name,
        "arguments": arguments or {},
        "arguments_text": arguments_text,
        "result": result,
        "status": status,
        "tool_call_id": tool_call_id,
        "raw": raw or {},
    }


def coerce_json_args(value: Any) -> tuple[dict[str, Any], str]:
    """
    Normalise a tool-argument payload into ``(parsed_dict, raw_text)``. A JSON
    string is parsed when it decodes to an object; anything non-dict is kept as
    text so nothing is silently dropped.
    """
    if isinstance(value, dict):
        return value, ""
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, dict):
                return parsed, ""
        except (json.JSONDecodeError, TypeError):
            pass
        return {}, value
    if value is None:
        return {}, ""
    return {}, str(value)
