"""
Passthrough adapter for traces that are already (close to) canonical: a JSON
array of step dicts using any of ``request``/``prompt``/``input`` and
``response``/``output``/``result``, optionally already carrying structured
``tool_calls``. This is the lowest-priority fallback so specific framework
formats win autodetection; it also covers hand-written / minimal logs.
"""

from __future__ import annotations

from typing import Any

from htir.adapters.base import CANONICAL_KEYS, TraceAdapter, register_adapter

_REQUEST_KEYS = ("request", "prompt", "input")
_RESPONSE_KEYS = ("response", "output", "result", "completion")


def _records(data: Any) -> list[dict] | None:
    if isinstance(data, dict) and isinstance(data.get("steps"), list):
        data = data["steps"]
    if isinstance(data, list) and data and all(isinstance(r, dict) for r in data):
        return data
    return None


@register_adapter
class RawStepsAdapter(TraceAdapter):
    name = "raw"
    aliases = ("passthrough", "steps")
    priority = 5  # low: only wins when no framework-specific adapter matches

    def detect(self, data: Any) -> bool:
        records = _records(data)
        if not records:
            return False
        keys = set().union(*(r.keys() for r in records))
        return bool(keys & set(_REQUEST_KEYS + _RESPONSE_KEYS + ("tool_calls",)))

    def parse(self, data: Any) -> list[dict[str, Any]]:
        records = _records(data) or []
        steps: list[dict[str, Any]] = []
        for r in records:
            request = next((str(r[k]) for k in _REQUEST_KEYS if r.get(k) is not None), "")
            response = next((str(r[k]) for k in _RESPONSE_KEYS if r.get(k) is not None), "")
            step: dict[str, Any] = {"request": request, "response": response}
            if r.get("tool_calls"):
                step["tool_calls"] = r["tool_calls"]
            for k in ("role_hint", "status_hint", "artifact_effects"):
                if r.get(k) is not None:
                    step[k] = r[k]
            extra = {k: v for k, v in r.items() if k not in CANONICAL_KEYS and k not in _REQUEST_KEYS + _RESPONSE_KEYS}
            if extra:
                step["metadata"] = extra
            steps.append(step)
        return steps
