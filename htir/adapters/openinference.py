"""
Adapter for OpenInference / OpenTelemetry GenAI span traces: a list of span
dicts, each with a flat ``attributes`` map (OpenInference's dotted-key
convention, e.g. ``openinference.span.kind``, ``input.value``,
``output.value``, ``tool.name``). This is the emerging cross-framework
standard emitted by tracers such as Arize Phoenix, OpenLLMetry, and the
OpenTelemetry GenAI semantic conventions, so one adapter covers many stacks.

Each span becomes a step. TOOL spans additionally produce a structured
``ToolCall``; span status (OK/ERROR) becomes the execution-status hint.
"""

from __future__ import annotations

from typing import Any

from htir.adapters.base import (
    TraceAdapter,
    canonical_step,
    coerce_json_args,
    register_adapter,
    tool_call,
)

_KIND_ATTR = "openinference.span.kind"
_ROLE_BY_KIND = {
    "TOOL": "tool_invocation",
    "RETRIEVER": "information_acquisition",
    "EMBEDDING": "information_acquisition",
    "AGENT": "orchestration_decision",
    "CHAIN": "orchestration_decision",
}


def _spans(data: Any) -> list[dict] | None:
    if isinstance(data, dict) and isinstance(data.get("spans"), list):
        data = data["spans"]
    if isinstance(data, list) and data and all(isinstance(s, dict) for s in data):
        return data
    return None


def _attrs(span: dict) -> dict:
    a = span.get("attributes")
    return a if isinstance(a, dict) else {}


def _kind(span: dict) -> str:
    return str(
        _attrs(span).get(_KIND_ATTR)
        or span.get("span_kind")
        or span.get("kind")
        or ""
    ).upper()


def _status(span: dict) -> str | None:
    raw = span.get("status_code") or span.get("status")
    if isinstance(raw, dict):
        raw = raw.get("status_code") or raw.get("code")
    if raw is None:
        return None
    s = str(raw).upper()
    if s in ("ERROR", "STATUS_CODE_ERROR", "2"):
        return "failure"
    if s in ("OK", "STATUS_CODE_OK", "1"):
        return "success"
    return None


@register_adapter
class OpenInferenceAdapter(TraceAdapter):
    name = "openinference"
    aliases = ("otel", "opentelemetry", "phoenix")
    priority = 70

    def detect(self, data: Any) -> bool:
        spans = _spans(data)
        if not spans:
            return False
        for s in spans:
            attrs = _attrs(s)
            if _KIND_ATTR in attrs or "input.value" in attrs or "output.value" in attrs:
                return True
            if ("span_kind" in s or "kind" in s) and "attributes" in s:
                return True
        return False

    def parse(self, data: Any) -> list[dict[str, Any]]:
        spans = _spans(data) or []
        # Preserve chronological order when start times are present.
        spans = sorted(spans, key=lambda s: s.get("start_time") or s.get("start_time_unix_nano") or 0) \
            if any(("start_time" in s or "start_time_unix_nano" in s) for s in spans) else spans

        steps: list[dict[str, Any]] = []
        for span in spans:
            attrs = _attrs(span)
            kind = _kind(span)
            request = str(attrs.get("input.value") or attrs.get("input") or "")
            response = str(attrs.get("output.value") or attrs.get("output") or "")
            status_hint = _status(span)

            tool_calls = None
            if kind == "TOOL":
                args, args_text = coerce_json_args(
                    attrs.get("tool.parameters") or attrs.get("input.value")
                )
                tool_calls = [
                    tool_call(
                        name=str(attrs.get("tool.name") or span.get("name") or "tool"),
                        arguments=args,
                        arguments_text=args_text,
                        result=response,
                        status=status_hint or "unknown",
                        raw=span,
                    )
                ]

            steps.append(
                canonical_step(
                    request=request,
                    response=response,
                    tool_calls=tool_calls,
                    role_hint=_ROLE_BY_KIND.get(kind),
                    status_hint=status_hint,
                    metadata={"span_name": span.get("name"), "span_kind": kind} if kind else None,
                )
            )
        return steps
