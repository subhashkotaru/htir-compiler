"""
Adapter for the ``{src, msg, tools, obs}`` turn format used by this repo's
own ``data/raw_traces`` (and similar minimal home-grown logs). Also accepts a
trial object ``{"steps": [ ...turns... ], ...meta}``.

User turns buffer into the next agent turn's request; agent turns become
steps; ``tools`` entries (``{fn, cmd}``) become structured ``ToolCall``s and
``obs`` becomes the tool result / observation. This supersedes the string-
concatenating ``htir.utils.io.normalize_turns`` by preserving tool structure.
"""

from __future__ import annotations

from typing import Any

from htir.adapters.base import (
    TraceAdapter,
    canonical_step,
    register_adapter,
    tool_call,
)


def _turns(data: Any) -> list[dict] | None:
    if isinstance(data, dict) and isinstance(data.get("steps"), list):
        data = data["steps"]
    if isinstance(data, list) and data and all(isinstance(t, dict) for t in data):
        return data
    return None


@register_adapter
class TurnsAdapter(TraceAdapter):
    name = "turns"
    aliases = ("src_msg", "trial")
    priority = 40

    def detect(self, data: Any) -> bool:
        turns = _turns(data)
        if not turns:
            return False
        return any(("src" in t or "msg" in t) for t in turns)

    def parse(self, data: Any) -> list[dict[str, Any]]:
        turns = _turns(data) or []
        steps: list[dict[str, Any]] = []
        pending_request: list[str] = []

        for turn in turns:
            src = turn.get("src", "agent")
            msg = str(turn.get("msg", "") or "")
            tools = turn.get("tools") or []
            obs = turn.get("obs")

            if src == "user":
                pending_request.append(f"[USER] {msg}")
                continue

            obs_text = "" if obs is None else str(obs)
            tool_calls = [
                tool_call(
                    name=str(t.get("fn") or "tool"),
                    arguments_text=str(t.get("cmd") or ""),
                    # A single observation trails the tool block; attach it to the
                    # last tool call so the call and its result stay linked.
                    result=obs_text if (i == len(tools) - 1 and obs_text) else "",
                    raw=t,
                )
                for i, t in enumerate(tools)
            ]
            response = msg
            if obs_text and not tools:
                response = f"{msg}\n\nObservation:\n{obs_text}" if msg else obs_text

            steps.append(
                canonical_step(
                    request="\n".join(pending_request) if pending_request else "(no prior context)",
                    response=response,
                    tool_calls=tool_calls or None,
                    metadata={"src": src, "has_obs": obs is not None},
                )
            )
            pending_request = []

        return steps
