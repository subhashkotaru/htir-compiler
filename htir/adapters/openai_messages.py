"""
Adapter for OpenAI-style chat traces: a list of chat-completion ``messages``
(``role`` in system/user/assistant/tool, with assistant ``tool_calls`` and
``tool`` result messages). Also handles the OpenAI Agents / Assistants habit
of wrapping the list as ``{"messages": [...]}``.

Each assistant turn becomes one step; preceding user/system messages form its
request; assistant ``tool_calls`` become structured ``ToolCall``s, and later
``role: tool`` messages are matched back to them by ``tool_call_id``.
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


def _messages(data: Any) -> list[dict] | None:
    if isinstance(data, dict) and isinstance(data.get("messages"), list):
        data = data["messages"]
    if isinstance(data, list) and all(isinstance(m, dict) for m in data) and data:
        return data
    return None


@register_adapter
class OpenAIMessagesAdapter(TraceAdapter):
    name = "openai"
    aliases = ("openai_messages", "chatml")
    priority = 50

    def detect(self, data: Any) -> bool:
        msgs = _messages(data)
        if not msgs:
            return False
        roles = {m.get("role") for m in msgs}
        if not roles <= {"system", "user", "assistant", "tool", "function", "developer"}:
            return False
        # Anthropic also uses user/assistant but with list-of-block content;
        # defer to that adapter when content is structured blocks.
        if any(isinstance(m.get("content"), list) for m in msgs) and "tool" not in roles:
            return False
        return "assistant" in roles or "tool" in roles

    def parse(self, data: Any) -> list[dict[str, Any]]:
        msgs = _messages(data) or []
        steps: list[dict[str, Any]] = []
        # tool_call_id -> (step_index, tool_call_index) for wiring results back.
        call_index: dict[str, tuple[int, int]] = {}
        pending_request: list[str] = []

        for msg in msgs:
            role = msg.get("role")
            content = _text(msg.get("content"))

            if role in ("user", "system", "developer"):
                if content:
                    pending_request.append(f"[{role.upper()}] {content}")
                continue

            if role in ("tool", "function"):
                # A tool result. Wire it back to the call that produced it.
                cid = str(msg.get("tool_call_id") or msg.get("name") or "")
                target = call_index.get(cid)
                if target is not None:
                    s_idx, c_idx = target
                    steps[s_idx]["tool_calls"][c_idx]["result"] = content
                    steps[s_idx]["tool_calls"][c_idx]["status"] = "success"
                else:
                    # Orphan tool result: keep it as its own step so nothing is lost.
                    steps.append(canonical_step(response=content, role_hint="tool_invocation"))
                continue

            # assistant turn -> a step
            tool_calls: list[dict[str, Any]] = []
            for tc in msg.get("tool_calls") or []:
                fn = tc.get("function") or {}
                args, args_text = coerce_json_args(fn.get("arguments"))
                cid = str(tc.get("id") or "")
                tool_calls.append(
                    tool_call(
                        name=str(fn.get("name") or tc.get("name") or "tool"),
                        arguments=args,
                        arguments_text=args_text,
                        tool_call_id=cid,
                        raw=tc,
                    )
                )
            step = canonical_step(
                request="\n".join(pending_request),
                response=content,
                tool_calls=tool_calls or None,
                role_hint="tool_invocation" if tool_calls else None,
            )
            pending_request = []
            steps.append(step)
            for c_idx, tc in enumerate(step.get("tool_calls", [])):
                if tc["tool_call_id"]:
                    call_index[tc["tool_call_id"]] = (len(steps) - 1, c_idx)

        return steps


def _text(content: Any) -> str:
    """Flatten message content (str, or a list of content parts) to text."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for p in content:
            if isinstance(p, dict):
                parts.append(str(p.get("text") or p.get("content") or ""))
            else:
                parts.append(str(p))
        return "\n".join(x for x in parts if x)
    return str(content)
