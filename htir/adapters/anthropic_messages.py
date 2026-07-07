"""
Adapter for Anthropic Messages-API traces: a list of ``{role, content}``
messages where ``content`` is a list of typed blocks (``text``, ``tool_use``,
``tool_result``). Assistant ``tool_use`` blocks become structured
``ToolCall``s; user ``tool_result`` blocks are matched back by ``tool_use_id``.
Also accepts ``{"messages": [...]}``.
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
    if isinstance(data, list) and data and all(isinstance(m, dict) for m in data):
        return data
    return None


def _blocks(content: Any) -> list[dict]:
    return [b for b in content if isinstance(b, dict)] if isinstance(content, list) else []


@register_adapter
class AnthropicMessagesAdapter(TraceAdapter):
    name = "anthropic"
    aliases = ("claude", "anthropic_messages")
    priority = 60

    def detect(self, data: Any) -> bool:
        msgs = _messages(data)
        if not msgs:
            return False
        # Signature: at least one message whose content is a list of typed
        # blocks including a tool_use/tool_result/text block.
        for m in msgs:
            for b in _blocks(m.get("content")):
                if b.get("type") in ("tool_use", "tool_result", "text"):
                    return m.get("role") in ("user", "assistant", None) or True
        return False

    def parse(self, data: Any) -> list[dict[str, Any]]:
        msgs = _messages(data) or []
        steps: list[dict[str, Any]] = []
        call_index: dict[str, tuple[int, int]] = {}
        pending_request: list[str] = []

        for msg in msgs:
            role = msg.get("role")
            blocks = _blocks(msg.get("content")) or (
                [{"type": "text", "text": msg.get("content")}] if isinstance(msg.get("content"), str) else []
            )

            texts = [str(b.get("text") or "") for b in blocks if b.get("type") == "text"]
            tool_uses = [b for b in blocks if b.get("type") == "tool_use"]
            tool_results = [b for b in blocks if b.get("type") == "tool_result"]

            # tool_result blocks (usually on a user turn) wire back to their call.
            for tr in tool_results:
                cid = str(tr.get("tool_use_id") or "")
                target = call_index.get(cid)
                result_text = _result_text(tr.get("content"))
                if target is not None:
                    s_idx, c_idx = target
                    steps[s_idx]["tool_calls"][c_idx]["result"] = result_text
                    steps[s_idx]["tool_calls"][c_idx]["status"] = (
                        "failure" if tr.get("is_error") else "success"
                    )
                elif result_text:
                    steps.append(canonical_step(response=result_text, role_hint="tool_invocation"))

            if role == "user":
                text = "\n".join(t for t in texts if t)
                if text:
                    pending_request.append(f"[USER] {text}")
                continue

            if role == "assistant" or tool_uses:
                tool_calls = []
                for tu in tool_uses:
                    args, args_text = coerce_json_args(tu.get("input"))
                    tool_calls.append(
                        tool_call(
                            name=str(tu.get("name") or "tool"),
                            arguments=args,
                            arguments_text=args_text,
                            tool_call_id=str(tu.get("id") or ""),
                            raw=tu,
                        )
                    )
                step = canonical_step(
                    request="\n".join(pending_request),
                    response="\n".join(t for t in texts if t),
                    tool_calls=tool_calls or None,
                    role_hint="tool_invocation" if tool_calls else None,
                )
                pending_request = []
                steps.append(step)
                for c_idx, tc in enumerate(step.get("tool_calls", [])):
                    if tc["tool_call_id"]:
                        call_index[tc["tool_call_id"]] = (len(steps) - 1, c_idx)

        return steps


def _result_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            str(b.get("text") or b.get("content") or "") for b in content if isinstance(b, dict)
        )
    return "" if content is None else str(content)
