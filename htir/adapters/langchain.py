"""
Adapter for LangChain / LangGraph traces expressed as serialized messages: a
list of ``{type|role, content, tool_calls}`` where ``type`` is one of
``human``/``ai``/``system``/``tool`` (LangChain ``BaseMessage`` dicts), or a
LangGraph state ``{"messages": [...]}``. AI messages carry ``tool_calls`` as
``{name, args, id}``; tool messages carry the result and a ``tool_call_id``.

Dependency-free: it parses the serialized dict form, so LangChain does not need
to be installed.
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

_TYPE_KEYS = ("type", "role")
_AI = {"ai", "aimessage", "assistant"}
_HUMAN = {"human", "humanmessage", "user"}
_SYSTEM = {"system", "systemmessage"}
_TOOL = {"tool", "toolmessage", "function"}


def _messages(data: Any) -> list[dict] | None:
    if isinstance(data, dict) and isinstance(data.get("messages"), list):
        data = data["messages"]
    if isinstance(data, list) and data and all(isinstance(m, dict) for m in data):
        return data
    return None


def _msg_type(m: dict) -> str:
    # Support both {"type": "human"} and the lc-serialized
    # {"id": [..., "HumanMessage"], "kwargs": {...}} form.
    for k in _TYPE_KEYS:
        if m.get(k):
            return str(m[k]).lower()
    lc_id = m.get("id")
    if isinstance(lc_id, list) and lc_id:
        return str(lc_id[-1]).lower()
    return ""


def _kwargs(m: dict) -> dict:
    return m.get("kwargs") if isinstance(m.get("kwargs"), dict) else m


@register_adapter
class LangChainAdapter(TraceAdapter):
    name = "langchain"
    aliases = ("langgraph", "lc")
    priority = 55

    def detect(self, data: Any) -> bool:
        msgs = _messages(data)
        if not msgs:
            return False
        # Match only on LangChain-specific markers, never on the OpenAI-style
        # ``role`` field (which the OpenAI adapter owns): a ``type`` of
        # human/ai/..., the lc-serialized ``id: [..., "HumanMessage"]`` form,
        # or an AI message whose tool_calls use LangChain's ``args`` key.
        for m in msgs:
            t = str(m.get("type") or "").lower()
            if t in (_AI | _HUMAN | _SYSTEM | _TOOL) or t.endswith("message"):
                return True
            lc_id = m.get("id")
            if isinstance(lc_id, list) and lc_id and str(lc_id[-1]).lower().endswith("message"):
                return True
            body = _kwargs(m)
            if any(isinstance(tc, dict) and "args" in tc for tc in body.get("tool_calls") or []):
                return True
        return False

    def parse(self, data: Any) -> list[dict[str, Any]]:
        msgs = _messages(data) or []
        steps: list[dict[str, Any]] = []
        call_index: dict[str, tuple[int, int]] = {}
        pending_request: list[str] = []

        for msg in msgs:
            mtype = _msg_type(msg)
            body = _kwargs(msg)
            content = _text(body.get("content"))

            if mtype in _HUMAN or mtype in _SYSTEM:
                if content:
                    pending_request.append(f"[{'USER' if mtype in _HUMAN else 'SYSTEM'}] {content}")
                continue

            if mtype in _TOOL:
                cid = str(body.get("tool_call_id") or body.get("name") or "")
                target = call_index.get(cid)
                if target is not None:
                    s_idx, c_idx = target
                    steps[s_idx]["tool_calls"][c_idx]["result"] = content
                    steps[s_idx]["tool_calls"][c_idx]["status"] = (
                        "failure" if body.get("status") == "error" else "success"
                    )
                else:
                    steps.append(canonical_step(response=content, role_hint="tool_invocation"))
                continue

            # ai / assistant turn -> a step
            tool_calls = []
            for tc in body.get("tool_calls") or []:
                args, args_text = coerce_json_args(tc.get("args"))
                tool_calls.append(
                    tool_call(
                        name=str(tc.get("name") or "tool"),
                        arguments=args,
                        arguments_text=args_text,
                        tool_call_id=str(tc.get("id") or ""),
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
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            str(p.get("text") or p.get("content") or "") if isinstance(p, dict) else str(p)
            for p in content
        )
    return str(content)
