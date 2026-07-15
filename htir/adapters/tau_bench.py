"""
Deterministic τ-bench trace adapter (AVG ``Parse_{S_d}`` for the ``tau_bench``
policy domain).

τ-bench (Sierra's tool-agent benchmark: ``retail`` and ``airline`` customer-
service tasks) records a trajectory as OpenAI chat ``messages`` -- a system
message carrying the *domain policy* (the agent SOP), user turns, assistant
turns that emit ``tool_calls``, and ``role: tool`` results. The generic
``openai`` adapter already carries those tool calls through, but -- like the
generic ``turns`` adapter for terminal traces -- it assigns no operation type,
execution status, or artifact effect, so with no API key every obligation
abstains (the "all-abstained -> falsely valid" pathology).

This adapter closes that gap *deterministically* (no LLM, fully offline), the
way ``htir.adapters.terminal`` does for Terminal-Bench. For each assistant step
it fills the three canonical-step annotation keys the compiler uses to
short-circuit the LLM pass:

* ``role_hint``   -- a ``tau_bench`` operation type, keyed off the tool name:
  a *read* (``get_*``/``find_*``/``search_*``/``list_*``/``calculate``), a
  *mutation* (``cancel_*``/``modify_*``/``book_*``/``update_*``/``return_*``/
  ``exchange_*``/``send_*``), a *transfer* to a human, a *plan* (``think``), or
  a plain conversational ``respond``.
* ``status_hint`` -- parsed from the tool result: τ-bench tool errors surface as
  ``"Error: ..."`` / ``"... not found"`` text, which reads as ``failure``; a
  non-empty non-error result reads as ``success``.
* ``artifact_effects`` -- a **mutation** records an ``artifact_change`` on the
  DB record it targets (``order_id`` / ``reservation_id`` / ``user_id`` from the
  tool arguments, else the tool name), and a **read** records a ``read_only``
  effect, so first-class artifact nodes + E_prov provenance edges are created.

The result is that the mechanical checkers can discharge execution-status and
artifact-provenance obligations, the ``authenticate-before-action`` domain
constraint (S_d.K_d) makes every mutation *policy-sensitive* (so an
un-policy-linked account-changing action is withheld rather than credited), and
the ``Omega_d`` policy artifact seeds one semantic policy-compliance obligation
per mutation -- the execution-free integrity signal a key turns on.

Source format: an OpenAI-messages trace, either a bare ``[...]`` list or the
τ-bench record wrapper ``{messages:[...], task_name, eval_result, meta}``.
Detection requires the message roles AND at least one τ-bench-looking tool name,
so this adapter wins over the generic ``openai`` adapter only for genuine
τ-bench traces.
"""

from __future__ import annotations

import re
from typing import Any

from htir.adapters.base import (
    TraceAdapter,
    canonical_step,
    coerce_json_args,
    register_adapter,
    tool_call,
)

# tau_bench operation-type names (kept in sync with domains/tau_bench.yaml).
ROLE_READ = "read_info"
ROLE_MUTATE = "mutate_state"
ROLE_TRANSFER = "transfer_to_human"
ROLE_PLAN = "orchestration_decision"
ROLE_RESPOND = "respond"
ROLE_FINAL = "final_submission"
ROLE_OTHER = "other"

# Tool-name -> operation-type classification. τ-bench tool names are verb-
# prefixed and stable across the retail/airline suites, so a small verb
# vocabulary (matched as a leading token, robust to the corpus's occasional
# typos like ``get_order_deails``) covers them. Mutation verbs change DB state;
# read verbs only retrieve it.
_MUTATE_VERBS = (
    "cancel", "modify", "book", "update", "return", "exchange", "send",
    "create", "delete", "remove", "add", "set", "place", "apply", "refund",
    "issue", "transfer_reservation",
)
_READ_VERBS = (
    "get", "find", "search", "list", "calculate", "check", "lookup", "view",
    "read", "retrieve", "fetch",
)
# Explicit tool names that don't follow the verb convention.
_PLAN_TOOLS = frozenset({"think", "thinking", "plan", "reason"})
_TRANSFER_TOOLS = frozenset({"transfer_to_human_agents", "transfer_to_human", "escalate"})

# Which argument keys name the DB record a mutation touches (for E_prov). Ordered
# most-specific first; the first present key wins as the artifact identifier.
_RESOURCE_ARG_KEYS = (
    "reservation_id", "order_id", "user_id", "payment_method_id",
    "product_id", "item_id", "flight_number", "confirmation_number",
)

# τ-bench tool-result failure markers (case-insensitive). τ-bench surfaces tool
# errors as free text rather than an exit code.
_ERROR_MARKERS = re.compile(
    r"^\s*error\b|\berror:|not found|cannot |could not |unable to |invalid |"
    r"no such |does not exist|is not |are not |must be |permission|not allowed|"
    r"not authorized|failed\b",
    re.IGNORECASE,
)


def _messages(data: Any) -> list[dict] | None:
    """Extract the OpenAI ``messages`` list from a bare list or τ-bench wrapper."""
    if isinstance(data, dict) and isinstance(data.get("messages"), list):
        data = data["messages"]
    if isinstance(data, list) and data and all(isinstance(m, dict) for m in data):
        return data
    return None


# τ-bench-signature tool-name stems (the actual retail/airline tool sets). A
# trace carrying one of these is unambiguously τ-bench, so this adapter only
# wins over the generic ``openai`` adapter for genuine τ-bench traces (a generic
# ``get_weather`` OpenAI trace must stay with ``openai``).
_TAU_SIGNATURE_TOOL_STEMS = (
    "find_user_id", "get_order_details", "get_product_details", "get_user_details",
    "get_user_profile", "list_all_product_types", "cancel_pending_order",
    "modify_pending_order", "exchange_delivered_order", "return_delivered_order",
    "modify_user_address", "get_reservation", "book_reservation", "cancel_reservation",
    "update_reservation", "search_direct_flight", "search_onestop_flight",
    "list_all_airports", "send_certificate", "transfer_to_human_agents",
)
# A τ-bench system prompt is the agent SOP; these phrases are stable across the
# retail/airline policies.
_TAU_POLICY_SIGNATURES = ("agent policy", "retail agent", "airline agent")


def _looks_like_tau(msgs: list[dict]) -> bool:
    """
    A τ-bench trace has OpenAI message roles AND a τ-bench signature: a
    recognised τ tool name, or an SOP system prompt alongside tool use. This is
    deliberately strict so the generic ``openai`` adapter keeps ordinary
    chat/tool traces.
    """
    roles = {m.get("role") for m in msgs}
    if not roles <= {"system", "user", "assistant", "tool", "function", "developer"}:
        return False

    has_tool_call = False
    for m in msgs:
        for tc in (m.get("tool_calls") or []):
            has_tool_call = True
            name = str((tc.get("function") or {}).get("name") or tc.get("name") or "").strip().lower()
            if any(name.startswith(stem) for stem in _TAU_SIGNATURE_TOOL_STEMS):
                return True

    if has_tool_call:
        for m in msgs:
            if m.get("role") in ("system", "developer"):
                text = _text(m.get("content")).lower()
                if any(sig in text for sig in _TAU_POLICY_SIGNATURES):
                    return True
    return False


def _tool_role(name: str) -> str:
    """Map a τ-bench tool name to a tau_bench operation type."""
    n = (name or "").strip().lower()
    if not n:
        return ROLE_OTHER
    if n in _TRANSFER_TOOLS:
        return ROLE_TRANSFER
    if n in _PLAN_TOOLS:
        return ROLE_PLAN
    head = n.split("_", 1)[0]
    # A leading mutation verb wins over a read verb (``send_certificate`` etc.).
    if head in _MUTATE_VERBS or any(n.startswith(v + "_") for v in _MUTATE_VERBS):
        return ROLE_MUTATE
    if head in _READ_VERBS or any(n.startswith(v + "_") for v in _READ_VERBS):
        return ROLE_READ
    return ROLE_OTHER


# Precedence when a step carries several tool calls: a state mutation is the
# most verification-relevant, then a transfer, then a read, then planning.
_ROLE_PRECEDENCE = {
    ROLE_MUTATE: 5, ROLE_TRANSFER: 4, ROLE_READ: 3, ROLE_PLAN: 2,
    ROLE_RESPOND: 1, ROLE_OTHER: 0,
}


def _resource_identifier(name: str, args: dict[str, Any]) -> str:
    """The DB record a call targets: a known id argument, else the tool name."""
    for key in _RESOURCE_ARG_KEYS:
        val = args.get(key)
        if val not in (None, ""):
            return f"{key}={val}"
    return name or "record"


def _parse_status(result_text: str) -> str | None:
    """
    Infer an ExecutionStatus name from a τ-bench tool result, or ``None``
    (-> 'unknown', abstain) when there is no result to judge. An error marker
    reads as ``failure``; any other non-empty result reads as ``success``.
    """
    text = (result_text or "").strip()
    if not text:
        return None
    if _ERROR_MARKERS.search(text):
        return "failure"
    return "success"


@register_adapter
class TauBenchAdapter(TraceAdapter):
    """
    Deterministic ``Parse_{S_d}`` for τ-bench (retail/airline) OpenAI-messages
    traces. Higher priority than the generic ``openai`` adapter (50) so genuine
    τ-bench traces route here, while leaving other OpenAI-messages traces to the
    generic adapter.
    """

    name = "tau_bench"
    aliases = ("tau", "taubench", "tau_retail", "tau_airline")
    priority = 55  # above openai (50) so τ-bench traces route here

    def detect(self, data: Any) -> bool:
        msgs = _messages(data)
        if not msgs:
            return False
        return _looks_like_tau(msgs)

    def parse(self, data: Any) -> list[dict[str, Any]]:
        msgs = _messages(data) or []
        steps: list[dict[str, Any]] = []
        # tool_call_id -> (step_index, tool_call_index) to wire results back.
        call_index: dict[str, tuple[int, int]] = {}
        pending_request: list[str] = []
        last_assistant_idx: int | None = None

        for msg in msgs:
            role = msg.get("role")
            content = _text(msg.get("content"))

            if role in ("user", "system", "developer"):
                if content and content.lower() != "null":
                    pending_request.append(f"[{str(role).upper()}] {content}")
                continue

            if role in ("tool", "function"):
                cid = str(msg.get("tool_call_id") or msg.get("name") or "")
                target = call_index.get(cid)
                status = _parse_status(content)
                if target is not None:
                    s_idx, c_idx = target
                    tc = steps[s_idx]["tool_calls"][c_idx]
                    tc["result"] = content
                    tc["status"] = status or "unknown"
                    # Fold the tool outcome into the step's status + artifact
                    # effect (the effect was recorded optimistically at the
                    # assistant turn; a failed result flips it to unknown so a
                    # failed mutation does not manufacture provenance).
                    _apply_result_to_step(steps[s_idx], c_idx, status)
                else:
                    steps.append(canonical_step(response=content, role_hint=ROLE_OTHER,
                                                status_hint=status))
                continue

            # assistant turn -> a step
            raw_calls = msg.get("tool_calls") or []
            tool_calls: list[dict[str, Any]] = []
            roles: list[str] = []
            effects: list[dict[str, Any]] = []
            for tc in raw_calls:
                fn = tc.get("function") or {}
                name = str(fn.get("name") or tc.get("name") or "")
                args, args_text = coerce_json_args(fn.get("arguments"))
                cid = str(tc.get("id") or "")
                tr = _tool_role(name)
                roles.append(tr)
                tool_calls.append(
                    tool_call(
                        name=name or "tool",
                        arguments=args,
                        arguments_text=args_text,
                        tool_call_id=cid,
                        raw=tc,
                    )
                )
                # Optimistic artifact effect; a later failed tool result clears
                # it via ``_apply_result_to_step``.
                if tr == ROLE_MUTATE:
                    ident = _resource_identifier(name, args)
                    effects.append({
                        "effect_category": "artifact_change",
                        "affected_resource": ident,
                        "observed_change": f"{name} modified {ident}",
                        "supporting_evidence": "",
                        "_tool_call_index": len(tool_calls) - 1,
                    })
                elif tr == ROLE_READ:
                    ident = _resource_identifier(name, args)
                    effects.append({
                        "effect_category": "read_only",
                        "affected_resource": ident,
                        "observed_change": f"{name} read {ident}",
                        "supporting_evidence": "",
                        "_tool_call_index": len(tool_calls) - 1,
                    })

            if roles:
                role = max(roles, key=lambda r: _ROLE_PRECEDENCE.get(r, 0))
            elif content and content.lower() != "null":
                role = ROLE_RESPOND
            else:
                role = ROLE_OTHER

            step = canonical_step(
                request="\n".join(pending_request) if pending_request else "(no prior context)",
                response="" if content.lower() == "null" else content,
                tool_calls=tool_calls or None,
                role_hint=role,
                artifact_effects=effects or None,
                metadata={"n_tool_calls": len(tool_calls)},
            )
            pending_request = []
            steps.append(step)
            last_assistant_idx = len(steps) - 1
            for c_idx, tc in enumerate(step.get("tool_calls", [])):
                if tc["tool_call_id"]:
                    call_index[tc["tool_call_id"]] = (len(steps) - 1, c_idx)

        # The trailing assistant turn with no tool call is the final response to
        # the customer -> the final submission (τ-bench ends on the agent's
        # closing message, judged by a DB check the trace doesn't contain).
        if last_assistant_idx is not None:
            last = steps[last_assistant_idx]
            if not last.get("tool_calls") and last.get("role_hint") in (ROLE_RESPOND, ROLE_OTHER, None):
                last["role_hint"] = ROLE_FINAL

        # Strip the internal ``_tool_call_index`` bookkeeping key from effects.
        for s in steps:
            for eff in s.get("artifact_effects") or []:
                eff.pop("_tool_call_index", None)

        return steps


def _apply_result_to_step(step: dict[str, Any], tool_call_index: int, status: str | None) -> None:
    """
    Fold a tool result's outcome back into the assistant step: set the step's
    ``status_hint`` (a mutation/read step inherits the outcome of its call) and,
    on a *failed* call, drop the optimistic artifact effect for that call so a
    failed mutation does not create a spurious provenance edge.
    """
    if status is not None and not step.get("status_hint"):
        step["status_hint"] = status
    if status == "failure":
        effects = step.get("artifact_effects") or []
        step["artifact_effects"] = [
            e for e in effects if e.get("_tool_call_index") != tool_call_index
        ] or None


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
