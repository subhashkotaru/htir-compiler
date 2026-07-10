"""
Deterministic terminal / CLI trace adapter (AVG ``Parse_{S_d}`` for the
``terminal_swe`` domain).

Terminal-Bench-style traces are free-text: an agent emits shell commands and
file edits, and the environment returns stdout/stderr with exit codes. The
generic ``turns`` adapter carries the tool calls through but assigns no
operation type, execution status, or artifact effect -- so with no API key the
LLM annotation pass is skipped, every step falls back to ``other``/
``tool_invocation``, no artifact binds, and every obligation abstains (the
"all-abstained -> falsely valid" pathology this adapter + the ``witness``
aggregation fix together resolve).

This adapter closes that gap *deterministically* (no LLM, fully offline). For
each agent step it fills the three canonical-step annotation keys the compiler
uses to short-circuit the LLM pass -- ``role_hint`` (an operation type in the
``terminal_swe`` vocabulary), ``status_hint`` (parsed from exit-code / error
markers in the observation), and ``artifact_effects`` (file mutations/reads so
first-class artifact nodes + E_prov provenance edges are created). The result
is that mechanical checkers can actually discharge execution-status and
artifact-provenance obligations instead of abstaining on all of them.

Source format: the same ``{steps:[{src,msg,tools,obs}], ...}`` turn schema as
``htir.adapters.turns`` (this repo's ``data/raw_traces`` and the
``yoonholee/terminalbench-trajectories`` HF dataset), where each ``tools``
entry is ``{fn, cmd}`` -- ``fn`` the tool name (Bash/Edit/Write/Read/...) and
``cmd`` the shell command or file path. Detection additionally requires that
the tools look like terminal/file operations, so this adapter wins over the
generic ``turns`` adapter only for genuinely terminal traces.

Exit-code parsing handles both the local ``data/raw_traces`` convention
(``Exit code N`` / ``[error] tool reported failure``) and the HF
Terminal-Bench convention (``<returncode>N</returncode>``).
"""

from __future__ import annotations

import re
from typing import Any

from htir.adapters.base import (
    TraceAdapter,
    canonical_step,
    register_adapter,
    tool_call,
)

# ---------------------------------------------------------------------------
# Tool-name -> operation-type (S_d.P_d for terminal_swe) classification.
#
# Names are matched case-insensitively against a small vocabulary that covers
# the common terminal-agent tool sets (Claude Code, Codex, generic shells).
# ---------------------------------------------------------------------------

_EDIT_TOOLS = frozenset({
    "write", "edit", "multiedit", "create", "create_file", "str_replace",
    "str_replace_editor", "apply_patch", "patch", "notebookedit", "insert",
})
_READ_TOOLS = frozenset({
    "read", "cat", "view", "open", "grep", "glob", "ls", "find", "search",
    "readfile", "read_file",
})
_SHELL_TOOLS = frozenset({
    "bash", "shell", "sh", "run", "exec", "execute", "terminal", "command",
    "run_command", "run_terminal_cmd", "run_shell_command",
})
_PLAN_TOOLS = frozenset({"todowrite", "todo", "plan", "think", "thinking"})

# terminal_swe operation-type names (kept in sync with domains/terminal_swe.yaml).
ROLE_EDIT = "edit_file"
ROLE_READ = "read_file"
ROLE_RUN = "run_command"
ROLE_TEST = "run_test"
ROLE_PLAN = "orchestration_decision"
ROLE_FINAL = "final_submission"
ROLE_OTHER = "other"

# A shell command counts as a test run when it invokes a recognised test
# runner. Substring/word match over the lowered command text.
_TEST_COMMAND = re.compile(
    r"\b(pytest|unittest|py\.test|nose2?|tox|"
    r"npm (?:run )?test|yarn test|jest|mocha|vitest|"
    r"go test|cargo test|rspec|phpunit|ctest|"
    r"make (?:test|check)|\./run_tests?|run_tests?\.sh|"
    r"rscript[^\n]*\btest\s*\(|source\([^)]*\)[^\n]*\btest\s*\()",
    re.IGNORECASE,
)

# Exit-code / failure markers in an observation. Ordered most-specific first.
_RETURNCODE_TAG = re.compile(r"<returncode>\s*(-?\d+)\s*</returncode>", re.IGNORECASE)
_EXIT_CODE = re.compile(r"\bexit(?:\s*code|ed with(?: code)?|status)?\s*[:=]?\s*(-?\d+)", re.IGNORECASE)
_ERROR_MARKERS = re.compile(
    r"\[error\]|tool reported failure|traceback \(most recent call last\)|"
    r"command not found|no such file or directory|permission denied|"
    r"segmentation fault|fatal:|panic:|assertionerror|"
    r"\b\d+ (?:failed|error)s?\b",
    re.IGNORECASE,
)
# A bare "$NN" observation is an elided/externalised reference, not real output.
_ELIDED_OBS = re.compile(r"^\$[0-9a-fx]+$")


def _turns(data: Any) -> list[dict] | None:
    if isinstance(data, dict) and isinstance(data.get("steps"), list):
        data = data["steps"]
    if isinstance(data, list) and data and all(isinstance(t, dict) for t in data):
        return data
    return None


def _tool_role(fn: str, cmd: str) -> str:
    """Map a single (tool name, command/path) to a terminal_swe operation type."""
    f = (fn or "").strip().lower()
    if f in _EDIT_TOOLS:
        return ROLE_EDIT
    if f in _SHELL_TOOLS:
        return ROLE_TEST if _TEST_COMMAND.search(cmd or "") else ROLE_RUN
    if f in _READ_TOOLS:
        return ROLE_READ
    if f in _PLAN_TOOLS:
        return ROLE_PLAN
    return ROLE_OTHER


# Precedence when a step carries several tool calls: a validation/test run is
# the most verification-relevant, then an edit, then a command, then a read,
# then planning. Higher wins.
_ROLE_PRECEDENCE = {
    ROLE_TEST: 5, ROLE_EDIT: 4, ROLE_RUN: 3, ROLE_READ: 2, ROLE_PLAN: 1, ROLE_OTHER: 0,
}


def _step_role(tools: list[dict]) -> str:
    roles = [_tool_role(str(t.get("fn") or ""), str(t.get("cmd") or "")) for t in tools]
    if not roles:
        return ROLE_OTHER
    return max(roles, key=lambda r: _ROLE_PRECEDENCE.get(r, 0))


def _parse_status(obs_text: str, response: str) -> str | None:
    """
    Infer an ExecutionStatus name ('success'/'failure') from an observation,
    or ``None`` (-> 'unknown', abstain) when the outcome is not determinable.

    Precedence: explicit ``<returncode>``/``exit code`` (0 = success, else
    failure) beats error-marker heuristics, which beat "output present, no
    error marker" (a conservative success only when there is real output).
    """
    text = f"{obs_text}\n{response}"
    m = _RETURNCODE_TAG.search(text) or _EXIT_CODE.search(text)
    if m:
        return "success" if m.group(1) == "0" else "failure"
    if _ERROR_MARKERS.search(text):
        return "failure"
    stripped = obs_text.strip()
    if not stripped or _ELIDED_OBS.match(stripped):
        # No usable observation (absent or an elided "$NN" reference).
        return None
    # Real output with no exit code and no error marker: a deterministic tool
    # (file read/edit) or a clean command. Conservatively 'success'.
    return "success"


def _artifact_effects(role: str, tools: list[dict], obs_text: str) -> list[dict[str, Any]]:
    """
    Emit ArtifactStateEvidence dicts for the file operations in a step so
    ``_extract_artifacts`` lifts first-class artifact nodes + E_prov edges.
    Only file-bearing tools (edit/read) produce artifact effects; command/test
    runs are verified via execution status, not artifact provenance.
    """
    effects: list[dict[str, Any]] = []
    for t in tools:
        fn = str(t.get("fn") or "")
        path = str(t.get("cmd") or "").strip()
        tool_role = _tool_role(fn, path)
        if tool_role == ROLE_EDIT and path:
            effects.append({
                "effect_category": "artifact_change",
                "affected_resource": path,
                "observed_change": f"{fn} modified {path}",
                "supporting_evidence": obs_text[:500],
            })
        elif tool_role == ROLE_READ and path:
            effects.append({
                "effect_category": "read_only",
                "affected_resource": path,
                "observed_change": f"{fn} read {path}",
                "supporting_evidence": obs_text[:500],
            })
    return effects


@register_adapter
class TerminalAdapter(TraceAdapter):
    """
    Deterministic ``Parse_{S_d}`` for terminal/CLI traces in the
    ``{src,msg,tools,obs}`` turn format. Higher priority than the generic
    ``turns`` adapter so it wins autodetection for traces whose tools are
    clearly terminal/file operations, while leaving non-terminal turn traces
    to ``turns``.
    """

    name = "terminal"
    aliases = ("terminal_swe", "cli", "terminalbench")
    priority = 45  # above turns (40) so terminal traces route here

    def detect(self, data: Any) -> bool:
        turns = _turns(data)
        if not turns:
            return False
        # Must be the turn shape AND carry at least one recognised terminal
        # tool, so we don't hijack non-terminal {src,msg} traces from `turns`.
        for t in turns:
            for tl in (t.get("tools") or []):
                fn = str(tl.get("fn") or "").strip().lower()
                if fn in _EDIT_TOOLS or fn in _READ_TOOLS or fn in _SHELL_TOOLS:
                    return True
        return False

    def parse(self, data: Any) -> list[dict[str, Any]]:
        turns = _turns(data) or []
        steps: list[dict[str, Any]] = []
        pending_request: list[str] = []
        # Index of the last agent step, so the trailing prose turn (no tools)
        # can be typed as the final submission.
        last_agent_idx: int | None = None

        for turn in turns:
            src = turn.get("src", "agent")
            msg = str(turn.get("msg", "") or "")
            tools = turn.get("tools") or []
            obs = turn.get("obs")

            if src == "user":
                pending_request.append(f"[USER] {msg}")
                continue

            obs_text = "" if obs is None else str(obs)
            role = _step_role(tools)
            status = _parse_status(obs_text, msg)
            effects = _artifact_effects(role, tools, obs_text)

            tool_calls = [
                tool_call(
                    name=str(t.get("fn") or "tool"),
                    arguments_text=str(t.get("cmd") or ""),
                    result=obs_text if (i == len(tools) - 1 and obs_text) else "",
                    status=status or "unknown",
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
                    role_hint=role,
                    status_hint=status,
                    artifact_effects=effects or None,
                    metadata={"src": src, "has_obs": obs is not None},
                )
            )
            last_agent_idx = len(steps) - 1
            pending_request = []

        # The trailing agent turn with no tool call is the final submission.
        if last_agent_idx is not None:
            last = steps[last_agent_idx]
            if not last.get("tool_calls") and last.get("role_hint") in (ROLE_OTHER, None):
                last["role_hint"] = ROLE_FINAL

        return steps
