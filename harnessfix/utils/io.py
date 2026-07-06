"""Small utility helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_raw_trace(path: str | Path) -> list[dict[str, Any]]:
    """
    Load a raw agent trace and return a list of step records normalised to the
    ``{request, response, ...}`` shape expected by the compiler.

    Three input shapes are accepted:

    1. A trial object ``{"steps": [{"src", "msg", "tools", "obs"}, ...], ...}``
       (the format used by this repo's ``data/raw_traces``). It is normalised
       via :func:`normalize_turns`.
    2. A JSON array of records. If the records already carry ``request`` /
       ``response`` they are returned as-is; if they carry ``src`` / ``msg``
       they are treated as turns and normalised.
    3. JSON-lines, one record per line (same per-record handling as 2).
    """
    steps, _ = load_task_trace(path)
    return steps


def load_task_trace(path: str | Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """
    Like :func:`load_raw_trace` but also returns the trial-level metadata
    (task_name, agent, model, reward, ...) when present. Returns
    ``(normalised_steps, meta)``.
    """
    p = Path(path)
    text = p.read_text(encoding="utf-8")

    data: Any = None
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        # JSON-lines fallback
        turns = [json.loads(line) for line in text.splitlines() if line.strip()]
        return _records_to_steps(turns), {}

    # Trial object with a nested "steps" list of raw turns.
    if isinstance(data, dict) and isinstance(data.get("steps"), list):
        meta = {k: v for k, v in data.items() if k != "steps"}
        return _records_to_steps(data["steps"]), meta

    if isinstance(data, list):
        return _records_to_steps(data), {}

    raise ValueError(f"Unrecognised raw-trace format in {path}")


def _records_to_steps(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Route records either through turn-normalisation or straight through."""
    if records and any("src" in r or "msg" in r for r in records):
        return normalize_turns(records)
    return records


def normalize_turns(turns: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Convert raw ``{src, msg, tools, obs}`` turns into ``{request, response,
    src, tools, has_obs}`` step records.

    User turns are buffered and become the ``request`` context of the next
    agent turn (prefixed with ``[USER]``); agent turns become steps whose
    ``response`` embeds any tool calls and observation. This mirrors the
    normalisation used to produce ``data/htir_outputs``.
    """
    steps: list[dict[str, Any]] = []
    pending_user: list[str] = []

    for turn in turns:
        src = turn.get("src", "agent")
        msg = str(turn.get("msg", "") or "")
        tools = turn.get("tools") or []
        obs = turn.get("obs")

        if src == "user":
            pending_user.append(f"[USER] {msg}")
            continue

        # Agent (or tool/system) turn -> emit a step.
        request = "\n".join(pending_user) if pending_user else "(no prior context)"
        pending_user = []

        response = msg
        if tools:
            tool_lines = "\n".join(
                f"  [{t.get('fn', '')}] {t.get('cmd', '')}".rstrip() for t in tools
            )
            response = f"{msg}\n\nTool calls:\n{tool_lines}"
        if obs is not None:
            response = f"{response}\n\nObservation:\n{obs}"

        steps.append(
            {
                "request": request,
                "response": response,
                "src": src,
                "tools": tools,
                "has_obs": obs is not None,
            }
        )

    return steps


def load_harness_snippets(harness_root: str | Path, extensions: tuple[str, ...] = (".py", ".yaml", ".json", ".md")) -> dict[str, str]:
    """
    Walk the harness directory and collect the text content of files with
    the given extensions.  Returns {relative_path: content}.
    """
    root = Path(harness_root)
    snippets: dict[str, str] = {}
    if not root.exists():
        return snippets
    for p in root.rglob("*"):
        if p.is_file() and p.suffix in extensions:
            rel = str(p.relative_to(root))
            try:
                snippets[rel] = p.read_text(encoding="utf-8", errors="replace")
            except OSError:
                pass
    return snippets


def truncate(text: str, max_chars: int = 3000) -> str:
    """Truncate long text for LLM context, preserving the tail."""
    if len(text) <= max_chars:
        return text
    head = max_chars // 2
    tail = max_chars - head - 30
    return text[:head] + "\n...[truncated]...\n" + text[-tail:]
