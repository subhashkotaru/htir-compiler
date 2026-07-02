"""Small utility helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_raw_trace(path: str | Path) -> list[dict[str, Any]]:
    """
    Load a raw agent trace from a JSON-lines or JSON-array file.
    Each record should have at least 'request' and 'response' fields.
    """
    p = Path(path)
    text = p.read_text(encoding="utf-8")

    # Try JSON array first
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return data
    except json.JSONDecodeError:
        pass

    # Fall back to JSON-lines
    records = []
    for line in text.splitlines():
        line = line.strip()
        if line:
            records.append(json.loads(line))
    return records


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
