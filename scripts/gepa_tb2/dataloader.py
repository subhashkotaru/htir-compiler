"""
TB2 split loader for GEPA (Track G).

Reads the *already-materialised* train/val/test split that the SkillOpt run
produced (``data/skillopt_runs/tb2_89/_generated_splits/...``), so GEPA
optimizes and is tested on the **exact same task partition** as SkillOpt --
the fairest possible head-to-head (same 27/9/53 tasks, same seed=42).

Falls back to carving a fresh deterministic ratio split from a flat task list
when no materialised split dir is given.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any


def _normalize(raw: dict) -> dict[str, Any]:
    task_name = str(raw.get("task_name") or raw.get("id") or raw.get("uid") or "").strip()
    return {
        "id": task_name,
        "task_name": task_name,
        "task_type": str(raw.get("task_type") or "terminal_bench"),
        "question": str(raw.get("question") or f"Solve Terminal-Bench 2.0 task '{task_name}'."),
    }


def _load_json_or_jsonl(path: Path) -> list[dict]:
    if path.is_dir():
        for cand in sorted(path.glob("*.json")) + sorted(path.glob("*.jsonl")):
            return _load_json_or_jsonl(cand)
        raise FileNotFoundError(f"no .json/.jsonl under {path}")
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".jsonl":
        return [json.loads(ln) for ln in text.splitlines() if ln.strip()]
    payload = json.loads(text)
    if not isinstance(payload, list):
        raise ValueError(f"expected a JSON array at top level of {path}")
    return payload


def load_materialized_split(split_dir: str) -> dict[str, list[dict]]:
    """Load ``{train,val,test}`` item lists from a materialised split dir."""
    root = Path(split_dir)
    out: dict[str, list[dict]] = {}
    for split in ("train", "val", "test"):
        sub = root / split
        rows = _load_json_or_jsonl(sub)
        out[split] = [_normalize(r) for r in rows if isinstance(r, dict)]
    return out


def carve_ratio_split(
    data_path: str, ratio: str = "3:1:6", seed: int = 42, limit: int = 0
) -> dict[str, list[dict]]:
    """Deterministic train/val/test carve from a flat task list (fallback)."""
    rows = [_normalize(r) for r in _load_json_or_jsonl(Path(data_path)) if isinstance(r, dict)]
    if limit and limit > 0:
        rows = rows[:limit]
    rng = random.Random(seed)
    rng.shuffle(rows)
    a, b, c = (int(x) for x in ratio.split(":"))
    n = len(rows)
    n_train = round(n * a / (a + b + c))
    n_val = round(n * b / (a + b + c))
    return {
        "train": rows[:n_train],
        "val": rows[n_train : n_train + n_val],
        "test": rows[n_train + n_val :],
    }
