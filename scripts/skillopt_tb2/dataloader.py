"""
Terminal-Bench 2.0 task loader for SkillOpt (Track S).

Items are thin task descriptors (``id`` / ``task_name``); the actual Docker
environment and verifier live inside harbor's ``terminal-bench@2.0`` dataset
and are resolved at rollout time by task name. The loader only owns the
train/val/test split SkillOpt's held-out gate needs.
"""

from __future__ import annotations

import json
from pathlib import Path

from skillopt.datasets.base import SplitDataLoader


def _normalize_item(raw: dict) -> dict:
    task_name = str(raw.get("task_name") or raw.get("id") or raw.get("uid") or "").strip()
    return {
        "id": task_name,
        "task_name": task_name,
        "task_type": str(raw.get("task_type") or "terminal_bench"),
        "question": str(raw.get("question") or f"Solve Terminal-Bench 2.0 task '{task_name}'."),
    }


class TerminalBenchDataLoader(SplitDataLoader):
    """
    Dataset-backed loader over a JSON/JSONL of TB2 task names.

    Supports SkillOpt's two split modes:

    * ``split_mode="ratio"`` -- deterministically carve train/val/test from a
      single ``data_path`` JSONL (the pilot default).
    * ``split_mode="split_dir"`` -- read an already-materialised
      ``split_dir/{train,val,test}/`` layout.
    """

    def load_split_items(self, split_path: str) -> list[dict]:
        path = Path(split_path)
        json_files = sorted(path.glob("*.json"))
        if json_files:
            payload = json.loads(json_files[0].read_text(encoding="utf-8"))
            if not isinstance(payload, list):
                raise ValueError(f"Expected JSON array at top level of {json_files[0]}")
            return [_normalize_item(row) for row in payload if isinstance(row, dict)]

        jsonl_files = sorted(path.glob("*.jsonl"))
        if jsonl_files:
            items: list[dict] = []
            with jsonl_files[0].open(encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    row = json.loads(line)
                    if isinstance(row, dict):
                        items.append(_normalize_item(row))
            return items

        raise FileNotFoundError(f"No .json or .jsonl file found in {split_path}")

    def load_raw_items(self, data_path: str) -> list[dict]:
        """Load the full task list for ``split_mode='ratio'``."""
        path = Path(data_path)
        if not path.exists():
            raise FileNotFoundError(f"TB2 task list not found: {data_path}")
        if path.is_dir():
            return self.load_split_items(str(path))
        if path.suffix == ".jsonl":
            items: list[dict] = []
            with path.open(encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    row = json.loads(line)
                    if isinstance(row, dict):
                        items.append(_normalize_item(row))
            return items
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError(f"Expected JSON array at top level of {path}")
        return [_normalize_item(row) for row in payload if isinstance(row, dict)]
