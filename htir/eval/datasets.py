"""
Terminal-Bench trajectory ingestion (avg.tex Sec. 4.1 data; experiment-plan P0).

Loads the ``yoonholee/terminalbench-trajectories`` HF dataset -- 52,104 traces
over 89 Terminal-Bench tasks, each a ``{steps:[{src,msg,tools,obs}], reward,
agent, model, task_name}`` record (the same turn schema as
``data/raw_traces``) -- or any local JSON/JSONL in that schema, and draws a
balanced solved/unsolved sample.

The HF ``datasets`` dependency is imported lazily inside
``load_terminalbench`` so importing this module (and ``import htir``) never
requires it or network access. The balanced sampler and label plumbing are
pure Python and unit-testable offline.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Iterable, Iterator

from htir.adapters import load_trace
from htir.eval.weak_labels import extract_reward, label_from_reward

HF_REPO = "yoonholee/terminalbench-trajectories"


def iter_local_traces(paths: Iterable[str | Path]) -> Iterator[dict[str, Any]]:
    """
    Yield raw trace dicts from local ``.json`` (one object, or an array/
    ``{traces:[...]}`` wrapper) or ``.jsonl`` files in the turn schema.
    """
    for path in paths:
        p = Path(path)
        text = p.read_text(encoding="utf-8")
        if p.suffix == ".jsonl":
            for line in text.splitlines():
                if line.strip():
                    yield json.loads(line)
            continue
        data = json.loads(text)
        if isinstance(data, dict) and isinstance(data.get("traces"), list):
            yield from data["traces"]
        elif isinstance(data, list):
            yield from data
        else:
            yield data


def load_terminalbench(
    repo: str = HF_REPO,
    *,
    split: str = "train",
    limit: int | None = None,
    streaming: bool = True,
) -> list[dict[str, Any]]:
    """
    Load Terminal-Bench trajectories from the HF hub as raw trace dicts.

    Requires the optional ``datasets`` dependency (``pip install datasets``);
    raises an informative ``ImportError`` otherwise. ``streaming`` (default)
    avoids materializing all 52k traces; pass ``limit`` to cap how many are
    pulled.
    """
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise ImportError(
            "load_terminalbench requires the 'datasets' package. "
            "Install it with: pip install datasets"
        ) from exc

    ds = load_dataset(repo, split=split, streaming=streaming)
    out: list[dict[str, Any]] = []
    for i, rec in enumerate(ds):
        if limit is not None and i >= limit:
            break
        out.append(dict(rec))
    return out


def balanced_sample(
    traces: list[dict[str, Any]],
    n: int,
    *,
    seed: int = 0,
) -> list[dict[str, Any]]:
    """
    Draw up to ``n`` traces with an equal split of solved (reward != 0) and
    unsolved (reward = 0), per avg.tex's matched-condition reporting. Traces
    with no recorded reward are excluded. If one class is smaller than ``n/2``,
    the sample is capped at ``2 * min(class sizes)`` so it stays balanced.
    Deterministic given ``seed``.
    """
    solved = [t for t in traces if label_from_reward(extract_reward(t)) == "valid"]
    unsolved = [t for t in traces if label_from_reward(extract_reward(t)) == "invalid"]

    rng = random.Random(seed)
    rng.shuffle(solved)
    rng.shuffle(unsolved)

    per_class = min(n // 2, len(solved), len(unsolved))
    sample = solved[:per_class] + unsolved[:per_class]
    rng.shuffle(sample)
    return sample


def to_canonical_steps(trace: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Canonical step dicts for one raw trace via the terminal adapter, ready for
    ``TraceAbstractionAgent.compile``. Thin convenience over ``load_trace`` so
    callers don't reach into the adapter layer.
    """
    return load_trace(trace, adapter="terminal")
