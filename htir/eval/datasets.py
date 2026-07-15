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
# τ-bench trajectory corpus: per-model JSONL of full retail/airline runs with a
# DB-match ``eval_result.score`` reward (parallel to the terminalbench set).
TAU_HF_REPO = "AgentSuite/tau-bench-trajectories"
TAU_HF_BASE = f"https://huggingface.co/datasets/{TAU_HF_REPO}/resolve/main/"


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


def normalize_steps_field(trace: dict[str, Any]) -> dict[str, Any]:
    """
    Return ``trace`` with its ``steps`` field guaranteed to be a list.

    The HF ``terminalbench-trajectories`` dataset (and JSONL caches of it)
    serialize the per-row ``steps`` as a *JSON string*, whereas the adapters
    expect a decoded ``list[dict]``. Decode it if needed; leave an already-list
    (this repo's ``data/raw_traces`` convention) untouched. A copy is returned
    only when a decode was necessary, so the caller's dict is never mutated.
    """
    steps = trace.get("steps")
    if isinstance(steps, str):
        try:
            decoded = json.loads(steps)
        except (json.JSONDecodeError, ValueError):
            decoded = []
        trace = {**trace, "steps": decoded if isinstance(decoded, list) else []}
    return trace


def _is_tau_shaped(trace: dict[str, Any]) -> bool:
    """A τ-bench record carries OpenAI ``messages`` and no ``steps`` turn list."""
    return (
        isinstance(trace, dict)
        and isinstance(trace.get("messages"), list)
        and not isinstance(trace.get("steps"), list)
    )


def to_canonical_steps(trace: dict[str, Any], adapter: str | None = None) -> list[dict[str, Any]]:
    """
    Canonical step dicts for one raw trace, ready for
    ``TraceAbstractionAgent.compile``. Thin convenience over ``load_trace`` so
    callers don't reach into the adapter layer.

    ``adapter`` selects the parser; when ``None`` (the default) it is chosen by
    the record's *shape* so a single call site serves both corpora:

    * a τ-bench record (an OpenAI ``messages`` list, no ``steps``) routes to the
      ``tau_bench`` adapter over its ``messages``;
    * a Terminal-Bench turn record (``steps`` list, JSON-string-decoded first)
      routes to the ``terminal`` adapter, exactly as before.
    """
    if adapter is None:
        adapter = "tau_bench" if _is_tau_shaped(trace) else "terminal"
    if adapter in ("tau_bench", "tau", "openai", "openai_messages"):
        payload = trace["messages"] if _is_tau_shaped(trace) else trace
        return load_trace(payload, adapter=adapter)
    return load_trace(normalize_steps_field(trace), adapter=adapter)


# ---------------------------------------------------------------------------
# τ-bench (Sierra retail/airline) trajectory ingestion
# ---------------------------------------------------------------------------

# The subset of AgentSuite/tau-bench-trajectories model files pulled by
# ``load_tau_bench(hf=True)`` when no explicit list is given. A spread of agent
# families keeps the solved/unsolved mix broad rather than one model's quirks.
TAU_DEFAULT_HF_FILES = (
    "claude-4-sonnet-thinking-off.jsonl",
    "gemini-2.5-flash-thinking-off.jsonl",
    "Qwen3-235B-A22B-FP8.jsonl",
    "DeepSeek-V3-0324.jsonl",
    "Kimi-K2-Instruct.jsonl",
)


def normalize_tau_record(rec: dict[str, Any]) -> dict[str, Any]:
    """
    Normalize a raw ``AgentSuite/tau-bench-trajectories`` record into the fields
    the eval harness expects, without dropping the original payload:

    * ``reward``     -- int in {0, 1} from ``eval_result.score`` /
      ``meta.is_correct`` (so ``extract_reward`` / ``balanced_sample`` work).
    * ``task_name``  -- the per-task id (``meta.id``, e.g. ``retail_12``), which
      the experiment writers use as an opaque trace id.
    * ``tau_domain`` -- ``retail`` / ``airline`` (the original ``task_name``),
      kept for per-domain slicing / the policy-drift stress test.
    * ``messages``   -- the OpenAI-messages trajectory, passed through untouched
      so ``to_canonical_steps`` routes it to the ``tau_bench`` adapter.
    """
    meta = rec.get("meta") if isinstance(rec.get("meta"), dict) else {}
    reward = extract_reward(rec)
    task_id = str(meta.get("id") or rec.get("task_name") or "")
    return {
        **rec,
        "reward": reward,
        "task_name": task_id,
        "tau_domain": str(rec.get("task_name") or ""),
        "messages": rec.get("messages") or [],
    }


def load_tau_bench(
    paths: Iterable[str | Path] | None = None,
    *,
    hf: bool = False,
    hf_files: Iterable[str] | None = None,
    repo: str = TAU_HF_REPO,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """
    Load τ-bench trajectories as normalized raw trace dicts.

    ``paths`` reads local ``.jsonl`` caches of the corpus (the offline default;
    see ``data/tau_cache``). ``hf=True`` streams the per-model JSONL files from
    the HF hub over plain HTTP (no ``datasets`` dependency); ``hf_files`` picks
    which model files (defaults to :data:`TAU_DEFAULT_HF_FILES`). ``limit`` caps
    the number of records returned.
    """
    out: list[dict[str, Any]] = []
    if hf:
        import requests  # local, so importing this module needs no network

        for fname in (hf_files or TAU_DEFAULT_HF_FILES):
            resp = requests.get(TAU_HF_BASE + fname, timeout=120, stream=True)
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line:
                    continue
                out.append(normalize_tau_record(json.loads(line)))
                if limit is not None and len(out) >= limit:
                    return out
        return out

    if not paths:
        raise ValueError("load_tau_bench needs local paths= or hf=True")
    for rec in iter_local_traces(paths):
        out.append(normalize_tau_record(rec))
        if limit is not None and len(out) >= limit:
            break
    return out
