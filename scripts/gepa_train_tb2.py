#!/usr/bin/env python
"""
Track G training driver: run GEPA's Pareto reflective prompt-evolution against
Terminal-Bench 2.0, on the same frozen terminus-2 + skill-injection harness as
SkillOpt (Track S).

DELIBERATELY OUTSIDE THE ``htir`` PACKAGE (like ``scripts/skillopt_train_tb2``):
the live harbor rollouts are kept out of the deterministic core. Reuses
``scripts.skillopt_tb2.rollout.run_batch`` for the harness, so GEPA vs. SkillOpt
isolates the *optimizer* (Pareto evolution vs. single-best reflect-and-gate).

Prerequisites::

    pip install "gepa==0.1.4"
    pip install --only-binary=:all: harbor   # same as Track M/S
    export OPENAI_API_KEY=<key>              # or .env
    # Docker daemon up for live rollouts

Usage::

    # Offline wiring check (no harbor / no API spend for rollouts):
    python scripts/gepa_train_tb2.py --config configs/gepa/terminal_bench.yaml --dry-run

    # Real (expensive) optimization:
    python scripts/gepa_train_tb2.py --config configs/gepa/terminal_bench.yaml

Deployed artifact: ``<out_root>/best_skill.md``. Head-to-head summary (GEPA best
vs. seed on the held-out test split): ``<out_root>/summary.json``.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _strip_frontmatter(text: str) -> str:
    """Return the body of a SKILL.md, dropping any leading ``---`` frontmatter.

    GEPA evolves the instruction body; ``run_batch`` re-wraps it in clean
    harbor skill frontmatter, so the frontmatter should not itself be mutated.
    """
    t = text.lstrip()
    if t.startswith("---"):
        parts = t.split("---", 2)
        if len(parts) == 3:
            return parts[2].strip()
    return text.strip()


def _mean(xs: list[float]) -> float:
    return statistics.fmean(xs) if xs else 0.0


def main(argv: list[str] | None = None) -> int:
    try:
        from dotenv import load_dotenv

        load_dotenv(REPO_ROOT / ".env")
    except ImportError:
        pass

    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--config", required=True, help="Path to GEPA YAML config")
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip harbor rollouts (stub zero scores) into a *_dryrun out_root. "
        "Exercises the full GEPA wiring with no docker/API spend.",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help="Allow resuming from an existing non-empty run_dir (default: refuse, "
        "to prevent the silent resume-skip that a reused output dir can cause).",
    )
    args = p.parse_args(argv)

    import yaml

    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = REPO_ROOT / cfg_path
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}

    def _resolve(rel: str | None) -> str:
        if not rel:
            return ""
        p2 = Path(rel)
        return str(p2 if p2.is_absolute() else (REPO_ROOT / p2).resolve())

    dry_run = bool(args.dry_run or cfg.get("dry_run"))

    # Strict dry-run / real separation: a dry-run NEVER writes into the real
    # out_root (this is exactly the reuse that caused SkillOpt's resume-skip).
    base_out = _resolve(cfg.get("out_root") or "data/gepa_runs/tb2_89")
    out_root = Path(base_out + ("_dryrun" if dry_run else ""))
    run_dir = out_root / "gepa_state"

    # Guard against the silent resume-skip: refuse to reuse a populated state
    # dir for a real run unless explicitly told to resume.
    if run_dir.exists() and any(run_dir.iterdir()) and not args.resume and not dry_run:
        print(
            f"[gepa_tb2] ERROR: run_dir already exists and is non-empty:\n  {run_dir}\n"
            "Refusing to start (a reused state dir silently skips optimization -- the exact\n"
            "bug that invalidated two SkillOpt runs). Delete it for a clean run, or pass\n"
            "--resume to continue the existing one.",
            file=sys.stderr,
        )
        return 2
    out_root.mkdir(parents=True, exist_ok=True)

    # -- Load the split (reuse SkillOpt's exact 27/9/53 partition when present) --
    from scripts.gepa_tb2 import (
        SKILL_COMPONENT,
        TerminalBenchGEPAAdapter,
        carve_ratio_split,
        load_materialized_split,
    )

    split_dir = _resolve(cfg.get("split_dir"))
    if split_dir and Path(split_dir).exists():
        splits = load_materialized_split(split_dir)
        split_src = f"materialized:{split_dir}"
    else:
        splits = carve_ratio_split(
            _resolve(cfg.get("data_path") or "data/skillopt/terminal_bench_tasks.json"),
            ratio=str(cfg.get("split_ratio") or "3:1:6"),
            seed=int(cfg.get("split_seed") or 42),
            limit=int(cfg.get("limit") or 0),
        )
        split_src = "carved"
    trainset, valset, testset = splits["train"], splits["val"], splits["test"]

    # -- Seed candidate = the same starter skill SkillOpt began from ----------
    skill_init_path = _resolve(cfg.get("skill_init") or "scripts/skills/skillopt-tb2-init/SKILL.md")
    seed_body = _strip_frontmatter(Path(skill_init_path).read_text(encoding="utf-8"))
    seed_candidate = {SKILL_COMPONENT: seed_body}

    harbor_model = str(cfg.get("harbor_model") or cfg.get("target_model") or "openai/gpt-4o-mini")
    reflection_lm = str(cfg.get("reflection_lm") or harbor_model)
    max_metric_calls = int(cfg.get("max_metric_calls") or 40)
    minibatch = int(cfg.get("reflection_minibatch_size") or 3)
    test_size = int(cfg.get("test_size") or 15)

    adapter = TerminalBenchGEPAAdapter(
        out_root=str(out_root),
        harbor_model=harbor_model,
        harbor_agent=str(cfg.get("harbor_agent") or "terminus-2"),
        harbor_env=str(cfg.get("harbor_env") or "docker"),
        harbor_max_retries=int(cfg.get("harbor_max_retries") or 0),
        workers=int(cfg.get("workers") or 1),
        dry_run=dry_run,
        inject_skill=bool(cfg.get("inject_skill", True)),
    )

    print(f"[gepa_tb2] config={cfg_path}", file=sys.stderr)
    print(f"[gepa_tb2] out_root={out_root}  dry_run={dry_run}", file=sys.stderr)
    print(f"[gepa_tb2] split={split_src}  train={len(trainset)} val={len(valset)} test={len(testset)}", file=sys.stderr)
    print(f"[gepa_tb2] harbor_model={harbor_model} agent={adapter.harbor_agent} workers={adapter.workers}", file=sys.stderr)
    print(f"[gepa_tb2] reflection_lm={reflection_lm} max_metric_calls={max_metric_calls} minibatch={minibatch}", file=sys.stderr)

    import gepa

    result = gepa.optimize(
        seed_candidate=seed_candidate,
        trainset=trainset,
        valset=valset,
        adapter=adapter,
        reflection_lm=reflection_lm,
        candidate_selection_strategy=str(cfg.get("candidate_selection_strategy") or "pareto"),
        reflection_minibatch_size=minibatch,
        max_metric_calls=max_metric_calls,
        run_dir=str(run_dir),
        seed=int(cfg.get("seed") or 0),
        display_progress_bar=False,
        raise_on_exception=False,
    )

    best_candidate = result.best_candidate
    best_skill = best_candidate[SKILL_COMPONENT] if isinstance(best_candidate, dict) else str(best_candidate)
    (out_root / "best_skill.md").write_text(best_skill + "\n", encoding="utf-8")

    print(
        f"[gepa_tb2] optimize done. candidates={result.num_candidates} "
        f"best_idx={result.best_idx} best_val={result.val_aggregate_scores[result.best_idx]:.4f} "
        f"total_metric_calls={result.total_metric_calls}",
        file=sys.stderr,
    )

    # -- Held-out test: seed vs. best on the same test tasks (SkillOpt-style) --
    test_slice = testset[: test_size if test_size > 0 else len(testset)]
    print(f"[gepa_tb2] held-out test eval on {len(test_slice)} tasks (seed vs best)...", file=sys.stderr)
    seed_eval = adapter.evaluate(test_slice, seed_candidate, capture_traces=False)
    best_eval = adapter.evaluate(test_slice, best_candidate if isinstance(best_candidate, dict) else {SKILL_COMPONENT: best_skill}, capture_traces=False)
    seed_hard = _mean([1.0 if s > 0 else 0.0 for s in seed_eval.scores])
    best_hard = _mean([1.0 if s > 0 else 0.0 for s in best_eval.scores])

    summary = {
        "experiment": "Track G -- GEPA (Pareto reflective evolution) on Terminal-Bench 2.0",
        "config": str(cfg_path),
        "dry_run": dry_run,
        "split_source": split_src,
        "n_train": len(trainset),
        "n_val": len(valset),
        "n_test_eval": len(test_slice),
        "harbor_model": harbor_model,
        "reflection_lm": reflection_lm,
        "max_metric_calls": max_metric_calls,
        "total_metric_calls": result.total_metric_calls,
        "num_candidates": result.num_candidates,
        "best_idx": result.best_idx,
        "best_val_aggregate": result.val_aggregate_scores[result.best_idx],
        "seed_val_aggregate": result.val_aggregate_scores[0] if result.val_aggregate_scores else None,
        "test_hard_seed": seed_hard,
        "test_hard_best": best_hard,
        "test_hard_delta": best_hard - seed_hard,
        "seed_scored_skill_unchanged": best_skill.strip() == seed_body.strip(),
    }
    (out_root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print("=" * 60, file=sys.stderr)
    print(f"[gepa_tb2] TEST (held-out, n={len(test_slice)}):", file=sys.stderr)
    print(f"    seed skill : hard={seed_hard:.4f}", file=sys.stderr)
    print(f"    GEPA best  : hard={best_hard:.4f}", file=sys.stderr)
    print(f"    delta      : {best_hard - seed_hard:+.4f}", file=sys.stderr)
    print(f"[gepa_tb2] best_skill -> {out_root / 'best_skill.md'}", file=sys.stderr)
    print(f"[gepa_tb2] summary   -> {out_root / 'summary.json'}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
