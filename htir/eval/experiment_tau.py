"""
τ-bench campaign driver (paper-completion plan, Phases 1-3 on the policy domain).

Runs the SA experiment suite on the Sierra **τ-bench** (retail/airline) policy
domain -- the second, policy-bearing domain the completion plan calls for -- and
packages the results the way avg.tex Sec. 4.7 asks for: **mean ± standard error
over ≥3 seeds** (WP-0.1 / :mod:`htir.eval.seeds`), with an optional **LLM
slice** (``--use-llm``) that turns on the semantic policy-compliance checker,
the execution-free arm, and the monolithic-LLM judge on a small subsample so the
otherwise-offline numbers are grounded against a real model (E1).

Experiments (``--experiments``):

* ``sa1``      -- Q1 graph vs. monolith on τ-bench (false-valid rate).
* ``sa3``      -- Q3 calibrated abstention on τ-bench (false-valid reduction,
  AUROC/ECE of the shared score).
* ``sa6``      -- stress tests on τ-bench: the **policy-drift**, large-tool-menu,
  and hidden-state perturbations the terminal grid had to defer.
* ``transfer`` -- the Q2 **cross-domain transfer matrix** (terminal_swe x
  tau_bench), which needs a real second domain and so did not exist before.

Every experiment reuses the committed ``run_sa*`` runners with
``spec=tau_bench`` (+ its Ω_d bundle); this driver only samples, sweeps seeds,
runs the LLM slice, and writes JSON. Method-preserving throughout: no AVG node,
edge, checker, or aggregation rule changes -- only data (a new domain), seeds,
and the LLM enablement the plan already names.

CLI::

    python -m htir.eval.experiment_tau --experiments sa1,sa3,sa6,transfer \\
        --seeds 0,1,2 --n 1000 --outdir data
    python -m htir.eval.experiment_tau --experiments sa1 --use-llm --llm-n 120
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Callable

from htir.eval.datasets import (
    balanced_sample,
    iter_local_traces,
    load_tau_bench,
    normalize_tau_record,
)
from htir.eval.seeds import format_aggregate, run_multiseed
from htir.eval.experiment_sa1 import run_sa1, format_table as sa1_table
from htir.eval.experiment_sa3 import run_sa3, format_table as sa3_table
from htir.eval.experiment_sa6 import run_sa6, format_table as sa6_table
from htir.eval.experiment_sa2 import run_transfer_matrix, format_transfer
from htir.models.domain import get_domain_spec, load_domain_artifacts

DEFAULT_TAU_CACHE = "data/tau_cache/tau_all.jsonl"
DEFAULT_TERMINAL_CACHE = "data/tau_cache/terminal_sample_600.jsonl"
# τ-bench traces are shorter than Terminal-Bench; ~14 median steps, so the
# long-horizon knee sits lower than terminal's 20.
TAU_LONG_HORIZON_STEPS = 12
LLM_MODEL = "openai/gpt-4o-mini"  # cheap-tier judge for the LLM slice (plan Sec. 5)


# ---------------------------------------------------------------------------
# Per-experiment metric extractors (what mean±SE aggregates over seeds)
# ---------------------------------------------------------------------------

def _sa1_metrics(res) -> dict[str, float]:
    out: dict[str, float] = {"base_rate_valid": res.base_rate_valid}
    for a in res.arms:
        out[f"{a.arm}.false_valid"] = a.overall.false_valid_rate
        out[f"{a.arm}.resolved_frac"] = a.overall.resolved_fraction
        out[f"{a.arm}.resolved_acc"] = a.overall.resolved_accuracy
        out[f"{a.arm}.false_valid.long"] = a.long_horizon.false_valid_rate
    return out


def _sa3_metrics(res) -> dict[str, float]:
    out = {
        "auroc_all": res.auroc_all if res.auroc_all is not None else None,
        "ece_all": res.ece_all,
        "false_valid_reduction.absolute": res.false_valid_reduction.get("absolute", 0.0),
        "false_valid_reduction.relative": res.false_valid_reduction.get("relative", 0.0),
    }
    for a in res.arms:
        out[f"{a.arm}.false_valid"] = a.decision_metrics.false_valid_rate
        out[f"{a.arm}.coverage"] = a.coverage
    return out


def _sa6_metrics(res) -> dict[str, float]:
    out: dict[str, float] = {}
    for pr in res.perturbations:
        by_arm = {c.arm: c for c in pr.cells}
        for arm, c in by_arm.items():
            out[f"{pr.perturbation}.{arm}.false_valid"] = c.false_valid_rate
            out[f"{pr.perturbation}.{arm}.catch"] = c.shortcut_catch_rate
    return out


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def _load_tau(cache: str, hf: bool) -> list[dict[str, Any]]:
    if hf:
        return load_tau_bench(hf=True)
    return load_tau_bench([cache])


def _load_terminal(cache: str) -> list[dict[str, Any]]:
    """Load a (pre-cached, balanced) Terminal-Bench sample for the transfer matrix."""
    return list(iter_local_traces([cache]))


def _multiseed_experiment(
    name: str,
    run_fn: Callable[[list[dict[str, Any]]], Any],
    extract: Callable[[Any], dict[str, float]],
    tau_traces: list[dict[str, Any]],
    *,
    n: int,
    seeds: list[int],
    table_fn: Callable[[Any], str],
    log: Any,
) -> dict[str, Any]:
    def sample_fn(seed: int) -> list[dict[str, Any]]:
        return balanced_sample(tau_traces, n, seed=seed)

    summary, per_seed = run_multiseed(sample_fn, run_fn, seeds, extract=extract, log=log)
    print(f"\n===== {name} (seed 0 table) =====", file=log)
    print(table_fn(per_seed[0]), file=log)
    print(f"\n----- {name} mean±SE over seeds {seeds} -----", file=log)
    print(format_aggregate(summary.aggregate), file=log)
    return {
        "experiment": name,
        "domain": "tau_bench",
        "n_per_seed": summary.n_per_seed,
        "seeds": seeds,
        "aggregate": {k: v.model_dump() for k, v in summary.aggregate.items()},
        "per_seed": [r.model_dump() for r in per_seed],
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="τ-bench experiment campaign (SA-1/3/6 + transfer)")
    p.add_argument("--experiments", type=str, default="sa1,sa3,sa6,transfer",
                   help="comma list from: sa1, sa3, sa6, transfer")
    p.add_argument("--cache", type=str, default=DEFAULT_TAU_CACHE, help="τ-bench raw/normalized JSONL")
    p.add_argument("--hf", action="store_true", help="stream τ-bench traces from the HF hub instead")
    p.add_argument("--terminal-cache", type=str, default=DEFAULT_TERMINAL_CACHE,
                   help="balanced Terminal-Bench JSONL for the transfer matrix")
    p.add_argument("--n", type=int, default=1000, help="balanced sample size per seed")
    p.add_argument("--seeds", type=str, default="0,1,2", help="comma list of seeds")
    p.add_argument("--use-llm", action="store_true", help="also run a small LLM-enabled slice")
    p.add_argument("--llm-n", type=int, default=120, help="balanced sample size for the LLM slice")
    p.add_argument("--model", type=str, default=LLM_MODEL, help="OpenRouter judge model for the LLM slice")
    p.add_argument("--outdir", type=str, default="data", help="where to write <exp>_tau_results.json")
    args = p.parse_args(argv)

    log = sys.stderr
    experiments = [e.strip() for e in args.experiments.split(",") if e.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip() != ""]
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    spec = get_domain_spec("tau_bench")
    omega = load_domain_artifacts("tau_bench")
    print(f"[tau] loading τ-bench traces ({'HF' if args.hf else args.cache}) ...", file=log)
    tau_traces = _load_tau(args.cache, args.hf)
    print(f"[tau] {len(tau_traces)} τ-bench traces; spec={spec.domain_id}, "
          f"Ω_d artifacts={0 if omega is None else len(omega.artifacts)}", file=log)

    t0 = time.time()

    if "sa1" in experiments:
        out = _multiseed_experiment(
            "SA-1", lambda tr: run_sa1(tr, spec=spec, long_horizon_steps=TAU_LONG_HORIZON_STEPS,
                                       progress_every=0),
            _sa1_metrics, tau_traces, n=args.n, seeds=seeds, table_fn=sa1_table, log=log,
        )
        # SA-8: paired-t significance on the false-valid gaps vs full AVG (incl.
        # the prm / agent_judge competitive baselines now in DEFAULT_ARMS).
        from htir.eval.experiment_sa1 import SIG_GAPS
        from htir.eval.seeds import MeanSE, paired_t_test
        agg_ms = {k: MeanSE(**v) for k, v in out["aggregate"].items()}
        sig = []
        for a_arm, b_arm in SIG_GAPS:
            ak, bk = f"{a_arm}.false_valid", f"{b_arm}.false_valid"
            if ak in agg_ms and bk in agg_ms:
                sig.append(paired_t_test(agg_ms[ak].values, agg_ms[bk].values,
                                         label=f"{a_arm}_vs_{b_arm}.false_valid",
                                         a=a_arm, b=b_arm).model_dump())
        out["significance"] = sig
        print("\n----- SA-1 significance (false_valid gap vs full AVG) -----", file=log)
        for g in sig:
            print(f"  {g['a']} - {g['b']} = {g['mean_diff']:+.3f}±{g['se_diff']:.3f} "
                  f"(t={g['t_stat']:.2f}, df={g['df']}, p={g['p_value']:.4f})", file=log)
        if args.use_llm:
            print("\n[tau] SA-1 LLM slice ...", file=log)
            llm_sample = balanced_sample(tau_traces, args.llm_n, seed=0)
            llm_res = run_sa1(llm_sample, spec=spec, use_llm=True, use_semantic=True,
                              long_horizon_steps=TAU_LONG_HORIZON_STEPS, model=args.model, progress_every=0)
            print(sa1_table(llm_res), file=log)
            out["llm_slice"] = llm_res.model_dump()
        (outdir / "sa1_tau_results.json").write_text(json.dumps(out, indent=2))
        print(f"[tau] wrote {outdir/'sa1_tau_results.json'}", file=log)

    if "sa3" in experiments:
        out = _multiseed_experiment(
            "SA-3", lambda tr: run_sa3(tr, spec=spec, progress_every=0),
            _sa3_metrics, tau_traces, n=args.n, seeds=seeds, table_fn=sa3_table, log=log,
        )
        if args.use_llm:
            print("\n[tau] SA-3 LLM slice ...", file=log)
            llm_sample = balanced_sample(tau_traces, args.llm_n, seed=0)
            llm_res = run_sa3(llm_sample, spec=spec, use_llm=True, model=args.model, progress_every=0)
            print(sa3_table(llm_res), file=log)
            out["llm_slice"] = llm_res.model_dump()
        (outdir / "sa3_tau_results.json").write_text(json.dumps(out, indent=2))
        print(f"[tau] wrote {outdir/'sa3_tau_results.json'}", file=log)

    if "sa6" in experiments:
        # SA-6 draws its base population (reward=1) from a sample; use a fixed
        # per-seed balanced sample so seeds vary the population like SA-1/3.
        out = _multiseed_experiment(
            "SA-6", lambda tr: run_sa6(tr, spec=spec, domain_artifacts=omega, progress_every=0),
            _sa6_metrics, tau_traces, n=args.n, seeds=seeds, table_fn=sa6_table, log=log,
        )
        if args.use_llm:
            print("\n[tau] SA-6 LLM slice ...", file=log)
            llm_sample = balanced_sample(tau_traces, args.llm_n, seed=0)
            llm_res = run_sa6(llm_sample, spec=spec, domain_artifacts=omega, use_llm=True,
                              model=args.model, progress_every=0)
            print(sa6_table(llm_res), file=log)
            out["llm_slice"] = llm_res.model_dump()
        (outdir / "sa6_tau_results.json").write_text(json.dumps(out, indent=2))
        print(f"[tau] wrote {outdir/'sa6_tau_results.json'}", file=log)

    if "transfer" in experiments:
        print("\n[tau] transfer matrix (terminal_swe x tau_bench) ...", file=log)
        terminal_traces = _load_terminal(args.terminal_cache)
        tau_sample = balanced_sample(tau_traces, min(args.n, len(tau_traces)), seed=0)
        domains = {
            "terminal_swe": {
                "spec": get_domain_spec("terminal_swe"),
                "omega": load_domain_artifacts("terminal_swe"),
                "traces": terminal_traces,
            },
            "tau_bench": {"spec": spec, "omega": omega, "traces": tau_sample},
        }
        tr = run_transfer_matrix(domains, use_llm=False, log=log)
        print(format_transfer(tr), file=log)
        (outdir / "transfer_tau_results.json").write_text(tr.model_dump_json(indent=2))
        print(f"[tau] wrote {outdir/'transfer_tau_results.json'}", file=log)

    print(f"\n[tau] campaign done in {time.time()-t0:.1f}s", file=log)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
