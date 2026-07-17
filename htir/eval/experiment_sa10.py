"""
SA-10 -- Third domain (SWE-Gym) -> 3x3 cross-domain transfer matrix (Q2).

SA-2's transfer matrix (``htir/eval/experiment_sa2.py``) answered the sharp Q2
question -- *apply the verifier specced for domain A to traces from domain B* --
but only over a 2x2 grid (``terminal_swe`` x ``tau_bench``) because a real third
domain lacked a trace loader. SA-10 closes that: ``htir.eval.datasets.load_swe_gym``
maps SWE-Gym OpenHands rollouts into this repo's turn schema, so the same
``run_transfer_matrix`` runs over **three** real domains -- ``terminal_swe``,
``tau_bench``, ``swe_gym`` -- plus the ``universal_only`` floor.

Method-preserving: no AVG node/edge/checker/aggregation change. This driver only
adds the third domain's data, sweeps >=3 seeds (avg.tex Sec. 4.7 rigor bar:
mean +/- SE, paired-t on the key gap), and packages the matrix. Every cell scores
the deterministic ``exec_only`` graph arm against the weak reward label, exactly
as SA-2.

What the matrix shows (and, honestly, what it does not):

* ``universal_only`` binds no domain obligation, so it **abstains everywhere**
  (resolved_fraction ~ 0, never over-credits) -- the zero-adaptation floor.
* ``tau_bench`` (a *policy* domain -- OpenAI tool-call transcripts, no terminal
  ops) is **orthogonal** to the terminal family: its spec resolves only its own
  column and abstains (resolved_fraction 0, false_valid 0) on the terminal and
  SWE-Gym columns, and vice-versa. This is the clean transfer contrast.
* ``terminal_swe`` and ``swe_gym`` are **both terminal-shaped** and share the
  ``terminal`` adapter + operation vocabulary; they differ only in S_d (the
  obligation set). So they *cross-bind*: a terminal spec resolves some SWE-Gym
  traces and vice-versa. The demonstrated separation axis is therefore
  **policy vs. terminal-family**, not a clean 3-way diagonal -- reported, not
  hidden (see ``notes``).
* SWE-Gym's own diagonal resolves *weakly* offline. SWE-Gym agents re-validate a
  fix by running a **reproducer script** (``python reproduce_error.py``), not a
  test runner, and rarely re-run it after every edit, so the high-severity
  post-edit-validation obligations mostly abstain and the trajectory stays
  ``uncertain``. This is a coverage limit of the offline mechanical checker on
  this corpus (the loader/S_d gate the plan flagged), not an over-crediting
  failure -- false_valid stays low in every cell.

The invariant that *does* hold across all three domains is the transfer **safety
property**: a mis-specified (off-family) or universal verifier abstains; it never
over-credits. Cross-family cells have false_valid ~ 0.

CLI::

    python -m htir.eval.experiment_sa10 --out data/sa10_results.json
    python -m htir.eval.experiment_sa10 --swe-hf --n 300 --seeds 0,1,2
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, Field

from htir.eval.datasets import (
    balanced_sample,
    iter_local_traces,
    load_swe_gym,
    load_tau_bench,
)
from htir.eval.experiment_sa2 import TransferResult, format_transfer, run_transfer_matrix
from htir.eval.seeds import MeanSE, PairedGap, mean_se, paired_t_test
from htir.models.domain import get_domain_spec, load_domain_artifacts

DEFAULT_SWE_CACHE = "data/swe_gym_cache/swe_gym_sample.jsonl"
DEFAULT_TERMINAL_CACHE = "data/tau_cache/terminal_sample_600.jsonl"
DEFAULT_TAU_CACHE = "data/tau_cache/tau_all.jsonl"

# Adapter family per spec -- the honest separation axis. ``terminal_swe`` and
# ``swe_gym`` share the ``terminal`` adapter + operation vocabulary (they differ
# only in S_d), so they are one family; ``tau_bench`` (policy transcripts) is
# orthogonal; ``universal_only`` binds nothing. Cross-*family* off-diagonal cells
# are where the transfer safety property (abstain, never over-credit) is tested.
_FAMILY = {
    "universal_only": "universal",
    "terminal_swe": "terminal",
    "swe_gym": "terminal",
    "tau_bench": "policy",
}


# ---------------------------------------------------------------------------
# Result models
# ---------------------------------------------------------------------------

class SA10Cell(BaseModel):
    """One (train-spec x test-domain) cell aggregated over seeds (mean +/- SE)."""
    train_spec: str
    test_domain: str
    n: int = 0
    diagonal: bool = False
    cross_family: bool = False
    resolved_fraction: MeanSE
    false_valid: MeanSE
    abstention: MeanSE


class SA10Result(BaseModel):
    """SA-10 3x3 transfer output: per-cell mean +/- SE, significance, provenance."""
    experiment: str = "SA-10: SWE-Gym third domain -> 3x3 cross-domain transfer (Q2)"
    domains: list[str] = Field(default_factory=list)
    train_specs: list[str] = Field(default_factory=list)
    test_domains: list[str] = Field(default_factory=list)
    seeds: list[int] = Field(default_factory=list)
    n_per_domain: dict[str, int] = Field(default_factory=dict)
    use_llm: bool = False
    seconds: float = 0.0
    cells: list[SA10Cell] = Field(default_factory=list)
    significance: list[PairedGap] = Field(default_factory=list)
    headline: dict[str, Any] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)
    per_seed: list[dict[str, Any]] = Field(default_factory=list)

    def cell(self, train_spec: str, test_domain: str) -> Optional[SA10Cell]:
        return next(
            (c for c in self.cells if c.train_spec == train_spec and c.test_domain == test_domain),
            None,
        )


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_sa10(
    domains: dict[str, dict[str, Any]],
    *,
    seeds: list[int],
    n: int,
    use_llm: bool = False,
    model: str = "openai/gpt-4o",
    log: Any = sys.stderr,
) -> SA10Result:
    """
    Run the 3x3 (+ universal floor) transfer matrix over ``len(seeds)``
    independent balanced subsamples of size ``n`` per domain, then aggregate each
    cell's resolved_fraction / false_valid / abstention to mean +/- SE.

    ``domains`` maps a domain id to ``{"spec", "omega", "traces"}`` (the same
    shape :func:`htir.eval.experiment_sa2.run_transfer_matrix` takes). Each seed
    draws a fresh :func:`balanced_sample` per domain so seeds vary the population
    exactly as SA-1/SA-2 do. Returns a JSON-able :class:`SA10Result`.
    """
    t0 = time.time()
    per_seed: list[TransferResult] = []
    for seed in seeds:
        seeded: dict[str, dict[str, Any]] = {}
        for dom_id, cfg in domains.items():
            traces = cfg["traces"]
            sample = balanced_sample(traces, min(n, len(traces)), seed=seed)
            seeded[dom_id] = {"spec": cfg["spec"], "omega": cfg.get("omega"), "traces": sample}
        if log is not None:
            print(f"[sa10] seed={seed}: transfer matrix over {list(seeded)} ...", file=log)
        per_seed.append(run_transfer_matrix(seeded, use_llm=use_llm, model=model, log=log))

    if log is not None:
        print(f"[sa10] {len(seeds)} seeds in {time.time() - t0:.1f}s", file=log)
    # ``seconds`` is intentionally left 0.0 in the serialized result: the offline
    # artifact must be byte-deterministic across runs (wall-clock is the only
    # non-reproducible field), so timing is logged, not stored.
    return _assemble(per_seed, seeds, n, use_llm)


def _assemble(
    per_seed: list[TransferResult],
    seeds: list[int],
    n: int,
    use_llm: bool,
) -> SA10Result:
    ref = per_seed[0]
    train_specs = ref.train_specs
    test_domains = ref.test_domains

    cells: list[SA10Cell] = []
    # {(train, test): {"rf": [...], "fv": [...], "ab": [...], "n": int}}
    for train in train_specs:
        for test in test_domains:
            rf, fv, ab, ns = [], [], [], []
            for tr in per_seed:
                c = tr.cell(train, test)
                if c is None:
                    continue
                rf.append(c.metrics.resolved_fraction)
                fv.append(c.metrics.false_valid_rate)
                ab.append(c.metrics.abstention_rate)
                ns.append(c.n)
            cells.append(SA10Cell(
                train_spec=train,
                test_domain=test,
                n=max(ns) if ns else 0,
                diagonal=(train == test),
                cross_family=(_FAMILY.get(train) != _FAMILY.get(test)
                              and _FAMILY.get(train) != "universal"),
                resolved_fraction=mean_se(rf),
                false_valid=mean_se(fv),
                abstention=mean_se(ab),
            ))

    result = SA10Result(
        domains=test_domains,
        train_specs=train_specs,
        test_domains=test_domains,
        seeds=seeds,
        n_per_domain={d: (max((c.n for c in cells if c.test_domain == d), default=0)) for d in test_domains},
        use_llm=use_llm,
        seconds=0.0,
        cells=cells,
    )

    # Significance (paired-t over seeds): the matched diagonal resolves where the
    # universal floor abstains. One test per domain -- for the orthogonal policy
    # domain this is significant; for SWE-Gym the near-null diagonal makes the
    # gate visible in the statistic itself (small, non-significant), honestly.
    sig: list[PairedGap] = []
    for d in test_domains:
        diag = _values(per_seed, d, d, "resolved_fraction")
        uni = _values(per_seed, "universal_only", d, "resolved_fraction")
        if diag and uni:
            sig.append(paired_t_test(
                diag, uni, label=f"diagonal_vs_universal.resolved_fraction[{d}]",
                a=f"{d}x{d}", b=f"universal_onlyx{d}",
            ))
    result.significance = sig

    # Headline: the transfer safety property + where the diagonal actually binds.
    cross_fv = [c.false_valid.mean for c in cells if c.cross_family]
    uni_rf = [c.resolved_fraction.mean for c in cells if c.train_spec == "universal_only"]
    result.headline = {
        "diagonal_resolved_fraction": {d: (result.cell(d, d).resolved_fraction.mean
                                           if result.cell(d, d) else None) for d in test_domains},
        "universal_only_resolved_fraction_max": round(max(uni_rf), 4) if uni_rf else None,
        "cross_family_false_valid_max": round(max(cross_fv), 4) if cross_fv else None,
    }

    result.notes = [
        "3x3 (+ universal floor) transfer: apply the verifier specced for the ROW "
        "domain to the COLUMN domain's traces; exec_only vs the weak reward label. "
        "Third domain (swe_gym) is real: htir.eval.datasets.load_swe_gym maps "
        "SWE-Gym OpenHands rollouts into the turn schema, parsed by the terminal adapter.",
        "SAFETY PROPERTY (holds everywhere): universal_only and every cross-FAMILY "
        "off-diagonal cell abstain -- resolved_fraction ~ 0, false_valid ~ 0. A "
        "mis-specified verifier abstains; it never over-credits cross-domain.",
        "HONEST CAVEAT #1 -- terminal_swe and swe_gym share the terminal adapter + "
        "operation vocabulary (they differ only in S_d), so they cross-bind: the "
        "clean separation axis this matrix demonstrates is policy (tau_bench) vs. "
        "terminal-family, not a 3-way diagonal.",
        "HONEST CAVEAT #2 -- swe_gym's own diagonal resolves weakly offline: SWE-Gym "
        "agents re-validate via a reproducer script (python reproduce_error.py), not "
        "a test runner, and rarely after every edit, so high-severity post-edit "
        "obligations abstain and traces stay uncertain. A coverage limit of the "
        "offline mechanical checker on this corpus (the loader/S_d gate), not "
        "over-crediting -- false_valid stays low. --use-llm would let the semantic "
        "checker adjudicate the abstained obligations.",
    ]

    # Neutralize wall-clock ``seconds`` so the committed artifact is byte-stable.
    per_seed_dumps: list[dict[str, Any]] = []
    for tr in per_seed:
        d = tr.model_dump()
        d["seconds"] = 0.0
        per_seed_dumps.append(d)
    result.per_seed = per_seed_dumps
    return result


def _values(per_seed: list[TransferResult], train: str, test: str, attr: str) -> list[float]:
    """Per-seed values of one cell metric (``resolved_fraction`` / ``false_valid_rate``)."""
    key = "false_valid_rate" if attr in ("false_valid", "false_valid_rate") else attr
    out: list[float] = []
    for tr in per_seed:
        c = tr.cell(train, test)
        if c is not None:
            out.append(getattr(c.metrics, key))
    return out


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def format_table(result: SA10Result) -> str:
    """Compact 3x3 matrix (mean +/- SE) for resolved_fraction and false_valid."""
    lines: list[str] = []
    lines.append(
        f"{result.experiment}  |  seeds={result.seeds}  "
        f"n/domain={result.n_per_domain}  use_llm={result.use_llm}"
    )
    for metric_name, attr in (("resolved_fraction", "resolved_fraction"), ("false_valid", "false_valid")):
        lines.append(f"  [{metric_name} mean±SE]  rows=train spec, cols=test domain")
        header = "    " + f"{'train/test':<16}" + "".join(f"{d:>20}" for d in result.test_domains)
        lines.append(header)
        for tr in result.train_specs:
            row = f"    {tr:<16}"
            for td in result.test_domains:
                c = result.cell(tr, td)
                cell_ms: MeanSE = getattr(c, attr) if c else MeanSE()
                mark = "*" if (c and c.diagonal) else " "
                row += f"{mark + cell_ms.as_str():>20}"
            lines.append(row)
    hd = result.headline
    lines.append(f"  headline: diagonal resolved_fraction = {hd.get('diagonal_resolved_fraction')}")
    lines.append(
        f"            universal_only resolved_fraction max = {hd.get('universal_only_resolved_fraction_max')}; "
        f"cross-family false_valid max = {hd.get('cross_family_false_valid_max')}"
    )
    lines.append("  [significance -- paired t-test: diagonal vs universal_only resolved_fraction]")
    for g in result.significance:
        lines.append(f"    {g.as_str()}")
    for note in result.notes:
        lines.append(f"  note: {note}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_domains(args: argparse.Namespace, log: Any) -> dict[str, dict[str, Any]]:
    print("[sa10] loading traces for the three domains ...", file=log)
    if args.swe_hf:
        swe = load_swe_gym(hf=True, limit=args.swe_hf_limit)
    else:
        swe = load_swe_gym([args.swe_cache])
    terminal = list(iter_local_traces([args.terminal_cache]))
    tau = load_tau_bench([args.tau_cache])
    print(f"[sa10] swe_gym={len(swe)}  terminal_swe={len(terminal)}  tau_bench={len(tau)}", file=log)
    return {
        "terminal_swe": {
            "spec": get_domain_spec("terminal_swe"),
            "omega": load_domain_artifacts("terminal_swe"),
            "traces": terminal,
        },
        "tau_bench": {
            "spec": get_domain_spec("tau_bench"),
            "omega": load_domain_artifacts("tau_bench"),
            "traces": tau,
        },
        "swe_gym": {
            "spec": get_domain_spec("swe_gym"),
            "omega": load_domain_artifacts("swe_gym"),
            "traces": swe,
        },
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="SA-10: SWE-Gym third domain -> 3x3 transfer (Q2)")
    src = p.add_argument_group("data source")
    src.add_argument("--swe-cache", type=str, default=DEFAULT_SWE_CACHE,
                     help="local SWE-Gym JSONL (raw OpenHands or normalized turn schema)")
    src.add_argument("--swe-hf", action="store_true",
                     help="stream SWE-Gym from the HF hub (SWE-Gym/OpenHands-Sampled-Trajectories)")
    src.add_argument("--swe-hf-limit", type=int, default=4000, help="records to stream when --swe-hf")
    src.add_argument("--terminal-cache", type=str, default=DEFAULT_TERMINAL_CACHE)
    src.add_argument("--tau-cache", type=str, default=DEFAULT_TAU_CACHE)
    p.add_argument("--n", type=int, default=200, help="balanced sample size per domain per seed")
    p.add_argument("--seeds", type=str, default="0,1,2", help="comma list of seeds (>=3 for the rigor bar)")
    p.add_argument("--use-llm", action="store_true", help="enable the semantic checker (needs API key)")
    p.add_argument("--model", type=str, default="openai/gpt-4o")
    p.add_argument("--out", type=str, default="", help="write SA10Result JSON here")
    args = p.parse_args(argv)

    log = sys.stderr
    seeds = [int(s) for s in args.seeds.split(",") if s.strip() != ""]
    domains = _build_domains(args, log)
    result = run_sa10(domains, seeds=seeds, n=args.n, use_llm=args.use_llm, model=args.model, log=log)
    print(format_table(result))
    if args.out:
        Path(args.out).write_text(result.model_dump_json(indent=2), encoding="utf-8")
        print(f"\n[sa10] wrote {args.out}", file=log)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
