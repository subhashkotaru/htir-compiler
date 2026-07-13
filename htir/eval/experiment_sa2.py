"""
SA-2 -- Q2: Universal-only vs. Universal + Adapters (transfer & sample
efficiency).

Runs the experiment from ``docs/experiment-plan.md`` (SA-2) / ``avg.tex``
Sec. 4 (question Q2, ablations #2 and #7). Where SA-1 held the domain spec
fixed and varied the *checker* configuration over one graph, SA-2 holds the
checker configuration fixed (the deterministic ``exec_only`` graph arm) and
varies the **domain adaptation** -- the spec ``S_d`` and the weak-supervision
bundle ``Omega_d`` -- recompiling each trace once per adaptation level:

* ``universal_only`` -- the cross-domain ``default`` spec: only universal
  obligation templates, expressed over a generic operation vocabulary that does
  not name terminal operations. This is the zero-adaptation baseline.
* ``adapter_*``      -- the ``terminal_swe`` spec, built up one adaptation unit
  at a time (execution-status binding -> provenance -> post-edit validation ->
  the full adapter). Each level adds domain obligation templates over the
  terminal operation vocabulary, so obligations actually bind to
  ``run_command`` / ``run_test`` / ``edit_file`` steps.
* ``adapter_full_omega`` -- the full adapter plus the ``Omega_d`` bundle loaded
  from ``htir/domains/terminal_swe.artifacts/`` (schema / policy / test), which
  seeds schema evidence and per-edit policy-compliance obligations.

The three *headline* arms named in the experiment plan are
``universal_only`` -> ``adapter_full`` -> ``adapter_full_omega``; the
intermediate ``adapter_*`` levels trace the **sample-efficiency curve**
(verifier resolution vs. amount of domain adaptation).

Metrics. Each arm's ``predicted_status`` is scored against the weak reward
label with :func:`htir.eval.weak_labels.evaluate_predictions`. Because a
universal verifier that never binds an obligation can only *abstain* (it never
false-credits, but it never catches anything either), the headline contrast for
Q2 is **resolved fraction** -- how much of the domain the verifier can actually
adjudicate -- alongside SA-1's false-valid / failure-flag precision. We also
report mean obligation counts (total and semantic-routed) so Omega_d's effect
is visible even where it is offline-inert: it adds coverage the semantic
checker would consume with an API key.

Offline reproducibility. With no API key (``use_llm=False``, the default) the
pipeline is deterministic: the semantic checker abstains rather than calling an
LLM. Two consequences, reported explicitly rather than hidden: (1) the
adaptation gain that shows up offline is the *mechanical* one -- binding
execution-status obligations to terminal operations; (2) domain templates and
``Omega_d`` artifacts whose obligations route to the semantic checker (policy
compliance, final-answer support) add obligation coverage but abstain, so they
do not move offline verdicts. Separating those needs a key (``--use-llm``),
exactly as in SA-1.

Transfer. A genuine cross-domain transfer matrix needs a second domain with a
real trace loader. The ``swe_gym`` spec exists and is adapter-compatible, but
the SWE-Gym HF trace schema is not yet wired into ``htir.eval.datasets``, so
this runner evaluates universal-vs-adapters *within* the terminal domain and
marks the transfer matrix as future work (see ``docs/experiment-plan.md``).

CLI::

    python -m htir.eval.experiment_sa2 --cache sa2_sample.jsonl --out sa2_results.json
    python -m htir.eval.experiment_sa2 --hf --n 3000 --out sa2_results.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Optional

from pydantic import BaseModel, Field

from htir.agents.baselines import VerifierArm, run_arm
from htir.agents.trace_abstraction import TraceAbstractionAgent
from htir.eval.datasets import iter_local_traces, to_canonical_steps
from htir.eval.weak_labels import (
    VerifierMetrics,
    evaluate_predictions,
    extract_reward,
    label_from_reward,
)
from htir.models.domain import (
    DomainArtifactBundle,
    DomainSpec,
    get_domain_spec,
    load_domain_artifacts,
)
from htir.models.htir import CheckerType, HTIR

# Long-horizon slice knee (Q2, like SA-1, expects the adaptation gain to matter
# most where verification is hardest -- long traces). Overridable from the CLI.
DEFAULT_LONG_HORIZON_STEPS = 20


# ---------------------------------------------------------------------------
# Adaptation levels (the SA-2 arms + sample-efficiency curve)
# ---------------------------------------------------------------------------

class AdaptationLevel(BaseModel):
    """
    One point on the universal->adapter adaptation axis: a domain spec, an
    optional Omega_d bundle, and a ``budget`` ordinal (how many domain
    adaptation units have been added). ``spec`` / ``omega`` are excluded from
    serialization (they are not JSON-friendly and are reconstructable from
    ``name``); the scored :class:`ArmReport` carries the reportable fields.
    """
    name: str
    description: str
    budget: int = Field(..., description="Cumulative domain-adaptation units added (0 = universal-only)")
    spec: DomainSpec = Field(..., exclude=True)
    omega: Optional[DomainArtifactBundle] = Field(default=None, exclude=True)

    model_config = {"arbitrary_types_allowed": True}


# Template ids grouped by the adaptation unit they represent, so the
# intermediate specs are built by *cumulatively* keeping domain-relevant
# templates over the terminal operation vocabulary. (Kept in sync with
# htir/domains/terminal_swe.yaml.)
_EXEC_STATUS_TEMPLATES = {"swe-command-status", "swe-test-status", "trig-explain-failure"}
_PROVENANCE_TEMPLATES = {"swe-edit-provenance"}
_POST_EDIT_TEMPLATES = {"swe-edit-then-validate"}


def _adapter_spec(full: DomainSpec, keep: set[str], domain_id: str) -> DomainSpec:
    """A terminal_swe spec restricted to ``keep`` obligation templates (same
    operation/artifact vocabulary, so roles bind identically -- only the
    obligation set narrows)."""
    spec = full.model_copy(deep=True)
    spec.domain_id = domain_id
    spec.obligation_templates = [t for t in full.obligation_templates if t.template_id in keep]
    return spec


def build_adaptation_levels(
    *, include_intermediate: bool = True, load_omega: bool = True
) -> list[AdaptationLevel]:
    """
    Construct the ordered adaptation axis for SA-2.

    ``level 0`` is the cross-domain ``default`` spec (universal-only). Levels
    1-3 build the ``terminal_swe`` adapter up one unit at a time
    (execution-status binding -> +provenance -> +post-edit validation); level 4
    is the full adapter; level 5 adds the ``Omega_d`` bundle. Pass
    ``include_intermediate=False`` to keep only the three headline arms named in
    the plan (universal-only, full adapter, full adapter + Omega_d).
    """
    default = get_domain_spec("default")
    full = get_domain_spec("terminal_swe")
    omega = load_domain_artifacts("terminal_swe") if load_omega else None

    exec_keep = set(_EXEC_STATUS_TEMPLATES)
    prov_keep = exec_keep | _PROVENANCE_TEMPLATES
    post_keep = prov_keep | _POST_EDIT_TEMPLATES

    levels: list[AdaptationLevel] = [
        AdaptationLevel(
            name="universal_only",
            description="default spec: universal templates only, generic vocabulary (zero adaptation)",
            budget=0,
            spec=default,
        ),
    ]
    if include_intermediate:
        levels += [
            AdaptationLevel(
                name="adapter_exec",
                description="terminal vocab + execution-status obligations (command/test status)",
                budget=1,
                spec=_adapter_spec(full, exec_keep, "terminal_swe_exec"),
            ),
            AdaptationLevel(
                name="adapter_prov",
                description="+ artifact-provenance obligation on edits",
                budget=2,
                spec=_adapter_spec(full, prov_keep, "terminal_swe_prov"),
            ),
            AdaptationLevel(
                name="adapter_postedit",
                description="+ post-edit-validation obligation",
                budget=3,
                spec=_adapter_spec(full, post_keep, "terminal_swe_postedit"),
            ),
        ]
    levels.append(
        AdaptationLevel(
            name="adapter_full",
            description="full terminal_swe adapter (all domain + universal-equivalent templates)",
            budget=4,
            spec=full,
        )
    )
    levels.append(
        AdaptationLevel(
            name="adapter_full_omega",
            description="full adapter + Omega_d bundle (schema / policy / test)",
            budget=5 if omega is None else 4 + len(omega.artifacts),
            spec=full,
            omega=omega,
        )
    )
    return levels


# ---------------------------------------------------------------------------
# Result models
# ---------------------------------------------------------------------------

class PerTraceRecord(BaseModel):
    """One trace's label, size, and each arm's predicted status + obligation counts."""
    task_id: str = ""
    reward: Optional[int] = None
    label: Optional[str] = None
    n_steps: int = 0
    predicted: dict[str, str] = Field(default_factory=dict)
    n_obligations: dict[str, int] = Field(default_factory=dict)
    n_semantic_obligations: dict[str, int] = Field(default_factory=dict)


class ArmReport(BaseModel):
    """Scored metrics for one adaptation level, overall and on the long-horizon slice."""
    arm: str
    description: str = ""
    budget: int = 0
    n_omega_artifacts: int = 0
    overall: VerifierMetrics
    long_horizon: VerifierMetrics
    mean_obligations: float = 0.0
    mean_semantic_obligations: float = 0.0
    total_llm_calls: int = 0
    mean_llm_calls: float = 0.0


class SA2Result(BaseModel):
    """Full SA-2 experiment output: config, per-arm reports, and provenance."""
    experiment: str = "SA-2: Universal-only vs. Universal + Adapters (Q2)"
    n_traces: int = 0
    n_labeled: int = 0
    base_rate_valid: float = Field(0.0, description="Fraction of labeled traces with label 'valid'")
    long_horizon_steps: int = DEFAULT_LONG_HORIZON_STEPS
    n_long_horizon: int = 0
    use_llm: bool = False
    seconds: float = 0.0
    arms: list[ArmReport] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


def _semantic_obligation_count(htir: HTIR) -> int:
    """How many obligations were routed to the SEMANTIC checker (cost proxy)."""
    return sum(1 for o in htir.obligations if o.checker == CheckerType.SEMANTIC)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_sa2(
    raw_traces: Iterable[dict[str, Any]],
    *,
    levels: list[AdaptationLevel] | None = None,
    use_llm: bool = False,
    long_horizon_steps: int = DEFAULT_LONG_HORIZON_STEPS,
    model: str = "openai/gpt-4o",
    progress_every: int = 250,
    log: Any = sys.stderr,
) -> SA2Result:
    """
    Execute SA-2 over ``raw_traces`` (turn-schema dicts with a ``reward``).

    Each trace is compiled *once per adaptation level* (unlike SA-1, the domain
    spec differs across arms, so the graph cannot be shared) through obligation
    generation, then scored with the deterministic ``exec_only`` graph arm via
    :func:`htir.agents.baselines.run_arm`. ``use_llm`` turns on the semantic
    checker so the policy/final-answer obligations Omega_d and the domain
    templates add can be discharged instead of abstaining.
    """
    levels = levels or build_adaptation_levels()
    agents = {lvl.name: TraceAbstractionAgent(model=model, domain_spec=lvl.spec) for lvl in levels}

    records: list[PerTraceRecord] = []
    t0 = time.time()

    for i, raw in enumerate(raw_traces):
        reward = extract_reward(raw)
        label = label_from_reward(reward)
        task_id = str(raw.get("task_name", "")) if isinstance(raw, dict) else ""

        try:
            steps = to_canonical_steps(raw)
        except Exception as exc:  # a malformed trace should not sink the run
            if log is not None:
                print(f"[sa2] skip trace {i} ({task_id}): {exc!r}", file=log)
            continue

        rec = PerTraceRecord(task_id=task_id, reward=reward, label=label)
        n_steps = 0
        for lvl in levels:
            try:
                htir = agents[lvl.name].compile(
                    task_id=task_id or f"trace-{i}",
                    raw_steps=steps,
                    harness_snippets={},
                    generate_obligations=True,
                    use_semantic_analysis=use_llm,
                    run_checks=False,
                    domain_artifacts=lvl.omega,
                )
            except Exception as exc:
                if log is not None:
                    print(f"[sa2] skip trace {i} arm {lvl.name}: {exc!r}", file=log)
                continue
            n_steps = len(htir.steps)
            agg = run_arm(
                htir, lvl.spec, VerifierArm.EXEC_ONLY,
                use_llm=use_llm, domain_artifacts=lvl.omega, model=model,
            )
            rec.predicted[lvl.name] = agg.predicted_status
            rec.n_obligations[lvl.name] = len(htir.obligations)
            rec.n_semantic_obligations[lvl.name] = _semantic_obligation_count(htir)

        rec.n_steps = n_steps
        records.append(rec)
        if progress_every and log is not None and (i + 1) % progress_every == 0:
            print(f"[sa2] compiled+scored {i + 1} traces...", file=log)

    return _assemble(records, levels, use_llm, long_horizon_steps, time.time() - t0)


def _assemble(
    records: list[PerTraceRecord],
    levels: list[AdaptationLevel],
    use_llm: bool,
    long_horizon_steps: int,
    seconds: float,
) -> SA2Result:
    labels = [r.label for r in records]
    n_labeled = sum(1 for l in labels if l is not None)
    n_valid = sum(1 for l in labels if l == "valid")
    long_idx = [i for i, r in enumerate(records) if r.n_steps >= long_horizon_steps]

    arm_reports: list[ArmReport] = []
    for lvl in levels:
        preds = [r.predicted.get(lvl.name, "uncertain") for r in records]
        overall = evaluate_predictions(preds, labels)
        lh_preds = [preds[i] for i in long_idx]
        lh_labels = [labels[i] for i in long_idx]
        long_horizon = evaluate_predictions(lh_preds, lh_labels)

        obl = [r.n_obligations.get(lvl.name, 0) for r in records]
        sem = [r.n_semantic_obligations.get(lvl.name, 0) for r in records]
        # Cost proxy (as in SA-1): would-issue LLM calls == one narrow call per
        # semantic-routed obligation for the graph arm.
        arm_reports.append(
            ArmReport(
                arm=lvl.name,
                description=lvl.description,
                budget=lvl.budget,
                n_omega_artifacts=len(lvl.omega.artifacts) if lvl.omega else 0,
                overall=overall,
                long_horizon=long_horizon,
                mean_obligations=statistics.fmean(obl) if obl else 0.0,
                mean_semantic_obligations=statistics.fmean(sem) if sem else 0.0,
                total_llm_calls=sum(sem),
                mean_llm_calls=statistics.fmean(sem) if sem else 0.0,
            )
        )

    notes: list[str] = [
        "Q2 headline metric is resolved_fraction: a universal verifier that binds no "
        "obligation can only abstain (false_valid 0, but resolves nothing). Adding the "
        "domain adapter lets obligations bind to terminal operations and be discharged.",
    ]
    if not use_llm:
        notes.append(
            "Offline run (no API key): the semantic checker abstains, so the adaptation "
            "gain seen here is the mechanical one (execution-status binding). Domain "
            "templates and Omega_d artifacts whose obligations route to the semantic "
            "checker (policy compliance, final-answer support) add obligation coverage "
            "(see mean_obligations / mean_semantic_obligations) but abstain, so they do "
            "not move offline verdicts. Separating them needs --use-llm (a key), as in SA-1."
        )
    notes.append(
        "Transfer matrix is future work: needs a second-domain trace loader. The swe_gym "
        "spec exists and is adapter-compatible, but the SWE-Gym HF trace schema is not yet "
        "wired into htir.eval.datasets, so this runs universal-vs-adapters within terminal."
    )

    return SA2Result(
        n_traces=len(records),
        n_labeled=n_labeled,
        base_rate_valid=(n_valid / n_labeled) if n_labeled else 0.0,
        long_horizon_steps=long_horizon_steps,
        n_long_horizon=len(long_idx),
        use_llm=use_llm,
        seconds=round(seconds, 2),
        arms=arm_reports,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def format_table(result: SA2Result) -> str:
    """A compact fixed-width results table for the terminal / logs."""
    lines: list[str] = []
    lines.append(
        f"SA-2: Universal-only vs. Adapters  |  n={result.n_traces} "
        f"(labeled {result.n_labeled}, base-rate valid {result.base_rate_valid:.2f})  |  "
        f"use_llm={result.use_llm}"
    )
    header = (
        f"{'arm':<20} {'bud':>3} {'res_frac':>8} {'res_acc':>8} {'false_valid':>11} "
        f"{'ff_prec':>8} {'ff_rec':>7} {'abstain':>8} {'obl/tr':>7} {'sem/tr':>7}"
    )

    def _rows(scope: str) -> list[str]:
        out = [f"  [{scope}]", "  " + header]
        for a in result.arms:
            m = getattr(a, scope)
            out.append(
                f"  {a.arm:<20} {a.budget:>3} {m.resolved_fraction:>8.3f} "
                f"{m.resolved_accuracy:>8.3f} {m.false_valid_rate:>11.3f} "
                f"{m.failure_flag_precision:>8.3f} {m.failure_flag_recall:>7.3f} "
                f"{m.abstention_rate:>8.3f} {a.mean_obligations:>7.2f} "
                f"{a.mean_semantic_obligations:>7.2f}"
            )
        return out

    lines += _rows("overall")
    lines.append(f"  [long-horizon >= {result.long_horizon_steps} steps]  n={result.n_long_horizon}")
    lines += _rows("long_horizon")[1:]
    for note in result.notes:
        lines.append(f"  note: {note}")
    return "\n".join(lines)


def _load_traces(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.hf:
        from htir.eval.datasets import balanced_sample, load_terminalbench
        raw = load_terminalbench(limit=args.hf_limit, streaming=True)
        return balanced_sample(raw, args.n, seed=args.seed)
    if not args.cache:
        raise SystemExit("provide --cache <jsonl> or --hf")
    traces = list(iter_local_traces([args.cache]))
    if args.n and len(traces) > args.n:
        traces = traces[: args.n]
    return traces


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="SA-2: Universal-only vs. Universal + Adapters (Q2)")
    src = p.add_argument_group("data source")
    src.add_argument("--cache", type=str, default="", help="local JSON/JSONL sample (turn schema)")
    src.add_argument("--hf", action="store_true", help="pull from the HF terminalbench dataset")
    src.add_argument("--hf-limit", type=int, default=9000, help="records to stream when --hf")
    src.add_argument("--n", type=int, default=3000, help="target balanced sample size")
    src.add_argument("--seed", type=int, default=0)
    p.add_argument("--use-llm", action="store_true", help="enable the semantic checker (needs API key)")
    p.add_argument("--headline-only", action="store_true",
                   help="only the 3 plan arms (universal / full adapter / +Omega_d), no intermediate curve")
    p.add_argument("--long-horizon-steps", type=int, default=DEFAULT_LONG_HORIZON_STEPS)
    p.add_argument("--model", type=str, default="openai/gpt-4o")
    p.add_argument("--out", type=str, default="", help="write SA2Result JSON here")
    args = p.parse_args(argv)

    traces = _load_traces(args)
    levels = build_adaptation_levels(include_intermediate=not args.headline_only)
    result = run_sa2(
        traces,
        levels=levels,
        use_llm=args.use_llm,
        long_horizon_steps=args.long_horizon_steps,
        model=args.model,
    )
    print(format_table(result))
    if args.out:
        Path(args.out).write_text(result.model_dump_json(indent=2), encoding="utf-8")
        print(f"\n[sa2] wrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
