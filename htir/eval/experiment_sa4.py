"""
SA-4 -- Q4a: Online Intervention, offline replay (avg.tex Sec. 4 question Q4;
avg.tex Sec. 3.11 "Online Intervention").

Q4a claims that *monitoring obligations on the partial graph* G_{tau<=t} lets
the harness intervene **timely** (early, before the trajectory finishes) and
**precisely** (on runs that were actually going to fail). This experiment
tests that claim by *replaying* each recorded Terminal-Bench trace prefix by
prefix through :meth:`TraceAbstractionAgent.compile_prefix`, exactly as
:func:`htir.agents.intervention.run_intervention_loop` does, and, at every
step t, reading the *active obligation set*
(:func:`htir.agents.intervention.active_obligations` -- the high-severity
obligations that are FAILED or ABSTAINED on G_{tau<=t}) and the intervention
:func:`htir.agents.intervention.select_intervention` would recommend
(iota_t). Nothing drives an agent; this is a recommendation trace scored
against the weak trajectory reward label.

We separate two intervention *classes* by evidence confidence, because offline
they behave very differently:

* ``any_active``       -- a step raised *any* active high-severity obligation.
  This is dominated by ABSTENTIONS (edit-not-yet-validated, unlinked policy
  action, untraceable final claim): insufficient-evidence flags, not proven
  failures. High recall, but near-base-rate precision -- the same
  weak-offline-calibration finding as SA-1/SA-3.
* ``failed_obligation`` -- a step raised an obligation with status FAILED: a
  mechanical check (a test/validation/command run) that *actually reported
  failure*. This is the confident, evidence-backed stop signal. Rarer offline,
  but **precise** (well above base rate) and **timely** (fires far earlier in
  the trace than the abstention flags).

For each class we report, against the weak label (invalid = "should have
intervened"):

* **intervention precision / recall** -- of the traces the class flags, what
  fraction truly failed (precision); of truly failed traces, what fraction it
  flags (recall). Precision above the invalid base rate is real signal.
* **timeliness** -- the first-fire position t*/T (how early into the trace the
  first flag lands; smaller = timelier).
* **steps-saved** -- the counterfactual "intervening at t* would have avoided
  the rest of the run": T - t* downstream steps on **invalid** traces (where
  stopping early is pure savings), reported absolute and as a fraction. On
  **valid** traces a flag is a false alarm, counted as a wasted interruption.

We also emit the **per-step active-obligation profile** (active-obligation
mass by normalized trace position and by template), and a worked
**per-step obligation walkthrough** of one exemplar trace -- a failed run whose
mechanical-failure obligation fires early -- as the qualitative deliverable.

Offline reproducibility. With no API key (``use_llm=False``, the default) the
whole replay is deterministic and issues zero LLM calls: the semantic and
integrity checkers abstain rather than calling a model, so the confident
``failed_obligation`` signal comes only from mechanical checks (exit codes /
test reports). The abstention-driven ``any_active`` flags therefore carry
little discriminative signal offline; enabling the semantic/integrity checker
(``--use-llm`` with a key) is what turns those abstentions into confident
catches. This gate is reported, not hidden.

CLI::

    python -m htir.eval.experiment_sa4 --cache sa4_sample.jsonl --out sa4_results.json
    python -m htir.eval.experiment_sa4 --hf --n 3000 --out sa4_results.json
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Optional

from pydantic import BaseModel, Field

from htir.agents.intervention import active_obligations, select_intervention
from htir.agents.trace_abstraction import TraceAbstractionAgent
from htir.eval.datasets import iter_local_traces, to_canonical_steps
from htir.eval.weak_labels import (
    LABEL_INVALID,
    LABEL_VALID,
    extract_reward,
    label_from_reward,
)
from htir.models.domain import TERMINAL_DOMAIN_SPEC, DomainSpec
from htir.models.htir import HTIR, Obligation, ObligationStatus

# The two intervention classes we contrast, in table order. "any_active" is the
# union (recall-oriented, abstention-dominated); "failed_obligation" is the
# confident mechanical-failure subset (precision-oriented, timely).
CLASS_ANY = "any_active"
CLASS_FAILED = "failed_obligation"
INTERVENTION_CLASSES = [CLASS_ANY, CLASS_FAILED]

# Normalized-position bins for the active-obligation profile (deciles).
_POSITION_BINS = 10

# The walkthrough exemplar prefers a compact trace (<= this many steps) so the
# per-step deliverable stays readable, then the earliest confident failure.
_WALK_MAX_STEPS = 40


class PerTraceInterventionRecord(BaseModel):
    """One trace's label, size, and the first step each class first fires at."""
    task_id: str = ""
    reward: Optional[int] = None
    label: Optional[str] = None
    n_steps: int = 0
    # class name -> earliest step t (1-indexed) an obligation of that class was
    # active on G_{tau<=t}; None if the class never fired on this trace.
    first_fire: dict[str, Optional[int]] = Field(default_factory=dict)
    fired_templates: list[str] = Field(default_factory=list)


class InterventionClassReport(BaseModel):
    """Precision/recall + timeliness + steps-saved for one intervention class."""
    name: str
    flagged: int = Field(0, description="labeled traces this class fired on (at any step)")
    tp: int = 0
    fp: int = 0
    fn: int = 0
    precision: float = Field(0.0, description="P(invalid | flagged): flags that were real failures")
    recall: float = Field(0.0, description="P(flagged | invalid): real failures the class catches")
    # Timeliness of the first flag, as a fraction t*/T of the way through the trace.
    first_fire_frac_mean: Optional[float] = None
    first_fire_frac_median: Optional[float] = None
    # Counterfactual steps saved on INVALID flagged traces (T - t*).
    steps_saved_mean: Optional[float] = None
    steps_saved_median: Optional[float] = None
    steps_saved_frac_mean: Optional[float] = Field(None, description="(T - t*)/T on invalid flagged traces")
    false_alarm_interruptions: int = Field(0, description="valid traces this class would have interrupted (FP)")


class PositionBin(BaseModel):
    """Active-obligation mass in one normalized-position decile of the trace."""
    bin_lo: float
    bin_hi: float
    active_obligations: int = Field(0, description="total (step, active-obligation) pairs landing in this decile")
    steps: int = Field(0, description="total steps landing in this decile")
    mean_active_per_step: float = 0.0


class TemplateFireReport(BaseModel):
    """How often one obligation template drove an intervention, and via what action."""
    template_id: str
    active_count: int = Field(0, description="(step, active-obligation) pairs from this template")
    n_traces: int = Field(0, description="distinct traces where it was ever active")
    failed_count: int = Field(0, description="of those, how many were status FAILED (vs ABSTAINED)")
    dominant_action: str = Field("", description="most common iota_t selected for it")


class WalkthroughActive(BaseModel):
    """One active obligation at one step of the walkthrough exemplar."""
    template_id: str = ""
    status: str = ""
    severity: str = ""
    action: str = Field("", description="iota_t select_intervention would recommend")


class WalkthroughStep(BaseModel):
    """One step of the per-step obligation walkthrough (avg.tex Sec. 3.11 example)."""
    step: int
    role: str = ""
    request_preview: str = ""
    active: list[WalkthroughActive] = Field(default_factory=list)


class Walkthrough(BaseModel):
    """A worked per-step replay of one exemplar trace (the qualitative deliverable)."""
    task_id: str = ""
    label: Optional[str] = None
    n_steps: int = 0
    first_failed_step: Optional[int] = None
    steps_saved: Optional[int] = None
    steps: list[WalkthroughStep] = Field(default_factory=list)


class SA4Result(BaseModel):
    """Full SA-4 output: config, per-class reports, profile, templates, walkthrough."""
    experiment: str = "SA-4: Online Intervention, offline replay (Q4a)"
    n_traces: int = 0
    n_labeled: int = 0
    base_rate_invalid: float = Field(0.0, description="Fraction of labeled traces with label 'invalid'")
    use_llm: bool = False
    domain_id: str = ""
    seconds: float = 0.0
    classes: list[InterventionClassReport] = Field(default_factory=list)
    position_profile: list[PositionBin] = Field(default_factory=list)
    template_fires: list[TemplateFireReport] = Field(default_factory=list)
    walkthrough: Optional[Walkthrough] = None
    notes: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Replay
# ---------------------------------------------------------------------------

class _StepScan(BaseModel):
    """Internal: the active obligations observed at one replayed prefix step."""
    step: int
    role: str
    request_preview: str
    active: list[Obligation]

    model_config = {"arbitrary_types_allowed": True}


def _classes_fired(active: list[Obligation]) -> set[str]:
    """Which intervention classes the active-obligation set at this step triggers."""
    fired: set[str] = set()
    if active:
        fired.add(CLASS_ANY)
        if any(o.status == ObligationStatus.FAILED for o in active):
            fired.add(CLASS_FAILED)
    return fired


def _replay(agent: TraceAbstractionAgent, task_id: str, steps: list[dict[str, Any]], **compile_kwargs: Any) -> list[_StepScan]:
    """
    Replay ``steps`` prefix by prefix (mirroring
    :func:`htir.agents.intervention.run_intervention_loop`, but retaining the
    typed :class:`Obligation` objects so we can read template/status/action).
    Returns one :class:`_StepScan` per step t=1..T.
    """
    scans: list[_StepScan] = []
    for t in range(1, len(steps) + 1):
        htir_prefix = agent.compile_prefix(task_id, steps, {}, t, **compile_kwargs)
        raw = steps[t - 1] if t - 1 < len(steps) else {}
        role = htir_prefix.steps[-1].role if htir_prefix.steps else ""
        req = str(raw.get("request", raw.get("prompt", raw.get("input", ""))))
        scans.append(_StepScan(
            step=t,
            role=str(role),
            request_preview=req[:120],
            active=active_obligations(htir_prefix),
        ))
    return scans


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_sa4(
    raw_traces: Iterable[dict[str, Any]],
    *,
    spec: DomainSpec | None = None,
    use_llm: bool = False,
    model: str = "openai/gpt-4o",
    progress_every: int = 100,
    log: Any = sys.stderr,
) -> SA4Result:
    """
    Execute SA-4 over ``raw_traces`` (turn-schema dicts with a ``reward``).

    Each trace is replayed prefix by prefix; at every step we record the active
    high-severity obligation set and the intervention it would trigger. Returns
    a fully populated :class:`SA4Result`. Offline (no LLM calls) by default.
    """
    spec = spec or TERMINAL_DOMAIN_SPEC
    agent = TraceAbstractionAgent(model=model, domain_spec=spec)
    compile_kwargs: dict[str, Any] = {"use_semantic_analysis": use_llm}

    records: list[PerTraceInterventionRecord] = []
    # Aggregates over all (step, active-obligation) pairs.
    pos_active = [0] * _POSITION_BINS
    pos_steps = [0] * _POSITION_BINS
    tmpl_active: dict[str, int] = {}
    tmpl_failed: dict[str, int] = {}
    tmpl_traces: dict[str, set[int]] = {}
    tmpl_actions: dict[str, dict[str, int]] = {}
    # Walkthrough candidate: an invalid trace whose failed_obligation fires
    # early (maximizing illustrated steps-saved), preferring a compact trace for
    # readability. Selected by the sort key (is_long, fire_fraction, n_steps).
    best_walk_key: Optional[tuple[int, float, int]] = None
    best_walk: Optional[tuple[int, list[_StepScan], PerTraceInterventionRecord]] = None

    t0 = time.time()
    for i, raw in enumerate(raw_traces):
        reward = extract_reward(raw)
        label = label_from_reward(reward)
        task_id = str(raw.get("task_name", "")) if isinstance(raw, dict) else ""

        try:
            steps = to_canonical_steps(raw)
            scans = _replay(agent, task_id or f"trace-{i}", steps, **compile_kwargs)
        except Exception as exc:  # a malformed trace should not sink the run
            if log is not None:
                print(f"[sa4] skip trace {i} ({task_id}): {exc!r}", file=log)
            continue

        T = len(scans)
        first_fire: dict[str, Optional[int]] = {c: None for c in INTERVENTION_CLASSES}
        fired_templates: set[str] = set()
        first_failed_step: Optional[int] = None

        for sc in scans:
            fired = _classes_fired(sc.active)
            for c in fired:
                if first_fire[c] is None:
                    first_fire[c] = sc.step
            if first_failed_step is None and CLASS_FAILED in fired:
                first_failed_step = sc.step

            # Position profile (normalized decile of this step within the trace).
            b = min(_POSITION_BINS - 1, int((sc.step - 1) / T * _POSITION_BINS)) if T else 0
            pos_steps[b] += 1
            pos_active[b] += len(sc.active)

            for o in sc.active:
                tid = o.template_id or "(none)"
                tmpl_active[tid] = tmpl_active.get(tid, 0) + 1
                if o.status == ObligationStatus.FAILED:
                    tmpl_failed[tid] = tmpl_failed.get(tid, 0) + 1
                tmpl_traces.setdefault(tid, set()).add(i)
                fired_templates.add(tid)
                act = _safe_action(o)
                actions = tmpl_actions.setdefault(tid, {})
                actions[act] = actions.get(act, 0) + 1

        records.append(PerTraceInterventionRecord(
            task_id=task_id,
            reward=reward,
            label=label,
            n_steps=T,
            first_fire=first_fire,
            fired_templates=sorted(fired_templates),
        ))

        # Track the best walkthrough exemplar (invalid; confident failure fires
        # early; compact trace preferred for readability).
        if label == LABEL_INVALID and first_failed_step is not None and T > 0:
            key = (1 if T > _WALK_MAX_STEPS else 0, first_failed_step / T, T)
            if best_walk_key is None or key < best_walk_key:
                best_walk_key = key
                best_walk = (first_failed_step, scans, records[-1])

        if progress_every and log is not None and (i + 1) % progress_every == 0:
            print(f"[sa4] replayed {i + 1} traces...", file=log)

    return _assemble(
        records, spec, use_llm, time.time() - t0,
        pos_active, pos_steps, tmpl_active, tmpl_failed, tmpl_traces, tmpl_actions, best_walk,
    )


def _safe_action(o: Obligation) -> str:
    """iota_t for one obligation via the default policy (reduces to alpha_i)."""
    return select_intervention(o, HTIR(task_id="_sa4")).value


def _stats(xs: list[float]) -> tuple[Optional[float], Optional[float]]:
    if not xs:
        return None, None
    return statistics.fmean(xs), statistics.median(xs)


def compute_class_report(labeled: list[PerTraceInterventionRecord], class_name: str) -> InterventionClassReport:
    """
    Precision/recall + timeliness + counterfactual steps-saved for one
    intervention class over the labeled per-trace records. A trace is "flagged"
    if ``first_fire[class_name]`` is set (the class fired at some step t). Tied
    to the weak label: invalid = "should have intervened", so a flag on an
    invalid trace is a true positive and on a valid trace a false alarm.
    Steps-saved (T - t*) is the counterfactual over invalid flagged traces
    only -- stopping the doomed run at its first flag.
    """
    flagged = [r for r in labeled if r.first_fire.get(class_name) is not None]
    tp = sum(1 for r in flagged if r.label == LABEL_INVALID)
    fp = sum(1 for r in flagged if r.label == LABEL_VALID)
    fn = sum(1 for r in labeled if r.label == LABEL_INVALID and r.first_fire.get(class_name) is None)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0

    fire_frac = [r.first_fire[class_name] / r.n_steps for r in flagged if r.n_steps]
    inv_flagged = [r for r in flagged if r.label == LABEL_INVALID and r.n_steps]
    steps_saved = [r.n_steps - r.first_fire[class_name] for r in inv_flagged]
    steps_saved_frac = [(r.n_steps - r.first_fire[class_name]) / r.n_steps for r in inv_flagged]

    ff_mean, ff_med = _stats(fire_frac)
    ss_mean, ss_med = _stats([float(x) for x in steps_saved])
    ssf_mean, _ = _stats(steps_saved_frac)

    return InterventionClassReport(
        name=class_name, flagged=len(flagged), tp=tp, fp=fp, fn=fn,
        precision=precision, recall=recall,
        first_fire_frac_mean=ff_mean, first_fire_frac_median=ff_med,
        steps_saved_mean=ss_mean, steps_saved_median=ss_med,
        steps_saved_frac_mean=ssf_mean,
        false_alarm_interruptions=fp,
    )


def _assemble(
    records: list[PerTraceInterventionRecord],
    spec: DomainSpec,
    use_llm: bool,
    seconds: float,
    pos_active: list[int],
    pos_steps: list[int],
    tmpl_active: dict[str, int],
    tmpl_failed: dict[str, int],
    tmpl_traces: dict[str, set[int]],
    tmpl_actions: dict[str, dict[str, int]],
    best_walk: Optional[tuple[int, list[Any], PerTraceInterventionRecord]],
) -> SA4Result:
    labeled = [r for r in records if r.label is not None]
    n_invalid = sum(1 for r in labeled if r.label == LABEL_INVALID)
    base_rate = (n_invalid / len(labeled)) if labeled else 0.0

    class_reports = [compute_class_report(labeled, c) for c in INTERVENTION_CLASSES]

    profile: list[PositionBin] = []
    for b in range(_POSITION_BINS):
        s = pos_steps[b]
        profile.append(PositionBin(
            bin_lo=round(b / _POSITION_BINS, 2),
            bin_hi=round((b + 1) / _POSITION_BINS, 2),
            active_obligations=pos_active[b],
            steps=s,
            mean_active_per_step=round(pos_active[b] / s, 3) if s else 0.0,
        ))

    template_fires: list[TemplateFireReport] = []
    for tid, cnt in sorted(tmpl_active.items(), key=lambda kv: kv[1], reverse=True):
        actions = tmpl_actions.get(tid, {})
        dominant = max(actions.items(), key=lambda kv: kv[1])[0] if actions else ""
        template_fires.append(TemplateFireReport(
            template_id=tid,
            active_count=cnt,
            n_traces=len(tmpl_traces.get(tid, set())),
            failed_count=tmpl_failed.get(tid, 0),
            dominant_action=dominant,
        ))

    walkthrough = _build_walkthrough(best_walk)

    notes: list[str] = []
    if not use_llm:
        notes.append(
            "Offline run (no API key): semantic + integrity checkers abstain, so the "
            "confident 'failed_obligation' signal is mechanical only (exit codes / test "
            "reports). 'any_active' is abstention-dominated -> near-base-rate precision. "
            "Enable --use-llm (with a key) to turn those abstentions into confident catches."
        )
    notes.append(
        "Intervention is recommendation-only (htir.agents.intervention): the replay never "
        "drives an agent. Precision/recall are vs. the weak trajectory reward label "
        "(invalid = 'should have intervened'); steps-saved is the counterfactual T - t* on "
        "invalid flagged traces (stopping the doomed run at the first flag)."
    )
    notes.append(
        "'failed_obligation' fires only when a mechanical check reported failure, so it is "
        "rarer but precise and timely; 'any_active' includes edit-not-yet-validated / "
        "unlinked-policy / untraceable-final-claim abstentions, which mostly land near the "
        "trace end (see position_profile)."
    )

    return SA4Result(
        n_traces=len(records),
        n_labeled=len(labeled),
        base_rate_invalid=base_rate,
        use_llm=use_llm,
        domain_id=spec.domain_id,
        seconds=round(seconds, 2),
        classes=class_reports,
        position_profile=profile,
        template_fires=template_fires,
        walkthrough=walkthrough,
        notes=notes,
    )


def _build_walkthrough(best_walk: Optional[tuple[int, list[Any], PerTraceInterventionRecord]]) -> Optional[Walkthrough]:
    if best_walk is None:
        return None
    first_failed, scans, rec = best_walk
    steps: list[WalkthroughStep] = []
    for sc in scans:
        actives = [
            WalkthroughActive(
                template_id=o.template_id or "(none)",
                status=o.status.value,
                severity=o.severity.value,
                action=_safe_action(o),
            )
            for o in sc.active
        ]
        steps.append(WalkthroughStep(
            step=sc.step, role=sc.role, request_preview=sc.request_preview, active=actives,
        ))
    return Walkthrough(
        task_id=rec.task_id,
        label=rec.label,
        n_steps=rec.n_steps,
        first_failed_step=first_failed,
        steps_saved=rec.n_steps - first_failed if rec.n_steps else None,
        steps=steps,
    )


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def format_table(result: SA4Result) -> str:
    """A compact fixed-width results table for the terminal / logs."""
    lines: list[str] = []
    lines.append(
        f"SA-4: Online Intervention (Q4a)  |  n={result.n_traces} "
        f"(labeled {result.n_labeled}, base-rate invalid {result.base_rate_invalid:.2f})  |  "
        f"domain={result.domain_id}  use_llm={result.use_llm}"
    )
    header = (
        f"{'class':<18} {'flagged':>7} {'prec':>6} {'recall':>7} "
        f"{'t*/T med':>9} {'saved med':>10} {'saved mean':>11} {'false_alarm':>11}"
    )
    lines.append("  " + header)
    for c in result.classes:
        ff = f"{c.first_fire_frac_median:.2f}" if c.first_fire_frac_median is not None else "  n/a"
        sm = f"{c.steps_saved_median:.1f}" if c.steps_saved_median is not None else "  n/a"
        sa = f"{c.steps_saved_mean:.1f}" if c.steps_saved_mean is not None else "  n/a"
        lines.append(
            f"  {c.name:<18} {c.flagged:>7} {c.precision:>6.3f} {c.recall:>7.3f} "
            f"{ff:>9} {sm:>10} {sa:>11} {c.false_alarm_interruptions:>11}"
        )

    lines.append("  [active-obligation profile by normalized trace position]")
    lines.append(f"    {'position':>12} {'steps':>7} {'active':>8} {'active/step':>12}")
    for p in result.position_profile:
        lines.append(
            f"    {p.bin_lo:>5.1f}-{p.bin_hi:<5.1f} {p.steps:>7} {p.active_obligations:>8} "
            f"{p.mean_active_per_step:>12.3f}"
        )

    lines.append("  [obligation templates driving interventions]")
    lines.append(f"    {'template_id':<44} {'active':>8} {'traces':>7} {'failed':>7} {'action':>9}")
    for t in result.template_fires:
        lines.append(
            f"    {t.template_id:<44} {t.active_count:>8} {t.n_traces:>7} {t.failed_count:>7} {t.dominant_action:>9}"
        )

    w = result.walkthrough
    if w is not None:
        lines.append(
            f"  [walkthrough: {w.task_id} ({w.label}), {w.n_steps} steps; "
            f"first mechanical-failure obligation at step {w.first_failed_step} "
            f"-> {w.steps_saved} downstream steps saved]"
        )
        for s in w.steps:
            n_failed = sum(1 for a in s.active if a.status == "failed")
            n_abst = len(s.active) - n_failed
            tag = f"{len(s.active)} active ({n_failed} failed, {n_abst} abstained)" if s.active else "-"
            lines.append(f"    step {s.step:>2} [{s.role:<16}] {tag}")

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
    p = argparse.ArgumentParser(description="SA-4: Online Intervention, offline replay (Q4a)")
    src = p.add_argument_group("data source")
    src.add_argument("--cache", type=str, default="", help="local JSON/JSONL sample (turn schema)")
    src.add_argument("--hf", action="store_true", help="pull from the HF terminalbench dataset")
    src.add_argument("--hf-limit", type=int, default=9000, help="records to stream when --hf")
    src.add_argument("--n", type=int, default=3000, help="target balanced sample size")
    src.add_argument("--seed", type=int, default=0)
    p.add_argument("--use-llm", action="store_true", help="enable the semantic/integrity checker")
    p.add_argument("--model", type=str, default="openai/gpt-4o")
    p.add_argument("--out", type=str, default="", help="write SA4Result JSON here")
    args = p.parse_args(argv)

    traces = _load_traces(args)
    result = run_sa4(traces, use_llm=args.use_llm, model=args.model)
    print(format_table(result))
    if args.out:
        Path(args.out).write_text(result.model_dump_json(indent=2), encoding="utf-8")
        print(f"\n[sa4] wrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
