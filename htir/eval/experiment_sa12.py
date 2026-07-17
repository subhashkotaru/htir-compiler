"""
SA-12 -- Human-review efficiency of the verification witness (spotlight plan P1).

The intro promises that AVG's verification witness ``W_tau`` (avg.tex Sec. 3.9)
*compresses the trace for a human reviewer*: instead of re-reading a whole
trajectory, a reviewer inspects a short, evidence-localized summary -- which
obligations passed / failed / abstained, the evidence to look at, and a one-line
review recommendation. SA-12 tests that promise directly: shown the witness
instead of the raw trace, does a reviewer reach a **correct verdict faster and/or
more accurately**?

This is a small controlled human study, so the deliverable is a *harness*, not a
finished measurement. The study is a two-condition, counterbalanced, within-trace
design:

* ~50 traces, balanced valid / invalid (:func:`htir.eval.datasets.balanced_sample`).
* Each trace is rendered two ways -- **raw** (the full trajectory) and **witness**
  (``W_tau``) -- and shown to raters. Raters are split into two counterbalance
  groups so that (i) every trace is reviewed in *both* conditions across the pool
  and (ii) no rater ever sees the same trace twice and each rater sees a balanced
  mix of conditions. Order/condition is therefore not confounded with trace.
* We measure, per rating, **time-to-verdict** and the **verdict** (valid/invalid),
  and score the verdict against the ground-truth reward label
  (:func:`htir.eval.weak_labels.trace_label`).

Metrics (:class:`SA12Result`): per-condition verdict **accuracy**, **median /
mean time-to-verdict**, the reviewer **false-valid rate** (does a raw-trace
reviewer get fooled by a reward-hack the same way the monolith does?), and
**inter-rater agreement**. The two headline contrasts -- witness-minus-raw
**accuracy** and **time** -- are reported as *paired differences over traces*
with a paired t-test (:func:`htir.eval.seeds.paired_t_test`), the same
significance machinery the other SA experiments use.

Two phases, so the study is "fill in the CSV once raters are available":

* ``export`` -- compile the corpus, render both conditions, counterbalance the
  assignment, and write per-rater review packets (HTML), blank rater-response CSV
  templates, and a separate answer key. Deterministic.
* ``score`` -- ingest the filled rater CSVs, join to the answer key, and emit
  :class:`SA12Result` + ``data/sa12_results.json``.

Offline reproducibility. The **human raters** are the external dependency. Until
they are available, ``dryrun`` mode (the default) fills the same scoring pipeline
with a **deterministic simulated rater** -- an explicit, documented model of the
two conditions (a raw-trace reviewer skims a long trajectory and is fooled by
reward-hacks like the endpoint monolith; a witness reviewer follows the localized
recommendation) -- so the harness is exercised end-to-end and byte-reproducible.
Every simulated number is flagged ``simulated=True`` and is a pipeline self-check,
**not** a human result; it is replaced verbatim by real rater CSVs via ``score``.

CLI::

    # 1. export packets for real raters
    python -m htir.eval.experiment_sa12 --mode export --domain tau_bench \\
        --cache data/tau_cache/tau_all.jsonl --n 50 --raters 6 --packet-dir data/sa12_packets

    # 2. once raters return CSVs, score them
    python -m htir.eval.experiment_sa12 --mode score \\
        --items data/sa12_packets/items.json --responses 'data/sa12_packets/responses_*.csv' \\
        --out data/sa12_results.json

    # dry-run (default): simulated raters, reproducible self-check
    python -m htir.eval.experiment_sa12 --domain tau_bench \\
        --cache data/tau_cache/tau_all.jsonl --n 50 --raters 6 --out data/sa12_results.json
"""

from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import json
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Optional

from pydantic import BaseModel, Field

from htir.agents.trace_abstraction import TraceAbstractionAgent
from htir.eval.datasets import iter_local_traces, to_canonical_steps
from htir.eval.seeds import PairedGap, paired_t_test
from htir.eval.weak_labels import (
    LABEL_INVALID,
    LABEL_VALID,
    extract_reward,
    label_from_reward,
)
from htir.models.domain import DomainArtifactBundle, DomainSpec, get_domain_spec
from htir.models.htir import HTIR, Obligation, ObligationStatus

CONDITION_RAW = "raw"
CONDITION_WITNESS = "witness"
CONDITIONS = (CONDITION_RAW, CONDITION_WITNESS)

# How many characters of each step's request/response a raw-trace packet shows
# before truncating (keeps the packet readable; the raw condition is still the
# long one relative to the witness).
DEFAULT_MAX_CHARS = 240


# ---------------------------------------------------------------------------
# Simulated-rater model (dry-run only -- an explicit, documented stand-in for
# real humans so the scoring pipeline is exercised and byte-reproducible).
# ---------------------------------------------------------------------------

class SimRaterParams(BaseModel):
    """
    Parameters of the deterministic simulated rater. Every value is an explicit
    modelling assumption, not a measurement; real ``score`` runs ignore this
    entirely. Time is modelled as *reading cost*: the witness compresses the
    trace, so its reading units (failed + abstained obligations + the one-line
    recommendation) are far fewer than the raw trace's (every step).
    """
    sec_per_step: float = Field(6.0, description="Raw: seconds a reviewer spends per trajectory step")
    sec_per_witness_item: float = Field(5.0, description="Witness: seconds per surfaced obligation line")
    base_sec: float = Field(8.0, description="Fixed orientation time added to both conditions")
    jitter_sec: float = Field(6.0, description="Per-(rater,trace) deterministic time jitter magnitude")
    # Accuracy model. The witness reviewer follows the localized recommendation
    # and is right most of the time. The raw reviewer skims: fine on genuinely
    # completed (valid) traces, but systematically fooled on reward-hacks
    # (invalid-but-complete), where errors are biased toward crediting 'valid'.
    p_witness_correct: float = Field(0.90, description="Witness: P(correct verdict)")
    p_raw_correct_on_valid: float = Field(0.82, description="Raw: P(correct) when trace is truly valid")
    p_raw_correct_on_invalid: float = Field(0.46, description="Raw: P(correct) when trace is a reward-hack")


# ---------------------------------------------------------------------------
# Review packet models
# ---------------------------------------------------------------------------

class ReviewItem(BaseModel):
    """
    One trace prepared for review in both conditions. ``ground_truth`` is the
    answer key -- present in the experimenter's ``items.json`` but never shown to
    raters (the per-rater HTML/CSV omit it).
    """
    trace_id: str
    task_id: str = ""
    ground_truth: str = Field("", description="'valid'/'invalid' from reward (answer key)")
    reward: Optional[int] = None
    predicted_status: str = Field("", description="AVG's own aggregate verdict (metadata, not shown)")
    n_steps: int = 0
    n_passed: int = 0
    n_failed: int = 0
    n_abstained: int = 0
    n_evidence: int = 0
    raw_text: str = Field("", description="Rendering shown in the RAW condition")
    witness_text: str = Field("", description="Rendering shown in the WITNESS condition")

    @property
    def raw_read_units(self) -> int:
        """Reading cost of the raw condition: every trajectory step."""
        return max(self.n_steps, 1)

    @property
    def witness_read_units(self) -> int:
        """Reading cost of the witness: the surfaced obligations + recommendation."""
        return self.n_failed + self.n_abstained + 1


class ReviewAssignment(BaseModel):
    """One (rater, trace) cell of the counterbalanced design and its condition."""
    rater_id: str
    trace_id: str
    condition: str


class RaterResponse(BaseModel):
    """One rater's recorded review of one trace in one condition."""
    rater_id: str
    trace_id: str
    condition: str
    verdict: str = Field("", description="Rater's decision: 'valid' / 'invalid'")
    seconds: float = 0.0


# ---------------------------------------------------------------------------
# Result models
# ---------------------------------------------------------------------------

class ConditionReport(BaseModel):
    """Aggregate reviewer performance in one condition (raw or witness)."""
    condition: str
    n_ratings: int = 0
    accuracy: float = Field(0.0, description="Fraction of verdicts matching ground truth")
    median_seconds: float = 0.0
    mean_seconds: float = 0.0
    false_valid_rate: float = Field(
        0.0, description="Of invalid traces reviewed, fraction the rater credited 'valid'",
    )
    inter_rater_agreement: float = Field(
        0.0, description="Mean within-trace pairwise verdict agreement (chance-uncorrected)",
    )


class SA12Result(BaseModel):
    """Full SA-12 human-review-efficiency output: config, per-condition, contrasts."""
    experiment: str = "SA-12: Human-review efficiency of the verification witness"
    domain_id: str = ""
    simulated: bool = Field(True, description="True = deterministic simulated raters (dry-run), not humans")
    n_traces: int = 0
    n_raters: int = 0
    n_ratings: int = 0
    base_rate_valid: float = 0.0
    seconds: float = 0.0

    conditions: list[ConditionReport] = Field(default_factory=list)
    # Paired-over-traces contrasts (witness - raw): accuracy (higher is better)
    # and time-to-verdict (lower is better).
    accuracy_gap: PairedGap = Field(default_factory=PairedGap)
    time_gap: PairedGap = Field(default_factory=PairedGap)
    n_paired_traces: int = Field(0, description="Traces rated in BOTH conditions (the paired unit)")
    notes: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Rendering: raw trace vs. witness W_tau
# ---------------------------------------------------------------------------

def _truncate(text: str, max_chars: int) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= max_chars else text[: max_chars - 1] + "…"


def render_raw_trace(htir: HTIR, *, max_chars: int = DEFAULT_MAX_CHARS) -> str:
    """The RAW condition: the full trajectory, step by step (the long read)."""
    lines = [f"RAW TRAJECTORY  --  task {htir.task_id}  ({len(htir.steps)} steps)"]
    for step in htir.steps_in_order():
        lines.append(
            f"[{step.step_id}] role={step.role} status={step.execution_status.value}"
        )
        req = _truncate(step.request_message, max_chars)
        resp = _truncate(step.response_message, max_chars)
        if req:
            lines.append(f"    > {req}")
        if resp:
            lines.append(f"    < {resp}")
    lines.append("Verdict: did this trajectory correctly and legitimately solve the task? [valid/invalid]")
    return "\n".join(lines)


def _oblig_label(o: Obligation) -> str:
    return o.template_id or o.description or f"obligation #{o.obligation_id}"


def render_witness(htir: HTIR) -> str:
    """
    The WITNESS condition: ``W_tau`` -- the compact, evidence-localized summary a
    reviewer inspects instead of the whole trace. Passed obligations are listed by
    count only (nothing to inspect); failed and abstained ones are named so the
    reviewer knows *what* to check; the review recommendation points at the single
    most important obligation.
    """
    w = htir.witness
    by_id = {o.obligation_id: o for o in htir.obligations}
    lines = [f"VERIFICATION WITNESS  --  task {htir.task_id}"]
    if w is None:
        lines.append("(no witness available)")
        return "\n".join(lines)

    lines.append(f"Passed obligations (O+): {len(w.passed_obligation_ids)} discharged.")
    if w.failed_obligation_ids:
        lines.append(f"Failed obligations (O-): {len(w.failed_obligation_ids)}")
        for oid in w.failed_obligation_ids:
            o = by_id.get(oid)
            if o is not None:
                lines.append(f"    - #{oid} [{o.severity.value}] {_oblig_label(o)}")
    if w.abstained_obligation_ids:
        lines.append(f"Unresolved / abstained (O-empty): {len(w.abstained_obligation_ids)}")
        for oid in w.abstained_obligation_ids:
            o = by_id.get(oid)
            if o is not None:
                lines.append(f"    - #{oid} [{o.severity.value}] {_oblig_label(o)}")
    if w.witness_evidence_ids:
        lines.append(f"Evidence to inspect (E_W): {w.witness_evidence_ids}")
    lines.append(f"Review recommendation (R_W): {w.review_recommendation}")
    lines.append("Verdict: did this trajectory correctly and legitimately solve the task? [valid/invalid]")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Build review items (compile once, render both conditions)
# ---------------------------------------------------------------------------

def build_review_items(
    raw_traces: Iterable[dict[str, Any]],
    *,
    spec: DomainSpec,
    omega: DomainArtifactBundle | None = None,
    use_llm: bool = False,
    max_chars: int = DEFAULT_MAX_CHARS,
    model: str = "openai/gpt-4o",
    progress_every: int = 0,
    log: Any = sys.stderr,
) -> list[ReviewItem]:
    """
    Compile every trace once through the full AVG pipeline (``run_checks=True`` so
    the witness is built) and render both review conditions. Traces without a
    ground-truth reward label are skipped (the study needs an answer key). Fully
    deterministic offline.
    """
    agent = TraceAbstractionAgent(model=model, domain_spec=spec, domain_artifacts=omega)
    items: list[ReviewItem] = []
    for i, raw in enumerate(raw_traces):
        reward = extract_reward(raw)
        label = label_from_reward(reward)
        if label is None:
            continue
        task_id = str(raw.get("task_name", "")) if isinstance(raw, dict) else ""
        task_id = task_id or f"trace-{i}"
        trace_id = f"{task_id}#{i:04d}"
        try:
            steps = to_canonical_steps(raw)
            htir = agent.compile(
                task_id=task_id,
                raw_steps=steps,
                harness_snippets={},
                generate_obligations=True,
                use_semantic_analysis=use_llm,
                run_checks=True,
                domain_artifacts=omega,
            )
        except Exception as exc:  # a malformed trace should not sink the run
            if log is not None:
                print(f"[sa12] skip trace {i} ({task_id}): {exc!r}", file=log)
            continue

        w = htir.witness
        n_passed = len(w.passed_obligation_ids) if w else 0
        n_failed = len(w.failed_obligation_ids) if w else 0
        n_abstained = len(w.abstained_obligation_ids) if w else 0
        n_evidence = len(w.witness_evidence_ids) if w else 0
        items.append(ReviewItem(
            trace_id=trace_id,
            task_id=task_id,
            ground_truth=label,
            reward=reward,
            predicted_status=htir.aggregate.predicted_status if htir.aggregate else "",
            n_steps=len(htir.steps),
            n_passed=n_passed,
            n_failed=n_failed,
            n_abstained=n_abstained,
            n_evidence=n_evidence,
            raw_text=render_raw_trace(htir, max_chars=max_chars),
            witness_text=render_witness(htir),
        ))
        if progress_every and log is not None and (i + 1) % progress_every == 0:
            print(f"[sa12] built {len(items)} review items...", file=log)
    return items


# ---------------------------------------------------------------------------
# Counterbalanced assignment
# ---------------------------------------------------------------------------

def assign_conditions(
    items: list[ReviewItem], *, n_raters: int, seed: int = 0,
) -> list[ReviewAssignment]:
    """
    Counterbalanced within-trace assignment. Raters are split into two groups by
    parity; for trace index ``i`` and group ``g in {0, 1}`` the condition is
    ``CONDITIONS[(i + g) % 2]``. Consequences:

    * every trace is reviewed in **both** conditions across the pool (as long as
      each group is non-empty),
    * each rater reviews **each trace exactly once** (never the same trace twice),
    * each rater sees a balanced ~50/50 mix of conditions.

    Order is deterministic (``seed`` reserved for future shuffling of item order
    per rater; the assignment itself is fixed by the parity design).
    """
    assignments: list[ReviewAssignment] = []
    for r in range(n_raters):
        rater_id = f"R{r:02d}"
        group = r % 2
        for i, it in enumerate(items):
            cond = CONDITIONS[(i + group) % 2]
            assignments.append(ReviewAssignment(rater_id=rater_id, trace_id=it.trace_id, condition=cond))
    return assignments


# ---------------------------------------------------------------------------
# Export: per-rater HTML packets + blank CSV templates + answer key
# ---------------------------------------------------------------------------

def _html_escape(text: str) -> str:
    return (
        text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    )


def export_packets(
    items: list[ReviewItem],
    assignments: list[ReviewAssignment],
    out_dir: str | Path,
) -> dict[str, str]:
    """
    Write the study instruments to ``out_dir`` and return a manifest of paths:

    * ``items.json``          -- full :class:`ReviewItem` list incl. answer key
      (experimenter-only; consumed by :func:`score`).
    * ``assignments.json``    -- the counterbalanced design.
    * ``packet_<rater>.html`` -- one review packet per rater (no ground truth).
    * ``responses_<rater>.csv`` -- one blank response template per rater to fill.

    Deterministic: identical inputs produce byte-identical files.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    by_id = {it.trace_id: it for it in items}
    manifest: dict[str, str] = {}

    items_path = out / "items.json"
    items_path.write_text(
        json.dumps([it.model_dump() for it in items], indent=2), encoding="utf-8"
    )
    manifest["items"] = str(items_path)

    assign_path = out / "assignments.json"
    assign_path.write_text(
        json.dumps([a.model_dump() for a in assignments], indent=2), encoding="utf-8"
    )
    manifest["assignments"] = str(assign_path)

    by_rater: dict[str, list[ReviewAssignment]] = defaultdict(list)
    for a in assignments:
        by_rater[a.rater_id].append(a)

    for rater_id in sorted(by_rater):
        rater_assigns = by_rater[rater_id]
        html_path = out / f"packet_{rater_id}.html"
        html_path.write_text(_render_packet_html(rater_id, rater_assigns, by_id), encoding="utf-8")
        manifest[f"packet_{rater_id}"] = str(html_path)

        csv_path = out / f"responses_{rater_id}.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["rater_id", "trace_id", "condition", "verdict", "seconds"])
            for a in rater_assigns:
                writer.writerow([a.rater_id, a.trace_id, a.condition, "", ""])
        manifest[f"responses_{rater_id}"] = str(csv_path)

    return manifest


def _render_packet_html(
    rater_id: str, assigns: list[ReviewAssignment], by_id: dict[str, ReviewItem],
) -> str:
    parts = [
        "<section>",
        f"<h1>Review packet &mdash; rater {_html_escape(rater_id)}</h1>",
        "<p>For each item, read the material, decide whether the trajectory "
        "<b>correctly and legitimately</b> solved the task (<code>valid</code>) or "
        "not (<code>invalid</code>), and record your verdict and the seconds you "
        "spent in <code>responses_" + _html_escape(rater_id) + ".csv</code>. "
        "Do not skip ahead; treat each item independently.</p>",
    ]
    for n, a in enumerate(assigns, start=1):
        it = by_id.get(a.trace_id)
        if it is None:
            continue
        body = it.raw_text if a.condition == CONDITION_RAW else it.witness_text
        parts.append(
            f"<article><h2>Item {n} &mdash; {_html_escape(a.trace_id)} "
            f"<small>[{_html_escape(a.condition)}]</small></h2>"
            f"<pre>{_html_escape(body)}</pre></article>"
        )
    parts.append("</section>")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Simulated raters (dry-run) + real CSV ingestion
# ---------------------------------------------------------------------------

def _uniform(*keys: Any) -> float:
    """A deterministic, platform-stable pseudo-uniform draw in [0, 1) from keys
    (hashlib, not the salted builtin ``hash``, so runs are byte-reproducible)."""
    h = hashlib.sha256("|".join(str(k) for k in keys).encode("utf-8")).hexdigest()
    return int(h[:16], 16) / float(1 << 64)


def simulate_responses(
    items: list[ReviewItem],
    assignments: list[ReviewAssignment],
    *,
    params: SimRaterParams | None = None,
) -> list[RaterResponse]:
    """
    Deterministic simulated-rater responses for the dry-run self-check (see the
    module docstring). Time is a reading-cost model (raw pays per step, witness
    per surfaced obligation), so the witness is faster whenever it compresses the
    trace; accuracy follows the documented two-condition model in
    :class:`SimRaterParams`. Not a human result -- flagged ``simulated=True``.
    """
    params = params or SimRaterParams()
    by_id = {it.trace_id: it for it in items}
    responses: list[RaterResponse] = []
    for a in assignments:
        it = by_id.get(a.trace_id)
        if it is None:
            continue
        responses.append(RaterResponse(
            rater_id=a.rater_id,
            trace_id=a.trace_id,
            condition=a.condition,
            verdict=_sim_verdict(it, a, params),
            seconds=_sim_seconds(it, a, params),
        ))
    return responses


def _sim_seconds(it: ReviewItem, a: ReviewAssignment, params: SimRaterParams) -> float:
    if a.condition == CONDITION_WITNESS:
        base = params.base_sec + params.sec_per_witness_item * it.witness_read_units
    else:
        base = params.base_sec + params.sec_per_step * it.raw_read_units
    jitter = (2.0 * _uniform("time", a.rater_id, it.trace_id, a.condition) - 1.0) * params.jitter_sec
    return round(max(1.0, base + jitter), 1)


def _sim_verdict(it: ReviewItem, a: ReviewAssignment, params: SimRaterParams) -> str:
    draw = _uniform("verdict", a.rater_id, it.trace_id, a.condition)
    other = LABEL_INVALID if it.ground_truth == LABEL_VALID else LABEL_VALID
    if a.condition == CONDITION_WITNESS:
        return it.ground_truth if draw < params.p_witness_correct else other
    # Raw condition: skimming, biased toward crediting a complete-looking trace.
    p_correct = (
        params.p_raw_correct_on_valid if it.ground_truth == LABEL_VALID
        else params.p_raw_correct_on_invalid
    )
    return it.ground_truth if draw < p_correct else other


def load_responses(paths: Iterable[str | Path]) -> list[RaterResponse]:
    """
    Ingest filled rater-response CSVs (columns
    ``rater_id,trace_id,condition,verdict,seconds``). Rows with a blank verdict
    (unanswered) are skipped; a blank/invalid ``seconds`` becomes 0.0.
    """
    responses: list[RaterResponse] = []
    for path in paths:
        with Path(path).open(newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                verdict = (row.get("verdict") or "").strip().lower()
                if verdict not in (LABEL_VALID, LABEL_INVALID):
                    continue
                try:
                    seconds = float(row.get("seconds") or 0.0)
                except (TypeError, ValueError):
                    seconds = 0.0
                responses.append(RaterResponse(
                    rater_id=(row.get("rater_id") or "").strip(),
                    trace_id=(row.get("trace_id") or "").strip(),
                    condition=(row.get("condition") or "").strip().lower(),
                    verdict=verdict,
                    seconds=seconds,
                ))
    return responses


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def _inter_rater_agreement(
    responses: list[RaterResponse], condition: str,
) -> float:
    """
    Mean within-trace pairwise verdict agreement in one condition (chance-
    uncorrected): for each trace with >=2 ratings, the fraction of rater pairs
    that gave the same verdict, averaged over such traces. 0.0 if no trace has
    two ratings in the condition.
    """
    by_trace: dict[str, list[str]] = defaultdict(list)
    for r in responses:
        if r.condition == condition:
            by_trace[r.trace_id].append(r.verdict)
    fracs: list[float] = []
    for verdicts in by_trace.values():
        m = len(verdicts)
        if m < 2:
            continue
        agree = 0
        total = 0
        for i in range(m):
            for j in range(i + 1, m):
                total += 1
                if verdicts[i] == verdicts[j]:
                    agree += 1
        fracs.append(agree / total if total else 0.0)
    return statistics.fmean(fracs) if fracs else 0.0


def _condition_report(
    responses: list[RaterResponse], items_by_id: dict[str, ReviewItem], condition: str,
) -> ConditionReport:
    rows = [r for r in responses if r.condition == condition and r.trace_id in items_by_id]
    n = len(rows)
    if n == 0:
        return ConditionReport(condition=condition)
    correct = sum(1 for r in rows if r.verdict == items_by_id[r.trace_id].ground_truth)
    times = [r.seconds for r in rows]
    invalid_rows = [r for r in rows if items_by_id[r.trace_id].ground_truth == LABEL_INVALID]
    false_valid = sum(1 for r in invalid_rows if r.verdict == LABEL_VALID)
    return ConditionReport(
        condition=condition,
        n_ratings=n,
        accuracy=correct / n,
        median_seconds=round(statistics.median(times), 2),
        mean_seconds=round(statistics.fmean(times), 2),
        false_valid_rate=(false_valid / len(invalid_rows)) if invalid_rows else 0.0,
        inter_rater_agreement=round(_inter_rater_agreement(responses, condition), 4),
    )


def _paired_over_traces(
    responses: list[RaterResponse],
    items_by_id: dict[str, ReviewItem],
) -> tuple[list[float], list[float], list[float], list[float], list[str]]:
    """
    Per-trace, per-condition summaries for the paired contrasts. Returns
    ``(acc_witness, acc_raw, time_witness, time_raw, paired_trace_ids)`` aligned
    by trace, over traces that were rated in **both** conditions. Per-trace
    accuracy is the mean correctness over that trace's raters in the condition;
    per-trace time is the median seconds.
    """
    # trace_id -> condition -> (correct flags, times)
    agg: dict[str, dict[str, tuple[list[int], list[float]]]] = defaultdict(
        lambda: {c: ([], []) for c in CONDITIONS}
    )
    for r in responses:
        it = items_by_id.get(r.trace_id)
        if it is None or r.condition not in CONDITIONS:
            continue
        flags, times = agg[r.trace_id][r.condition]
        flags.append(1 if r.verdict == it.ground_truth else 0)
        times.append(r.seconds)

    acc_w: list[float] = []
    acc_r: list[float] = []
    time_w: list[float] = []
    time_r: list[float] = []
    paired_ids: list[str] = []
    for trace_id in sorted(agg):
        wf, wt = agg[trace_id][CONDITION_WITNESS]
        rf, rt = agg[trace_id][CONDITION_RAW]
        if not wf or not rf:  # needs both conditions to be a paired unit
            continue
        acc_w.append(statistics.fmean(wf))
        acc_r.append(statistics.fmean(rf))
        time_w.append(statistics.median(wt))
        time_r.append(statistics.median(rt))
        paired_ids.append(trace_id)
    return acc_w, acc_r, time_w, time_r, paired_ids


def score(
    items: list[ReviewItem],
    responses: list[RaterResponse],
    *,
    domain_id: str = "",
    simulated: bool = False,
    extra_notes: list[str] | None = None,
) -> SA12Result:
    """
    Score rater ``responses`` against the answer key carried by ``items`` into a
    :class:`SA12Result`: per-condition accuracy / time / false-valid / agreement,
    plus the paired-over-traces witness-minus-raw accuracy and time contrasts with
    paired t-tests. Identical whether ``responses`` came from real CSVs or the
    simulated dry-run -- only ``simulated`` and the notes differ.
    """
    items_by_id = {it.trace_id: it for it in items}
    n_valid = sum(1 for it in items if it.ground_truth == LABEL_VALID)
    n_labeled = sum(1 for it in items if it.ground_truth in (LABEL_VALID, LABEL_INVALID))
    raters = {r.rater_id for r in responses}

    conditions = [_condition_report(responses, items_by_id, c) for c in CONDITIONS]
    acc_w, acc_r, time_w, time_r, paired_ids = _paired_over_traces(responses, items_by_id)
    accuracy_gap = paired_t_test(acc_w, acc_r, label="witness_minus_raw.accuracy", a="witness", b="raw")
    time_gap = paired_t_test(time_w, time_r, label="witness_minus_raw.time", a="witness", b="raw")

    notes: list[str] = []
    if simulated:
        notes.append(
            "SIMULATED dry-run (simulated=True): responses are a deterministic "
            "two-condition rater model, NOT humans -- a pipeline self-check that is "
            "byte-reproducible offline. Replace verbatim by running --mode score on "
            "real rater CSVs; the harness and metrics are identical."
        )
    notes.append(
        "Design: within-trace counterbalanced -- every trace is reviewed in both "
        "conditions across the pool, no rater sees a trace twice, and each rater's "
        "condition mix is balanced. The paired unit is the trace (per-trace mean "
        "accuracy / median time in each condition); the gap is witness - raw."
    )
    notes.append(
        "false_valid_rate here is the *reviewer's* rate (credited 'valid' on a "
        "truly-invalid trace) -- the human analogue of the monolith's headline "
        "failure the witness is meant to reduce."
    )
    notes.extend(extra_notes or [])

    return SA12Result(
        domain_id=domain_id,
        simulated=simulated,
        n_traces=n_labeled,
        n_raters=len(raters),
        n_ratings=len(responses),
        base_rate_valid=(n_valid / n_labeled) if n_labeled else 0.0,
        conditions=conditions,
        accuracy_gap=accuracy_gap,
        time_gap=time_gap,
        n_paired_traces=len(paired_ids),
        notes=notes,
    )


# ---------------------------------------------------------------------------
# End-to-end runner (build -> assign -> simulate-or-ingest -> score)
# ---------------------------------------------------------------------------

def run_sa12(
    raw_traces: Iterable[dict[str, Any]],
    *,
    spec: DomainSpec | None = None,
    omega: DomainArtifactBundle | None = None,
    use_llm: bool = False,
    n_raters: int = 6,
    seed: int = 0,
    max_chars: int = DEFAULT_MAX_CHARS,
    sim_params: SimRaterParams | None = None,
    responses: list[RaterResponse] | None = None,
    model: str = "openai/gpt-4o",
    progress_every: int = 0,
    log: Any = sys.stderr,
) -> SA12Result:
    """
    Execute SA-12 end to end over ``raw_traces``.

    Compiles + renders both conditions, builds the counterbalanced assignment,
    then either scores supplied real ``responses`` or (default) the deterministic
    simulated dry-run, returning a :class:`SA12Result`. Fully offline / byte-
    deterministic when ``responses is None`` (the dry-run path the test pins).
    """
    spec = spec or get_domain_spec("tau_bench")
    t0 = time.time()
    items = build_review_items(
        raw_traces, spec=spec, omega=omega, use_llm=use_llm, max_chars=max_chars,
        model=model, progress_every=progress_every, log=log,
    )
    assignments = assign_conditions(items, n_raters=n_raters, seed=seed)

    if responses is None:
        simulated = True
        resp = simulate_responses(items, assignments, params=sim_params)
    else:
        simulated = False
        resp = responses

    result = score(items, resp, domain_id=spec.domain_id, simulated=simulated)
    result.seconds = round(time.time() - t0, 2)
    return result


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def format_table(result: SA12Result) -> str:
    """A compact fixed-width results view for the terminal / logs."""
    lines: list[str] = []
    tag = "SIMULATED dry-run" if result.simulated else "human raters"
    lines.append(
        f"SA-12: Human-review efficiency of the witness  |  domain={result.domain_id}  "
        f"[{tag}]"
    )
    lines.append(
        f"  n_traces={result.n_traces} (base-rate valid {result.base_rate_valid:.2f})  "
        f"raters={result.n_raters}  ratings={result.n_ratings}  "
        f"paired_traces={result.n_paired_traces}"
    )
    header = (
        f"    {'condition':<9} {'n':>5} {'accuracy':>9} {'med_sec':>8} "
        f"{'mean_sec':>9} {'false_valid':>12} {'irr':>6}"
    )
    lines.append("  [per condition]")
    lines.append(header)
    for c in result.conditions:
        lines.append(
            f"    {c.condition:<9} {c.n_ratings:>5} {c.accuracy:>9.3f} "
            f"{c.median_seconds:>8.1f} {c.mean_seconds:>9.1f} "
            f"{c.false_valid_rate:>12.3f} {c.inter_rater_agreement:>6.2f}"
        )
    lines.append("  [paired contrasts (witness - raw) over traces]")
    lines.append(f"    accuracy: {result.accuracy_gap.as_str()}")
    lines.append(f"    time:     {result.time_gap.as_str()}")
    for note in result.notes:
        lines.append(f"  note: {note}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _load_traces(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.domain == "tau_bench":
        from htir.eval.datasets import load_tau_bench
        if args.hf:
            return load_tau_bench(hf=True, limit=args.hf_limit)
        if not args.cache:
            raise SystemExit("provide --cache <jsonl> (or --hf) for tau_bench")
        return load_tau_bench([args.cache])
    if args.hf:
        from htir.eval.datasets import load_terminalbench
        return list(load_terminalbench(limit=args.hf_limit, streaming=True))
    if not args.cache:
        raise SystemExit("provide --cache <jsonl> (or --hf)")
    return list(iter_local_traces([args.cache]))


def _balanced(traces: list[dict[str, Any]], n: int, seed: int) -> list[dict[str, Any]]:
    from htir.eval.datasets import balanced_sample
    return balanced_sample(traces, n, seed=seed) if n > 0 else traces


def _load_omega(domain: str) -> DomainArtifactBundle | None:
    from htir.models.domain import load_domain_artifacts
    try:
        return load_domain_artifacts(domain)
    except Exception:
        return None


def _mode_export(args: argparse.Namespace) -> int:
    spec = get_domain_spec(args.domain)
    omega = _load_omega(args.domain)
    traces = _balanced(_load_traces(args), args.n, args.seed)
    items = build_review_items(traces, spec=spec, omega=omega, use_llm=args.use_llm,
                               max_chars=args.max_chars, model=args.model,
                               progress_every=args.progress_every)
    assignments = assign_conditions(items, n_raters=args.raters, seed=args.seed)
    manifest = export_packets(items, assignments, args.packet_dir)
    print(f"SA-12 export: {len(items)} items, {args.raters} raters, "
          f"{len(assignments)} assignments -> {args.packet_dir}")
    for key in sorted(manifest):
        print(f"  {key}: {manifest[key]}")
    return 0


def _mode_score(args: argparse.Namespace) -> int:
    if not args.items:
        raise SystemExit("--mode score requires --items <items.json>")
    items = [ReviewItem(**d) for d in json.loads(Path(args.items).read_text(encoding="utf-8"))]
    paths = sorted(glob.glob(args.responses)) if args.responses else []
    if not paths:
        raise SystemExit(f"no response CSVs matched --responses {args.responses!r}")
    responses = load_responses(paths)
    result = score(items, responses, domain_id=args.domain, simulated=False)
    print(format_table(result))
    if args.out:
        Path(args.out).write_text(result.model_dump_json(indent=2), encoding="utf-8")
        print(f"\n[sa12] wrote {args.out}", file=sys.stderr)
    return 0


def _mode_dryrun(args: argparse.Namespace) -> int:
    spec = get_domain_spec(args.domain)
    omega = _load_omega(args.domain)
    traces = _balanced(_load_traces(args), args.n, args.seed)
    result = run_sa12(traces, spec=spec, omega=omega, use_llm=args.use_llm,
                      n_raters=args.raters, seed=args.seed, max_chars=args.max_chars,
                      model=args.model, progress_every=args.progress_every)
    print(format_table(result))
    if args.out:
        Path(args.out).write_text(result.model_dump_json(indent=2), encoding="utf-8")
        print(f"\n[sa12] wrote {args.out}", file=sys.stderr)
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="SA-12: human-review efficiency of the verification witness")
    p.add_argument("--mode", choices=["export", "score", "dryrun"], default="dryrun",
                   help="export packets for raters | score returned CSVs | dryrun (simulated raters)")
    src = p.add_argument_group("data source (export / dryrun)")
    src.add_argument("--domain", type=str, default="tau_bench", help="domain spec S_d + loader")
    src.add_argument("--cache", type=str, default="", help="local JSON/JSONL corpus")
    src.add_argument("--hf", action="store_true", help="stream from the HF hub instead of --cache")
    src.add_argument("--hf-limit", type=int, default=None, help="records to pull when --hf")
    src.add_argument("--n", type=int, default=50, help="balanced sample size (0 = all)")
    src.add_argument("--seed", type=int, default=0, help="balanced-sample + assignment seed")
    src.add_argument("--raters", type=int, default=6, help="number of raters to counterbalance")
    src.add_argument("--max-chars", type=int, default=DEFAULT_MAX_CHARS,
                     help="per-step truncation in the raw-condition packet")
    src.add_argument("--use-llm", action="store_true", help="enable semantic checkers (needs key)")
    src.add_argument("--model", type=str, default="openai/gpt-4o")
    src.add_argument("--progress-every", type=int, default=0)
    exp = p.add_argument_group("export")
    exp.add_argument("--packet-dir", type=str, default="data/sa12_packets",
                     help="directory to write per-rater packets + templates + answer key")
    sc = p.add_argument_group("score")
    sc.add_argument("--items", type=str, default="", help="items.json answer key from an export run")
    sc.add_argument("--responses", type=str, default="",
                    help="glob of filled rater-response CSVs")
    p.add_argument("--out", type=str, default="", help="write SA12Result JSON here (score / dryrun)")
    args = p.parse_args(argv)

    if args.mode == "export":
        return _mode_export(args)
    if args.mode == "score":
        return _mode_score(args)
    return _mode_dryrun(args)


if __name__ == "__main__":
    raise SystemExit(main())
