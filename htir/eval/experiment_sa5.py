"""
SA-5 -- Q4b: Offline Harness Improvement Loop (avg.tex Sec. 4 question Q4;
avg.tex Sec. 3.12 "Offline Harness Improvement").

Q4b claims that **verification witnesses drive domain-spec edits that
generalize to held-out traces without changing the base model**. This
experiment formalizes and scales the offline loop built from the Step-8
primitives (:mod:`htir.agents.harness_improvement`):

    mine_recurring_failures -> score_config -> accept_edit -> apply_domain_spec_edit

over a *recorded witness corpus* mined from Terminal-Bench traces.

Witness-driven failure detection (the "repeated witness patterns identify
missing domain obligations" claim of avg.tex Sec. 3.12). Each experience trace
is compiled under the *baseline* domain spec S_d^(0); a trace is tagged
``hidden_test_failure`` exactly when its verification witness credits it
**valid** while the recorded reward says the run actually **failed** -- the
"agents frequently pass a narrow test but fail hidden tests" blind spot named
in avg.tex Sec. 3.12, and the same false-valid mode SA-1/SA-3 quantified. This
is a mechanical, auditable witness pattern, never a fabricated tag.

The loop (three experience batches). After each batch its witness records are
folded into the accumulating corpus; ``mine_recurring_failures`` proposes a
stronger validation obligation template when the tag recurs above threshold;
``score_config`` scores the current vs. edited harness config (J_hat); and
``accept_edit`` gates the edit on J_hat improvement + safety. Accepted edits
grow S_d via ``apply_domain_spec_edit``. We record, per batch, the J_hat
trajectory for the **AVG offline loop** vs. a frozen **no-offline-loop**
baseline (ablation #5, avg.tex Sec. 4.6), and the edit-acceptance rate.

The mined remediation is terminal-bound (:data:`HIDDEN_TEST_TEMPLATE`): it
fires on ``run_command`` operations and, via a registered mechanical checker
(:func:`_check_hidden_test_validation`), **passes** only when the trajectory
contains a genuine successful test run and otherwise **abstains** -- withholding
credit for the "success claimed via shell commands, never a real test" pattern
rather than fabricating a failure. Because it only ever abstains (never fails),
it can never turn a valid trajectory into a false *invalid*; it can only move an
over-credited "valid" to "uncertain".

Held-out generalization (the Q4b test). After the loop we re-compile a held-out
set of traces from **task families never seen in the experience batches** under
S_d^(0) and under the grown S_d^(final), and report:

* **false-valid rate before vs. after** -- does the obligation the loop learned
  from other tasks reduce over-crediting on unseen tasks? (generalization);
* **negative transfer** -- the new false-veto rate it introduces on genuinely
  valid held-out traces (harmful degradation; ~0 by construction since the
  remediation only abstains) and the valid-coverage it costs;
* a **spec-growth table** -- the obligation-set size on one held-out exemplar
  trace across spec versions S_d^(0), S_d^(1), ... (obligations grow as the
  domain spec accretes accepted templates).

Offline reproducibility. With no API key (``use_llm=False``, the default) the
whole loop is deterministic and issues zero LLM calls. Offline, valid and
invalid traces are *mechanically near-indistinguishable* (the AUROC~0.5 finding
of SA-1/SA-3), so the remediation withholds credit from the "no genuine test"
population as a whole; the residual false-valids are traces that *did* run a
passing test yet still failed the hidden grader -- the irreducible shortcut/
overfit core that only the semantic/integrity checker (``--use-llm``) can
refute. This gate is reported, not hidden.

CLI::

    python -m htir.eval.experiment_sa5 --cache sa5_sample.jsonl --out sa5_results.json
    python -m htir.eval.experiment_sa5 --hf --n 3000 --out sa5_results.json
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Optional

from pydantic import BaseModel, Field

from htir.agents.checker_registry import CheckerContext, register_checker
from htir.agents.harness_improvement import (
    EditTarget,
    HarnessConfig,
    ProposedEdit,
    WitnessCorpus,
    WitnessRecord,
    accept_edit,
    apply_domain_spec_edit,
    mine_recurring_failures,
    score_config,
)
from htir.agents.trace_abstraction import TraceAbstractionAgent
from htir.eval.datasets import iter_local_traces, to_canonical_steps
from htir.eval.weak_labels import (
    LABEL_INVALID,
    LABEL_VALID,
    VerifierMetrics,
    evaluate_predictions,
    extract_reward,
    label_from_reward,
)
from htir.models.domain import TERMINAL_DOMAIN_SPEC, DomainSpec, ObligationTemplate
from htir.models.htir import (
    CheckerResult,
    EscalationRule,
    EvidenceType,
    ExecutionStatus,
    HTIR,
    ObligationScope,
    ObligationStatus,
    Severity,
)
from htir.agents.witness import HIGH_SEVERITIES, STATUS_VALID

# The recurring-failure tag the offline loop mines for: a witness credited the
# trajectory valid, but the recorded reward says it failed (avg.tex Sec. 3.12,
# "pass a narrow test but fail hidden tests"). Detected from witnesses only.
HIDDEN_TEST_FAILURE_TAG = "hidden_test_failure"

# The remediation template id (kept identical to the module-neutral default in
# harness_improvement so the loop's vocabulary is stable) and the operation it
# triggers on in the terminal domain.
HIDDEN_TEST_TEMPLATE_ID = "harness-hidden-test-validation"
_TRIGGER_OPERATION = "run_command"


# ---------------------------------------------------------------------------
# The mined remediation: template + its mechanical checker contract.
# ---------------------------------------------------------------------------

HIDDEN_TEST_TEMPLATE = ObligationTemplate(
    template_id=HIDDEN_TEST_TEMPLATE_ID,
    description=(
        "Stronger validation obligation mined offline: a trajectory credited "
        "successful must be corroborated by a genuine test/validation run, not "
        "merely passing shell commands."
    ),
    claim_template="A visible success is corroborated by a genuine test/validation run.",
    scope=ObligationScope.DOMAIN,
    trigger=_TRIGGER_OPERATION,
    required_evidence=EvidenceType.EXECUTABLE,
    severity=Severity.HIGH,
    escalation=EscalationRule.ESCALATE,
    target_claim_type="execution_status",
)

# The domain-bound tag -> template map handed to the Step-8 primitives so the
# proposed/scored template is the terminal one above rather than the neutral
# default (whose 'validation' trigger does not exist in the terminal vocabulary).
TERMINAL_FAILURE_TEMPLATES: dict[str, ObligationTemplate] = {
    HIDDEN_TEST_FAILURE_TAG: HIDDEN_TEST_TEMPLATE,
}


@register_checker(template_id=HIDDEN_TEST_TEMPLATE_ID)
def _check_hidden_test_validation(ctx: CheckerContext) -> CheckerResult:
    """
    Checker contract (q_i) for the mined ``harness-hidden-test-validation``
    obligation. It consults the trajectory's validation operations (the E_val
    neighbourhood): **pass** when a genuine test run succeeded somewhere in the
    trajectory (the visible success is corroborated), otherwise **abstain** --
    there is no held-out/test validation evidence, so credit is withheld rather
    than a failure fabricated. It never returns fail, so it can only move an
    over-credited trajectory to "uncertain", never to a false "invalid".
    """
    for step in ctx.htir.steps:
        if "test" in step.role.lower() and step.execution_status == ExecutionStatus.SUCCESS:
            return CheckerResult(
                p_pass=1.0, p_fail=0.0, p_abstain=0.0, score=1.0,
                evidence_used=list(ctx.obligation.candidate_evidence_ids),
            )
    return CheckerResult(p_pass=0.0, p_fail=0.0, p_abstain=1.0, score=0.0, evidence_used=[])


# ---------------------------------------------------------------------------
# Result models
# ---------------------------------------------------------------------------

class BatchRecord(BaseModel):
    """One experience batch of the offline loop."""
    batch: int
    n_traces: int = 0
    n_tagged: int = Field(0, description="traces tagged hidden_test_failure in this batch")
    corpus_size: int = Field(0, description="accumulated corpus size after folding in this batch")
    proposals_considered: int = Field(0, description="edits mine_recurring_failures proposed this batch")
    edits_accepted: int = Field(0, description="of those, how many accept_edit gated through")
    jhat_loop: float = Field(0.0, description="J_hat of the AVG offline-loop config after this batch")
    jhat_noloop: float = Field(0.0, description="J_hat of the frozen no-offline-loop config (ablation #5)")
    active_template_ids: list[str] = Field(default_factory=list)


class SpecGrowthRow(BaseModel):
    """Obligation-set size on the held-out exemplar under one spec version."""
    version: int
    n_templates: int = Field(0, description="|S_d.B_d| obligation templates at this version")
    added_template_id: str = Field("", description="template accepted to reach this version ('' for the base)")
    exemplar_obligations: int = Field(0, description="|O(G_tau)| on the exemplar at this version")
    exemplar_high_severity: int = Field(0, description="of those, HIGH/CRITICAL severity")
    exemplar_status: str = Field("", description="aggregate verdict on the exemplar at this version")


class HeldoutEval(BaseModel):
    """Held-out (unseen task families) evaluation under a single spec version."""
    label: str = Field("", description="'baseline S_d^(0)' or 'grown S_d^(final)'")
    metrics: VerifierMetrics = Field(default_factory=VerifierMetrics)
    valid_kept_rate: float = Field(0.0, description="P(pred valid | truth valid) = valid_recall")
    triggering_traces: int = Field(0, description="held-out traces the remediation fires on (>=1 run_command)")
    genuinely_tested_valid_kept: int = Field(
        0, description="truth-valid, test-bearing traces still credited valid (targeting: credit retained)"
    )


class SA5Result(BaseModel):
    """Full SA-5 output: loop trajectory, held-out generalization, spec growth."""
    experiment: str = "SA-5: Offline Harness Improvement Loop (Q4b)"
    n_traces: int = 0
    n_experience: int = 0
    n_holdout: int = 0
    n_batches: int = 0
    use_llm: bool = False
    domain_id: str = ""
    seconds: float = 0.0

    total_proposals: int = 0
    total_accepted: int = 0
    acceptance_rate: float = 0.0
    final_template_ids: list[str] = Field(default_factory=list)

    batches: list[BatchRecord] = Field(default_factory=list)
    heldout_baseline: HeldoutEval = Field(default_factory=HeldoutEval)
    heldout_grown: HeldoutEval = Field(default_factory=HeldoutEval)

    # Generalization headline (held-out, unseen task families).
    false_valid_before: float = 0.0
    false_valid_after: float = 0.0
    false_valid_reduction: float = Field(0.0, description="relative reduction (before-after)/before")
    false_veto_before: float = Field(0.0, description="valid->invalid rate under baseline S_d^(0)")
    false_veto_after: float = Field(0.0, description="valid->invalid rate under grown S_d^(final)")
    negative_transfer: float = Field(
        0.0, description="edit-induced negative transfer = false_veto_after - false_veto_before"
    )
    valid_coverage_cost: float = Field(
        0.0, description="valid_kept lost to the edit = valid_kept_before - valid_kept_after"
    )

    spec_growth: list[SpecGrowthRow] = Field(default_factory=list)
    exemplar_task_id: str = ""

    notes: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Compilation helpers
# ---------------------------------------------------------------------------

class _CompiledTrace(BaseModel):
    """A trace's baseline compile, cached so batches/tagging reuse one pass."""
    task_id: str = ""
    reward: Optional[int] = None
    label: Optional[str] = None
    predicted_status: str = ""
    tagged: bool = False
    triggers_remediation: bool = False
    has_passing_test: bool = False
    witness_record: WitnessRecord

    model_config = {"arbitrary_types_allowed": True}


def _high_severity_unresolved_ids(htir: HTIR) -> list[int]:
    """Ids of HIGH/CRITICAL obligations that failed or abstained (O- u O-empty)."""
    return sorted(
        o.obligation_id
        for o in htir.obligations
        if o.severity in HIGH_SEVERITIES
        and o.status in (ObligationStatus.FAILED, ObligationStatus.ABSTAINED)
    )


def _compile(agent: TraceAbstractionAgent, task_id: str, raw: dict[str, Any], **kw: Any) -> HTIR:
    steps = to_canonical_steps(raw)
    return agent.compile(task_id, steps, {}, run_checks=True, **kw)


def _to_witness_record(htir: HTIR, task_id: str, reward: Optional[int], tagged: bool) -> WitnessRecord:
    outcome = "resolved" if (reward is not None and reward != 0) else "failed"
    return WitnessRecord(
        trace_id=task_id,
        witness=htir.witness,
        task_outcome=outcome,
        aggregate=htir.aggregate,
        high_severity_obligation_ids=_high_severity_unresolved_ids(htir),
        failure_tags=[HIDDEN_TEST_FAILURE_TAG] if tagged else [],
    )


def _compile_experience_trace(
    agent: TraceAbstractionAgent, task_id: str, raw: dict[str, Any], **kw: Any
) -> _CompiledTrace:
    """Compile one experience trace under the baseline spec and tag it."""
    reward = extract_reward(raw)
    label = label_from_reward(reward)
    htir = _compile(agent, task_id, raw, **kw)
    status = htir.aggregate.predicted_status if htir.aggregate else ""
    # Witness-driven detector: credited valid but the run actually failed.
    tagged = status == STATUS_VALID and label == LABEL_INVALID
    triggers = any(s.role == _TRIGGER_OPERATION for s in htir.steps)
    has_test = any(
        "test" in s.role.lower() and s.execution_status == ExecutionStatus.SUCCESS for s in htir.steps
    )
    return _CompiledTrace(
        task_id=task_id,
        reward=reward,
        label=label,
        predicted_status=status,
        tagged=tagged,
        triggers_remediation=triggers,
        has_passing_test=has_test,
        witness_record=_to_witness_record(htir, task_id, reward, tagged),
    )


# ---------------------------------------------------------------------------
# Data split (by task family, so held-out tasks are genuinely unseen)
# ---------------------------------------------------------------------------

def _family_is_holdout(task_name: str, *, seed: int, holdout_fraction: float) -> bool:
    """Deterministically assign a whole task family to the held-out split."""
    h = hashlib.sha1(f"{seed}:{task_name}".encode()).hexdigest()
    return (int(h[:8], 16) / 0xFFFFFFFF) < holdout_fraction


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_sa5(
    raw_traces: Iterable[dict[str, Any]],
    *,
    spec: DomainSpec | None = None,
    n_batches: int = 3,
    holdout_fraction: float = 0.3,
    seed: int = 0,
    epsilon: float = 0.01,
    min_recurrence: float = 0.02,
    use_llm: bool = False,
    model: str = "openai/gpt-4o",
    progress_every: int = 500,
    log: Any = sys.stderr,
) -> SA5Result:
    """
    Execute SA-5 over ``raw_traces`` (turn-schema dicts with a ``reward``).

    Splits traces by task family into experience / held-out, mines the
    accumulating experience-witness corpus over ``n_batches`` batches, gates
    and applies domain-spec edits, then evaluates generalization on the unseen
    held-out families. Offline (no LLM) by default.
    """
    base_spec = spec or TERMINAL_DOMAIN_SPEC
    base_agent = TraceAbstractionAgent(model=model, domain_spec=base_spec)
    compile_kwargs: dict[str, Any] = {"use_semantic_analysis": use_llm}

    experience: list[_CompiledTrace] = []
    holdout_raw: list[dict[str, Any]] = []

    t0 = time.time()
    seen = 0
    for i, raw in enumerate(raw_traces):
        if not isinstance(raw, dict):
            continue
        task_name = str(raw.get("task_name", "")) or f"trace-{i}"
        label = label_from_reward(extract_reward(raw))
        if label is None:  # unlabeled traces cannot ground the loop or its eval
            continue
        seen += 1
        if _family_is_holdout(task_name, seed=seed, holdout_fraction=holdout_fraction):
            holdout_raw.append(raw)
            continue
        try:
            experience.append(_compile_experience_trace(base_agent, task_name, raw, **compile_kwargs))
        except Exception as exc:  # a malformed trace must not sink the run
            if log is not None:
                print(f"[sa5] skip experience trace {i} ({task_name}): {exc!r}", file=log)
        if progress_every and log is not None and seen % progress_every == 0:
            print(f"[sa5] scanned {seen} traces (exp={len(experience)}, holdout={len(holdout_raw)})...", file=log)

    # ------------------------------------------------------------------
    # The offline loop over experience batches.
    # ------------------------------------------------------------------
    batches = _split_batches(experience, n_batches)
    corpus = WitnessCorpus(records=[])
    current_spec = base_spec
    loop_config = HarnessConfig()          # AVG offline loop: edited as accepted
    noloop_config = HarnessConfig()        # ablation #5: frozen, never edited
    spec_versions: list[tuple[str, DomainSpec]] = [("", base_spec)]  # (added_template_id, spec)

    total_proposals = 0
    total_accepted = 0
    batch_records: list[BatchRecord] = []

    for b, batch in enumerate(batches, start=1):
        n_tagged = sum(1 for c in batch if c.tagged)
        corpus.records.extend(c.witness_record for c in batch)

        proposals = mine_recurring_failures(
            corpus, min_fraction=min_recurrence, known_templates=TERMINAL_FAILURE_TEMPLATES
        )
        # Only novel proposals (a template not already enforced) are gated; the
        # miner re-proposes an already-active template every batch, which is not
        # a new decision.
        novel = [
            e for e in proposals
            if e.obligation_template is not None
            and e.obligation_template.template_id not in loop_config.active_obligation_template_ids
        ]
        accepted_here = 0
        for edit in novel:
            tmpl = edit.obligation_template
            total_proposals += 1
            baseline_j = score_config(corpus, loop_config, known_templates=TERMINAL_FAILURE_TEMPLATES)
            candidate_ids = frozenset({*loop_config.active_obligation_template_ids, tmpl.template_id})
            candidate_config = loop_config.model_copy(update={"active_obligation_template_ids": candidate_ids})
            edited_j = score_config(corpus, candidate_config, known_templates=TERMINAL_FAILURE_TEMPLATES)
            if accept_edit(baseline_j, edited_j, epsilon=epsilon, safe=True):
                loop_config = candidate_config
                current_spec = apply_domain_spec_edit(current_spec, edit)
                spec_versions.append((tmpl.template_id, current_spec))
                total_accepted += 1
                accepted_here += 1

        batch_records.append(BatchRecord(
            batch=b,
            n_traces=len(batch),
            n_tagged=n_tagged,
            corpus_size=len(corpus.records),
            proposals_considered=len(novel),
            edits_accepted=accepted_here,
            jhat_loop=round(score_config(corpus, loop_config, known_templates=TERMINAL_FAILURE_TEMPLATES), 4),
            jhat_noloop=round(score_config(corpus, noloop_config, known_templates=TERMINAL_FAILURE_TEMPLATES), 4),
            active_template_ids=sorted(loop_config.active_obligation_template_ids),
        ))

    # ------------------------------------------------------------------
    # Held-out generalization on unseen task families.
    # ------------------------------------------------------------------
    heldout_baseline = _eval_holdout(holdout_raw, base_spec, "baseline S_d^(0)", model, compile_kwargs, log)
    heldout_grown = _eval_holdout(holdout_raw, current_spec, "grown S_d^(final)", model, compile_kwargs, log)

    fv_before = heldout_baseline.metrics.false_valid_rate
    fv_after = heldout_grown.metrics.false_valid_rate
    reduction = ((fv_before - fv_after) / fv_before) if fv_before else 0.0

    # ------------------------------------------------------------------
    # Spec-growth table on one held-out exemplar.
    # ------------------------------------------------------------------
    exemplar = _pick_exemplar(holdout_raw, base_agent, compile_kwargs)
    spec_growth = _spec_growth(exemplar, spec_versions, model, compile_kwargs) if exemplar else []
    exemplar_id = str(exemplar.get("task_name", "")) if exemplar else ""

    result = SA5Result(
        n_traces=seen,
        n_experience=len(experience),
        n_holdout=len(holdout_raw),
        n_batches=len(batches),
        use_llm=use_llm,
        domain_id=base_spec.domain_id,
        seconds=round(time.time() - t0, 2),
        total_proposals=total_proposals,
        total_accepted=total_accepted,
        acceptance_rate=round(total_accepted / total_proposals, 4) if total_proposals else 0.0,
        final_template_ids=sorted(loop_config.active_obligation_template_ids),
        batches=batch_records,
        heldout_baseline=heldout_baseline,
        heldout_grown=heldout_grown,
        false_valid_before=fv_before,
        false_valid_after=fv_after,
        false_valid_reduction=round(reduction, 4),
        false_veto_before=heldout_baseline.metrics.false_invalid_rate,
        false_veto_after=heldout_grown.metrics.false_invalid_rate,
        negative_transfer=round(
            heldout_grown.metrics.false_invalid_rate - heldout_baseline.metrics.false_invalid_rate, 4
        ),
        valid_coverage_cost=round(
            heldout_baseline.valid_kept_rate - heldout_grown.valid_kept_rate, 4
        ),
        spec_growth=spec_growth,
        exemplar_task_id=exemplar_id,
        notes=_notes(use_llm),
    )
    return result


def _split_batches(experience: list[_CompiledTrace], n_batches: int) -> list[list[_CompiledTrace]]:
    """Split the experience traces into ``n_batches`` contiguous batches."""
    if n_batches <= 0 or not experience:
        return [experience] if experience else []
    size = len(experience) // n_batches
    batches: list[list[_CompiledTrace]] = []
    for b in range(n_batches):
        lo = b * size
        hi = len(experience) if b == n_batches - 1 else (b + 1) * size
        batches.append(experience[lo:hi])
    return batches


def _eval_holdout(
    holdout_raw: list[dict[str, Any]],
    spec: DomainSpec,
    label: str,
    model: str,
    compile_kwargs: dict[str, Any],
    log: Any,
) -> HeldoutEval:
    """Compile every held-out trace under ``spec`` and score it vs. weak labels."""
    agent = TraceAbstractionAgent(model=model, domain_spec=spec)
    preds: list[str] = []
    labels: list[Optional[str]] = []
    triggering = 0
    tested_valid_kept = 0
    for i, raw in enumerate(holdout_raw):
        task_name = str(raw.get("task_name", "")) or f"holdout-{i}"
        try:
            htir = _compile(agent, task_name, raw, **compile_kwargs)
        except Exception as exc:
            if log is not None:
                print(f"[sa5] skip holdout trace {i} ({task_name}): {exc!r}", file=log)
            continue
        status = htir.aggregate.predicted_status if htir.aggregate else "uncertain"
        lab = label_from_reward(extract_reward(raw))
        preds.append(status)
        labels.append(lab)
        if any(s.role == _TRIGGER_OPERATION for s in htir.steps):
            triggering += 1
        has_test = any(
            "test" in s.role.lower() and s.execution_status == ExecutionStatus.SUCCESS for s in htir.steps
        )
        if lab == LABEL_VALID and has_test and status == STATUS_VALID:
            tested_valid_kept += 1

    metrics = evaluate_predictions(preds, labels)
    return HeldoutEval(
        label=label,
        metrics=metrics,
        valid_kept_rate=metrics.valid_recall,
        triggering_traces=triggering,
        genuinely_tested_valid_kept=tested_valid_kept,
    )


def _pick_exemplar(
    holdout_raw: list[dict[str, Any]], base_agent: TraceAbstractionAgent, compile_kwargs: dict[str, Any]
) -> Optional[dict[str, Any]]:
    """
    A held-out exemplar for the spec-growth table: prefer a compact false-valid
    blind-spot trace (credited valid under S_d^(0), truly failed, triggers the
    remediation, ran no genuine test) so growing S_d visibly withdraws its
    credit. Falls back to the first held-out trace that triggers the remediation.
    """
    best: Optional[tuple[tuple[int, int], dict[str, Any]]] = None
    fallback: Optional[dict[str, Any]] = None
    for i, raw in enumerate(holdout_raw):
        task_name = str(raw.get("task_name", "")) or f"holdout-{i}"
        try:
            htir = _compile(base_agent, task_name, raw, **compile_kwargs)
        except Exception:
            continue
        triggers = any(s.role == _TRIGGER_OPERATION for s in htir.steps)
        if triggers and fallback is None:
            fallback = raw
        status = htir.aggregate.predicted_status if htir.aggregate else ""
        lab = label_from_reward(extract_reward(raw))
        has_test = any(
            "test" in s.role.lower() and s.execution_status == ExecutionStatus.SUCCESS for s in htir.steps
        )
        if status == STATUS_VALID and lab == LABEL_INVALID and triggers and not has_test:
            key = (len(htir.steps), i)  # prefer compact + deterministic
            if best is None or key < best[0]:
                best = (key, raw)
    return best[1] if best is not None else fallback


def _spec_growth(
    exemplar: dict[str, Any],
    spec_versions: list[tuple[str, DomainSpec]],
    model: str,
    compile_kwargs: dict[str, Any],
) -> list[SpecGrowthRow]:
    """Compile the exemplar under each accepted spec version; count obligations."""
    task_name = str(exemplar.get("task_name", "")) or "exemplar"
    rows: list[SpecGrowthRow] = []
    for version, (added_id, spec) in enumerate(spec_versions):
        agent = TraceAbstractionAgent(model=model, domain_spec=spec)
        htir = _compile(agent, task_name, exemplar, **compile_kwargs)
        n_high = sum(1 for o in htir.obligations if o.severity in HIGH_SEVERITIES)
        rows.append(SpecGrowthRow(
            version=version,
            n_templates=len(spec.obligation_templates),
            added_template_id=added_id,
            exemplar_obligations=len(htir.obligations),
            exemplar_high_severity=n_high,
            exemplar_status=htir.aggregate.predicted_status if htir.aggregate else "",
        ))
    return rows


def _notes(use_llm: bool) -> list[str]:
    notes: list[str] = []
    if not use_llm:
        notes.append(
            "Offline run (no API key): the loop is fully deterministic and issues zero LLM "
            "calls. Valid and invalid traces are mechanically near-indistinguishable offline "
            "(the AUROC~0.5 finding of SA-1/SA-3), so the mined obligation withholds credit "
            "from the whole 'no genuine test run' population; residual false-valids are traces "
            "that DID run a passing test yet still failed the hidden grader -- the irreducible "
            "shortcut/overfit core the semantic/integrity checker (--use-llm) targets."
        )
    notes.append(
        "Failure detection is witness-driven (avg.tex Sec. 3.12): a trace is tagged "
        "'hidden_test_failure' iff its verification witness credited it valid while the "
        "recorded reward says it failed. The remediation is mined + gated + applied through the "
        "Step-8 primitives (mine_recurring_failures / score_config / accept_edit / "
        "apply_domain_spec_edit); the base agent is never changed."
    )
    notes.append(
        "The mined obligation only ever abstains (via its registered checker), so it can never "
        "turn a valid trajectory into a false 'invalid' -- it only moves an over-credited 'valid' "
        "to 'uncertain'. Negative transfer is therefore the (near-zero) new false-veto rate plus "
        "the valid coverage it costs; traces that ran a genuine test keep their credit (targeting)."
    )
    notes.append(
        "J_hat trajectory contrasts the AVG offline loop vs. a frozen no-offline-loop config "
        "(ablation #5, avg.tex Sec. 4.6). This whole experiment is the 'no online loop' mode "
        "(ablation #6): the verifier is used purely as an outer-loop scoring function, the "
        "complement of SA-4's online intervention."
    )
    return notes


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def format_table(result: SA5Result) -> str:
    """A compact fixed-width results summary for the terminal / logs."""
    lines: list[str] = []
    lines.append(
        f"SA-5: Offline Harness Improvement Loop (Q4b)  |  n={result.n_traces} "
        f"(experience {result.n_experience}, held-out {result.n_holdout} over unseen task families)  |  "
        f"domain={result.domain_id}  use_llm={result.use_llm}  {result.seconds}s"
    )

    lines.append("  [offline loop -- J_hat learning curve + edit acceptance]")
    header = (
        f"    {'batch':>5} {'traces':>6} {'tagged':>6} {'corpus':>6} "
        f"{'proposed':>8} {'accepted':>8} {'J_loop':>9} {'J_noloop':>9}  templates"
    )
    lines.append(header)
    for br in result.batches:
        lines.append(
            f"    {br.batch:>5} {br.n_traces:>6} {br.n_tagged:>6} {br.corpus_size:>6} "
            f"{br.proposals_considered:>8} {br.edits_accepted:>8} {br.jhat_loop:>9.3f} "
            f"{br.jhat_noloop:>9.3f}  {','.join(br.active_template_ids) or '-'}"
        )
    lines.append(
        f"    -> {result.total_accepted}/{result.total_proposals} edits accepted "
        f"(rate {result.acceptance_rate:.2f}); final S_d templates: "
        f"{','.join(result.final_template_ids) or '-'}"
    )

    lines.append("  [held-out generalization -- unseen task families, before vs. after]")
    lines.append(
        f"    {'':<22} {'false_valid':>11} {'valid_kept':>10} {'veto_rate':>9} "
        f"{'abstain':>8} {'res_acc':>8}"
    )
    for ev in (result.heldout_baseline, result.heldout_grown):
        m = ev.metrics
        lines.append(
            f"    {ev.label:<22} {m.false_valid_rate:>11.3f} {ev.valid_kept_rate:>10.3f} "
            f"{m.false_invalid_rate:>9.3f} {m.abstention_rate:>8.3f} {m.resolved_accuracy:>8.3f}"
        )
    lines.append(
        f"    -> false-valid {result.false_valid_before:.3f} -> {result.false_valid_after:.3f} "
        f"({result.false_valid_reduction * 100:.0f}% reduction) on unseen tasks; "
        f"edit-induced negative transfer (dfalse-veto) {result.negative_transfer:+.3f}; "
        f"valid-coverage cost {result.valid_coverage_cost:+.3f}; "
        f"genuinely-tested valids kept: {result.heldout_grown.genuinely_tested_valid_kept}"
    )

    lines.append(f"  [spec growth on held-out exemplar '{result.exemplar_task_id}']")
    lines.append(
        f"    {'version':>7} {'|B_d|':>6} {'obligations':>11} {'high_sev':>8} "
        f"{'verdict':>10}  added_template"
    )
    for row in result.spec_growth:
        lines.append(
            f"    {row.version:>7} {row.n_templates:>6} {row.exemplar_obligations:>11} "
            f"{row.exemplar_high_severity:>8} {row.exemplar_status:>10}  {row.added_template_id or '(base)'}"
        )

    for note in result.notes:
        lines.append(f"  note: {note}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

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
    p = argparse.ArgumentParser(description="SA-5: Offline Harness Improvement Loop (Q4b)")
    src = p.add_argument_group("data source")
    src.add_argument("--cache", type=str, default="", help="local JSON/JSONL sample (turn schema)")
    src.add_argument("--hf", action="store_true", help="pull from the HF terminalbench dataset")
    src.add_argument("--hf-limit", type=int, default=9000, help="records to stream when --hf")
    src.add_argument("--n", type=int, default=3000, help="target balanced sample size")
    src.add_argument("--seed", type=int, default=0)
    p.add_argument("--batches", type=int, default=3, help="number of experience batches")
    p.add_argument("--holdout-fraction", type=float, default=0.3, help="fraction of task families held out")
    p.add_argument(
        "--min-recurrence", type=float, default=0.02,
        help="min fraction of the corpus a failure tag must recur in to be mined",
    )
    p.add_argument("--use-llm", action="store_true", help="enable the semantic/integrity checker")
    p.add_argument("--model", type=str, default="openai/gpt-4o")
    p.add_argument("--out", type=str, default="", help="write SA5Result JSON here")
    args = p.parse_args(argv)

    traces = _load_traces(args)
    result = run_sa5(
        traces,
        n_batches=args.batches,
        holdout_fraction=args.holdout_fraction,
        seed=args.seed,
        min_recurrence=args.min_recurrence,
        use_llm=args.use_llm,
        model=args.model,
    )
    print(format_table(result))
    if args.out:
        Path(args.out).write_text(result.model_dump_json(indent=2), encoding="utf-8")
        print(f"\n[sa5] wrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
