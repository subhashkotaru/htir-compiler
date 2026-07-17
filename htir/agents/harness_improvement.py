"""
Offline harness improvement (AVG Step 8, avg.tex Sec. 3.11 "Offline Harness
Improvement").

Verification witnesses accumulated across many traces (Step 6 output) also
provide offline feedback for improving the harness

    h = (p, s, m, r)

(prompts, skills, memory policies, runtime rules). Given a proposed edit
``Delta h``, it is accepted only if it improves verifier-conditioned
performance on held-out traces and is safe:

    Accept(Delta h) = I[J_hat(h+Delta h) > J_hat(h) + epsilon AND Safe(Delta h) = 1]

This module is an **offline analysis over a recorded corpus of witnesses**.
It never re-runs a live agent and never auto-applies an edit -- it only
*proposes* edits (to the domain spec ``S_d`` or, out of this repo's scope, to
the harness itself) and scores/gates them. Applying a proposal is a separate,
explicit call (``apply_domain_spec_edit``); nothing here mutates a
``DomainSpec`` in place.

Two edit targets exist per avg.tex Sec. 3.11's examples ("agents frequently
pass a narrow test but fail hidden tests" / "produce CSV without checking
headers"): (i) the domain spec ``S_d`` (add obligation templates -- cheapest,
self-contained, and what this module implements end-to-end) and (ii) the
harness ``h`` itself (prompts/skills/memory/runtime rules -- out-of-repo,
represented here only as an opaque, unapplied ``HarnessConfig``/
``ProposedEdit`` so the interface exists without pretending to act on it).

Entry points: ``mine_recurring_failures``, ``score_config``, ``accept_edit``,
``apply_domain_spec_edit``.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field

from htir.models.domain import DomainSpec, ObligationTemplate
from htir.models.htir import (
    AggregateResult,
    EscalationRule,
    EvidenceType,
    ObligationScope,
    Severity,
    VerificationWitness,
)

# ---------------------------------------------------------------------------
# h = (p, s, m, r) and the witness corpus Step 8 mines over
# ---------------------------------------------------------------------------


class HarnessConfig(BaseModel):
    """
    h = (p, s, m, r): prompts, skills, memory policies, and runtime rules
    (avg.tex Sec. 3.11). The harness itself is out of this repo's scope (it
    lives wherever the agent harness runs), so this is an opaque, descriptive
    snapshot -- nothing here executes prompts/skills/rules.

    ``active_obligation_template_ids`` additionally records which obligation
    templates the *current* h + S_d combination is understood to enforce, so
    ``score_config`` can tell whether a recurring failure category is already
    covered (caught) or still a blind spot (missed).
    """
    prompts: dict[str, str] = Field(default_factory=dict)
    skills: dict[str, str] = Field(default_factory=dict)
    memory_policies: dict[str, str] = Field(default_factory=dict)
    runtime_rules: dict[str, str] = Field(default_factory=dict)
    active_obligation_template_ids: frozenset[str] = Field(default_factory=frozenset)


class WitnessRecord(BaseModel):
    """
    One (trace_id, VerificationWitness, task_outcome) entry (avg.tex Sec.
    3.11), extended with the small amount of extra digest information
    ``score_config`` needs that a bare witness doesn't carry on its own
    (severity/coverage/cost/policy signal, and mined recurring-failure tags).
    """
    trace_id: str
    witness: VerificationWitness
    task_outcome: str = Field("", description="External ground-truth evaluation, e.g. 'resolved' / 'failed'")
    aggregate: Optional[AggregateResult] = None
    high_severity_obligation_ids: list[int] = Field(
        default_factory=list, description="Obligation ids in O-/O-empty whose severity is HIGH/CRITICAL"
    )
    policy_violation_ids: list[int] = Field(default_factory=list, description="Obligation ids that are policy-constraint failures")
    cost: float = Field(0.0, description="Verification cost incurred for this trace (checker/human-review time, etc.)")
    failure_tags: list[str] = Field(
        default_factory=list,
        description="Recurring-failure categories observed for this trace, e.g. 'hidden_test_failure'",
    )


class WitnessCorpus(BaseModel):
    """A recorded collection of ``WitnessRecord``s -- Step 6 output across many traces."""
    records: list[WitnessRecord] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Proposed edits
# ---------------------------------------------------------------------------


class EditTarget(str, Enum):
    DOMAIN_SPEC = "domain_spec"
    HARNESS = "harness"


class ProposedEdit(BaseModel):
    """
    A proposed ``Delta h`` (or ``Delta S_d``). ``obligation_template`` is set
    for ``DOMAIN_SPEC``-targeted edits (applied via ``apply_domain_spec_edit``);
    ``harness_delta`` is an opaque free-form description for ``HARNESS``-
    targeted edits, which this module never applies (out of repo scope).
    """
    target: EditTarget
    description: str = ""
    obligation_template: Optional[ObligationTemplate] = None
    harness_delta: Optional[dict] = None


# Recognised recurring-failure tags -> the domain-spec template that guards
# against them, per avg.tex Sec. 3.11's two named examples. Unrecognised tags
# propose nothing -- mining never invents an obligation out of thin air.
_KNOWN_FAILURE_TEMPLATES: dict[str, ObligationTemplate] = {
    "hidden_test_failure": ObligationTemplate(
        template_id="harness-hidden-test-validation",
        claim_template="A visible test pass is corroborated by a hidden/held-out validation run.",
        scope=ObligationScope.DOMAIN,
        trigger="validation",
        required_evidence=EvidenceType.EXECUTABLE,
        severity=Severity.HIGH,
        escalation=EscalationRule.ESCALATE,
        target_claim_type="execution_status",
    ),
    "csv_missing_header": ObligationTemplate(
        template_id="harness-csv-schema-check",
        claim_template="Produced CSV artifacts declare and match an expected header schema.",
        scope=ObligationScope.DOMAIN,
        trigger="artifact_edit",
        required_evidence=EvidenceType.SCHEMA,
        severity=Severity.MEDIUM,
        escalation=EscalationRule.REQUEST_EVIDENCE,
        target_claim_type="artifact_provenance",
    ),
}

# A failure tag must recur in at least this fraction of the corpus to be
# treated as a real pattern rather than one-off noise.
MIN_RECURRENCE_FRACTION = 0.5


def mine_recurring_failures(
    corpus: WitnessCorpus,
    *,
    min_fraction: float = MIN_RECURRENCE_FRACTION,
    known_templates: dict[str, ObligationTemplate] | None = None,
) -> list[ProposedEdit]:
    """
    Scan ``corpus.records[*].failure_tags`` for tags recurring in at least
    ``min_fraction`` of the traces, and propose a stronger domain-spec
    obligation template for each recognised one (avg.tex Sec. 3.11: "agents
    frequently pass a narrow test but fail hidden tests" -> a stronger
    validation obligation; "produce CSV without checking headers" -> a
    CSV-schema obligation). A tag with no known template mapping, or that
    doesn't recur often enough, proposes nothing -- this stays a mechanical,
    auditable pattern-match, not a generative process.

    ``known_templates`` maps a recognised failure tag to the obligation
    template that guards against it. It defaults to the domain-neutral
    :data:`_KNOWN_FAILURE_TEMPLATES`; a concrete domain passes a specialised
    map so the remediation template it proposes actually *binds* to that
    domain's operation vocabulary (e.g. a terminal task triggers the
    hidden-test obligation on its ``run_test`` operation rather than the
    neutral ``validation`` trigger). Unrecognised tags still propose nothing.
    """
    templates = known_templates if known_templates is not None else _KNOWN_FAILURE_TEMPLATES
    n = len(corpus.records)
    if n == 0:
        return []

    counts: dict[str, int] = {}
    for record in corpus.records:
        for tag in set(record.failure_tags):
            counts[tag] = counts.get(tag, 0) + 1

    proposals: list[ProposedEdit] = []
    for tag in sorted(counts):
        count = counts[tag]
        if count / n < min_fraction:
            continue
        template = templates.get(tag)
        if template is None:
            continue
        proposals.append(
            ProposedEdit(
                target=EditTarget.DOMAIN_SPEC,
                description=(
                    f"{count}/{n} traces show recurring '{tag}' failures; add obligation "
                    f"template '{template.template_id}' to S_d."
                ),
                obligation_template=template,
            )
        )
    return proposals


def apply_domain_spec_edit(spec: DomainSpec, edit: ProposedEdit) -> DomainSpec:
    """
    Apply a ``DOMAIN_SPEC``-targeted ``ProposedEdit`` by returning a *new*
    ``DomainSpec`` with the proposed obligation template appended (``spec``
    is never mutated in place). Idempotent: applying the same edit twice
    returns a spec with the template appearing only once.
    """
    if edit.target != EditTarget.DOMAIN_SPEC or edit.obligation_template is None:
        raise ValueError("apply_domain_spec_edit requires a DOMAIN_SPEC-targeted edit with an obligation_template")

    if any(t.template_id == edit.obligation_template.template_id for t in spec.obligation_templates):
        return spec.model_copy(deep=True)

    return spec.model_copy(update={
        "obligation_templates": [*spec.obligation_templates, edit.obligation_template],
    })


# ---------------------------------------------------------------------------
# Score J_hat and the acceptance gate
# ---------------------------------------------------------------------------

# Explicit, auditable weights for J_hat (avg.tex Sec. 3.11: "task success,
# failed obligations, unresolved high-severity obligations, evidence
# coverage, cost, and policy violations").
W_TASK_SUCCESS = 1.0
W_FAILED_OBLIGATIONS = 0.2
W_UNRESOLVED_HIGH_SEVERITY = 0.3
W_EVIDENCE_COVERAGE = 0.5
W_COST = 0.1
W_POLICY_VIOLATIONS = 0.4

# Reward/penalty for a recurring-failure tag that is/isn't covered by an
# active obligation template in `config` -- this is what makes J_hat rise
# once a stronger obligation template starts catching a previously-blind
# failure category.
W_FAILURE_CAUGHT = 0.5
W_FAILURE_MISSED = 1.0

DEFAULT_ACCEPT_EPSILON = 0.01


def score_config(
    corpus: WitnessCorpus,
    config: HarnessConfig,
    *,
    known_templates: dict[str, ObligationTemplate] | None = None,
) -> float:
    """
    J_hat(h) (avg.tex Sec. 3.11): task success, failed obligations,
    unresolved high-severity obligations, evidence coverage, cost, and
    policy violations, averaged over ``corpus``, plus a catch/miss term for
    recurring failure tags against ``config.active_obligation_template_ids``
    -- a tag whose mapped template is active is rewarded (caught); one whose
    mapped template is not active is penalized (missed). Returns 0.0 for an
    empty corpus.

    ``known_templates`` (the tag -> guarding-template map) must match the one
    passed to :func:`mine_recurring_failures` so the catch/miss term keys on
    the same template ids the loop actually applies; it defaults to the
    domain-neutral :data:`_KNOWN_FAILURE_TEMPLATES`.
    """
    templates = known_templates if known_templates is not None else _KNOWN_FAILURE_TEMPLATES
    records = corpus.records
    n = len(records)
    if n == 0:
        return 0.0

    task_success_rate = sum(1 for r in records if r.task_outcome == "resolved") / n
    avg_failed = sum(len(r.witness.failed_obligation_ids) for r in records) / n
    avg_unresolved_high = sum(len(r.high_severity_obligation_ids) for r in records) / n
    avg_coverage = sum((r.aggregate.evidence_coverage if r.aggregate else 1.0) for r in records) / n
    avg_cost = sum(r.cost for r in records) / n
    avg_policy_violations = sum(len(r.policy_violation_ids) for r in records) / n

    caught = 0
    missed = 0
    for record in records:
        for tag in record.failure_tags:
            template_id = templates.get(tag)
            template_id = template_id.template_id if template_id is not None else None
            if template_id is not None and template_id in config.active_obligation_template_ids:
                caught += 1
            else:
                missed += 1

    return (
        W_TASK_SUCCESS * task_success_rate
        - W_FAILED_OBLIGATIONS * avg_failed
        - W_UNRESOLVED_HIGH_SEVERITY * avg_unresolved_high
        + W_EVIDENCE_COVERAGE * avg_coverage
        - W_COST * avg_cost
        - W_POLICY_VIOLATIONS * avg_policy_violations
        + W_FAILURE_CAUGHT * caught
        - W_FAILURE_MISSED * missed
    )


def accept_edit(
    baseline_j: float, edited_j: float, epsilon: float = DEFAULT_ACCEPT_EPSILON, safe: bool = True
) -> bool:
    """
    Accept(Delta h) = I[J_hat(h+Delta h) > J_hat(h) + epsilon AND Safe(Delta h) = 1]
    (avg.tex Sec. 3.11). ``safe`` is the caller-supplied ``Safe(Delta h)``
    judgement (out of scope to compute here -- e.g. a held-out regression
    suite or a human sign-off); this function only implements the gate.
    """
    return bool(safe) and (edited_j > baseline_j + epsilon)
