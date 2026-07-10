"""
Smoke test for the deterministic (non-LLM) parts of the HTIR/AVG pipeline:
artifact extraction (E_prov), the Step-3 analysis layer (well-formedness +
analysis modules), and claim/obligation construction.

Builds synthetic HTIRs by hand (no OpenRouter/LLM calls) so this runs without
``OPENROUTER_API_KEY``, then exercises:

  * ``TraceAbstractionAgent._extract_artifacts`` -- artifact nodes +
    ``ArtifactProvenanceLink`` (E_prov).
  * ``htir.agents.analysis.enrich`` -- well-formedness issues and the
    provenance / dependency / validation / state-transition / policy-linking
    / integrity analysis modules (avg.tex Sec. 3.4-3.5).
  * ``htir.agents.obligations.build_claims_and_obligations`` -- claim /
    evidence / obligation nodes, the support edges, and obligations seeded
    from unresolved well-formedness issues.
  * a regression test for the ``role.value`` crash in
    ``TraceAbstractionAgent._infer_control_flow_batch`` (``role`` is a plain
    ``str``, not an enum).
"""

from __future__ import annotations

import re

from htir.agents.analysis import enrich
from htir.agents.checking import check_obligations
from htir.agents.harness_improvement import (
    EditTarget,
    HarnessConfig,
    WitnessCorpus,
    WitnessRecord,
    accept_edit,
    apply_domain_spec_edit,
    mine_recurring_failures,
    score_config,
)
from htir.agents.intervention import active_obligations, run_intervention_loop, select_intervention
from htir.agents.obligations import _template_triggers, build_claims_and_obligations
from htir.agents.trace_abstraction import TraceAbstractionAgent
from htir.agents.witness import aggregate, build_witness
from htir.models.domain import (
    DEFAULT_DOMAIN_SPEC,
    ArtifactKind,
    Constraint,
    DomainArtifact,
    DomainArtifactBundle,
    DomainSpec,
    ObligationTemplate,
)
from htir.models.htir import (
    HTIR,
    ArtifactEffect,
    ArtifactStateEvidence,
    CheckerResult,
    CheckerType,
    ClaimStatus,
    EscalationRule,
    EvidenceType,
    ExecutionStatus,
    InterventionAction,
    Obligation,
    ObligationStatus,
    ProvenanceRelation,
    Severity,
    SupportPolarity,
    TraceStep,
    VerificationWitness,
)


def _build_synthetic_htir() -> HTIR:
    steps = [
        TraceStep(
            step_id=1,
            request_message="run pytest",
            response_message="1 failed, 0 passed",
            role="validation",
            execution_status=ExecutionStatus.FAILURE,
            artifact_state_effects=[
                ArtifactStateEvidence(
                    effect_category=ArtifactEffect.ARTIFACT_CHANGE,
                    affected_resource="test_report",
                    observed_change="test suite run, 1 failure",
                )
            ],
        ),
        TraceStep(
            step_id=2,
            request_message="edit parser.py to fix failing test",
            response_message="applied patch",
            role="artifact_editing",
            execution_status=ExecutionStatus.SUCCESS,
            artifact_state_effects=[
                ArtifactStateEvidence(
                    effect_category=ArtifactEffect.READ_ONLY,
                    affected_resource="test_report",
                    observed_change="inspected failure",
                ),
                ArtifactStateEvidence(
                    effect_category=ArtifactEffect.ARTIFACT_CHANGE,
                    affected_resource="parser.py",
                    observed_change="patched parsing logic",
                ),
            ],
        ),
        TraceStep(
            step_id=3,
            request_message="run pytest again",
            response_message="2 passed",
            role="validation",
            execution_status=ExecutionStatus.SUCCESS,
            artifact_state_effects=[
                ArtifactStateEvidence(
                    effect_category=ArtifactEffect.READ_ONLY,
                    affected_resource="parser.py",
                    observed_change="revalidated",
                ),
                ArtifactStateEvidence(
                    effect_category=ArtifactEffect.ARTIFACT_CHANGE,
                    affected_resource="test_report",
                    observed_change="test suite run, all passed",
                ),
            ],
        ),
        TraceStep(
            step_id=4,
            request_message="submit final answer",
            response_message="Fixed the parser bug; tests now pass.",
            role="final_submission",
            execution_status=ExecutionStatus.SUCCESS,
        ),
    ]
    return HTIR(
        task_id="synthetic-smoke-test",
        domain_id=DEFAULT_DOMAIN_SPEC.domain_id,
        steps=steps,
    )


def test_artifact_extraction_populates_provenance_links():
    htir = _build_synthetic_htir()
    agent = TraceAbstractionAgent(domain_spec=DEFAULT_DOMAIN_SPEC)

    agent._extract_artifacts(htir)

    assert len(htir.artifacts) == 2  # test_report, parser.py
    assert len(htir.provenance_links) == 5  # 3 effects on test_report + 2 on parser.py

    relations = {(lk.step_id, lk.artifact_id): lk.relation for lk in htir.provenance_links}
    test_report = next(a for a in htir.artifacts if a.identifier == "test_report")
    parser_py = next(a for a in htir.artifacts if a.identifier == "parser.py")

    assert relations[(1, test_report.artifact_id)] == ProvenanceRelation.CREATED
    assert relations[(2, test_report.artifact_id)] == ProvenanceRelation.READ
    assert relations[(3, test_report.artifact_id)] == ProvenanceRelation.MODIFIED
    assert relations[(2, parser_py.artifact_id)] == ProvenanceRelation.CREATED
    assert relations[(3, parser_py.artifact_id)] == ProvenanceRelation.READ


def test_claims_and_obligations_populate_dependency_links():
    htir = _build_synthetic_htir()
    agent = TraceAbstractionAgent(domain_spec=DEFAULT_DOMAIN_SPEC)
    agent._extract_artifacts(htir)

    # E_val / E_cons / E_causal now live in the Step-3 analysis layer, which
    # must run before obligation generation consumes it.
    enrich(htir, DEFAULT_DOMAIN_SPEC)
    build_claims_and_obligations(htir, DEFAULT_DOMAIN_SPEC)

    assert len(htir.claims) > 0
    assert len(htir.evidence) > 0
    assert len(htir.obligations) > 0
    assert len(htir.support_links) > 0
    assert len(htir.validation_links) > 0
    assert len(htir.constraint_links) > 0

    # E_causal (dependency): step 2 consumes test_report produced by step 1;
    # step 3 consumes parser.py produced by step 2.
    dep_pairs = {(lk.source_step_id, lk.target_step_id) for lk in htir.dependency_links}
    assert (2, 1) in dep_pairs
    assert (3, 2) in dep_pairs


def test_infer_control_flow_batch_role_is_plain_string(monkeypatch):
    """
    Regression test for the ``role.value`` crash: ``TraceStep.role`` was
    refactored from an enum to a plain ``str``, but
    ``_infer_control_flow_batch`` did ``s.role.value`` in its f-string,
    raising ``AttributeError: 'str' object has no attribute 'value'``.

    Stubs ``chat_json`` so the f-string is exercised (and the fix verified)
    without a network call.
    """
    steps = [
        TraceStep(
            step_id=1,
            request_message="run tool",
            response_message="tool output",
            role="tool_invocation",
            execution_status=ExecutionStatus.SUCCESS,
        ),
        TraceStep(
            step_id=2,
            request_message="run validation",
            response_message="1 failed",
            role="validation",
            execution_status=ExecutionStatus.FAILURE,
        ),
    ]

    def _fake_chat_json(messages, schema, model=None, max_tokens=None, **kwargs):
        return schema()  # empty _ControlFlowLinkList

    monkeypatch.setattr("htir.agents.trace_abstraction.chat_json", _fake_chat_json)

    agent = TraceAbstractionAgent(domain_spec=DEFAULT_DOMAIN_SPEC)
    links = agent._infer_control_flow_batch(steps, harness_context="")

    assert links == []


def test_state_transition_pattern_matched_for_fail_edit_pass_trace():
    """State-transition analysis recognises failing-validation -> edit -> passing-validation."""
    htir = _build_synthetic_htir()
    agent = TraceAbstractionAgent(domain_spec=DEFAULT_DOMAIN_SPEC)
    agent._extract_artifacts(htir)

    enrich(htir, DEFAULT_DOMAIN_SPEC)

    matched = [p for p in htir.state_transitions if p.matched]
    assert len(matched) == 1
    assert matched[0].pattern_name == "failing_validation_edit_revalidation"
    assert matched[0].step_ids == [1, 2, 3]


def test_policy_sensitive_step_without_policy_artifact_emits_unresolved_obligation():
    """
    Well-formedness rule (e) + policy-linking analysis: a policy-sensitive
    step (governed by a domain constraint) with no linked policy artifact
    must emit an unresolved obligation, not a task failure.
    """
    spec = DomainSpec(
        domain_id="policy-test",
        constraints=[
            Constraint(
                constraint_id="policy-required",
                description="Final answers must cite an applicable policy.",
                severity=Severity.HIGH,
                applies_to_operations=["final_submission"],
            )
        ],
    )
    steps = [
        TraceStep(
            step_id=1,
            request_message="submit final answer",
            response_message="Done.",
            role="final_submission",
            execution_status=ExecutionStatus.SUCCESS,
        ),
    ]
    htir = HTIR(task_id="policy-smoke-test", domain_id=spec.domain_id, steps=steps)

    enrich(htir, spec)

    issues = [i for i in htir.wellformedness if i.rule_id == "policy_action_unlinked"]
    assert len(issues) == 1
    assert issues[0].offending_node_ids == [1]

    build_claims_and_obligations(htir, spec)
    seeded = [o for o in htir.obligations if o.template_id == "wellformedness:policy_action_unlinked"]
    assert len(seeded) == 1
    assert seeded[0].checker == CheckerType.ABSTENTION


def test_integrity_flags_direct_test_modification():
    """Integrity analysis flags a step that mutates a test artifact directly (tamper)."""
    steps = [
        TraceStep(
            step_id=1,
            request_message="edit test_report.py",
            response_message="patched test assertions directly",
            role="artifact_editing",
            execution_status=ExecutionStatus.SUCCESS,
            artifact_state_effects=[
                ArtifactStateEvidence(
                    effect_category=ArtifactEffect.ARTIFACT_CHANGE,
                    affected_resource="test_report.py",
                    observed_change="modified assertions to force a pass",
                )
            ],
        ),
    ]
    htir = HTIR(task_id="integrity-smoke-test", domain_id=DEFAULT_DOMAIN_SPEC.domain_id, steps=steps)
    agent = TraceAbstractionAgent(domain_spec=DEFAULT_DOMAIN_SPEC)
    agent._extract_artifacts(htir)

    enrich(htir, DEFAULT_DOMAIN_SPEC)

    issues = [i for i in htir.wellformedness if i.rule_id == "integrity_test_modified"]
    assert len(issues) == 1
    assert issues[0].offending_node_ids[0] == 1


def _build_compiled_synthetic_htir():
    """``_build_synthetic_htir`` run through artifact extraction + enrich + obligations."""
    htir = _build_synthetic_htir()
    agent = TraceAbstractionAgent(domain_spec=DEFAULT_DOMAIN_SPEC)
    agent._extract_artifacts(htir)
    enrich(htir, DEFAULT_DOMAIN_SPEC)
    build_claims_and_obligations(htir, DEFAULT_DOMAIN_SPEC)
    return htir


def test_failing_step_evidence_refutes_its_execution_status_claim():
    """B1: a FAILURE execution-status claim is REFUTED, not SUPPORTED, by its evidence."""
    htir = _build_compiled_synthetic_htir()

    step1_claim = next(
        c for c in htir.claims if c.claim_type == "execution_status" and c.source_step_id == 1
    )
    links = [lk for lk in htir.support_links if lk.claim_id == step1_claim.claim_id]
    assert links, "expected at least one support link for the failing step's execution claim"
    assert all(lk.polarity == SupportPolarity.REFUTES for lk in links)


def test_obligation_candidate_evidence_matches_required_evidence_type():
    """B2: candidate_evidence_ids (E_i) only contain evidence of the required type r_i."""
    htir = _build_compiled_synthetic_htir()
    evidence_type_by_id = {e.evidence_id: e.evidence_type for e in htir.evidence}

    for ob in htir.obligations:
        for ev_id in ob.candidate_evidence_ids:
            assert evidence_type_by_id[ev_id] == ob.required_evidence


def test_validation_obligation_anchors_on_execution_status_not_artifact_provenance():
    """
    B3: step 1 and step 3 are both validations that also produce/mutate an
    artifact (test_report), so ``claim_ids[-1]`` (the old, positional logic)
    would incorrectly land on the artifact_provenance claim. The
    ``uni-validation-backed`` template declares ``target_claim_type:
    execution_status``, so its obligation must anchor on that step's
    execution_status claim instead.
    """
    htir = _build_compiled_synthetic_htir()

    validation_obligations = [o for o in htir.obligations if o.template_id == "uni-validation-backed"]
    assert validation_obligations

    claims_by_id = {c.claim_id: c for c in htir.claims}
    for ob in validation_obligations:
        claim = claims_by_id[ob.claim_id]
        assert claim.claim_type == "execution_status"


def test_explain_failure_obligation_has_routed_checker():
    """B4: trig-explain-failure (r_i = log) must be routed to a real checker, not UNASSIGNED."""
    htir = _build_compiled_synthetic_htir()

    explain_failure = [o for o in htir.obligations if o.template_id == "trig-explain-failure"]
    assert explain_failure
    assert all(o.checker != CheckerType.UNASSIGNED for o in explain_failure)


def test_build_claims_and_obligations_is_idempotent():
    """B5: calling build_claims_and_obligations twice must not duplicate nodes/edges."""
    htir = _build_synthetic_htir()
    agent = TraceAbstractionAgent(domain_spec=DEFAULT_DOMAIN_SPEC)
    agent._extract_artifacts(htir)
    enrich(htir, DEFAULT_DOMAIN_SPEC)

    build_claims_and_obligations(htir, DEFAULT_DOMAIN_SPEC)
    first_counts = (
        len(htir.claims), len(htir.evidence), len(htir.obligations), len(htir.support_links),
    )

    build_claims_and_obligations(htir, DEFAULT_DOMAIN_SPEC)
    second_counts = (
        len(htir.claims), len(htir.evidence), len(htir.obligations), len(htir.support_links),
    )

    assert first_counts == second_counts


def test_template_trigger_is_exact_not_substring():
    """B6: a 'decision' trigger must not fire on the 'orchestration_decision' operation type."""
    template = ObligationTemplate(
        template_id="t-decision",
        claim_template="unused",
        trigger="decision",
        required_evidence=EvidenceType.NONE,
    )
    step = TraceStep(
        step_id=1,
        request_message="plan next action",
        response_message="decided to run tests",
        role="orchestration_decision",
        execution_status=ExecutionStatus.SUCCESS,
    )
    assert _template_triggers(template, step) is False


def test_integrity_finding_escalates_to_veto():
    """B7: a HIGH-severity integrity finding seeds an obligation escalating to VETO."""
    steps = [
        TraceStep(
            step_id=1,
            request_message="edit test_report.py",
            response_message="patched test assertions directly",
            role="artifact_editing",
            execution_status=ExecutionStatus.SUCCESS,
            artifact_state_effects=[
                ArtifactStateEvidence(
                    effect_category=ArtifactEffect.ARTIFACT_CHANGE,
                    affected_resource="test_report.py",
                    observed_change="modified assertions to force a pass",
                )
            ],
        ),
    ]
    htir = HTIR(task_id="integrity-escalation-test", domain_id=DEFAULT_DOMAIN_SPEC.domain_id, steps=steps)
    agent = TraceAbstractionAgent(domain_spec=DEFAULT_DOMAIN_SPEC)
    agent._extract_artifacts(htir)
    enrich(htir, DEFAULT_DOMAIN_SPEC)
    build_claims_and_obligations(htir, DEFAULT_DOMAIN_SPEC)

    from htir.models.htir import EscalationRule

    seeded = [o for o in htir.obligations if o.template_id == "wellformedness:integrity_test_modified"]
    assert len(seeded) == 1
    assert seeded[0].escalation == EscalationRule.VETO


def test_obligations_deduped_by_claim_and_template():
    """B8: two templates with the same trigger firing on the same claim don't duplicate obligations."""
    spec = DomainSpec(
        domain_id="dedup-test",
        operation_types=DEFAULT_DOMAIN_SPEC.operation_types,
        obligation_templates=[
            ObligationTemplate(
                template_id="dup-a",
                claim_template="Final answer is supported.",
                trigger="final_submission",
                required_evidence=EvidenceType.NONE,
                target_claim_type="final_answer_support",
            ),
            ObligationTemplate(
                template_id="dup-a",
                claim_template="Final answer is supported (duplicate template id).",
                trigger="final_submission",
                required_evidence=EvidenceType.NONE,
                target_claim_type="final_answer_support",
            ),
        ],
    )
    steps = [
        TraceStep(
            step_id=1,
            request_message="submit final answer",
            response_message="Done.",
            role="final_submission",
            execution_status=ExecutionStatus.SUCCESS,
        ),
    ]
    htir = HTIR(task_id="dedup-smoke-test", domain_id=spec.domain_id, steps=steps)
    enrich(htir, spec)
    build_claims_and_obligations(htir, spec)

    matching = [o for o in htir.obligations if o.template_id == "dup-a"]
    assert len(matching) == 1


# ---------------------------------------------------------------------------
# Step 5 -- checker execution (htir.agents.checking)
# ---------------------------------------------------------------------------

def test_post_edit_validation_obligation_passes_mechanically():
    """
    Step 5: the trig-post-edit-validation obligation anchored on the edit
    step's (step 2, parser.py) artifact_provenance claim is PASSED
    mechanically, because the later validation step 3 revalidated it with
    ExecutionStatus.SUCCESS.
    """
    htir = _build_compiled_synthetic_htir()
    check_obligations(htir, DEFAULT_DOMAIN_SPEC)

    claims_by_id = {c.claim_id: c for c in htir.claims}
    post_edit = [
        o for o in htir.obligations
        if o.template_id == "trig-post-edit-validation"
        and claims_by_id[o.claim_id].source_step_id == 2
    ]
    assert len(post_edit) == 1
    ob = post_edit[0]
    assert ob.status == ObligationStatus.PASSED
    assert ob.result is not None
    assert ob.result.p_pass == 1.0


def test_failing_step_execution_claim_is_refuted_after_checking():
    """Step 5: the failing step-1 execution-status claim ends up REFUTED, never SUPPORTED."""
    htir = _build_compiled_synthetic_htir()
    check_obligations(htir, DEFAULT_DOMAIN_SPEC)

    step1_claim = next(
        c for c in htir.claims if c.claim_type == "execution_status" and c.source_step_id == 1
    )
    assert step1_claim.status == ClaimStatus.REFUTED


def test_wellformedness_seeded_obligation_abstains():
    """Step 5: a well-formedness-seeded (ABSTENTION-routed) obligation abstains with p_abstain==1.0."""
    htir = _build_compiled_synthetic_htir()
    check_obligations(htir, DEFAULT_DOMAIN_SPEC)

    seeded = [o for o in htir.obligations if o.checker == CheckerType.ABSTENTION]
    assert seeded, "expected at least one ABSTENTION-routed obligation in the synthetic trace"
    for ob in seeded:
        assert ob.status == ObligationStatus.ABSTAINED
        assert ob.result.p_abstain == 1.0


def test_semantic_obligations_abstain_without_llm_call(monkeypatch):
    """
    Step 5: with use_semantic=False (the default), every SEMANTIC obligation
    abstains and chat_json is never invoked.
    """
    htir = _build_compiled_synthetic_htir()

    def _fail_if_called(*args, **kwargs):
        raise AssertionError("chat_json must not be called when use_semantic=False")

    monkeypatch.setattr("htir.agents.checking.chat_json", _fail_if_called)

    check_obligations(htir, DEFAULT_DOMAIN_SPEC, use_semantic=False)

    semantic_obligations = [o for o in htir.obligations if o.checker == CheckerType.SEMANTIC]
    assert semantic_obligations
    for ob in semantic_obligations:
        assert ob.status == ObligationStatus.ABSTAINED
        assert ob.result.p_abstain == 1.0


def test_check_obligations_is_idempotent():
    """Step 5: calling check_obligations twice yields identical results/statuses."""
    htir = _build_compiled_synthetic_htir()
    check_obligations(htir, DEFAULT_DOMAIN_SPEC)

    first = [(o.status, o.result.model_dump()) for o in htir.obligations]
    first_claim_statuses = [c.status for c in htir.claims]

    check_obligations(htir, DEFAULT_DOMAIN_SPEC)

    second = [(o.status, o.result.model_dump()) for o in htir.obligations]
    second_claim_statuses = [c.status for c in htir.claims]

    assert first == second
    assert first_claim_statuses == second_claim_statuses


# ---------------------------------------------------------------------------
# Step 6 -- aggregation z_tau + verification witness W_tau (htir.agents.witness)
# ---------------------------------------------------------------------------

def _obligation(
    obligation_id: int, status: ObligationStatus, severity: Severity,
    *, p_pass: float = 0.0, p_fail: float = 0.0, p_abstain: float = 0.0,
    evidence_used: list[int] | None = None, candidate_evidence_ids: list[int] | None = None,
) -> Obligation:
    return Obligation(
        obligation_id=obligation_id,
        claim_id=obligation_id,
        severity=severity,
        status=status,
        candidate_evidence_ids=candidate_evidence_ids or [],
        result=CheckerResult(p_pass=p_pass, p_fail=p_fail, p_abstain=p_abstain, evidence_used=evidence_used or []),
    )


def test_aggregate_all_pass_trace_is_valid():
    """Step 6: a trace with only passed obligations aggregates to 'valid' with an empty O-."""
    htir = HTIR(task_id="agg-all-pass", obligations=[
        _obligation(1, ObligationStatus.PASSED, Severity.LOW, p_pass=1.0),
        _obligation(2, ObligationStatus.PASSED, Severity.HIGH, p_pass=1.0),
    ])

    z = aggregate(htir)
    assert z.predicted_status == "valid"

    w = build_witness(htir)
    assert w.failed_obligation_ids == []
    assert w.passed_obligation_ids == [1, 2]


def test_aggregate_forced_failed_critical_obligation_is_invalid():
    """Step 6: a failed CRITICAL obligation vetoes to 'invalid' even with many passes."""
    htir = HTIR(task_id="agg-veto", obligations=[
        _obligation(1, ObligationStatus.PASSED, Severity.LOW, p_pass=1.0),
        _obligation(2, ObligationStatus.PASSED, Severity.LOW, p_pass=1.0),
        _obligation(3, ObligationStatus.PASSED, Severity.LOW, p_pass=1.0),
        _obligation(4, ObligationStatus.FAILED, Severity.CRITICAL, p_fail=1.0, evidence_used=[10]),
    ])

    z = aggregate(htir)
    assert z.predicted_status == "invalid"

    w = build_witness(htir)
    assert w.failed_obligation_ids == [4]
    assert 10 in w.witness_evidence_ids


def test_aggregate_abstain_heavy_high_severity_is_uncertain():
    """Step 6: no failures but many abstained high-severity obligations => 'uncertain', not 'valid'."""
    htir = HTIR(task_id="agg-uncertain", obligations=[
        _obligation(1, ObligationStatus.ABSTAINED, Severity.HIGH, p_abstain=1.0),
        _obligation(2, ObligationStatus.ABSTAINED, Severity.HIGH, p_abstain=1.0),
        _obligation(3, ObligationStatus.PASSED, Severity.LOW, p_pass=1.0),
    ])

    z = aggregate(htir)
    assert z.predicted_status == "uncertain"


def test_aggregate_all_abstained_no_pass_is_uncertain_not_valid():
    """Regression (over-crediting bug): a trace that binds only low/medium
    obligations, all of which ABSTAIN, has zero supporting evidence and must
    aggregate to 'uncertain', never 'valid'. Reproduces scratch_results.json
    real_traces[0] (56 uni-tool-schema obligations, all abstained, coverage
    0.0) which previously fell through to 'valid'."""
    htir = HTIR(task_id="agg-all-abstain", obligations=[
        _obligation(i, ObligationStatus.ABSTAINED, Severity.LOW, p_abstain=1.0)
        for i in range(1, 6)
    ])

    z = aggregate(htir)
    assert z.predicted_status == "uncertain"
    assert z.uncertainty == 1.0


def test_aggregate_zero_obligations_is_uncertain_not_valid():
    """Regression: a trace that binds *no* obligations verified nothing, so it
    cannot be credited as 'valid'. Reproduces scratch_results.json
    real_traces[2] (0 obligations) which previously aggregated to 'valid'."""
    htir = HTIR(task_id="agg-empty", obligations=[])

    z = aggregate(htir)
    assert z.predicted_status == "uncertain"


def test_aggregate_broad_low_severity_abstention_with_one_pass_is_uncertain():
    """A single incidental pass must not mask that most of the trajectory went
    unverified: with abstention above BROAD_ABSTAIN_FRACTION_THRESHOLD across
    all severities, the status is 'uncertain' even though no high-severity
    obligation exists to trigger the high-severity rule."""
    htir = HTIR(task_id="agg-broad-abstain", obligations=[
        _obligation(1, ObligationStatus.PASSED, Severity.LOW, p_pass=1.0),
        _obligation(2, ObligationStatus.ABSTAINED, Severity.LOW, p_abstain=1.0),
        _obligation(3, ObligationStatus.ABSTAINED, Severity.MEDIUM, p_abstain=1.0),
        _obligation(4, ObligationStatus.ABSTAINED, Severity.LOW, p_abstain=1.0),
    ])

    z = aggregate(htir)
    assert z.predicted_status == "uncertain"


def test_build_witness_is_deterministic_and_idempotent():
    """Step 6: build_witness makes no LLM call and re-running it yields an identical witness."""
    htir = _build_compiled_synthetic_htir()
    check_obligations(htir, DEFAULT_DOMAIN_SPEC)
    aggregate(htir)

    first = build_witness(htir).model_dump()
    second = build_witness(htir).model_dump()

    assert first == second


def test_compile_run_checks_flag_populates_aggregate_and_witness(monkeypatch):
    """Step 6 wiring: TraceAbstractionAgent.compile(run_checks=True) fills HTIR.aggregate/witness."""
    def _fake_chat_json(messages, schema, model=None, max_tokens=None, **kwargs):
        return schema(role="final_submission", execution_status=ExecutionStatus.SUCCESS, artifact_effects=[])

    monkeypatch.setattr("htir.agents.trace_abstraction.chat_json", _fake_chat_json)

    agent = TraceAbstractionAgent(domain_spec=DEFAULT_DOMAIN_SPEC)
    htir = agent.compile(
        task_id="compile-run-checks",
        raw_steps=[{"request": "submit", "response": "final answer"}],
        harness_snippets={},
        run_checks=True,
    )

    assert htir.aggregate is not None
    assert htir.witness is not None
    assert htir.aggregate.predicted_status in ("valid", "invalid", "uncertain")
    assert all(o.status != ObligationStatus.PENDING for o in htir.obligations)


# ---------------------------------------------------------------------------
# Verifier arms / baselines (htir.agents.baselines, avg.tex Sec. 4.3)
# ---------------------------------------------------------------------------

def _compiled_terminal_htir():
    """Compile the synthetic terminal trace offline through obligation gen."""
    from htir.adapters import load_trace
    from htir.models.domain import get_domain_spec
    from tests.test_adapters import TERMINAL_TRACE

    spec = get_domain_spec("terminal_swe")
    steps = load_trace(TERMINAL_TRACE)
    htir = TraceAbstractionAgent(domain_spec=spec).compile(
        task_id="arm", raw_steps=steps, harness_snippets={},
    )
    return htir, spec


def test_run_all_arms_returns_every_arm_offline():
    from htir.agents.baselines import VerifierArm, run_all_arms

    htir, spec = _compiled_terminal_htir()
    results = run_all_arms(htir, spec, use_llm=False)
    assert set(results) == set(VerifierArm)
    for agg in results.values():
        assert agg.predicted_status in ("valid", "invalid", "uncertain")


def test_exec_free_abstains_more_than_exec_only():
    """Disabling mechanical evidence (exec-free) can only raise abstention /
    uncertainty relative to exec-only over the same graph."""
    from htir.agents.baselines import VerifierArm, run_arm

    htir, spec = _compiled_terminal_htir()
    exec_only = run_arm(htir, spec, VerifierArm.EXEC_ONLY, use_llm=False)
    exec_free = run_arm(htir, spec, VerifierArm.EXEC_FREE, use_llm=False)
    assert exec_free.uncertainty >= exec_only.uncertainty


def test_arms_do_not_mutate_caller_graph():
    """run_arm works on a copy by default: obligation statuses on the caller's
    graph are untouched so arms can be compared independently."""
    from htir.agents.baselines import VerifierArm, run_arm

    htir, spec = _compiled_terminal_htir()
    before = [o.status for o in htir.obligations]
    run_arm(htir, spec, VerifierArm.EXEC_ONLY, use_llm=False)
    assert [o.status for o in htir.obligations] == before  # all still PENDING


def test_monolithic_judge_is_endpoint_oriented():
    """The monolithic baseline trusts the last observable validation outcome:
    a trace ending in a passing test reads 'valid'; ending in a failing test
    reads 'invalid' -- with no obligation graph or abstention."""
    from htir.agents.baselines import monolithic_judge

    passing = HTIR(task_id="mono-pass", steps=[
        TraceStep(step_id=1, request_message="edit", response_message="done",
                  role="edit_file", execution_status=ExecutionStatus.SUCCESS),
        TraceStep(step_id=2, request_message="pytest", response_message="ok",
                  role="run_test", execution_status=ExecutionStatus.SUCCESS),
    ])
    failing = HTIR(task_id="mono-fail", steps=[
        TraceStep(step_id=1, request_message="edit", response_message="done",
                  role="edit_file", execution_status=ExecutionStatus.SUCCESS),
        TraceStep(step_id=2, request_message="pytest", response_message="1 failed",
                  role="run_test", execution_status=ExecutionStatus.FAILURE),
    ])
    assert monolithic_judge(passing).predicted_status == "valid"
    assert monolithic_judge(failing).predicted_status == "invalid"
    assert monolithic_judge(passing).evidence_coverage == 0.0  # no localization


# ---------------------------------------------------------------------------
# Work item A -- Omega_d weak domain artifacts (htir.models.domain)
# ---------------------------------------------------------------------------

def _policy_governed_spec() -> DomainSpec:
    return DomainSpec(
        domain_id="policy-omega-test",
        operation_types=DEFAULT_DOMAIN_SPEC.operation_types,
        constraints=[
            Constraint(
                constraint_id="policy-required",
                description="Final answers must cite an applicable policy.",
                severity=Severity.HIGH,
                applies_to_operations=["final_submission"],
            )
        ],
    )


def test_omega_bundle_produces_policy_compliance_obligation_with_candidate_evidence():
    """
    Work item A: with a loaded Omega_d policy artifact, a policy-sensitive
    step gets a real omega-policy-compliance obligation whose candidate
    evidence points at that artifact's content, still PENDING (Step 5's job
    to discharge).
    """
    spec = _policy_governed_spec()
    steps = [
        TraceStep(
            step_id=1, request_message="submit", response_message="Done, fabricated the results.",
            role="final_submission", execution_status=ExecutionStatus.SUCCESS,
        ),
    ]
    htir = HTIR(task_id="omega-policy-test", domain_id=spec.domain_id, steps=steps)
    bundle = DomainArtifactBundle(
        domain_id=spec.domain_id,
        artifacts=[
            DomainArtifact(
                artifact_kind=ArtifactKind.POLICY,
                identifier="no-fabrication-policy",
                content="Agents must not fabricate results.",
            )
        ],
    )

    enrich(htir, spec, domain_artifacts=bundle)
    build_claims_and_obligations(htir, spec, domain_artifacts=bundle)

    omega_obs = [o for o in htir.obligations if o.template_id == "omega-policy-compliance"]
    assert len(omega_obs) == 1
    ob = omega_obs[0]
    assert ob.status == ObligationStatus.PENDING
    assert ob.checker == CheckerType.SEMANTIC
    assert ob.candidate_evidence_ids, "expected E_i to point at the Omega_d policy artifact"

    evidence_by_id = {e.evidence_id for e in htir.evidence}
    assert set(ob.candidate_evidence_ids) <= evidence_by_id
    pointed_evidence = next(e for e in htir.evidence if e.evidence_id in ob.candidate_evidence_ids)
    assert "no-fabrication-policy" in pointed_evidence.description
    assert "fabricate" in pointed_evidence.content

    dep_reasons = [lk.reason for lk in htir.dependency_links if lk.source_step_id == 1]
    assert any("no-fabrication-policy" in r for r in dep_reasons)


def test_omega_bundle_absent_leaves_obligations_unchanged():
    """Work item A: with no bundle loaded, behavior is identical to before Omega_d existed."""
    spec = _policy_governed_spec()
    steps = [
        TraceStep(
            step_id=1, request_message="submit", response_message="Done.",
            role="final_submission", execution_status=ExecutionStatus.SUCCESS,
        ),
    ]
    htir = HTIR(task_id="omega-absent-test", domain_id=spec.domain_id, steps=steps)

    enrich(htir, spec)
    build_claims_and_obligations(htir, spec)

    assert not [o for o in htir.obligations if o.template_id == "omega-policy-compliance"]
    assert not any("Omega_d" in lk.reason for lk in htir.dependency_links)


def test_omega_schema_evidence_resolves_schema_obligation():
    """
    Work item A: an Omega_d schema artifact matching an artifact type's
    schema_hint gives the swe-generated-schema obligation real E_i, and Step
    5's schema checker then passes mechanically by consuming it.
    """
    from htir.models.domain import TERMINAL_DOMAIN_SPEC

    steps = [
        TraceStep(
            step_id=1,
            request_message="edit implementation",
            response_message="wrote test report",
            role="edit_file",
            execution_status=ExecutionStatus.SUCCESS,
            artifact_state_effects=[
                ArtifactStateEvidence(
                    effect_category=ArtifactEffect.ARTIFACT_CHANGE,
                    affected_resource="test_report",
                    observed_change="generated a test report",
                )
            ],
        ),
    ]
    htir = HTIR(task_id="omega-schema-test", domain_id=TERMINAL_DOMAIN_SPEC.domain_id, steps=steps)
    agent = TraceAbstractionAgent(domain_spec=TERMINAL_DOMAIN_SPEC)
    agent._extract_artifacts(htir)
    enrich(htir, TERMINAL_DOMAIN_SPEC)

    bundle = DomainArtifactBundle(
        domain_id=TERMINAL_DOMAIN_SPEC.domain_id,
        artifacts=[
            DomainArtifact(
                artifact_kind=ArtifactKind.SCHEMA,
                identifier="test_report",
                content="{pass_count:int, fail_count:int, exit_code:int}",
            )
        ],
    )
    build_claims_and_obligations(htir, TERMINAL_DOMAIN_SPEC, domain_artifacts=bundle)

    schema_obs = [o for o in htir.obligations if o.template_id == "swe-generated-schema"]
    assert len(schema_obs) == 1
    assert schema_obs[0].candidate_evidence_ids
    schema_evidence = next(e for e in htir.evidence if e.evidence_id in schema_obs[0].candidate_evidence_ids)
    assert schema_evidence.evidence_type == EvidenceType.SCHEMA

    check_obligations(htir, TERMINAL_DOMAIN_SPEC, domain_artifacts=bundle)
    assert schema_obs[0].status == ObligationStatus.PASSED


# ---------------------------------------------------------------------------
# Step 7 -- online intervention iota_t (htir.agents.intervention)
# ---------------------------------------------------------------------------

# Raw (request/response) form of the fail -> edit -> pass synthetic trace
# (mirrors _build_synthetic_htir), for replay through TraceAbstractionAgent.compile_prefix.
_INTERVENTION_RAW_STEPS = [
    {"request": "run pytest", "response": "1 failed, 0 passed"},
    {"request": "edit parser.py to fix failing test", "response": "applied patch"},
    {"request": "run pytest again", "response": "2 passed"},
    {"request": "submit final answer", "response": "Fixed the parser bug; tests now pass."},
]

_INTERVENTION_ANNOTATIONS: dict[int, dict] = {
    1: dict(
        role="validation", execution_status=ExecutionStatus.FAILURE,
        artifact_effects=[{
            "effect_category": "artifact_change", "affected_resource": "test_report",
            "observed_change": "test suite run, 1 failure",
        }],
    ),
    2: dict(
        role="artifact_editing", execution_status=ExecutionStatus.SUCCESS,
        artifact_effects=[
            {"effect_category": "read_only", "affected_resource": "test_report", "observed_change": "inspected failure"},
            {"effect_category": "artifact_change", "affected_resource": "parser.py", "observed_change": "patched parsing logic"},
        ],
    ),
    3: dict(
        role="validation", execution_status=ExecutionStatus.SUCCESS,
        artifact_effects=[
            {"effect_category": "read_only", "affected_resource": "parser.py", "observed_change": "revalidated"},
            {"effect_category": "artifact_change", "affected_resource": "test_report", "observed_change": "test suite run, all passed"},
        ],
    ),
    4: dict(role="final_submission", execution_status=ExecutionStatus.SUCCESS, artifact_effects=[]),
}


def _fake_annotation_chat_json(messages, schema, model=None, max_tokens=None, **kwargs):
    """Stubs TraceAbstractionAgent._annotate_step deterministically; other schemas get empty defaults."""
    user_content = next((m["content"] for m in messages if m.get("role") == "user"), "")
    match = re.match(r"Step (\d+):", user_content)
    if match and schema.__name__ == "_StepAnnotation":
        return schema(**_INTERVENTION_ANNOTATIONS[int(match.group(1))])
    return schema()


def test_active_obligation_gets_repair_intervention_before_revalidation(monkeypatch):
    """
    Step 7: replaying the fail->edit->pass trace up to step 2 (right after the
    edit, before revalidation) surfaces the post-edit-validation obligation
    (anchored on the edit's artifact_provenance claim) as active, and its
    deterministic intervention follows its escalation rule (repair, per
    trig-post-edit-validation in domains/default.yaml).
    """
    monkeypatch.setattr("htir.agents.trace_abstraction.chat_json", _fake_annotation_chat_json)
    agent = TraceAbstractionAgent(domain_spec=DEFAULT_DOMAIN_SPEC)

    htir_prefix = agent.compile_prefix("intervention-test", _INTERVENTION_RAW_STEPS, {}, 2)
    claims_by_id = {c.claim_id: c for c in htir_prefix.claims}

    actives = active_obligations(htir_prefix)
    edit_obligation = next(
        o for o in actives
        if o.template_id == "trig-post-edit-validation" and claims_by_id[o.claim_id].source_step_id == 2
    )
    assert edit_obligation.status == ObligationStatus.ABSTAINED

    action = select_intervention(edit_obligation, htir_prefix)
    assert action == InterventionAction.REPAIR


def test_active_obligation_resolves_once_revalidation_passes(monkeypatch):
    """Step 7: once the revalidation step is included, that same obligation is no longer active."""
    monkeypatch.setattr("htir.agents.trace_abstraction.chat_json", _fake_annotation_chat_json)
    agent = TraceAbstractionAgent(domain_spec=DEFAULT_DOMAIN_SPEC)

    htir_prefix = agent.compile_prefix("intervention-test", _INTERVENTION_RAW_STEPS, {}, 3)
    claims_by_id = {c.claim_id: c for c in htir_prefix.claims}

    remaining = [
        o for o in active_obligations(htir_prefix)
        if o.template_id == "trig-post-edit-validation" and claims_by_id[o.claim_id].source_step_id == 2
    ]
    assert remaining == []


def test_select_intervention_defaults_to_obligation_escalation():
    """Step 7: with the default benefit/cost/risk functions, iota_t reduces to alpha_i (escalation)."""
    ob = _obligation(1, ObligationStatus.FAILED, Severity.HIGH, p_fail=1.0)
    ob.escalation = EscalationRule.VETO
    htir = HTIR(task_id="intervention-unit-test")

    assert select_intervention(ob, htir) == InterventionAction.VETO


def test_run_intervention_loop_is_deterministic_and_idempotent(monkeypatch):
    """Step 7: replaying the same trace twice yields an identical intervention log."""
    monkeypatch.setattr("htir.agents.trace_abstraction.chat_json", _fake_annotation_chat_json)
    agent = TraceAbstractionAgent(domain_spec=DEFAULT_DOMAIN_SPEC)

    htir1 = run_intervention_loop(agent, "loop-test", _INTERVENTION_RAW_STEPS, {})
    log1 = [entry.model_dump() for entry in htir1.intervention_log]

    htir2 = run_intervention_loop(agent, "loop-test", _INTERVENTION_RAW_STEPS, {})
    log2 = [entry.model_dump() for entry in htir2.intervention_log]

    assert log1, "expected at least one recorded intervention"
    assert log1 == log2


# ---------------------------------------------------------------------------
# Step 8 -- offline harness improvement (htir.agents.harness_improvement)
# ---------------------------------------------------------------------------

def _hidden_test_failure_corpus(n: int = 4) -> WitnessCorpus:
    return WitnessCorpus(records=[
        WitnessRecord(
            trace_id=f"trace-{i}",
            witness=VerificationWitness(passed_obligation_ids=[i], review_recommendation="ok"),
            task_outcome="resolved",
            failure_tags=["hidden_test_failure"],
        )
        for i in range(n)
    ])


def test_mine_recurring_failures_proposes_stronger_validation_and_raises_score():
    """
    Step 8: a corpus where every trace shows a recurring hidden-test-failure
    tag proposes a stronger validation obligation template; applying it to
    S_d and re-scoring (as an active obligation) raises J_hat, and
    accept_edit gates it through.
    """
    corpus = _hidden_test_failure_corpus()

    proposals = mine_recurring_failures(corpus)
    assert len(proposals) == 1
    edit = proposals[0]
    assert edit.target == EditTarget.DOMAIN_SPEC
    assert edit.obligation_template is not None
    assert edit.obligation_template.template_id == "harness-hidden-test-validation"

    baseline_config = HarnessConfig()
    baseline_j = score_config(corpus, baseline_config)

    edited_spec = apply_domain_spec_edit(DEFAULT_DOMAIN_SPEC, edit)
    assert edit.obligation_template.template_id in {t.template_id for t in edited_spec.obligation_templates}
    # Original spec must not be mutated in place.
    assert edit.obligation_template.template_id not in {t.template_id for t in DEFAULT_DOMAIN_SPEC.obligation_templates}

    edited_config = HarnessConfig(active_obligation_template_ids=frozenset({edit.obligation_template.template_id}))
    edited_j = score_config(corpus, edited_config)

    assert edited_j > baseline_j
    assert accept_edit(baseline_j, edited_j, epsilon=0.01, safe=True) is True


def test_mine_recurring_failures_no_signal_proposes_nothing():
    """Step 8: a corpus with no recognised recurring-failure tags proposes nothing."""
    corpus = WitnessCorpus(records=[
        WitnessRecord(trace_id="a", witness=VerificationWitness(), task_outcome="resolved"),
        WitnessRecord(trace_id="b", witness=VerificationWitness(), task_outcome="resolved", failure_tags=["unmapped_tag"]),
    ])
    assert mine_recurring_failures(corpus) == []


def test_mine_recurring_failures_below_threshold_proposes_nothing():
    """Step 8: a tag appearing in too few traces is treated as noise, not a pattern."""
    corpus = WitnessCorpus(records=[
        WitnessRecord(trace_id="a", witness=VerificationWitness(), task_outcome="resolved", failure_tags=["hidden_test_failure"]),
        WitnessRecord(trace_id="b", witness=VerificationWitness(), task_outcome="resolved"),
        WitnessRecord(trace_id="c", witness=VerificationWitness(), task_outcome="resolved"),
        WitnessRecord(trace_id="d", witness=VerificationWitness(), task_outcome="resolved"),
    ])
    assert mine_recurring_failures(corpus) == []


def test_apply_domain_spec_edit_is_idempotent():
    """Step 8: applying the same proposed edit twice does not duplicate the obligation template."""
    corpus = _hidden_test_failure_corpus()
    edit = mine_recurring_failures(corpus)[0]

    once = apply_domain_spec_edit(DEFAULT_DOMAIN_SPEC, edit)
    twice = apply_domain_spec_edit(once, edit)

    matching = [t for t in twice.obligation_templates if t.template_id == edit.obligation_template.template_id]
    assert len(matching) == 1


def test_accept_edit_gate_requires_improvement_and_safety():
    """Step 8: Accept(Delta h) = I[J_hat improves by > epsilon AND Safe(Delta h)]."""
    assert accept_edit(1.0, 1.5, epsilon=0.01, safe=True) is True
    assert accept_edit(1.0, 1.005, epsilon=0.01, safe=True) is False  # improvement too small
    assert accept_edit(1.0, 2.0, epsilon=0.01, safe=False) is False  # unsafe, even if it improves
