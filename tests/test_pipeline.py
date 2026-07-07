"""
Smoke test for the deterministic (non-LLM) parts of the HTIR/AVG pipeline:
artifact extraction (E_prov), dependency linking (E_causal), and claim /
obligation construction.

Builds a synthetic HTIR by hand (no OpenRouter/LLM calls) so it can run
without ``OPENROUTER_API_KEY``, then exercises:

  * ``TraceAbstractionAgent._extract_artifacts`` -- artifact nodes +
    ``ArtifactProvenanceLink`` (E_prov).
  * ``harnessfix.agents.obligations.build_claims_and_obligations`` -- claim /
    evidence / obligation nodes and the support / validation / constraint /
    dependency (E_causal) edges.
"""

from __future__ import annotations

from harnessfix.agents.obligations import build_claims_and_obligations
from harnessfix.agents.trace_abstraction import TraceAbstractionAgent
from harnessfix.models.domain import DEFAULT_DOMAIN_SPEC
from harnessfix.models.htir import (
    HTIR,
    ArtifactEffect,
    ArtifactStateEvidence,
    ExecutionStatus,
    ProvenanceRelation,
    TraceStep,
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

    build_claims_and_obligations(htir, DEFAULT_DOMAIN_SPEC)

    assert len(htir.claims) > 0
    assert len(htir.evidence) > 0
    assert len(htir.obligations) > 0
    assert len(htir.support_links) > 0
    assert len(htir.validation_links) > 0
    assert len(htir.constraint_links) > 0

    # E_causal (dependency): step 2 consumes test_report produced by step 1;
    # step 3 consumes parser.py produced by step 2.
    assert len(htir.dependency_links) == 2
    dep_pairs = {(lk.source_step_id, lk.target_step_id) for lk in htir.dependency_links}
    assert (2, 1) in dep_pairs
    assert (3, 2) in dep_pairs
