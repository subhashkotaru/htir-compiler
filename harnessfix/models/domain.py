"""
Domain specifications (AVG S_d).

A domain specification describes how to interpret observable events in a
particular environment:

    S_d = (P_d, R_d, K_d, B_d)

where P_d is the set of operation types, R_d the artifact types, K_d the
domain constraints, and B_d the obligation templates. Replacing the old fixed
``StepRole`` enum, operation types now come from the active spec, so the same
verification layer can be adapted to terminal, data, or policy-heavy domains
through small, declarative specs.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field

from harnessfix.models.htir import (
    EscalationRule,
    EvidenceType,
    ObligationScope,
    Severity,
)

# Directory holding the YAML domain specs (the recommended editing surface).
DOMAINS_DIR = Path(__file__).resolve().parent.parent / "domains"


# ---------------------------------------------------------------------------
# Spec components
# ---------------------------------------------------------------------------

class OperationType(BaseModel):
    """An element of P_d: a kind of operation a step may perform."""
    name: str
    description: str = ""


class ArtifactTypeSpec(BaseModel):
    """An element of R_d: a kind of artifact the domain produces/consumes."""
    name: str
    description: str = ""
    schema_hint: str = Field("", description="Optional description of expected structure/schema")


class Constraint(BaseModel):
    """An element of K_d: a domain rule that must hold."""
    constraint_id: str
    description: str
    severity: Severity = Severity.MEDIUM
    applies_to_operations: list[str] = Field(
        default_factory=list,
        description="Operation type names this constraint governs (empty = all)",
    )


class ObligationTemplate(BaseModel):
    """
    An element of B_d: a template that, when its trigger fires, instantiates a
    concrete Obligation over a claim in the graph.
    """
    template_id: str
    description: str = ""
    claim_template: str = Field(..., description="Human-readable claim the obligation checks")
    scope: ObligationScope = ObligationScope.DOMAIN
    trigger: str = Field(
        "",
        description="Operation type / event that triggers instantiation "
        "(e.g. 'final_submission', 'artifact_edit', 'failed_step'). Empty = always.",
    )
    required_evidence: EvidenceType = EvidenceType.SEMANTIC
    severity: Severity = Severity.MEDIUM
    escalation: EscalationRule = EscalationRule.REQUEST_EVIDENCE
    target_claim_type: str = Field(
        "",
        description="``ClaimNode.claim_type`` this template's obligations should anchor "
        "on (e.g. 'execution_status', 'artifact_provenance', 'final_answer_support'). "
        "Empty = no existing claim of that type is expected; a synthetic claim typed "
        "``template_id`` is created instead.",
    )


class DomainSpec(BaseModel):
    """The full domain specification S_d."""
    domain_id: str
    description: str = ""
    operation_types: list[OperationType] = Field(default_factory=list)      # P_d
    artifact_types: list[ArtifactTypeSpec] = Field(default_factory=list)    # R_d
    constraints: list[Constraint] = Field(default_factory=list)            # K_d
    obligation_templates: list[ObligationTemplate] = Field(default_factory=list)  # B_d

    def operation_type_names(self) -> list[str]:
        return [op.name for op in self.operation_types]

    def artifact_type_names(self) -> list[str]:
        return [at.name for at in self.artifact_types]

    def templates_for_scope(self, scope: ObligationScope) -> list[ObligationTemplate]:
        return [t for t in self.obligation_templates if t.scope == scope]


# ---------------------------------------------------------------------------
# YAML loading (the recommended surface for adding/adapting domains)
# ---------------------------------------------------------------------------

def load_domain_spec(path: str | Path) -> DomainSpec:
    """Load a single domain spec from a YAML file."""
    import yaml  # imported lazily so pyyaml stays an optional dependency

    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return DomainSpec.model_validate(data)


def load_domain_specs(directory: str | Path = DOMAINS_DIR) -> dict[str, DomainSpec]:
    """
    Discover and load every ``*.yaml`` / ``*.yml`` domain spec in ``directory``,
    keyed by ``domain_id``. Files that fail to parse are skipped so one bad
    spec cannot break the whole registry.
    """
    specs: dict[str, DomainSpec] = {}
    d = Path(directory)
    if not d.exists():
        return specs
    for p in sorted([*d.glob("*.yaml"), *d.glob("*.yml")]):
        try:
            spec = load_domain_spec(p)
        except Exception:
            continue
        specs[spec.domain_id] = spec
    return specs


# Minimal in-code fallback so imports never hard-fail if pyyaml or the YAML
# files are unavailable. The YAML files are the source of truth otherwise.
_FALLBACK_DEFAULT = DomainSpec(
    domain_id="default",
    description="Fallback default spec (pyyaml or domains/ unavailable).",
    operation_types=[
        OperationType(name="information_acquisition"),
        OperationType(name="tool_invocation"),
        OperationType(name="artifact_editing"),
        OperationType(name="validation"),
        OperationType(name="orchestration_decision"),
        OperationType(name="final_submission"),
        OperationType(name="other"),
    ],
    artifact_types=[ArtifactTypeSpec(name="state")],
)

# Registry of available specs, keyed by domain_id. Loaded from YAML at import,
# falling back to the in-code default if nothing could be loaded.
DOMAIN_SPECS: dict[str, DomainSpec] = load_domain_specs()
if "default" not in DOMAIN_SPECS:
    DOMAIN_SPECS["default"] = _FALLBACK_DEFAULT

DEFAULT_DOMAIN_SPEC: DomainSpec = DOMAIN_SPECS["default"]
TERMINAL_DOMAIN_SPEC: DomainSpec = DOMAIN_SPECS.get("terminal_swe", DEFAULT_DOMAIN_SPEC)


def get_domain_spec(domain_id: str) -> DomainSpec:
    """Look up a domain spec by id, defaulting to the default spec."""
    return DOMAIN_SPECS.get(domain_id, DEFAULT_DOMAIN_SPEC)
