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

from enum import Enum
from pathlib import Path

from pydantic import BaseModel, Field

from htir.models.htir import (
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
    requires_prior: list[str] = Field(
        default_factory=list,
        description="Operation-type names that must SUCCESSFULLY occur before a governed "
        "step (a precondition ordering, e.g. authenticate-before-action). When set, the "
        "constraint's obligation is a *mechanical* precondition check (structural, no LLM); "
        "when empty, it is a narrow semantic check against the constraint's policy text.",
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


# ---------------------------------------------------------------------------
# Domain artifacts Omega_d (avg.tex Sec. 2, "Domain artifacts")
# ---------------------------------------------------------------------------
#
# Omega_d = {schemas, manuals, logs, policies, tests, historical traces,
# counterexamples}: weak supervision content a checker can compare a claim
# against. Distinct from S_d, which only *declares* that a constraint or
# obligation exists -- Omega_d supplies the actual content (a policy's text,
# a schema's structure, ...). Optional: a domain with no Omega_d bundle
# behaves exactly as it did before this was introduced (see
# ``load_domain_artifacts`` returning ``None``).


class ArtifactKind(str, Enum):
    """The kinds of weak supervision artifacts an Omega_d bundle may contain."""
    SCHEMA = "schema"
    MANUAL = "manual"
    LOG = "log"
    POLICY = "policy"
    TEST = "test"
    HISTORICAL_TRACE = "historical_trace"
    COUNTEREXAMPLE = "counterexample"


class DomainArtifact(BaseModel):
    """A single element of Omega_d."""
    artifact_kind: ArtifactKind
    identifier: str = Field(..., description="Human/tool identifier, e.g. a policy or schema name")
    content: str = Field("", description="The artifact's actual content (policy text, schema body, log excerpt, ...)")
    metadata: dict = Field(default_factory=dict)


class DomainArtifactBundle(BaseModel):
    """Omega_d for one domain: the full set of weak supervision artifacts."""
    domain_id: str
    artifacts: list[DomainArtifact] = Field(default_factory=list)

    def by_kind(self, kind: ArtifactKind) -> list[DomainArtifact]:
        return [a for a in self.artifacts if a.artifact_kind == kind]

    def get(self, kind: ArtifactKind, identifier: str) -> DomainArtifact | None:
        for a in self.artifacts:
            if a.artifact_kind == kind and a.identifier == identifier:
                return a
        return None


def load_domain_artifacts(domain_id: str, directory: str | Path = DOMAINS_DIR) -> DomainArtifactBundle | None:
    """
    Discover and load Omega_d for ``domain_id`` from
    ``<directory>/<domain_id>.artifacts/*.yaml`` (mirrors ``load_domain_spec``'s
    YAML-as-source-of-truth convention). Each YAML file is one
    ``DomainArtifact``. Returns ``None`` when the directory is absent, empty,
    or nothing parses -- callers must treat a missing bundle as "no Omega_d
    available" and fall back to today's behavior (abstain rather than fake
    evidence), never raise.
    """
    import yaml  # imported lazily so pyyaml stays an optional dependency

    art_dir = Path(directory) / f"{domain_id}.artifacts"
    if not art_dir.exists():
        return None

    artifacts: list[DomainArtifact] = []
    for p in sorted([*art_dir.glob("*.yaml"), *art_dir.glob("*.yml")]):
        try:
            data = yaml.safe_load(p.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                continue
            artifacts.append(DomainArtifact.model_validate(data))
        except Exception:
            continue

    if not artifacts:
        return None
    return DomainArtifactBundle(domain_id=domain_id, artifacts=artifacts)
