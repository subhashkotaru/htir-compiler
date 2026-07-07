"""
Mechanical-checker registry (AVG Step 5 extension point).

The checker *classes* (mechanical / semantic / abstention) are fixed by the
AVG proposal, but which mechanical rule discharges a given obligation is
open-ended and domain-specific. This registry lets a domain author or a
third-party package add a mechanical checker for a new claim type, template,
or required-evidence kind **without editing ``htir.agents.checking``** -- e.g.
a JSON-Schema validator, a SQL linter, an HTTP-status checker, a numeric
tolerance check.

A checker is a callable ``(CheckerContext) -> CheckerResult`` that inspects
only the obligation's *local* neighbourhood (its claim, that claim's step and
artifacts, and the r_i-typed candidate evidence). Register it by the key it
answers to::

    from htir.agents.checker_registry import CheckerContext, register_checker
    from htir.models.htir import CheckerResult

    @register_checker(claim_type="sql_result")
    def check_sql(ctx: CheckerContext) -> CheckerResult:
        ...

Third-party packages can register at import via the ``htir.checkers``
entry-point group (see ``load_entry_point_checkers``).

Resolution precedence (first match wins), mirroring the built-ins:
1. a checker registered for the obligation's ``required_evidence`` (r_i);
2. a checker registered for its ``template_id``;
3. a checker registered for its claim's ``claim_type``.
Nothing matches -> the caller abstains (never fakes a pass).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Optional

from htir.models.domain import DomainArtifactBundle, DomainSpec
from htir.models.htir import (
    CheckerResult,
    ClaimNode,
    EvidenceNode,
    EvidenceType,
    HTIR,
    Obligation,
)


@dataclass
class CheckerContext:
    """The local graph context handed to a mechanical checker."""
    htir: HTIR
    spec: DomainSpec
    obligation: Obligation
    claim: ClaimNode
    evidence_by_id: dict[int, EvidenceNode]
    domain_artifacts: Optional[DomainArtifactBundle] = None


MechanicalChecker = Callable[[CheckerContext], CheckerResult]

_BY_TEMPLATE: dict[str, MechanicalChecker] = {}
_BY_CLAIM_TYPE: dict[str, MechanicalChecker] = {}
_BY_EVIDENCE: dict[EvidenceType, MechanicalChecker] = {}


def _as_iter(value) -> Iterable:
    if value is None:
        return ()
    if isinstance(value, (str, EvidenceType)):
        return (value,)
    return value


def register_checker(
    fn: MechanicalChecker | None = None,
    *,
    template_id=None,
    claim_type=None,
    required_evidence=None,
):
    """
    Register a mechanical checker under one or more keys. Usable as a
    decorator (``@register_checker(claim_type="x")``) or called directly.
    Each of ``template_id`` / ``claim_type`` / ``required_evidence`` may be a
    single value or an iterable. Returns the function so it can decorate.
    """
    def deco(f: MechanicalChecker) -> MechanicalChecker:
        for t in _as_iter(template_id):
            _BY_TEMPLATE[t] = f
        for c in _as_iter(claim_type):
            _BY_CLAIM_TYPE[c] = f
        for e in _as_iter(required_evidence):
            _BY_EVIDENCE[e] = f
        return f

    return deco(fn) if fn is not None else deco


def resolve_checker(obligation: Obligation, claim: Optional[ClaimNode]) -> Optional[MechanicalChecker]:
    """Find the mechanical checker for ``obligation`` (see module precedence)."""
    if obligation.required_evidence in _BY_EVIDENCE:
        return _BY_EVIDENCE[obligation.required_evidence]
    if obligation.template_id and obligation.template_id in _BY_TEMPLATE:
        return _BY_TEMPLATE[obligation.template_id]
    # Defensive catch: a validation-flavoured template on a provenance claim
    # routes to the post-edit-validation checker even if its exact template id
    # was not registered.
    if (
        claim is not None
        and claim.claim_type == "artifact_provenance"
        and obligation.template_id
        and "validat" in obligation.template_id
        and "trig-post-edit-validation" in _BY_TEMPLATE
    ):
        return _BY_TEMPLATE["trig-post-edit-validation"]
    if claim is not None and claim.claim_type in _BY_CLAIM_TYPE:
        return _BY_CLAIM_TYPE[claim.claim_type]
    return None


def registered_checkers() -> dict[str, list[str]]:
    """Introspection: the keys currently registered, by kind."""
    return {
        "template_id": sorted(_BY_TEMPLATE),
        "claim_type": sorted(_BY_CLAIM_TYPE),
        "required_evidence": sorted(e.value for e in _BY_EVIDENCE),
    }


def load_entry_point_checkers() -> list[str]:
    """
    Discover and register third-party checkers advertised under the
    ``htir.checkers`` entry-point group. Returns the names loaded. Failures
    are ignored so one broken plugin cannot break checking.
    """
    loaded: list[str] = []
    try:
        from importlib.metadata import entry_points
    except ImportError:  # pragma: no cover
        return loaded
    try:
        eps = entry_points(group="htir.checkers")
    except TypeError:  # pragma: no cover - older API
        eps = entry_points().get("htir.checkers", [])  # type: ignore[attr-defined]
    for ep in eps:
        try:
            ep.load()
            loaded.append(ep.name)
        except Exception:
            continue
    return loaded
