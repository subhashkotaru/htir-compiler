# AVG -> HTIR mapping

This file tracks, for every symbol in the AVG proposal (`avg.tex`), the
concrete class/field in this codebase that realises it, and any known
deviation. HarnessFix concepts that are *not* part of AVG are called out
explicitly so they are not mistaken for proposal constructs.

Scope: this repo implements the full AVG pipeline, Steps 1-8: `raw trace ->
typed events -> verification graph G_tau` (Steps 1-2), well-formedness checks
and the analysis modules (Step 3, `htir.agents.analysis`), claim /
evidence / obligation generation (Step 4,
`htir.agents.obligations.build_claims_and_obligations`), checker
execution (Step 5, `htir.agents.checking.check_obligations`),
aggregation + verification witness (Step 6, `htir.agents.witness`),
online intervention (Step 7, `htir.agents.intervention`, over a
replayed *recorded* trace via `TraceAbstractionAgent.compile_prefix`), and
offline harness improvement (Step 8,
`htir.agents.harness_improvement`, over a recorded witness corpus).
`Omega_d` weak domain artifacts (avg.tex Sec. 2, work item A,
`htir.models.domain.DomainArtifact`/`DomainArtifactBundle`) are an
optional side input threaded through Steps 3-5; every new stage/parameter is
flag-gated and defaults to the pre-existing (`None`/`False`) behavior, so
`data/htir_outputs/*.json` fixtures do not drift unless a bundle or flag is
explicitly supplied. See "Out of scope" at the bottom for what's still a
placeholder/deliberately unautomated.

## Node kinds

| AVG concept | Class | Notes |
|---|---|---|
| Operation node | `TraceStep` (htir/models/htir.py) | One recoverable execution step. `role` is drawn from the active domain spec's operation vocabulary (`S_d.P_d`), not a fixed enum. |
| Artifact node | `ArtifactNode` | First-class produced/consumed object. `artifact_type` drawn from `S_d.R_d`. |
| Claim node | `ClaimNode` | Checkable statement induced from the trajectory; untrusted until discharged by an obligation. |
| Evidence node | `EvidenceNode` | Points to artifacts/graph neighbourhoods that may support or refute claims. |
| Obligation | `Obligation` | See obligation tuple below. |

## Edge families

| AVG edge | Class | Field on `HTIR` | Notes |
|---|---|---|---|
| `E_temp` (temporal) | `TemporalLink` | `temporal_links` | Preserves original execution order. |
| `E_prov` (provenance) | `ArtifactProvenanceLink` | `provenance_links` | Artifact-centric: links each artifact to the operation that `created`/`read`/`modified` it (avg.tex Sec. 3, Provenance analysis). Populated deterministically in `TraceAbstractionAgent._extract_artifacts`. |
| `E_causal` (dependency) | `DependencyLink` | `dependency_links` | "Which operations depend on earlier artifacts" (avg.tex Sec. 3, Dependency analysis). Populated by the Step-3 analysis module `htir/agents/analysis.py::link_dependencies`: consumer step -> producer step of a consumed artifact, edit -> most recent failing validation, and final answer -> policy artifact. |
| `E_sup` (support) | `SupportLink` | `support_links` | Evidence supports/refutes a claim. |
| `E_cons` (constraint) | `ConstraintLink` | `constraint_links` | Domain constraint (`S_d.K_d`) governing a step/artifact. |
| `E_val` (validation) | `ValidationLink` | `validation_links` | Validation operation -> the step/artifact it validates. |

## HarnessFix extensions (NOT AVG constructs)

These are carried over from the original HarnessFix paper for harness-code
attribution / diagnostics. They are additive, optional, and must not be
confused with the AVG edges/nodes above:

| Class / field | Purpose | AVG equivalent |
|---|---|---|
| `InputReuseLink` (`input_reuse_links`) | Step->step: how a later request reused an earlier step's content. | None. Previously mislabelled `E_prov`; renamed to avoid confusion with the real, artifact-centric `ArtifactProvenanceLink`. |
| `ControlFlowLink` (`control_flow_links`) | Step->step: harness controller transition (retry/delegate/finalize/...) that produced the next step. | None. Previously mislabelled `E_causal`; renamed/annotated to avoid confusion with `DependencyLink`. |
| `HarnessLayer`, `HarnessLayerFacet` (`TraceStep.harness_layer_facet`), `HarnessCodeRef` | The seven ETCLOVG harness layers implicated by a step's local evidence, and pointers to the harness source responsible. | None. Off by default (`TraceAbstractionAgent.compile(attach_harness_layers=False)`); costs one LLM call per step when enabled. |

## Obligation tuple `o_i = (c_i, r_i, E_i, q_i, rho_i, alpha_i)`

| Symbol | Field on `Obligation` | Type |
|---|---|---|
| `c_i` (claim) | `claim_id` | int (-> `ClaimNode`) |
| `r_i` (required evidence) | `required_evidence` | `EvidenceType` |
| `E_i` (candidate evidence) | `candidate_evidence_ids` | list[int] (-> `EvidenceNode`) |
| `q_i` (checker) | `checker` | `CheckerType` |
| `rho_i` (severity) | `severity` | `Severity` |
| `alpha_i` (escalation) | `escalation` | `EscalationRule` |

`EscalationRule` (alpha_i) intentionally shares its action vocabulary
(`accept`/`request-evidence`/`rerank`/`veto`/`repair`/`clarify`/`escalate`)
with `InterventionAction`, the online intervention `iota_t` from avg.tex Sec.
3.11. They are distinct mechanisms: `alpha_i` is fixed per-obligation at
generation time; `iota_t` is chosen online per step from the active
obligation set by `htir.agents.intervention.select_intervention`
(Step 7). The default policy simply follows `alpha_i` (see "Steps 5-8" below
for the pluggable benefit/cost/risk interface this leaves room for).

`CheckerResult` (`q_i(o_i, G_tau) = (p+, p-, p_abstain, s, eta)`) is filled by
`htir.agents.checking.check_obligations` (Step 5); `Obligation.status`
is its argmax with a conservative (abstain > fail > pass) tie-break, and
`ClaimNode.status` is derived per claim across all of its obligations so a
failed obligation or a pre-existing `REFUTES` support edge from Step 4 always
wins (never `SUPPORTED`).

## Domain spec `S_d = (P_d, R_d, K_d, B_d)`

| Symbol | Class | Notes |
|---|---|---|
| `P_d` (operation types) | `OperationType`, `DomainSpec.operation_types` | |
| `R_d` (artifact types) | `ArtifactTypeSpec`, `DomainSpec.artifact_types` | |
| `K_d` (constraints) | `Constraint`, `DomainSpec.constraints` | |
| `B_d` (obligation templates) | `ObligationTemplate`, `DomainSpec.obligation_templates` | Split into universal / domain / trajectory-triggered via `ObligationScope`. |

Domain specs are YAML-driven (`htir/domains/*.yaml`), loaded via
`htir.models.domain.load_domain_spec`.

## Omega_d weak domain artifacts (avg.tex Sec. 2, work item A)

| Symbol | Class | Notes |
|---|---|---|
| `Omega_d` | `DomainArtifactBundle`, `htir.models.domain` | `{schemas, manuals, logs, policies, tests, historical traces, counterexamples}`. Loadable via `load_domain_artifacts(domain_id)`, auto-discovered from `htir/domains/<domain_id>.artifacts/*.yaml`. Returns `None` when absent -- every downstream pass must (and does) behave identically to before `Omega_d` existed in that case. |
| element of `Omega_d` | `DomainArtifact` | `artifact_kind` (`ArtifactKind`), `identifier`, `content`, `metadata`. Never added as an `HTIR` `ArtifactNode` -- only consulted as evidence content (see below), so fixtures don't bloat. |

Threaded (all optionally, default `None`) through
`TraceAbstractionAgent.__init__`/`compile`, `analysis.enrich`/`link_policy`
(adds a real dependency edge, avg.tex Sec. 3, from every policy-sensitive
step to each loaded `policy` artifact), `obligations.
build_claims_and_obligations` (injects `SCHEMA` evidence for produced
artifacts whose type's `schema_hint` resolves against a loaded `schema`
artifact, and emits one `omega-policy-compliance` obligation per
policy-sensitive-step x `policy` artifact, `PENDING` until Step 5), and
`checking.check_obligations` (the schema/semantic checkers consume that
evidence; a schema checker abstains, never fakes a pass, when no matching
artifact is available).

## Steps 5-8: checking, aggregation, witness, intervention, harness improvement

| AVG step | Symbol(s) | Module | Notes |
|---|---|---|---|
| Step 5 (checking, Sec. 3.8) | `q_i(o_i, G_tau) = (p+, p-, p_abstain, s, eta)` | `htir.agents.checking.check_obligations` | Three checker classes routed by `Obligation.checker`: `MECHANICAL` (execution-status, provenance, post-edit-validation via `ValidationLink`, explained-failure via `DependencyLink`, schema), `SEMANTIC` (narrow LLM judge, gated behind `use_semantic`, abstains without a call when off), `ABSTENTION`/`UNASSIGNED` (`p_abstain=1.0`). |
| Step 6 (aggregation, Sec. 3.9) | `z_tau = (y_hat, u_hat, c_hat, eta_hat)` | `htir.agents.witness.aggregate` -> `HTIR.aggregate: AggregateResult` | Severity-aware: a failed HIGH/CRITICAL obligation vetoes to `invalid`; many abstained HIGH/CRITICAL obligations (see `UNCERTAIN_ABSTAIN_*_THRESHOLD`) give `uncertain`; otherwise `valid`. |
| Step 6 (witness, Sec. 3.10) | `W_tau = (O+, O-, O-empty, E_W, R_W)` | `htir.agents.witness.build_witness` -> `HTIR.witness: VerificationWitness` | `R_W` is a short, deterministic (no LLM) template string ending in an "inspect: <obligation>" pointer. |
| Step 7 (online intervention, Sec. 3.11) | `iota_t*` (over `G_{tau<=t}`) | `htir.agents.intervention` (`active_obligations`, `select_intervention`, `run_intervention_loop`) -> `HTIR.intervention_log: list[InterventionLogEntry]` | Runs over a *replayed recorded trace* via `TraceAbstractionAgent.compile_prefix` (re-runs the deterministic pipeline per prefix), not a live agent. Default policy reduces the paper's `argmax E[r_hat - beta*Cost - gamma*Risk]` to `alpha_i` via pluggable `benefit_fn`/`cost_fn`/`risk_fn`. Pure recommendation trace -- never drives an agent. |
| Step 8 (offline harness improvement, Sec. 3.12) | `h = (p, s, m, r)`, `Accept(Delta h)` | `htir.agents.harness_improvement` (`HarnessConfig`, `WitnessCorpus`, `mine_recurring_failures`, `score_config`, `accept_edit`, `apply_domain_spec_edit`) | Offline analysis over a recorded corpus of witnesses; mines recurring `failure_tags` into proposed `S_d` obligation-template edits (avg.tex's two named examples: hidden-test-only passes, CSV-missing-header) and gates them via `Accept(Delta h) = I[J_hat(h+Delta h) > J_hat(h)+epsilon AND Safe(Delta h)]`. Never auto-applies an edit or touches a live agent; harness (`h`) edits themselves stay out-of-repo/unapplied (`ProposedEdit.harness_delta`). |

All of the above are flag-gated on `TraceAbstractionAgent.compile`
(`run_checks=False` by default) or are separate opt-in entry points
(`compile_prefix`, `run_intervention_loop`, `harness_improvement.*`) --
Step 1-4 output is unchanged when they are not used.

## Out of scope / deliberately unautomated

Every AVG step (1-8) has a concrete implementation now; what's left is
scoped out on purpose, not an unbuilt seam:

- **True incremental graph compilation.** Step 7 uses the "simplest" option
  from its handoff -- `TraceAbstractionAgent.compile_prefix` re-runs the
  whole deterministic pipeline over `raw_steps[:t]` rather than
  incrementally updating an existing `HTIR`. Fine for offline replay of a
  recorded trace; a true incremental update is future work if perf matters.
- **Live-agent intervention.** `htir.agents.intervention` only
  produces a recommendation trace (`InterventionLogEntry`) over replayed
  recorded traces. It never drives a live agent/harness loop.
- **Learned benefit/cost/risk estimators.** `select_intervention`'s
  `argmax E[r_hat - beta*Cost - gamma*Risk]` uses simple constant
  `cost_fn`/`risk_fn` and a `benefit_fn` that reduces to `alpha_i`
  (`Obligation.escalation`) by default; the interface is pluggable so a
  learned estimator can be substituted without changing callers.
- **Auto-applied harness edits.** Step 8 (`harness_improvement.py`) only
  *proposes* and *scores* edits (`ProposedEdit`, `score_config`,
  `accept_edit`); it never writes to a live harness. `EditTarget.HARNESS`
  edits are represented as an opaque, unapplied `harness_delta` -- only
  `EditTarget.DOMAIN_SPEC` edits (`apply_domain_spec_edit`) are actually
  applicable, since the harness proper lives outside this repo.
- **Semantic-prose review recommendations.** `VerificationWitness.
  review_recommendation` (`R_W`) is a deterministic template string by
  design (avg.tex Sec. 3.10 calls this out explicitly); an LLM-authored
  prose upgrade gated behind `use_semantic` is a possible future addition,
  not built here.

