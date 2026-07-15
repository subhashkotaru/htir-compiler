# Handoff: Harden AVG Step 4 (obligation generation)

> **STATUS: COMPLETED (2026-07-07) — historical planning document.**
> Every item below has been implemented and verified (36 tests pass in
> `~/.venv`): the Step-4 bug fixes (B1–…), all seven "Code cleanup" items,
> Step 5 (`htir/agents/checking.py`), and Step 6
> (`htir/agents/witness.py`). The two items this doc calls "out of
> scope for the whole handoff" — online intervention `iota_t` (Step 7,
> `htir/agents/intervention.py`) and offline harness improvement
> (Step 8, `htir/agents/harness_improvement.py`) — were subsequently
> built as well. This file is retained as a design record; for the current
> AVG↔code mapping see [`avg-mapping.md`](avg-mapping.md). Read the
> imperative "Goal / TODO / out of scope" framing below as past tense.

Audience: an AI agent continuing this repo. Scope: **make obligation
generation correct and faithful to avg.tex Sec. 3.6 ("Generating Verification
Obligations")**. Do NOT build checker execution (Step 5) or the witness
(Step 6) here — obligations must still be emitted `status=PENDING`,
`result=None`.

Primary file: `htir/agents/obligations.py::build_claims_and_obligations`.
Consumes enrichment from `htir/agents/analysis.py` (already run by
`TraceAbstractionAgent.compile`). Ground truth: `avg.tex` Sec. 3.6, the
obligation tuple in Sec. 2, and the checker-routing rules in Sec. 3.7.

Guardrail: the deterministic pipeline must stay reproducible. Run
`python -m pytest tests/` (in `~/.venv`) after every change — all existing
tests must keep passing, and the checked-in `data/htir_outputs/*.json`
fixtures must not drift unless a change is explicitly intended.

---

## Bugs to fix (ordered by severity)

### B1 — Support edges have wrong polarity and are an untyped cross-product [HIGH]
`obligations.py:177-187`. E_sup is built as the full cross-product of every
step-local evidence node × every step-local claim, always
`polarity=SUPPORTS`. Two defects:
- A **failing** execution-status evidence node is linked as *supporting* a
  "completed successfully" claim. Polarity must depend on evidence vs. claim
  (e.g. `ExecutionStatus.FAILURE` evidence → `REFUTES` a success claim).
- Evidence is linked to *unrelated* claims at the same step (e.g. an
  execution-status evidence "supports" an artifact-provenance claim). This
  destroys the evidence-localization the paper relies on (Sec. 3.8).

Fix: link evidence to a claim only when they concern the same thing (match on
`claim.claim_type` ↔ evidence semantics, or on shared `artifact_ids` /
`step_ids`), and set polarity from the evidence (failure/refuting outcomes →
`REFUTES`).

### B2 — Obligation `candidate_evidence_ids` (E_i) ignore required evidence type (r_i) [HIGH]
`obligations.py:206`. `_emit` sets `candidate_evidence_ids =
evidence_by_step[step_id]` — *all* evidence at the step regardless of type.
AVG defines E_i as candidate evidence **of type r_i**. A `uni-tool-schema`
obligation (r_i = SCHEMA) is handed executable/artifact/log evidence too.

Fix: filter `evidence_by_step[step_id]` to nodes whose `evidence_type`
matches (or is compatible with) `template.required_evidence`. If none match,
leave E_i empty — that is the correct signal that the obligation is
uncovered (Coverage analysis reads this).

### B3 — Obligation anchored on "last claim of the step" [HIGH]
`obligations.py:220-230`. `target_claim = claim_ids[-1]` picks whatever claim
happened to be appended last. It only lands on the right claim for
`final_submission` by luck; for `validation`/`tool_invocation` triggers it
anchors on the arbitrary `execution_status` claim rather than a claim that
matches the template's `claim_template`.

Fix: select (or synthesize) the claim whose `claim_type` corresponds to the
template. Prefer an explicit mapping template_id → claim_type over positional
indexing. When no matching claim exists, create one (the existing synthetic
fallback path) with `claim_type = template.template_id`.

### B4 — Whole classes of obligations get no checker routed [MEDIUM]
`obligations.py:64-70` `_checker_for_evidence`: `LOG`, `MANUAL`, and `NONE`
fall through to `UNASSIGNED`. So `trig-explain-failure` (r_i = `log`, see
`domains/default.yaml`) is emitted with `checker=UNASSIGNED` and will never be
picked up by Step 5. Decide routing per avg.tex Sec. 3.7: LOG is observability
evidence → route to `MECHANICAL` (or `SEMANTIC` if it needs judgement);
`MANUAL` → `ABSTENTION` (needs a human). Only genuinely-no-evidence should
stay `UNASSIGNED`/route to `ABSTENTION`.

### B5 — `build_claims_and_obligations` is not idempotent [MEDIUM]
It appends to `htir.claims/evidence/obligations/support_links` without
clearing. A second call (or a re-compile of the same HTIR) silently doubles
every node. Fix: either clear those lists at the top, or assert they are
empty, or make the function detect prior population and no-op. Add a
regression test.

### B6 — Template trigger matching is too loose / case-fragile [MEDIUM]
`obligations.py:277-288` `_template_triggers`. Final line
`return step.role == trig or trig in step.role.lower()`:
- Substring match causes false positives: a template `trigger: decision`
  fires on `orchestration_decision`; `trigger: edit` fires on
  `artifact_editing`. Triggers that name an operation type should match the
  operation-type **name exactly**, not by substring.
- `trig` is not lowercased on the left of `trig in step.role.lower()`, so an
  uppercase trigger never matches.
- The reserved event triggers (`artifact_edit`, `failed_step`) shadow any
  real operation type of the same name. Low risk, but document/guard it.

Fix: exact-match against `spec.operation_type_names()` for operation triggers;
keep `artifact_edit`/`failed_step`/`""` as explicitly reserved events.

### B7 — Well-formedness-seeded obligations flatten severity/escalation [LOW]
`obligations.py:238-258`. Every seeded obligation gets
`escalation=REQUEST_EVIDENCE` and `scope=TRAJECTORY_TRIGGERED`, even HIGH
integrity findings (`integrity_test_modified`, `integrity_artifact_deleted`)
that should escalate to `VETO`/`ESCALATE`. Map escalation from the issue
(severity or `rule_id`).

### B8 — Redundant / duplicate obligations on the same step [LOW]
`always` templates (`trigger: ""`) plus multiple templates that fire on the
same trigger (e.g. `trig-post-edit-validation` and domain
`swe-edit-then-validate` both on an edit) produce overlapping obligations with
no dedup. Decide the intended semantics (dedup by (claim_id, template_id)? by
required_evidence?) and enforce it. At minimum, dedup identical
(claim_id, template_id) pairs.

### B9 — Redundant well-formedness check vs. an upstream invariant [LOW]
`analysis.py:144-156` rule `artifact_mutation_missing_provenance`: for any
`created`/`modified` provenance link, `_extract_artifacts`
(`trace_abstraction.py:339-342`) already guarantees `produced_by_step_id` is
set, so this branch is unreachable. Either remove it or make it check a real
gap (e.g. before/after states genuinely absent). Note it only affects
enrichment, but the seeded obligation depends on it.

---

## Plan for Step 4 (do in this order)

1. **Introduce a claim-type ↔ template mapping.** Add a small table (dict or
   a field on `ObligationTemplate`, e.g. `target_claim_type`) so B3 and B1 can
   anchor obligations and support edges on the right claim deterministically.
   This is the keystone; B1/B2/B3 all depend on it.

2. **Rework evidence typing + E_sup (fixes B1, B2).** Give `EvidenceNode`s a
   clear type at creation (already done), then: (a) filter E_i by r_i in
   `_emit`; (b) rewrite the support-edge pass to connect evidence↔claim only
   when related, with correct polarity. Add helpers
   `_evidence_supports(claim, evidence) -> SupportPolarity | None`.

3. **Fix checker routing (B4).** Extend `_checker_for_evidence` to cover
   LOG/MANUAL; add a unit test asserting every emitted obligation has a
   `checker != UNASSIGNED` unless intentionally so.

4. **Tighten triggering (B6) and dedup (B8).** Exact operation-type matching;
   dedup (claim_id, template_id).

5. **Idempotency + severity mapping (B5, B7).** Clear-or-assert at entry; map
   seeded-obligation escalation from issue severity.

6. **Coverage sanity.** After the above, re-run `compute_coverage` and confirm
   `covered_obligations` reflects the r_i-filtered E_i (it will drop — that is
   correct, not a regression).

### Acceptance criteria
- All existing tests pass; `data/htir_outputs/*.json` unchanged (or diffs
  reviewed and intended).
- New tests (add to `tests/test_pipeline.py`):
  - A failing step's evidence `REFUTES` (not supports) its success claim.
  - Every non-abstention obligation's `candidate_evidence_ids` are all of the
    obligation's `required_evidence` type.
  - A `validation`-triggered obligation anchors on a validation claim, not the
    execution-status claim.
  - `trig-explain-failure` obligations have a routed checker (not UNASSIGNED).
  - Calling `build_claims_and_obligations` twice does not duplicate nodes.
  - `trigger: decision`-style substrings do NOT fire on
    `orchestration_decision`.
- No new LLM calls in the deterministic path (Step 4 stays mechanical).

### Boundary with Step 5
Step 4 still emits obligations `status=PENDING`, `result=None`. Do not run
checkers *inside* `build_claims_and_obligations`. Step 5 (below) is a separate
pass that reads the finished obligation set and fills results. Finishing Step 4
cleanly (correct r_i-typed E_i, routed checker, right claim anchor) is what
makes Step 5 tractable.

Still out of scope for the whole handoff: online intervention `iota_t`
(avg.tex Sec. 3.10) and offline harness improvement (Sec. 3.11). Leave
`InterventionAction` as the defined-but-unwired enum it is.

---

## Code cleanup (separate commit from the bug fixes)

- **Stale docstring.** `trace_abstraction.py:2` says "Section III-A of the
  paper" — a HarnessFix reference; `avg.tex` has no such section. Replace with
  the AVG Step 1–2 description.
- **Stale doc.** `docs/avg-mapping.md:8-11` and its "Out of scope" section say
  Steps 3–6 are unbuilt; Steps 3–4 are now built. Update the scope statement
  and move the still-unbuilt items (checking, aggregation, witness, online
  loop) into the remaining list.
- **Confusing LLM schema name.** `trace_abstraction.py:70-82` names the
  input-reuse LLM schema `_ProvenanceLink` / `_ProvenanceLinkList` even though
  it maps to `InputReuseLink` (explicitly NOT AVG provenance). Rename to
  `_InputReuseLink` / `_InputReuseLinkList` to match the model and avoid
  collision with the real `ArtifactProvenanceLink`.
- **Unused loop variables.** `analysis.py` rule (f) `for i, step in
  enumerate(ordered)` (`check_wellformedness`) — `i` is unused; use `for step
  in ordered`. Sweep for other unused `enumerate` indices.
- **Operator-precedence readability.** `trace_abstraction.py:369-371`
  `_infer_artifact_type`: `if "." in ident and "/" in ident or ident.endswith(
  ...)` relies on `and` binding tighter than `or`. Parenthesize to make intent
  explicit even though behavior is correct.
- **ID-space hazard (document, don't necessarily change).** Evidence, claim,
  and obligation IDs each restart at 1 (`_Counter`), overlapping artifact and
  step IDs. Safe today because every cross-reference is typed by field name,
  but add a one-line note in the module docstring so no one later mixes them
  in a shared container (cf. `WellFormednessIssue.offending_node_ids`, which
  already mixes step/artifact ids by necessity).
- **Dead-comment check.** `obligations.py:291-294` trailing comment about moved
  functions is fine to keep; verify no other comment references
  `_link_constraint`/`_link_dependencies` as if still local.

Commit hygiene: one commit for bug fixes (with the new tests), one for
cleanup, so the fixture-affecting change is isolated and reviewable.

---

# Step 5 — Checking obligations (avg.tex Sec. 3.7)

Goal: discharge each obligation by running the checker routed to it in Step 4,
filling `Obligation.result` (a `CheckerResult`) and `Obligation.status`
(`PASSED`/`FAILED`/`ABSTAINED`), and propagating outcomes to
`ClaimNode.status` (`SUPPORTED`/`REFUTED`/`UNRESOLVED`). This is the central
verification act; do it only after Step 4 is hardened.

New file: `htir/agents/checking.py`. Do NOT touch graph construction or
obligation generation. Entry point suggestion:
`check_obligations(htir, spec, *, use_semantic=False, model=DEFAULT_MODEL) -> HTIR`.

### Checker contract
A checker consumes an obligation and its **local** graph context (the claim,
its candidate evidence `E_i`, and the immediate neighbourhood — producing
step, consumed/produced artifacts, validation/dependency edges touching it)
and returns `CheckerResult(p_pass, p_fail, p_abstain, score, evidence_used)`
with the three probabilities summing to 1. Keep context local (avg.tex Sec.
3.7): pass only the claim + linked evidence, never the whole trace.

### Three checker classes (route by `Obligation.checker` from Step 4)
1. **Mechanical (`CheckerType.MECHANICAL`)** — deterministic, no LLM. Examples
   the repo can support today:
   - execution-status / exit-code checks: read the EXECUTABLE evidence node's
     step, pass iff `ExecutionStatus.SUCCESS`, fail on FAILURE/TIMEOUT/BLOCKED.
   - provenance checks: claim "artifact X produced by step N" passes iff an
     `ArtifactProvenanceLink(created|modified)` exists for (N, X).
   - post-edit-validation (`trig-post-edit-validation` / `swe-edit-then-
     validate`): pass iff a `ValidationLink`/state-transition shows a
     successful revalidation after the edit; fail if revalidation failed;
     abstain if none exists.
   - schema checks (`uni-tool-schema`, `swe-generated-schema`): if the domain
     artifact type has a `schema_hint`, do the structural check you can;
     otherwise abstain (do not fake a pass).
   Mechanical checkers MUST be deterministic and reproducible (fixtures).
2. **Semantic (`CheckerType.SEMANTIC`)** — narrow LLM judge over a single
   claim–evidence pair, gated behind `use_semantic` (mirror the `enrich`
   pattern). Examples: "does this diff plausibly address this failing
   assertion", "is this final sentence supported by this retrieved passage",
   "does this action follow this policy excerpt". Reuse `llm.chat_json` with a
   small pydantic result schema `{verdict: pass|fail|abstain, confidence:
   float, rationale: str}`; map to `CheckerResult`. When `use_semantic=False`,
   a semantic obligation abstains (see class 3) — never silently passes.
3. **Abstention (`CheckerType.ABSTENTION`)** — emits `p_abstain=1.0`. This is
   the normal outcome for well-formedness-seeded obligations and for any
   obligation with insufficient evidence. It is a first-class result, not a
   failure (avg.tex Sec. 3.7).

### Effects to write back
- Set `obligation.result` and derive `obligation.status` from the argmax of
  the three probabilities (define a tie-break: abstain > fail > pass, i.e. be
  conservative).
- Update the linked `ClaimNode.status`: PASSED→SUPPORTED, FAILED→REFUTED,
  ABSTAINED→UNRESOLVED. Respect existing REFUTES support edges from Step 4
  (a claim with refuting evidence should not end SUPPORTED).
- Populate `CheckerResult.evidence_used` (`eta_i`) with the evidence-node ids
  the checker actually consulted (subset of `candidate_evidence_ids`).

### Pitfalls (learn from the Step-4 bugs)
- Do not re-derive evidence inside the checker; consume the r_i-typed `E_i`
  Step 4 provides. If `E_i` is empty, that is a signal to abstain, not to
  search the whole graph.
- Keep mechanical and semantic strictly separated so `use_semantic=False`
  stays byte-for-byte reproducible.
- Idempotency: re-running `check_obligations` must overwrite results, not
  accumulate. Guard/clear.

### Tests (add to `tests/test_pipeline.py`)
- On the synthetic fail→edit→pass trace: the post-edit-validation obligation
  is `PASSED` mechanically; the failing step-1 execution claim is `REFUTED`.
- A well-formedness-seeded obligation is `ABSTAINED` with `p_abstain==1.0`.
- With `use_semantic=False`, every SEMANTIC obligation is `ABSTAINED` (no LLM
  call — assert `chat_json` is not invoked, via monkeypatch).
- `check_obligations` run twice yields identical results (idempotent).

---

# Step 6 — Aggregation + verification witness (avg.tex Sec. 3.8–3.9)

Goal: collapse the checked obligations into a trajectory-level status and emit
the verification witness `W_tau`, the stated output of AVG. Depends on Step 5.

### New models (add to `htir/models/htir.py`)
- `AggregateResult` for `z_tau = (y_hat, u_hat, c_hat, eta_hat)`:
  - `predicted_status: str` (e.g. `valid` / `invalid` / `uncertain`),
  - `uncertainty: float`,
  - `evidence_coverage: float` (reuse/roll up `CoverageReport`),
  - `aggregated_evidence_ids: list[int]`.
- `VerificationWitness` for `W_tau = (O+, O-, O∅, E_W, R_W)`:
  - `passed_obligation_ids: list[int]`,
  - `failed_obligation_ids: list[int]`,
  - `abstained_obligation_ids: list[int]`,
  - `witness_evidence_ids: list[int]` (`E_W`),
  - `review_recommendation: str` (`R_W`).
- Add optional fields on `HTIR`: `aggregate: Optional[AggregateResult] = None`
  and `witness: Optional[VerificationWitness] = None` (default None →
  serialized fixtures stay backward-compatible).

### New file: `htir/agents/witness.py`
Entry points: `aggregate(htir) -> AggregateResult` and
`build_witness(htir) -> VerificationWitness`.

### Aggregation rules (severity-aware — avg.tex Sec. 3.8)
- A **failed high/critical-severity** obligation vetoes success →
  `predicted_status = invalid`, even if many low-severity obligations pass
  (the paper's "modified tests but suite passes" example).
- No failures but **many abstained high-severity** obligations →
  `predicted_status = uncertain`, not `valid`.
- Otherwise `valid`. Compute `uncertainty` from the mass of abstentions
  weighted by severity; `evidence_coverage` from `CoverageReport`.
- Make the thresholds explicit constants at the top of the module so they are
  auditable and testable.

### Witness (avg.tex Sec. 3.9)
- Partition obligations by `status` into O+/O-/O∅.
- `E_W` = union of `result.evidence_used` across the failed + abstained
  obligations (the evidence a reviewer needs), plus the evidence behind any
  vetoing obligation.
- `R_W` = a short deterministic template string summarising: overall status,
  the vetoing obligation(s) if any, and the count/kind of unresolved
  high-severity obligations, ending with "inspect: <the single most important
  unresolved/failed obligation>". Keep it mechanical (no LLM) so it is
  reproducible; a semantic prose upgrade can be gated behind `use_semantic`
  later.

### Wiring
Extend `TraceAbstractionAgent.compile` so that when `generate_obligations`
(and a new `run_checks=True` flag) is set, the tail becomes:
`enrich → build_claims_and_obligations → compute_coverage → check_obligations
→ aggregate → build_witness`. Keep each stage independently callable and each
new stage behind a flag so the current Step-1..4 output is unchanged when the
flags are off.

### Tests
- Fail→edit→pass synthetic trace with a forced failed critical obligation →
  witness `predicted_status = invalid` and that obligation id in
  `failed_obligation_ids` and in `E_W`.
- All-pass trace → `valid`, empty O-, non-empty O+.
- Trace with only abstentions on high-severity → `uncertain`.
- `build_witness` is deterministic (no LLM) and idempotent.

---

# Suggested overall sequencing for the implementing agent
1. Step 4 bug fixes + tests (one commit).
2. Cleanup (one commit).
3. Step 5 `checking.py` + models already exist (`CheckerResult`) + tests.
4. Step 6 new models + `witness.py` + `compile` wiring + tests.
5. Regenerate `data/htir_outputs/*.json` only if intended, and review the diff.
Keep every new heavy/LLM pass behind a flag defaulting off; the deterministic
path must remain reproducible at each step.
