# AVG Proposal vs. Implementation — Gap Analysis

Audit of the implemented experiments (SA-1 … SA-6) against the proposal in
`avg.tex` (Sec. 4). **Read-only analysis — nothing implemented here.** Companion
to `docs/experiment-plan.md` (which maps the intended packages) and
`docs/avg-mapping.md` (which maps proposal symbols to code).

**Audit date:** 2026-07-13

---

## TL;DR

Every reported experiment ran on the **same 3,000 Terminal-Bench traces, a
single domain (`terminal_swe`), a single seed (`seed=0`), with
`use_llm=False`**. That one configuration cascades into several proposal claims
being *structurally untested* rather than merely under-powered. The three
highest-leverage fixes are: (1) turn on `use_llm=True`, (2) wire a real second
domain, (3) hand-label a gold slice.

Verified facts (from `data/sa*_results.json` and the eval code):

| Experiment | n_traces | domain | use_llm | seeds |
|---|---|---|---|---|
| SA-1 … SA-6 | 3000 | `terminal_swe` | `False` | 1 (seed=0) |

---

## The overarching gap: offline + no-LLM + single-domain

This single fact cascades into several proposal claims being **structurally
untested**, not just under-powered.

1. **Semantic checkers are never exercised.** With `use_semantic=False`,
   `htir/agents/checking.py`'s semantic family always abstains without a call.
   So offline:
   - The **`EXEC_FREE` (execution-free) arm is effectively empty** — it can only
     abstain. Sec. 4.3 lists it as a real baseline; right now it is a null arm.
   - **`AVG_FULL` collapses to `EXEC_ONLY`** offline (semantic only adds
     abstentions). So SA-1's headline "AVG beats monolith" is really "mechanical
     checkers beat an endpoint heuristic" — the semantic-evidence half of the
     framework (Sec. 3.7) contributes nothing to any reported number.
2. **The monolithic baseline is a strawman.** `baselines.monolithic_judge`
   defaults to `_endpoint_monolithic_verdict` (trust the last validation
   status). The **LLM judge (`_llm_monolithic_verdict`) never ran.** Sec. 4.3 /
   Ablation #1 intend the comparison against a *model* judge over the full
   trace — that is the honest baseline AVG must beat, and it is missing from
   results.
3. **One benchmark only.** Sec. 4.1 lists six domains; only Terminal-Bench has
   data. This is what makes the results section thin.

---

## Gaps by proposal section

### Sec. 4.1 Benchmarks — 5 of 6 domains missing
- No adapter/spec/data for **MCPVerse, DAComp, JourneyBench, R2E-Gym/SWE, and
  the enterprise case study**.
- `htir/domains/swe_gym.yaml` spec exists but **no trace loader is wired**
  (`htir/eval/datasets.py` is Terminal-Bench only), so even the "closest" second
  domain is not runnable.
- Consequence: none of the domain-specific stressors the suite was chosen for
  (policy adherence, large action spaces, enterprise data) are exercised.

### Sec. 4.2 Evaluation protocols — none fully realized
- **Zero-shot transfer:** impossible with one domain.
- **Few-artifact adaptation:** SA-2 sweeps an artifact budget, but `Ω_d`
  obligations route to semantic checkers → **offline-inert** (they add coverage
  but abstain). The adaptation gain measured is only the mechanical binding, not
  the artifact-grounded one the protocol targets.
- **Online trace adaptation:** not implemented.
- **The three operating modes** (`Base` / `Base + Online AVG` /
  `Base + Online AVG + Offline Harness Updates`) are **never run end-to-end** —
  there is no live agent, so no mode actually changes task success.

### Sec. 4.3 Baselines — 3 of 6 missing or hollow
- Missing entirely: **SkillOpt / prompt-skill optimization**, and
  **Meta-Harness / Life-Harness** harness-optimization comparators.
- Hollow offline: **monolithic-LLM** (never called), **execution-free** (null
  arm).

### Sec. 4.4 Metrics — task metrics and gold-dependent verifier metrics absent
- **Task metrics** (success rate, partial completion, policy-violation rate,
  rollback rate, cost, wall-clock, tool-budget) are **not measured** — offline
  replay changes nothing, so there is no task-outcome delta. "Cost" is proxied
  by LLM-call *counts*, not real cost/latency.
- **Verifier metrics on weak labels only.** AUROC/ECE (SA-3) are computed against
  **reward-derived weak labels**, not the hand-labeled **100-trace gold slice**
  the plan's P0 called for (`htir/eval/weak_labels.py` explicitly marks gold as
  out of scope). Calibration numbers inherit label noise.
- **Evidence-localization quality is not measured at all** (needs
  step/obligation gold).

### Sec. 4.5 Ablations — 2 of 7 not real arms
- Done: #1 graph/monolith (SA-1), #2 adapters (SA-2, offline-limited), #3
  no-abstention (SA-3), #4 no-integrity (SA-6), #5 no-offline-loop (SA-5).
- **#7 template-free adaptation:** not implemented as an arm (only referenced in
  prose).
- **#6 no-online-loop:** only mentioned in SA-5; not a clean standalone arm.

### Sec. 4.6 Stress tests — 2 of 5
- Done (SA-6): shortcut opportunities, artifact inconsistency.
- Missing/deferred: **policy drift** (needs a policy-bearing domain), **large
  tool menus**, **hidden-state mismatch** (SA-6's `fabricated_final` came back
  null).

### Sec. 4.7 Statistical reporting — under-powered
- **Single seed (seed=0), single 3k subsample, no mean ± standard error, no
  multiple seeds.** The proposal explicitly requires mean/SE over seeds or
  subsamples and cost-normalized curves across matched conditions. Current
  results are point estimates.

---

## Framework-level gaps (vs. Sec. 3)
- **Online intervention & harness improvement are recommendation-only over
  replayed recorded traces** — no live harness loop (documented in
  `docs/avg-mapping.md`). This means Sec. 3.10/3.11's *causal* claims —
  "intervention avoided the failure," "edit improved the agent" — are
  counterfactual/offline, never demonstrated on a live run.
- **`R_W` review recommendation is a deterministic template**, not the
  LLM-authored witness prose Sec. 3.9 implies (minor).

---

## Priority ranking for a robust results section

1. **Turn on `use_llm=True`** and re-run SA-1/SA-3 — cheapest fix that makes
   three things real at once: semantic checkers, the execution-free arm, and the
   monolithic-LLM baseline. Without it the central "graph vs. monolith" claim
   rests on a strawman.
2. **Wire a real second domain** (SWE-Gym/R2E-Gym is closest — spec already
   exists) to unlock Q2 transfer, the zero-shot protocol, and the transfer
   matrix.
3. **Hand-label the 100-trace gold slice** so AUROC/ECE/localization stop
   relying on weak labels.
4. **Multi-seed + standard error** across all six experiments (Sec. 4.7
   compliance).
5. **Add the missing baselines** (SkillOpt/Meta-Harness) and **stress tests**
   (policy drift, large tool menus) — these need the second domain, so they
   follow (2).
