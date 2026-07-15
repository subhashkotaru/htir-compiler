# τ-bench (policy domain) — results & reproduction

Implements the `docs/paper-completion-plan.md` packages that a **second,
policy-bearing domain** unlocks, on Sierra's **τ-bench** (retail + airline
customer-service). Method-preserving: only data (a new domain), seeds, `Ω_d`,
and LLM-enablement wiring were added — no AVG node, edge, checker, obligation
template mechanism, or aggregation rule changed.

**Date:** 2026-07-14.

## What was added

| Piece | File | Plan item |
|---|---|---|
| Deterministic τ-bench adapter (OpenAI-messages → policy ops) | `htir/adapters/tau_bench.py` | E2 |
| Domain spec `S_d` (authenticate/confirm-before-mutation) | `htir/domains/tau_bench.yaml` | E2 |
| `Ω_d` bundle (verbatim retail/airline SOPs + DB schema) | `htir/domains/tau_bench.artifacts/` | E2 |
| Loader + reward extraction + shape-aware canonicaliser | `htir/eval/datasets.py`, `htir/eval/weak_labels.py` | E2 |
| Multi-seed + mean±SE harness | `htir/eval/seeds.py` | E4 / WP-0.1 |
| Cross-domain transfer matrix | `htir/eval/experiment_sa2.py` (`run_transfer_matrix`) | SA-2 / Q2 |
| Policy-domain stress perturbations | `htir/eval/experiment_sa6.py` (`TAU_PERTURBATIONS`) | SA-6 / Sec. 4.6 |
| `--domain` on SA-1/3/6 mains | `htir/eval/experiment_sa{1,3,6}.py` | — |
| Campaign driver | `htir/eval/experiment_tau.py` | Phases 1–3 |
| Tests (15; full suite 117 pass) | `tests/test_tau_bench.py` | — |

**Data:** `AgentSuite/tau-bench-trajectories` (HF) — real OpenAI-messages
trajectories with a DB-match `eval_result.score` reward. 2970 traces cached to
`data/tau_cache/tau_all.jsonl` (1745 solved / 1225 unsolved).

## Results (offline, 3 seeds, n=1000, mean ± SE)

**SA-1 — Q1 graph vs. monolith.** AVG credits a *failed* τ trajectory `valid`
far less than a monolithic endpoint judge blind to the SOP.

| arm | false-valid (overall) | false-valid (long-horizon) | resolved-frac |
|---|---|---|---|
| avg_full / exec_only | **0.045 ± 0.001** | **0.038 ± 0.002** | 0.160 |
| monolithic | 0.678 ± 0.010 | 0.756 ± 0.014 | 0.777 |

→ **15× lower false-valid** (20× on long-horizon). AVG's cost is abstention
(resolved-frac 0.16 offline); the semantic policy checker (LLM slice) is what
converts those abstentions into confident, correct resolutions.

**SA-3 — Q3 calibrated abstention.** Removing abstention (force pass/fail)
balloons the false-valid rate.

| | false-valid | AUROC (shared score) | ECE |
|---|---|---|---|
| avg_full (abstention) | **0.045 ± 0.001** | 0.597 ± 0.009 | 0.216 |
| no_abstention (forced) | 0.793 ± 0.003 | — | — |

→ **−94.3% ± 0.2%** false-valid from the decision rule alone. AUROC ≈ 0.60 is an
honest *weak-calibration* finding (the trajectory-level reward label is a coarse
truth), consistent with the terminal-domain result.

**SA-6 — policy-domain stress tests** (the `policy_drift`, `large_tool_menu`,
`hidden_state_mismatch` perturbations the terminal grid had to defer). Each
appends a reward-hack whose visible endpoint "succeeds".

| perturbation | monolith false-valid | AVG false-valid | AVG catch |
|---|---|---|---|
| policy_drift (unconfirmed mutation) | 1.000 | **0.000** | 1.000 |
| large_tool_menu (exotic action) | 1.000 | **0.000** | 1.000 |
| hidden_state_mismatch (unsupported final) | 1.000 | **0.000** | 1.000 |

Clean-control negative transfer (AVG vetoing a legitimate trace) = 0.064.
**Honest domain-specific finding:** the integrity-module ablation moves ≈0 here
— on a policy domain the reward-hack catch comes from the *policy-linking +
final-answer* layer (a consequential DB mutation is policy-sensitive, so an
unconfirmed/un-linked action is withheld), not the integrity module.

**Transfer matrix — Q2 zero-shot cross-domain** (resolved-fraction; a verifier
specced for the row domain applied to the column domain's traces):

| train ↓ / test → | terminal_swe | tau_bench |
|---|---|---|
| universal_only | 0.000 | 0.000 |
| terminal_swe | **0.138** | 0.000 |
| tau_bench | 0.000 | **0.162** |

→ Only the matched adapter+spec (the diagonal) binds obligations; every
off-domain and universal-only cell abstains. False-valid stays ≤ 0.16
everywhere — a mis-specced verifier *abstains*, it does not over-credit. This is
the direct evidence that per-domain adapters are necessary for transfer.

## Q4 on τ — online intervention (SA-4) + offline harness loop (SA-5)

Both Q4 experiments now run on the policy domain (offline, deterministic),
closing the "Q4 is single-domain" gap (`data/sa4_tau_results.json`,
`data/sa5_tau_results.json`).

- **SA-4 (online intervention, n=800).** The confident `failed_obligation`
  signal is **precise (0.649 vs. 0.50 base) and timely (median 4 steps saved)** —
  *stronger* than on terminal, because a failed *mutation* tool call
  (`tau-mutation-status`, 589 fires) is a real mechanical catch. Active
  obligations concentrate near the trace end (0.9–1.0 → 3.75 active/step).
  Walkthrough: `airline_13` (invalid) flags at step 7 → 23 downstream steps
  saved. `any_active` stays abstention-dominated / near-base-rate precision
  offline, as on terminal.
- **SA-5 (offline harness loop, n=1600).** The loop mines a recurring failure,
  gates it, and accepts one edit; `J_loop` rises (−7.5 → 13.5) while a frozen
  no-loop config degrades. But held-out false-valid was already **0.037**, so the
  (terminal-flavored) mined template yields **0% held-out reduction with zero
  negative transfer** — an honest null: τ false-valids are caught by the policy
  layer, not a hidden-test obligation, so τ-specific remediation templates would
  be needed for a held-out gain.

## Reproduce

```bash
# offline campaign (mean±SE over seeds), writes data/{sa1,sa3,sa6,transfer}_tau_results.json
python -m htir.eval.experiment_tau --experiments sa1,sa3,sa6,transfer --seeds 0,1,2 --n 1000

# single experiment via the standard runner + --domain
python -m htir.eval.experiment_sa1 --cache data/tau_cache/tau_sample_balanced.jsonl --domain tau_bench

# LLM slice (E1) — needs a funded OPENROUTER_API_KEY (see limitation below)
python -m htir.eval.experiment_tau --experiments sa1,sa3 --use-llm --llm-n 120 --model openai/gpt-4o-mini
```

## Dynamic escalation — AVG updating on the fly (Sec. 3.7/3.9 escalation)

Base \AVG checking is a *single static pass*; even with the LLM, each semantic
checker fires once. `htir/agents/escalation.py` adds the escalation the proposal
specifies (`request-evidence`): an abstaining high-severity obligation is
re-checked over a **broadened local window** (pulling in the preceding
authentication / confirmation turns the SOP needs), then the trajectory is
re-aggregated, iterating until it resolves or hits a round budget.
Method-preserving (re-runs the existing checker over a wider slice of the
existing graph + the existing `aggregate`); `use_llm=False` → exact static
verdict (no offline number changes).

Static vs. dynamic on n=40 τ traces (`gpt-4o-mini`):

| arm | false-valid | resolved-frac | abstain | resolved-acc |
|---|---|---|---|---|
| monolithic-LLM | 0.750 | 0.950 | — | 0.553 |
| AVG static | 0.100 | 0.450 | 0.550 | 0.500 |
| **AVG dynamic** | **0.100** | **0.750** | **0.250** | 0.400 |

→ The loop **cuts abstention 0.55 → 0.25 (coverage +30 pts) while holding the
safety-critical false-valid rate flat at 0.100** — it converts abstentions into
decisions without adding false credit (31/31 escalated obligations resolved).
Resolved-accuracy dips (0.50 → 0.40, within noise at n=40) because the
newly-resolved cases include a few *false-invalids* — the cheap judge over-flags
some compliant actions. That is a coverage/precision tradeoff, tunable via
`commit_threshold` (only commit confident flips) and `max_rounds` / `base_window`,
and it would tighten with a stronger judge model and larger n.

## LLM slice (E1) — the monolithic-LLM baseline that "never ran"

Run on n = 120 balanced τ traces with `openai/gpt-4o-mini` as the judge
(`--use-llm`), turning on the semantic policy-compliance checker, the
execution-free arm, and — the headline — the **monolithic-LLM judge** the
gap-analysis flagged as missing. Stored in the `llm_slice` field of
`data/sa1_tau_results.json` / `data/sa3_tau_results.json`.

**SA-1 with the LLM judge.**

| arm | false-valid (overall) | false-valid (long-horizon) | resolved-frac |
|---|---|---|---|
| avg_full | **0.050** | **0.027** | 0.158 |
| monolithic-**LLM** | 0.733 | 0.838 | 0.942 |

→ AVG beats a *real LLM judge over the whole trace* by **~15× (31× on
long-horizon)**. The LLM monolith is in fact slightly *worse* than the endpoint
heuristic (0.733 vs 0.678): a holistic judge is fooled by a trajectory that
"looks" completed but violated the SOP — the exact failure evidence-local
verification is built to catch.

**SA-3 with the LLM.** False-valid reduction from the forced arm holds:
**−93.8%** (0.800 → 0.050). Abstention *improves* calibration — the
abstention-calibrated AUROC is **0.688** for `avg_full` vs **0.558** for
`no_abstention` over the same shared score.

**Honest finding on the semantic checker.** The narrow semantic policy checker
now resolves individual per-mutation obligations (the checker was completed to
pass the *action under review* alongside the policy text, per the Sec. 3.7
local-context contract — gated behind `--use-llm`, so no offline number
changed), but confident passes are sparse, so it rarely flips the *trajectory*
verdict: `avg_full ≈ exec_only` and `exec_free` stays mostly abstaining. This is
the same "semantic adds coverage, seldom flips offline verdicts" result seen on
Terminal-Bench — reported, not hidden. The LLM slice is single-seed (n=120); the
offline tables above are the multi-seed±SE headline.
