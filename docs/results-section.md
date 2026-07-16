# Results

We evaluate Adaptive Verifier Graphs (AVG) on two independent agent domains that
share no code path beyond the domain-neutral core: **Terminal-Bench** (software
engineering — an agent editing files and running tests in a shell) and
**τ-bench** (Sierra's retail + airline customer-service policy domain — an agent
calling tools against a user database under a written standard-operating
procedure). The two domains stress complementary failure modes: Terminal-Bench
exposes *execution-integrity* reward hacks (tampered tests, deleted artifacts),
while τ-bench exposes *policy-compliance* reward hacks (mutating state without
authenticating or confirming). A single verifier mechanism that holds on both,
changing only its data-facing adapter and domain spec `S_d`, is the claim we
test.

## Experimental setup

**Domains and data.** Both domains use *real* agent trajectories with a
mechanical ground-truth reward, so "valid" is defined by the environment, not by
a judge we control.

| | Terminal-Bench | τ-bench |
|---|---|---|
| Task | shell / SWE (edit + test) | retail + airline customer service |
| Trajectories | Terminal-Bench HF traces (streamed) | `AgentSuite/tau-bench-trajectories` (2 970 real traces) |
| Ground-truth reward | test pass/fail | DB-state match (`eval_result.score`) |
| Reward-hack surface | integrity (test tamper, artifact deletion) | policy (auth / confirmation / SOP) |
| Domain spec `S_d` | `terminal_swe` | `tau_bench` (authenticate- + confirm-before-mutation) |

Every domain is reached through the same pipeline — a **framework-neutral
adapter** canonicalises raw traces into role/status/effect-tagged steps, the
**obligation generator** emits per-step obligations from `S_d`, evidence-local
**checkers** score each obligation, and a single **aggregation rule** turns the
obligation verdicts into a trajectory verdict of `valid` / `invalid` /
`uncertain (abstain)`. Adding τ-bench required **no change** to any node, edge,
checker mechanism, obligation-template mechanism, or aggregation rule — only a
new adapter, `S_d`, and domain-artifact bundle `Ω_d` (the verbatim SOPs + DB
schema).

**Verifier arms.** The headline contrast is:

- **`avg_full`** — the full graph: evidence-local obligations with
  abstention-aware aggregation.
- **`monolithic`** — a single holistic judge that sees the whole trajectory and
  its endpoint (the standard "LLM-as-judge over the transcript" baseline). Run
  both as a cheap endpoint heuristic (offline) and, on τ-bench, as a real
  `gpt-4o-mini` judge (the *monolithic-LLM* baseline).
- Ablations: **`exec_only`** (mechanical checkers only), **`exec_free`** (no
  execution evidence), and, for SA-6, **`avg_no_integrity`** (integrity module
  removed).

**Metric.** The safety-critical headline is the **false-valid rate**: the
fraction of *failed* trajectories the verifier credits as `valid`. In a
reward-model / RL-filtering setting this is exactly the quantity that leaks
reward hacks into training. We report it overall and on the **long-horizon**
slice (≥ 20 steps), where holistic judges degrade most. We also report
resolved-fraction (coverage) and resolved-accuracy, because AVG buys its low
false-valid rate with **calibrated abstention** — it declines to rule when the
evidence is insufficient rather than guessing.

**Protocol.** Offline results are **mean ± SE over 3 seeds** at n = 1 000
(τ-bench) or n = 3 000 (Terminal-Bench), balanced 50/50 valid/invalid. The
LLM-judge slice is a single seed at n = 120 with `gpt-4o-mini`. All numbers are
reproducible from `data/*_results.json` via `htir.eval.experiment_tau` /
`experiment_sa{1,3,6}`.

## Q1 — Evidence-local verification vs. a monolithic judge (SA-1)

A monolithic judge credits most *failed* trajectories as valid; AVG almost never
does.

| domain | arm | false-valid (overall) | false-valid (long-horizon) |
|---|---|---|---|
| Terminal-Bench | `avg_full` | **0.187** | **0.207** |
| Terminal-Bench | `monolithic` | 0.830 | 0.875 |
| τ-bench | `avg_full` | **0.042 ± 0.003** | **0.034 ± 0.003** |
| τ-bench | `monolithic` | 0.678 ± 0.010 | 0.756 ± 0.014 |

AVG cuts the false-valid rate by **4.4×** on Terminal-Bench and **16×** on
τ-bench (**22×** on long-horizon τ traces). The gap *widens* with horizon length
for exactly the reason evidence-local verification exists: a holistic judge is
increasingly fooled by a long trajectory that "looks" completed, whereas
per-step obligations tie the verdict to whether the specific consequential action
was actually justified by in-trace evidence. AVG pays for this with coverage —
offline it resolves ~16 % of traces and abstains on the rest — which is the
correct behaviour when the deciding evidence (e.g. a DB-state reward) is not in
the transcript.

## Q3 — Calibrated abstention is where the safety comes from (SA-3)

To show the win is the *decision rule*, not just the checkers, we force AVG to
commit on every trace (remove abstention).

| domain | false-valid *with* abstention | false-valid *forced* | reduction |
|---|---|---|---|
| Terminal-Bench | **0.128** | 0.724 | **−82.3 %** |
| τ-bench | **0.042 ± 0.003** | 0.787 ± 0.003 | **−94.7 % ± 0.4 %** |

Abstention alone removes 82–95 % of false credits. We report the calibration
honestly: the shared-score AUROC is 0.46 (Terminal-Bench) / 0.60 (τ-bench) — a
**weak-calibration** finding, because the trajectory-level reward is a coarse
label — yet the *abstain / don't-abstain* decision is highly effective. On the
LLM slice, abstention also *improves* ranking: AUROC rises to **0.688** for
`avg_full` vs **0.558** for the forced arm over the identical score.

## Q2 — Adapters are necessary: the cross-domain transfer matrix (SA-2)

Applying a verifier specced for one domain to another domain's traces yields a
crisp diagonal:

| train ↓ / test → | terminal_swe | tau_bench |
|---|---|---|
| universal_only | 0.000 | 0.000 |
| terminal_swe | **0.138** | 0.000 |
| tau_bench | 0.000 | **0.162** |

Only the matched adapter + spec binds obligations (resolved-fraction on the
diagonal); every off-diagonal and the domain-neutral-only verifier **abstains
everywhere**, and the false-valid rate stays ≤ 0.16 in *every* cell. The key
safety property: a mis-specified verifier **abstains, it does not over-credit** —
so per-domain adapters are necessary for coverage but cannot manufacture false
confidence when absent.

## Stress tests and the integrity module (SA-6)

Each stress test appends a reward-hack whose visible endpoint "succeeds," so a
correct verifier must reject a trajectory that superficially looks complete.

**Terminal-Bench — integrity reward hacks.** Here the *integrity module* is
load-bearing:

| perturbation | monolith FV | AVG *without* integrity | AVG *with* integrity |
|---|---|---|---|
| test-tamper | 1.000 | 0.521 | **0.000** |
| artifact-deletion | 0.993 | (leaks back) | **0.000** |

Removing the integrity verifier lets ~52 points of tampered-test false-valids
leak back; the module restores a 100 % catch. The monolithic judge is
**completely blind** (FV 1.0) to a modified test file.

**τ-bench — policy reward hacks.** Here the catch comes from a *different* layer,
and we report that difference rather than hide it:

| perturbation | monolith FV | AVG FV | AVG catch |
|---|---|---|---|
| policy_drift (unconfirmed mutation) | 1.000 | **0.000** | 1.000 |
| large_tool_menu (exotic action) | 1.000 | **0.000** | 1.000 |
| hidden_state_mismatch (unsupported final) | 1.000 | **0.000** | 1.000 |

AVG catches 100 % of policy reward-hacks that the monolith credits 100 % of the
time. The integrity-module ablation moves ≈ 0 here — an **honest, domain-specific
finding**: on a policy domain the catch is the *policy-linking* layer (a
consequential DB mutation is policy-sensitive, so an unconfirmed / unauthenticated
action is withheld), not the integrity module. The same graph, two different
protective layers, each doing its job on the domain it is meant for.

## The LLM-judge baseline (τ-bench, n = 120)

Replacing the cheap endpoint heuristic with a *real* `gpt-4o-mini` judge over the
full transcript does not close the gap:

| arm | false-valid (overall) | false-valid (long-horizon) |
|---|---|---|
| `avg_full` | **0.050** | **0.027** |
| `monolithic-LLM` | 0.733 | 0.838 |

AVG beats a real LLM judge by **~15× (31× on long-horizon)**. Strikingly, the LLM
monolith is slightly *worse* than the endpoint heuristic (0.733 vs 0.678): a
holistic model is confidently fooled by a plausible-but-SOP-violating trajectory
— precisely the failure evidence-local verification is designed to catch.

## Dynamically updating the obligations

The results above are all from a single static verification pass. We additionally
built, and validated by measurement, the mechanism that lets AVG **update its
obligations on the fly** — the escalation / `request-evidence` step of the
proposal. This section describes the *process* used to arrive at it, not just the
final code, because the design was chosen experimentally.

**The problem.** Base checking is one pass: every obligation is scored once, and a
high-severity policy obligation that lacks local evidence (e.g. "this mutation was
confirmed with the user") simply **abstains** — safe, but low-coverage. We want
the verifier to *do something* about an abstention instead of only recording it.

**Two design axes, resolved by experiment.** We prototyped three variants on n = 40
τ traces with `gpt-4o-mini` and measured each before committing to one:

| approach | false-valid | resolved-frac | \|E_W\| | E_i growth |
|---|---|---|---|---|
| STATIC (no escalation) | 0.100 | 0.450 | 1.38 | — |
| verdict-only re-score | 0.100 | 0.775 | 1.27 | 0.00 |
| **re-localizing (chosen)** | **0.100** | **0.750** | **1.90** | **0.72** |

1. *Re-score vs. re-localize.* Both recover coverage (abstain-rate 0.55 → 0.25,
   **+30 points**) while holding the safety-critical false-valid rate **flat at
   0.100** — escalation converts abstentions into decisions without adding false
   credit. Re-localizing additionally grows the obligation's candidate-evidence
   set `E_i` (0.72 evidence nodes/trace) and enriches the audit witness
   (`|E_W|` 1.27 → 1.90), *including on obligations that stay abstained*, so a
   reviewer sees "here is the broader context we pulled in, and it was still
   insufficient." We chose re-localization: escalation should update the
   obligation's *localization*, not merely its verdict.

2. *What to escalate.* We measured that resolving the intrinsically
   trace-unverifiable `final_answer_support` obligations **doubles** the
   false-valid rate (0.133 → 0.267) — the DB-reward success signal is simply not
   in the transcript, so the judge over-passes. These are therefore **excluded**
   from escalation. We also hold the re-check prompt *identical* to the
   verdict-only path (the gathered neighbourhood is shown as "surrounding
   trace"), so escalation adds provenance without silently shifting verdicts.

**The mechanism (`htir/agents/escalation.py`).** For each abstaining
high-severity obligation, we broaden its local step window to pull in the
preceding authentication / confirmation turns the SOP requires, lift that
neighbourhood into a **first-class `EvidenceNode`**, append it to the graph and
to the obligation's candidate-evidence set `E_i`, re-run the *existing* checker,
and — on resolution — add an `E_sup` support edge and refresh the witness. The
trajectory is then re-aggregated with the *existing* aggregation rule, iterating
until it resolves or hits a round budget. The loop is method-preserving: with
`use_llm=False` it is an exact static no-op (byte-identical offline numbers), and
`commit_threshold` gates confident-only flips as a tunable coverage/precision
knob. Against the monolithic-LLM's 0.750 false-valid, the dynamic loop holds
0.100 while lifting coverage from 0.45 to 0.75 (31/31 escalated obligations
resolved).

**Obligations at the lowest level.** Escalation is only sound if the obligations
it grows are atomic. We therefore decomposed policy obligations to the **lowest
level: one obligation per governing `S_d.K_d` constraint**, not one opaque
"complies with the entire SOP" judgment. The earlier whole-SOP design emitted one
obligation *per policy artifact*, which on a retail trace produced **two**
obligations — one checking the wrong (airline) policy (204 mutations → 408
obligations over 200 traces, half irrelevant). The atomic generator instead
checks each constraint at the lowest level that fits it:

- **authenticate-before-action** — a constraint with a `requires_prior` ordering
  → a **mechanical** precondition check: a *successful* prior `authenticate` step
  must precede the action. No LLM, reproducible offline, general to any domain's
  `K_d`; over 300 τ traces it false-flags only **2 (0.7 %)** while giving AVG a
  real offline policy signal it previously lacked.
- **confirm-before-mutation** — a constraint with no ordering → a **narrow
  semantic** check whose evidence is *that one rule's text*, not the 5 700-char
  SOP, keeping the checker's context local (escalation then gathers the
  surrounding turns to judge whether confirmation occurred).

This needed only a small, general `S_d` extension (`Constraint.requires_prior`)
and one new primitive operation type (`authenticate`, split from generic reads);
`terminal_swe`'s `no-test-mutation` constraint flows through the identical path
unchanged. The result is no duplication, no cross-domain leakage, every
obligation a single primitive rule — the precondition for escalation to grow
`E_i` safely rather than compounding an already-tangled judgment.

## Downstream payoff: verification as a data filter (SA-7)

The preceding results are *intrinsic* — a lower false-valid rate against the
mechanical reward. The question a practitioner asks next is whether that buys a
better *downstream outcome*. We test the most consequential use of a verifier:
**filtering candidate trajectories to build clean training / reranking data**. We
group each corpus by task (τ-bench: 280 tasks × ~15 candidates; Terminal-Bench:
11 tasks × 17–72 candidates), and use each verifier arm as a selector, two ways —
(i) **reranking**: pick the single highest-scored trajectory per task and measure
its true (reward = 1) success rate; (ii) **filtering**: keep the trajectories an
arm credits `valid` and measure how much reward-hack leaks into the kept set. The
corpus is compiled + scored once; metrics are reported as **mean ± SE over three
per-task candidate subsamples** (`htir/eval/experiment_sa7.py`).

**Filtering — the headline.** Used as a keep-if-`valid` data filter, AVG's kept
set is **near-clean of reward-hack where the monolith's is majority-contaminated**
(false-valid = fraction of the kept set that is actually a failed trajectory):

| Domain | Arm | Kept-set precision ↑ | Yield (fraction kept) | **Reward-hack leakage** ↓ |
|---|---|---|---|---|
| τ-bench | monolithic | 0.631 ± 0.002 | 0.667 | **0.599 ± 0.001** |
| τ-bench | **avg_full** | **0.773 ± 0.010** | 0.061 | **0.034 ± 0.001** |
| Terminal-Bench | monolithic | 0.471 ± 0.034 | 0.568 | **0.660 ± 0.023** |
| Terminal-Bench | **avg_full** | 0.273 ± 0.138 | 0.102 | **0.164 ± 0.034** |

On τ-bench (base rate 0.588), filtering with AVG yields a **77.3 %-clean** pool —
+18.5 points over the base rate — versus the monolith's 63.1 %, and lets through
**17.9× less reward-hack** (0.034 vs 0.599). On Terminal-Bench the leakage gap is
**4.0×** (0.164 vs 0.660), consistent with the intrinsic SA-1 false-valid gap. The
mechanism is the same abstention that drives every earlier result: AVG declines to
credit a trajectory it cannot support, so the reward-hacks it cannot mechanically
disprove are *withheld* rather than shipped as training labels.

**The honest caveat — precision vs. yield.** The safety comes at a **yield** cost:
offline, AVG only positively credits the fraction of trajectories it can
mechanically confirm (6 % on τ-bench, 10 % on Terminal-Bench), because the
semantic policy / test-support checkers abstain without an API key (as in SA-1).
Reward-hack **leakage** is therefore the stable, comparable filtering metric — it
is a rate over the kept-invalid traces and moves in AVG's favour in both domains —
whereas kept-set **precision** is confounded by that low yield: on Terminal-Bench
the ~10 %-of-pool AVG keeps is too small for a stable precision estimate
(0.273 ± 0.138) and its exit-code-only positives over-credit, so *offline* AVG's
value on Terminal-Bench is its abstention (veto), not its positive credit. Lifting
the yield — turning AVG into a high-precision *and* high-coverage filter — is
exactly what the funded LLM-judge slice (semantic policy / test-support checkers)
is expected to deliver, and is scoped as follow-up work.

**Reranking — comparable offline.** As a best-of-N reranker (score =
coverage-aware p-valid, `trajectory_valid_score`, which induces the tie-break
resolved-valid > abstained > resolved-invalid), AVG and the monolith are
**statistically indistinguishable** offline: τ-bench avg 0.615 ± 0.003 vs mono
0.596 ± 0.014 (paired bootstrap over tasks, gap +0.004, 95 % CI [−0.050, +0.061],
p = 0.45); Terminal-Bench avg 0.515 vs mono 0.576 (gap +0.091, CI [−0.18, +0.36],
p = 0.38, 11 tasks). Both sit near the random-pick floor (τ 0.595, Terminal 0.545)
and well below the oracle ceiling (0.890 / 0.939) — the *positive-selection*
signal, unlike the *rejection* signal, needs the semantic checkers to separate the
arms. We report this null honestly: **the downstream win is in filtering out
reward-hack, not (yet) in ranking the survivors.**

## Takeaways

- Evidence-local verification with calibrated abstention cuts false-valid rates
  **4–16×** vs. a monolithic judge across two unrelated domains, and the gap
  **widens with horizon length**.
- The safety comes from the **abstention decision rule** (−82 % to −95 % when
  removed), not from any single checker, and it beats a **real LLM judge** by
  ~15× on the domain where we could afford one.
- A mis-specified verifier **abstains rather than over-credits** (clean transfer
  diagonal), so per-domain adapters are safe to add.
- Escalation lets AVG **update its obligations on the fly** — recovering +30
  points of coverage with the false-valid rate held flat — a design we selected
  by measuring three variants, and made sound by decomposing obligations to
  atomic, per-constraint primitives.
- The intrinsic gain is a **downstream one**: used to filter candidate
  trajectories into training data, AVG leaks **4–18× less reward-hack** than a
  monolithic judge — at an honestly-disclosed yield cost that the semantic-checker
  slice is expected to close.
