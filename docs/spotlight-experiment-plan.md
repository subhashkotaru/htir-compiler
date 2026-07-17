# AVG Spotlight Experiment Plan

Implement the following:

The work remaining to lift the AVG experiments section to an ICLR-spotlight bar,
packaged so **one sub-agent can pick one package and work independently**. This
is the *forward* plan; the original build-out (SA-1…SA-6) lives in
`docs/experiment-plan.md` and its numbers are in `docs/results-section.md`.
Narrative rationale (why these, why not more breadth) is in
`docs/spotlight-plan-artifact.html`.

Each package below is self-contained: claim, arms, data, reuse, new code,
metrics, deliverable, run command, done-criteria, and external blockers. Pick
one, read only its section plus **§0 Conventions**, and go.

---

## §0 Conventions — read this once, applies to every package

**Environment.** Activate the project venv: `source ~/.venv/bin/activate` (has
`pytest` + `pydantic`; `~/avir_env` also works). The deterministic pipeline needs
**no API key**. LLM arms need an OpenRouter/OpenAI key in the environment and the
`[llm]` extra; without a key they must degrade gracefully (skip, don't crash).

**Reuse before you build — the infra already exists.** Do **not** re-implement
these; import them:

| Need | Use | Location |
|---|---|---|
| Verifier arms (avg_full / exec_only / exec_free / monolithic) | `VerifierArm`, `run_arm`, `run_all_arms` | `htir/agents/baselines.py` |
| Trajectory verdict for a trace | `TraceAbstractionAgent.compile(run_checks=True)` | `htir/agents/trace_abstraction.py` |
| Headline metric | `VerifierMetrics` (`false_valid_rate`, `resolved_accuracy`, `resolved_fraction`), `evaluate_predictions` | `htir/eval/weak_labels.py` |
| Weak labels from reward | `label_from_reward`, `extract_reward`, `trace_label` | `htir/eval/weak_labels.py` |
| Calibration / frontier | `roc_auc`, `reliability_bins`, `expected_calibration_error`, `risk_coverage_curve`, `trajectory_valid_score` | `htir/eval/calibration.py` |
| Data loading + balancing | `load_terminalbench`, `load_tau_bench`, `balanced_sample`, `iter_local_traces`, `to_canonical_steps`, `normalize_tau_record` | `htir/eval/datasets.py` |
| Multi-seed helpers | `htir/eval/seeds.py` | `htir/eval/seeds.py` |
| Domain specs `S_d` | `default.yaml`, `terminal_swe.yaml`, `tau_bench.yaml`, `swe_gym.yaml` | `htir/domains/` |
| Adapters | `terminal.py`, `tau_bench.py`, generic `turns`/`raw` | `htir/adapters/` |

**Experiment file pattern.** Copy the shape of `htir/eval/experiment_sa1.py`:
a `run_saN(...)` function returning a pydantic result model, a `format_table(...)`,
an `argparse` `main(argv)` with `--out` writing the result JSON, and a
`if __name__ == "__main__"` guard. Runnable as
`python -m htir.eval.experiment_saN`.

**Output contract — every package delivers three things:**
1. **Result JSON** written to `data/saN[_tau]_results.json` (reproducible).
2. **A results paragraph** appended to `docs/results-section.md` in the same
   voice as the existing sections (claim → table → honest caveat).
3. **A regression test** in `tests/` that runs the package offline (`use_llm=False`)
   on a tiny fixture and asserts the headline number is in range. Offline path
   must be byte-deterministic.

**Rigor bar (non-negotiable for headline numbers).** ≥3 seeds, report
**mean ± SE**, and a significance statement on the key gap (paired bootstrap or
t-test over per-seed values). Matched model / token / cost budget across arms —
and disclose it. Terminal-Bench is currently single-seed; any package touching it
adds seeds.

**Scope guardrail.** These packages are the agreed critical path. Adding *more*
domains or *more* judge variants beyond what a package specifies is explicitly
out of scope (see the artifact §7) — depth over breadth.

---

## Package summary

| ID | Question / deliverable | Priority | Type | Blocking deps | Shared files (coordinate) |
|----|------------------------|----------|------|---------------|---------------------------|
| **SA-7** | Downstream payoff: best-of-N reranking / filtering | **P0** | new file | none | — |
| **SA-8** | Competitive baselines: PRM + Agent-as-a-Judge arms | **P0** | extends | none (offline PRM) / key (judge) | `baselines.py`, `experiment_sa1.py` |
| **SA-9** | Scale the LLM-judge slice (n≥500 × 3 seeds) | **P0** | extends | **funded LLM key** | `experiment_tau.py` |
| **SA-10** | Third domain (SWE-Gym) → 3×3 transfer | P1 | extends | none | `datasets.py`, `experiment_sa2.py` |
| **SA-11** | Selective-verification frontier (Fig 2) + calibration reframe | P1 | new file | none | `results-section.md` |
| **SA-12** | Human-review efficiency of the witness | P1 | new file + protocol | human raters | — |
| **SA-13** | Cost curves + robustness sweep (appendix) | P2 | extends | none | — |

Two P0 packages (SA-8, SA-9) and SA-10 touch shared files — see **§Coordination**
at the end before editing them.

---

## SA-7 — Downstream payoff: best-of-N reranking / filtering  **(P0, highest leverage)**

**Claim (thesis clause 5, Table 5).** Lower false-valid rate is not just an
intrinsic metric — using AVG as a selector produces a measurably better outcome
than using a monolithic judge or a PRM. This is the single item that moves the
paper from "better verifier" to spotlight.

**Method.**
- **Reranking (primary).** Group the corpus by task so each task has N candidate
  trajectories with mixed ground-truth reward. For each selector arm
  (`avg_full`, `monolithic`, and the SA-8 `prm` arm if available), pick the
  best-scored trajectory per task; report the **true success rate of the picks**.
  AVG's tie-break under abstention: prefer a resolved-valid over an abstained
  over a resolved-invalid (document the rule).
- **Filtering (stronger variant).** Filter the pool to traces each arm credits
  `valid`; report the filtered-in set's **true-success rate** (this is exactly
  "how much reward-hack leaks into training data"). Report yield (fraction kept)
  alongside, so a high-precision/low-yield arm isn't flattered.

**Data.** τ-bench `AgentSuite/tau-bench-trajectories` grouped by `task_id`
(strongest, has real per-task candidates) and the Terminal-Bench sample grouped
by `task_name`. Reuse `datasets.balanced_sample` / `load_tau_bench`. Need ≥ a few
candidates per task — filter to tasks meeting a min-N.

**Reuse.** `run_arm` for scores/verdicts, `weak_labels.trace_label` for
ground truth, `datasets` loaders.

**New code.** `htir/eval/experiment_sa7.py` (self-contained), `data/sa7_results.json`,
`data/sa7_tau_results.json`, `tests/test_experiment_sa7.py`.

**Metrics.** Selected-pick success rate per arm; filtering precision + yield;
Δ vs. monolith with mean±SE over seeds and a significance test.

**Deliverable.** Table 5 (selector arm × domain → downstream success) + a
results-section paragraph. If the reranking gain is small, the filtering-precision
result is the fallback headline — report whichever is real, honestly.

**Run.** `python -m htir.eval.experiment_sa7 --domain tau_bench --n 1000 --min-candidates 3 --seeds 3 --out data/sa7_tau_results.json`

**Done when.** Table 5 exists with ≥3 seeds, a significance statement, and a
paragraph; offline (heuristic-monolith) path is deterministic and tested.

**Feasibility.** Offline arms fully runnable now. The `prm` selector depends on
SA-8; run avg_full vs. monolithic first and add the PRM column when SA-8 lands.

---

## SA-8 — Competitive baselines: PRM + Agent-as-a-Judge  **(P0)**

**Claim (Table 1, Fig 1).** AVG beats not just a strawman monolith but the two
baseline *categories* the 2026 field will demand: a process reward model and an
evidence-gathering agent-judge.

**Method — add two arms to the existing arm framework.**
- **`prm` arm** — a process/step reward model that scores each step and
  aggregates to a trajectory verdict (threshold on min/mean step score). Prefer
  an **offline** realization first (an open PRM checkpoint via `transformers`, or
  a deterministic step-heuristic PRM) so it runs without a key; an LLM step-scorer
  is the richer version behind `--use-llm`.
- **`agent_judge` arm** — an Agent-as-a-Judge style judge that may *gather*
  evidence from the trace (multi-hop over steps/artifacts) before committing,
  vs. the one-shot `monolithic` judge. LLM-backed; degrade gracefully without a key.

**Reuse / seam.** Extend `VerifierArm` in `htir/agents/baselines.py` with
`PRM = "prm"` and `AGENT_JUDGE = "agent_judge"`; add their branches in `run_arm`
(mirror how `MONOLITHIC` is special-cased at `baselines.py:94`). Wire both into
`experiment_sa1.py`'s arm loop and `format_table`. **Do not modify existing arm
behavior** — additive only.

**New code.** `htir/agents/baselines.py` (+2 arms + judge/PRM helpers),
`experiment_sa1.py` (arm list + table), `tests/test_baselines.py` (offline PRM
verdict on a fixture). Optionally a small `htir/agents/prm.py` if the PRM logic is
sizeable.

**Metrics.** Same as SA-1 — false_valid overall + long-horizon, resolved_frac,
resolved_acc, cost — now with `prm` and `agent_judge` rows. Matched token budget
between `agent_judge`, `monolithic`, and AVG's semantic checker.

**Deliverable.** Extended Table 1 + the Fig 1 teaser arms + a paragraph noting how
each competitor fails (PRM over-commits on weak-label steps; agent-judge still
fooled by plausible-but-invalid long traces).

**Run.** `python -m htir.eval.experiment_sa1 --cache <sample> --out data/sa1_results.json`
(offline PRM); add `--use-llm` for `agent_judge`.

**Done when.** Both arms appear in Table 1 for both domains, offline PRM is
tested/deterministic, and the agent-judge arm runs when a key is present.

**Feasibility.** Offline PRM: runnable now. Agent-judge + LLM-PRM: need a key
(same blocker as SA-9). Deliver the offline half first.

---

## SA-9 — Scale the LLM-judge slice  **(P0, external blocker)**

**Claim.** The "AVG beats a real `gpt-4o-mini` judge ~15×" headline currently
rests on n=120 single-seed (τ-bench). Scale so it survives a rigor challenge.

**Method.** Re-run the existing LLM-monolith comparison (`experiment_tau.py`
LLM slice) at **n ≥ 500 × 3 seeds** on τ-bench; add Terminal-Bench LLM slice if
budget allows. No new mechanism — this is a scale + statistics pass.

**Reuse.** `experiment_tau.py` already has the LLM slice and the
`monolithic-LLM` arm; add seed looping (`htir/eval/seeds.py`) and mean±SE
aggregation. `data/tau_cache/` caches responses to cut cost on re-runs.

**New code.** Edits to `experiment_tau.py` (seed loop, SE aggregation),
refreshed `data/sa*_tau_results.json`, updated paragraph in `results-section.md`.

**Metrics.** false_valid overall + long-horizon, mean±SE over 3 seeds,
significance on the AVG-vs-LLM-judge gap; report total token cost.

**Done when.** The 15× (and 31× long-horizon) headline is restated at n≥500 with
error bars and a significance statement, or revised to the true value.

**Feasibility. Blocked on a funded LLM key** — per project memory the OpenRouter
key is unfunded, which is why this slice is small. Code path is ready; this
package is "fund key → run → aggregate." Flag to the human owner; do the
aggregation/seed plumbing now so it's a one-command run once funded.

---

## SA-10 — Third domain (SWE-Gym) → 3×3 transfer  **(P1)**

**Claim (Table 3).** A 3×3 transfer diagonal is far more convincing than 2×2 for
"the matched adapter is necessary; a mis-specified verifier abstains, it does not
over-credit."

**Method.** Add SWE-Gym as the third domain via the **existing** terminal-shaped
path — `htir/domains/swe_gym.yaml` already exists. Provide a dataset loader
(`load_swe_gym` in `datasets.py`, mirroring `load_terminalbench`) mapping SWE-Gym
traces to canonical steps + reward labels, then run the SA-2 transfer matrix over
{`terminal_swe`, `tau_bench`, `swe_gym`} × the same test domains, plus
`universal_only`.

**Reuse.** `swe_gym.yaml`, the terminal adapter (`adapters/terminal.py` — SWE-Gym
traces are terminal-shaped), `experiment_sa2.py` matrix runner, `balanced_sample`.

**New code.** `datasets.load_swe_gym` (+ HF schema handling; lazy import),
`experiment_sa2.py` (add the third domain to the grid),
`data/sa2_results.json` refresh, `tests/` loader test on a fixture.

**Metrics.** Per-cell resolved_fraction (binds only on the diagonal) and
false_valid ≤ small in every cell — the transfer safety property.

**Deliverable.** 3×3 transfer matrix (Table 3) + paragraph.

**Done when.** The matrix has three real domains, off-diagonal + universal_only
abstain everywhere, and no cell over-credits.

**Feasibility.** Needs the SWE-Gym trace source (HF) and a confirmed schema —
the one real unknown. If the schema doesn't map cleanly to the terminal adapter,
scope to a 2-domain-train × 3-domain-test row and note the loader as the gate.

---

## SA-11 — Selective-verification frontier (Fig 2) + calibration reframe  **(P1)**

**Claim (Fig 2).** The ~16% coverage / 84% abstain rate is a *knob*, not a
weakness: sweeping the abstention threshold traces a false-valid-vs-coverage
frontier on which AVG dominates the judges — including at **matched coverage**.
This is also the honest home for the calibration story (AUROC 0.46/0.60 is a
coarse-label artifact; the abstain decision is what's calibrated).

**Method.** Sweep AVG's decision threshold over `trajectory_valid_score`; at each
operating point compute (coverage, false_valid) and overlay each baseline arm as
a point/curve. Add a matched-coverage comparison: fix coverage to each baseline's
operating point and read AVG's false_valid there.

**Reuse — mostly a re-plot.** `calibration.risk_coverage_curve`,
`trajectory_valid_score`, `roc_auc`, `expected_calibration_error`,
`reliability_bins` already exist. Use SA-1 / SA-3 result data.

**New code.** `htir/eval/experiment_sa11.py` (frontier extraction + a matplotlib
figure written to `docs/` or `data/`), `data/sa11_results.json`,
`tests/test_experiment_sa11.py`. Rewrite the SA-3 calibration paragraph in
`results-section.md` to lead with the frontier.

**Metrics.** Frontier points per arm; AVG false_valid at matched coverage vs.
each baseline; ECE + reliability diagram in an appendix.

**Deliverable.** Fig 2 (the coverage answer) + reframed calibration paragraph.

**Done when.** Fig 2 shows AVG on/above the frontier at matched coverage for both
domains, and the calibration text no longer reads as a liability.

**Feasibility.** Fully runnable now from existing result JSON — lowest-risk P1.

---

## SA-12 — Human-review efficiency of the witness  **(P1)**

**Claim (intro promise: "witness compresses the trace").** Reviewers reach a
correct verdict faster and/or more accurately from the verification witness
`W_τ` than from the raw trace.

**Method.** Small controlled study, ~50 traces balanced valid/invalid, each shown
in two conditions (raw trace vs. witness `W_τ`) to raters (counterbalanced).
Measure **time-to-verdict** and **verdict accuracy** vs. ground-truth reward.
Even n≈50 with 2–3 raters lands the claim if effect size is clear.

**Reuse.** `VerificationWitness` / `witness.py` to render `W_τ`; existing traces
+ `trace_label` for ground truth.

**New code.** `htir/eval/experiment_sa12.py` that exports the two-condition review
packets (HTML/JSON), a scoring script that ingests rater responses →
`data/sa12_results.json`, and a short protocol note in `docs/`.

**Metrics.** Median time-to-verdict (raw vs. witness), accuracy, inter-rater
agreement; report as paired differences with a significance test.

**Deliverable.** A human-efficiency table/figure + paragraph tied to the witness
contribution.

**Done when.** The two-condition packets generate deterministically and the
scoring script produces the table from rater CSVs.

**Feasibility.** Code/harness runnable now; the **human raters** are the external
dependency — build the harness so the study is a fill-in-the-CSV once raters are
available. Keep it honest about n.

---

## SA-13 — Cost curves + robustness sweep  **(P2, appendix)**

**Claim.** Sec 4.7 already promises cost-normalized curves; deliver them, plus a
small robustness grid, so the average-vs-reliability story is complete.

**Method.**
- **Cost curves.** Re-emit SA-1/SA-8 arms with per-arm token/compute cost →
  false_valid-vs-cost curve (uses `_llm_calls_for_arm` already in
  `experiment_sa1.py`).
- **Robustness sweep.** Extend SA-6's perturbation harness with noisy-log
  injection and tool-schema shift; report false_valid delta per perturbation.

**Reuse.** `experiment_sa1.py` cost accounting, `experiment_sa6.py` perturbation
harness.

**New code.** Small additions to SA-1/SA-6 or a thin `experiment_sa13.py`;
`data/sa13_results.json`; appendix figures.

**Done when.** A cost-normalized curve and a noisy-log/schema-shift grid exist for
the appendix.

**Feasibility.** Fully offline; do last.

---

## Coordination — shared-file seams (avoid collisions between agents)

- **`htir/agents/baselines.py`** — SA-8 adds `VerifierArm.PRM` / `.AGENT_JUDGE`.
  Any other agent touching arms must rebase on SA-8's enum. Additive only; never
  change existing arm flags in `_ARM_FLAGS`.
- **`htir/eval/experiment_sa1.py`** — SA-8 (new arms in the loop + table) and
  SA-13 (cost curve). If run concurrently, split: SA-8 owns the arm list,
  SA-13 owns the cost-emit path.
- **`htir/eval/datasets.py`** — SA-10 adds `load_swe_gym`; independent of SA-7's
  grouping helpers, but both import from here — additive functions, no signature
  changes to existing loaders.
- **`htir/eval/experiment_tau.py`** — SA-9 only (seed loop + SE). Leave alone
  elsewhere.
- **`docs/results-section.md`** — every package appends one paragraph. To avoid
  merge churn, append under a clearly labeled `## <ID> — …` heading matching the
  package ID; SA-11 additionally *rewrites* the existing SA-3 calibration
  paragraph (flag in the PR).

## Global definition of done (whole effort)

The section is spotlight-shaped when: **Table 5 (SA-7)** shows a real downstream
gain; **Table 1 (SA-8)** benchmarks AVG against PRM + agent-judge; **Fig 2
(SA-11)** answers the coverage objection; every headline number carries ≥3 seeds +
mean±SE + a significance statement; and the limitations paragraph names
offline-replay, low coverage, weak labels, and the 3-domain scope up front.
SA-9/10/12/13 strengthen but are not gating for the three-move minimum.
