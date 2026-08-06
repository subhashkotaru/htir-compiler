# Live harness-optimization baselines: Meta-Harness (Track M) + SkillOpt (Track S)

Companion to `docs/spotlight-experiment-plan.md` and the M7 gap it names:
`avg.tex` §4.3/§4.5 list prompt/skill- and harness-optimization baselines
(SkillOpt, Meta-Harness, Life-Harness) that were never run, while `docs/
build-roadmap.md` B4 deferred them as "no runnable comparators exist... frame
as related-work, not arms." As of 2026-07, both SkillOpt and Meta-Harness are
real, released, open-source systems (see citations below), so that deferral no
longer holds — this doc plans running them for real.

Unlike the SA-8 competitive baselines (PRM, Agent-as-a-Judge), which just
*re-score* trajectories AVG already has, these are harness-**optimization**
methods: their entire value is changing what the agent does, so a real
comparison requires *executing* an agent live, not replaying a recorded trace.
That live-execution capability does not exist anywhere else in this repo
(`htir.eval.datasets` only loads static, pre-recorded corpora) — it is
intentionally kept **outside** the `htir` package (`scripts/`, `data/
live_traces/`) so the deterministic core stays dependency-free and offline.

## Status

| item | status | where |
|------|--------|-------|
| **Track M** offline analysis half (SA-14) | ✅ done, tested offline | `htir/eval/experiment_sa14.py`, `htir/eval/datasets.py` (`load_meta_harness_tb2`), `tests/test_experiment_sa14.py` |
| **Track M** live capture driver | ✅ done, **validated against real `harbor run` output** for `meta_harness`/`base_agent` **and** the `codex` installed agent | `scripts/live_meta_harness_tb2.py` |
| **Track M** real captured data | ✅ `codex` (10/10 tasks) vs. `meta_harness` (6/10 tasks, 4 dropped to a real OpenAI-side content-policy block) on `openai/gpt-5.4-mini`, same 10-task set — see "2026-07-20 codex vs. meta_harness" below | `data/live_traces/meta_harness_tb2/*.jsonl` |
| **Track S** (SkillOpt) | ✅ scaffolded (not live-run yet) | `scripts/skillopt_tb2/`, `scripts/skillopt_train_tb2.py`, `configs/skillopt/terminal_bench_pilot.yaml`, `htir/eval/experiment_sa15.py`, `tests/test_experiment_sa15.py` |
| `avg.tex` §4.3/§4.5 text update | ⏳ not started — do after the *full* 89-task capture lands, not the 15-task pilot | — |

**2026-07-17 validation update.** All of the "unverified" caveats below this
point are now resolved by actually running both harnesses against a real
`harbor==0.19.0` install (docker sandbox, `openai/gpt-4o-mini`). Kept the
original caveats struck through rather than deleted, so the history of what
was assumed vs. verified stays visible.
- CLI flags: real flags are `--jobs-dir` (not `--output-dir`), `-l`/`--n-tasks`
  (not `--limit`), and `-a <import-path>` directly (not a separate
  `--agent-import-path` -- that flag from the artifact's own README doesn't
  exist in `harbor==0.19.0`; it's been folded into `-a`).
- Output schema: `<jobs-dir>/<job>/<task>__<id>/result.json` is a `TrialResult`
  (reward lives at `verifier_result.rewards`, e.g. `{"reward": 0.0}`) with a
  sibling `agent/trajectory.json` holding the ATIF-format transcript
  (`{schema_version, agent, steps:[...]}` -- not an OpenAI `messages` list).
  `scripts/live_meta_harness_tb2.py::_trajectory_to_messages` converts ATIF
  steps to OpenAI-style turns so the existing adapter/normalizer code didn't
  need to change.
- `terminus`/`terminus-1` are listed in `harbor run --help` but are **not**
  wired into this harbor version's `AgentFactory` (raises `Unknown agent
  type`). Use `terminus-2` as `--baseline-agent` -- which is also the fairer
  baseline anyway, since Meta-Harness's `AgentHarness` literally subclasses
  `harbor.agents.terminus_2.Terminus2`.
**Known issues surfaced by the (now-deleted) 2026-07-17/18 pilot.** Kept here
so the next capture doesn't have to rediscover them; the pilot's actual result
numbers (task-success rates, false-valid deltas, costs) were deleted with the
trace data on 2026-07-18 rather than carried forward as stale/low-n claims.
- `htir/utils/llm.py` needed an `OPENAI_API_KEY`-direct fallback (it was
  hardcoded to `OPENROUTER_API_KEY` only) — already fixed, no action needed;
  see `get_client()`'s docstring.
- With `--use-llm` on, `avg_full`/`exec_only` can still abstain on effectively
  every real terminal-bench trace even though the semantic checker resolves
  *some* per-obligation checks — `aggregate()`'s
  `BROAD_ABSTAIN_FRACTION_THRESHOLD = 0.5` requires >=50% of *all* obligations
  on a trajectory to resolve before the trajectory-level verdict leaves
  `uncertain`, and the generic mechanical obligations synthesized for this
  domain ("a shell command completed successfully") don't map cleanly onto
  real terminal-bench task-completion criteria. Getting `avg_full` to actually
  resolve on this domain needs better-targeted obligation *generation*
  (tying claims to the task's own verifier script), not a threshold tweak —
  flag this as a real prerequisite if the next capture wants `avg_full` to do
  more than abstain.
- `--include-task-name` (repeatable, forwards to harbor's `-i`) exists on
  `scripts/live_meta_harness_tb2.py` specifically to pin a follow-up run to
  the *same* task set as an earlier one, for a fair paired comparison instead
  of resampling a fresh, unrelated task slice.
- A reproducible hang: agent execution can go idle at 0% CPU well past
  harbor's own 900s `AgentTimeoutError` and never gets cancelled, requiring a
  manual kill after ~30 min stuck. It correlates with retry/backoff churn
  (rate limits, content-policy retries) more than with plain task difficulty,
  and may be a real harbor/litellm cancellation bug (an `asyncio.
  CancelledError` that never actually unwinds the in-flight aiohttp request)
  rather than a one-off. Worth a closer look before a large/unattended run.
- More expensive models (e.g. gpt-5.4) can have a much lower default org TPM
  limit than gpt-4o-mini, and concurrency (`-n`) can exhaust it — tune `-n`
  down or expect `RateLimitError`s on non-gpt-4o-mini models.
- Some task prompts can trigger `litellm.ContentPolicyViolationError` as an
  OpenAI-side false-positive unrelated to anything in this repo — worth a
  retry-once-then-drop policy for a large run rather than treating it as a
  real task failure.
- `-a agent:AgentHarness` needs the artifact checkout's directory on
  `PYTHONPATH`, not just as `cwd` -- `harbor` is an installed console-script
  entry point, so its cwd is not automatically on `sys.path` the way running
  `python foo.py` would be. `run_harbor()` sets `PYTHONPATH` explicitly.
- The Meta-Harness artifact's `agent.py` calls two `Terminus2` private helpers
  (`_setup_episode_logging`, `_record_asciinema_marker`) that harbor removed
  starting at `harbor==0.9.0` (confirmed via an AST diff of `Terminus2`'s
  methods across every harbor wheel 0.2.0-0.19.0 -- these are the *only* two
  removed). Both are optional debug-logging helpers with no effect on agent
  behavior; restored verbatim (harbor==0.8.0's implementation) as a disclosed
  local patch in `vendor/meta-harness-tbench2-artifact/agent.py` (search for
  "HTIR compat patch" in that file). This is the only modification made to
  the vendored artifact.

---

## Track M — Meta-Harness vs. Base Agent on Terminal-Bench 2.0

**Source.** Lee et al., *Meta-Harness: End-to-End Optimization of Model
Harnesses*, arXiv:2603.28052 (2026). Artifact:
[`stanford-iris-lab/meta-harness-tbench2-artifact`](https://github.com/stanford-iris-lab/meta-harness-tbench2-artifact)
— a **fixed, already-discovered** agent scaffold for Terminal-Bench 2.0
(76.4% reported, Claude Opus 4.6, 89 tasks × 5 trials). Because the search
already happened, we do not need to reproduce Meta-Harness's (expensive)
outer-loop search — we run the discovered scaffold as a fixed baseline arm,
exactly like running any other agent.

**Claim.** Does a harness-optimization method's discovered scaffold produce
(a) a real Terminal-Bench 2.0 success-rate delta over a plain baseline agent,
and (b) does AVG's own verifier read those two harnesses consistently with
that ground truth (e.g. does `false_valid_rate` spike on whichever harness
actually performs worse)?

**Method.** Two harness variants, same model, same task slice:
- `meta_harness` — Meta-Harness's released scaffold, invoked via harbor's
  `--agent-import-path` against a local checkout of the artifact repo.
- `base_agent` — a stock harbor agent (`--agent <name>`; the right built-in to
  use as a fair, non-scaffolded baseline needs confirming against your
  installed harbor version's agent catalog — see caveats).

Both run against `terminal-bench@2.0` under a **matched model** — default
`openai/gpt-4o-mini` per this round's ask. Each captured trial's outcome
(pass/fail) is the ground-truth reward; its transcript is normalized into the
turn schema and re-verified by AVG's arms (`avg_full`/`exec_only`/`monolithic`)
so both a task-outcome delta and a verifier-quality delta are reported side by
side.

**Reuse.** `htir.adapters.terminal` (zero new adapter code — the normalizer
only rewrites harbor's transcript into the `{steps:[{src,msg,tools,obs}]}` turn
schema this adapter already parses, exactly as `normalize_swe_gym_record` does
for OpenHands), `htir.agents.baselines.run_arm`, `htir.eval.weak_labels.
evaluate_predictions`, `htir.eval.seeds.{mean_se,paired_t_test}`.

**New code (done).**
- `htir/eval/datasets.py`: `normalize_meta_harness_record`, `load_meta_harness_tb2`
  (offline, unit-tested — mirrors the SWE-Gym section's pattern).
- `htir/eval/experiment_sa14.py`: `run_sa14`/`format_table`/CLI — groups
  captured trials by `harness`, reports task-success rate (task-level
  bootstrap mean±SE) and each verifier arm's false-valid/resolved-accuracy/
  abstention rate per harness, plus a paired-t significance gap
  (`meta_harness - base_agent`) on both task success and `avg_full`'s
  false-valid rate.
- `tests/test_experiment_sa14.py`: synthetic TB2-shaped fixture (no live
  calls), asserts the normalizer, the offline monolith-over-credits/avg-
  abstains contrast, determinism, and the CLI.
- `scripts/live_meta_harness_tb2.py`: the **only** place in the repo that
  shells out to `harbor run`. Not imported by `htir`; not run by tests/CI.
  Now validated end-to-end (see the 2026-07-17 status note above).
- `vendor/meta-harness-tbench2-artifact/agent.py`: two-method compat patch
  (disclosed above), needed only to bridge a harbor-version API drift.

**Environment used for the validated run.** A dedicated conda env, not the
system/base Python (`mamba create -n htir python=3.12`), so `pip install
harbor` and friends never touch `base`:
```bash
mamba create -n htir python=3.12
mamba run -n htir pip install --only-binary=:all: -e ".[dev,live]"
mamba run -n htir pip install --only-binary=:all: anthropic tenacity  # meta-harness artifact's own extra deps
git clone https://github.com/stanford-iris-lab/meta-harness-tbench2-artifact vendor/meta-harness-tbench2-artifact
# then apply the two-method compat patch to vendor/.../agent.py (see above)
```
`--only-binary=:all:` matters: a plain `pip install harbor` can try to build
one of `litellm`'s transitive deps from source via `maturin`/`rustc`, which
failed in this environment on a Homebrew rust/llvm ABI mismatch unrelated to
harbor itself.

**Remaining before the full-scale (89-task) capture.**
- No capture currently exists (the 2026-07-17/18 pilots were deleted
  2026-07-18 to start clean) — the next run is the first live one on this
  round. Start with a small `--n-tasks`/`--n-attempts 1` sanity pass before
  scaling to `--n-tasks 0 --n-attempts <k>` (the full 89-task suite); each task
  takes 1-20+ min depending on difficulty, and some CoreWars/optimization
  tasks can run the agent for 150+ trajectory steps before hitting the task
  timeout without solving it — see "known issues" above for the hang bug this
  can trigger.
- `avg.tex` §4.3 (baselines) / §4.5 (ablations) text is **not** updated yet —
  do that once the full-scale numbers exist, not the pilot (don't claim a
  result from 15 tasks as if it were the 89-task suite).

**Run (validated invocation).**
```bash
mamba run -n htir python scripts/live_meta_harness_tb2.py --dry-run \
    --n-tasks 2 --n-attempts 1                                    # 1. sanity-check the command
mamba run -n htir python scripts/live_meta_harness_tb2.py \
    --n-tasks 2 --n-attempts 1                                    # 2. cheap pilot; inspect unparsed files
mamba run -n htir python scripts/live_meta_harness_tb2.py \
    --n-tasks 0 --n-attempts 5                                    # 3. full capture (both harnesses, default -m openai/gpt-4o-mini)

mamba run -n htir python -m htir.eval.experiment_sa14 \
    --cache data/live_traces/meta_harness_tb2/meta_harness_openai_gpt-4o-mini.jsonl \
    --cache data/live_traces/meta_harness_tb2/base_agent_openai_gpt-4o-mini.jsonl \
    --seeds 0,1,2 --out data/sa14_results.json
```

**Done when.** `data/sa14_results.json` holds real (not synthetic) numbers for
both harnesses from the **full** 89-task capture, `docs/results-section.md`
has a paragraph in the existing voice, and `avg.tex` names Meta-Harness as an
actually-run baseline rather than a Related-Work-only citation.

**Rigor caveat to carry into the writeup.** Meta-Harness's scaffold was
discovered/reported against Claude Opus 4.6; running it under `openai/
gpt-4o-mini` is an out-of-distribution transfer test of the discovered
harness, not a reproduction of the paper's 76.4% headline. Say so explicitly.
Also: the mean±SE here comes from task-level bootstrap resampling of *one*
captured run, not independent live re-executions (each re-run costs real
budget) — disclosed in `SA14Result.notes`, keep that disclosure in any
downstream write-up too.

**2026-07-19/20 -- `codex` vs. `meta_harness` on `openai/gpt-5.4-mini`.** A
third harness variant, harbor's built-in **`codex`** installed agent (OpenAI's
own Codex CLI product -- a separate, closed agentic tool with its own loop/
tool implementations, *not* a Terminus2 subclass like `meta_harness`/
`base_agent` are), run with `--agent-kwarg reasoning_effort=medium` and
`--max-retries 0`.
- `scripts/live_meta_harness_tb2.py` needed two real fixes from this run,
  both now in place:
  1. `harness_label()` -- previously *any* `--baseline-agent` under
     `--harness base_agent` was hardcoded to the label/filename `"base_agent"`
     regardless of which agent actually ran, so a `codex` run would have
     silently overwritten (and been indistinguishable from) a `terminus-2`
     run. Non-default baseline agents now get their own label (`codex`,
     `claude_code`, ...).
  2. `_load_harbor_output` -- trials that crash *before* the agent does
     anything (`ApiUsageLimitError`/quota, `RateLimitError`, etc.) still
     produce a `verifier_result` (harness auto-fails to `reward=0.0`) and a
     `trajectory.json` with a few steps -- but those steps are just the
     echoed system prompt/task instruction (`source in {"system","user"}`),
     never a real model turn. Recording those as genuine reward=0 task
     failures would silently contaminate the capture with zero-signal rows.
     Fixed by requiring at least one `source == "agent"` step before keeping
     a trial; a mid-run infra exception (e.g. `AgentTimeoutError`) *after*
     real agent activity is still kept, since the partial trajectory + the
     reward the harness computed is a legitimate outcome, not an artifact.
     Caught this live: the first `codex` attempt hit `ApiUsageLimitError` on
     **all 10/10** trials (an OpenAI account-level quota block, ~$0 actually
     billed since no completions were generated) -- without this fix those
     10 zero-signal rows would have been recorded as if they were real task
     failures.
- `vendor/meta-harness-tbench2-artifact/agent.py` needed one more disclosed
  compat patch: `gpt-5.4-mini` sometimes emits each `commands[]` tool-call
  entry as a bare keystrokes string instead of the expected
  `{"keystrokes": ..., "duration": ...}` dict (a different quirk from the
  Haiku-double-encoding one the artifact already handles) -- crashed
  `pytorch-model-cli` with `AttributeError: 'str' object has no attribute
  'get'` on the first attempt; patched to accept a bare string, re-ran just
  that one task afterward (see "HTIR compat patch" markers in that file).
- **Task-matched capture**, same 10 TB2 tasks for both harnesses
  (`--include-task-name` x10): `break-filter-js-from-html`, `gpt2-codegolf`,
  `largest-eigenval`, `llm-inference-batching-scheduler`,
  `log-summary-date-ranges`, `merge-diff-arc-agi-task`, `pytorch-model-cli`,
  `reshard-c4-data`, `winning-avg-corewars`, `write-compressor`.
- **Results** (`data/sa14_results_codex_vs_meta_harness_gpt54mini.json`, 10
  bootstrap seeds):

  | harness | valid trials | real task success | `prm` false_valid | `monolithic`/`agent_judge` false_valid |
  |---|---|---|---|---|
  | `codex` | 10/10 | **40.0%** (4/10) | 100.0% | 79.6% |
  | `meta_harness` | 6/10 (4 dropped, all `ContentPolicyViolationError`) | **33.3%** (2/6) | 30.0% | 0.0% |

  `avg_full`/`exec_only` abstain 100% on both, as in every other Track M
  slice (no test/mechanical evidence in the captured turn shape without
  `--use-llm` -- see the earlier "known issues" note on `aggregate()`'s
  threshold).
  `codex` narrowly beats `meta_harness` on real task success here, but the
  more striking gap is verification quality: naive judges (`prm`,
  `monolithic`, `agent_judge`) are dramatically worse at catching `codex`'s
  failures (`prm` never once correctly flags a `codex` failure;
  `monolithic`/`agent_judge` over-credit ~80% of them) than `meta_harness`'s
  (`monolithic`/`agent_judge` catch 100% of `meta_harness`'s failures, `prm`
  catches 70%) -- plausibly because `codex`'s own trajectories are much
  longer/more tool-call-dense (up to 669 messages on one trial vs.
  `meta_harness`'s 15-87), which seems to make naive endpoint/PRM heuristics
  easier to fool.
- **Cost:** ~$3.11 (`codex`, 10 real trials after the quota block cleared) +
  ~$1.50 (`meta_harness`, 6 valid + 1 re-run trial) = **~$4.61** for this
  slice (excludes the ~$0 quota-blocked attempt).
- 4/10 tasks are still not captured for `meta_harness` on this model
  (`break-filter-js-from-html`, `merge-diff-arc-agi-task`, `write-compressor`,
  `winning-avg-corewars` -- all real OpenAI-side `ContentPolicyViolationError`
  false positives on the prompt itself, not fixable from this repo).
  `base_agent` (terminus-2) has not yet been run on this same 10-task/model
  slice -- would complete the 3-way comparison.

---

## Track S — SkillOpt on Terminal-Bench 2.0 (scaffolded)

**Source.** Yang et al., *SkillOpt: Executive Strategy for Self-Evolving Agent
Skills*, arXiv:2605.23904 (2026). `pip install skillopt`
(`microsoft/SkillOpt`, MIT). Unlike Meta-Harness, this is the **real training
loop** (rollout → reflect → aggregate → select → update → evaluate) — there is
no pre-discovered Terminal-Bench artifact to just download (confirmed: none of
SkillOpt's six built-in benchmarks is Terminal-Bench, and no community
`best_skill.md` for TB2 has been published).

**Why Terminal-Bench (changed from the earlier tau-bench sketch).** We already
have harbor + Docker + TB2 capture plumbing from Track M, and SkillOpt's
Codex/Claude-Code harness results in the paper are the same *shape* as
injecting a skill into a TB2 agent run. Targeting TB2 also lets SA-15 compare
`skillopt` vs `no_skill` on the same benchmark SA-14 uses for Meta-Harness.

**Implemented (offline / no live spend yet).**

1. `scripts/skillopt_tb2/` -- SkillOpt `EnvAdapter` + dataloader + harbor
   rollout helper. Each rollout item is a TB2 task name; `rollout` shells out
   to `harbor run -d terminal-bench@2.0 -i <task> --skill <candidate>` and
   returns the verifier reward as SkillOpt's hard/soft score. `--dry-run` /
   `HTIR_SKILLOPT_DRY_RUN=1` stubs scores without calling harbor.
2. `scripts/skillopt_train_tb2.py` -- thin driver that instantiates our
   adapter and hands it to `skillopt.engine.trainer.ReflACTTrainer` (bypasses
   SkillOpt's hard-coded built-in env registry).
3. `configs/skillopt/terminal_bench_pilot.yaml` -- deliberately tiny pilot
   (limit=12 tasks, 1 epoch, batch_size=2) with
   `azure_openai_auth_mode: openai_compatible` so a plain `OPENAI_API_KEY`
   works.
4. `scripts/skills/skillopt-tb2-init/SKILL.md` -- minimal starting skill.
5. `data/skillopt_tb2_tasks/tasks.jsonl` -- 41 TB2 task names (from the
   Track M capture set) as the initial split source.
6. `htir/eval/experiment_sa15.py` + `load_skillopt_tb2` -- offline
   `skillopt` vs `no_skill` comparison (mirrors SA-14).

**Not run yet.** Live SkillOpt training against harbor is deferred until the
API/Docker budget is confirmed -- each TB2 rollout is a real container trial.
Offline wiring check:

```bash
python scripts/skillopt_train_tb2.py \
  --config configs/skillopt/terminal_bench_pilot.yaml --dry-run
```
