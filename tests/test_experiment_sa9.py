"""
SA-9 regression test: the scaled LLM-judge slice plumbing.

SA-9 restates the "AVG beats a real `gpt-4o-mini` judge ~15x" headline at the
paper rigor bar -- n>=500 x >=3 seeds, mean±SE, a paired-t significance
statement, and measured token cost. The funded LLM run is blocked on a credited
key, so this test exercises the *offline* path (``use_llm=False``): the seed
loop, the mean±SE aggregation, the significance test, and the token accounting,
on a tiny synthetic τ fixture. The offline path is byte-deterministic and issues
zero real LLM calls, so the headline direction (AVG << monolith on false-valid)
and the token cost (0) are asserted in range.

All offline (no LLM, no network, no corpus cache).
"""

from __future__ import annotations

import copy
import json

import pytest

from htir.eval.experiment_tau import SA9_SIG_GAPS, format_sa9, run_sa9
from htir.eval.datasets import normalize_tau_record
from htir.models.domain import get_domain_spec
from htir.utils.llm import get_usage, reset_usage

from tests.test_tau_bench import TAU_TRACE


def _pool(n_each: int = 40) -> list[dict]:
    """A balanced synthetic τ pool. The valid trace is credited by reward; the
    invalid variant is reward=0 but still *ends on a successful mutation*, so the
    endpoint monolith credits it ``valid`` (a false-valid) while AVG withholds the
    policy-unlinked mutation -- the contrast SA-9 measures."""
    valid = normalize_tau_record(TAU_TRACE)
    bad = copy.deepcopy(TAU_TRACE)
    bad["eval_result"] = {"score": 0.0, "db_match": False}
    bad["meta"] = {"id": "retail_bad", "is_correct": False}
    invalid = normalize_tau_record(bad)
    return [valid, invalid] * n_each


SPEC = get_domain_spec("tau_bench")


def test_sa9_offline_runs_three_seeds_and_is_flagged():
    out = run_sa9(_pool(), spec=SPEC, seeds=[0, 1, 2], n=16, use_llm=False, log=None)
    # three seeds actually swept
    assert out["seeds"] == [0, 1, 2]
    assert len(out["n_per_seed"]) == 3 and all(nn == 16 for nn in out["n_per_seed"])
    assert len(out["per_seed"]) == 3
    # honest degraded flags (no funded key)
    assert out["real_llm"] is False
    assert out["status"] == "degraded-no-key"
    assert out["headline_pending"] is True


def test_sa9_headline_direction_and_range():
    out = run_sa9(_pool(), spec=SPEC, seeds=[0, 1, 2], n=16, use_llm=False, log=None)
    agg = out["aggregate"]
    avg = agg["avg_full.false_valid"]["mean"]
    mono = agg["monolithic.false_valid"]["mean"]
    # AVG withholds the policy-unlinked mutation -> ~zero false-valid; the endpoint
    # monolith credits the reward-0 trace that ends on a success -> high false-valid.
    assert avg <= 0.10
    assert mono >= 0.5
    assert avg < mono  # the headline direction
    # each aggregated metric retains its 3 per-seed values (mean±SE is well-defined)
    assert agg["avg_full.false_valid"]["n"] == 3


def test_sa9_significance_structure():
    out = run_sa9(_pool(), spec=SPEC, seeds=[0, 1, 2], n=16, use_llm=False, log=None)
    sig = out["significance"]
    # one overall + one long-horizon test per SA9 gap
    labels = {g["label"] for g in sig}
    for a_arm, b_arm in SA9_SIG_GAPS:
        assert f"{a_arm}_vs_{b_arm}.false_valid.overall" in labels
        assert f"{a_arm}_vs_{b_arm}.false_valid.long" in labels
    # the monolith-vs-AVG overall gap is positive (monolith over-credits more)
    mono_overall = next(g for g in sig if g["label"] == "monolithic_vs_avg_full.false_valid.overall")
    assert mono_overall["mean_diff"] > 0
    assert mono_overall["n_seeds"] == 3


def test_sa9_token_cost_zero_offline():
    reset_usage()
    out = run_sa9(_pool(), spec=SPEC, seeds=[0, 1, 2], n=16, use_llm=False, log=None)
    # no real LLM calls offline -> zero measured token cost
    tc = out["token_cost"]
    assert tc == {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    # and the accumulator itself stayed at zero
    assert get_usage().total_tokens == 0


def test_sa9_offline_byte_deterministic():
    a = run_sa9(_pool(), spec=SPEC, seeds=[0, 1, 2], n=16, use_llm=False, log=None)
    b = run_sa9(_pool(), spec=SPEC, seeds=[0, 1, 2], n=16, use_llm=False, log=None)
    # per_seed drops the wall-clock field, so the whole result serializes identically
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


def test_sa9_format_runs():
    out = run_sa9(_pool(), spec=SPEC, seeds=[0, 1, 2], n=16, use_llm=False, log=None)
    text = format_sa9(out)
    assert "SA-9" in text and "false_valid mean±SE" in text
    assert "headline pending" in text.lower()


def test_token_accounting_reset_and_zero():
    reset_usage()
    u = get_usage()
    assert (u.calls, u.prompt_tokens, u.completion_tokens, u.total_tokens) == (0, 0, 0, 0)
