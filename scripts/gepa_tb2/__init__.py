"""
Track G: GEPA on Terminal-Bench 2.0.

Wires GEPA's Pareto reflective prompt-evolution onto the same frozen
``terminus-2`` + skill-injection harness SkillOpt (Track S) uses, reusing
``scripts.skillopt_tb2.rollout.run_batch`` for the harbor rollouts. See
``scripts/gepa_train_tb2.py`` for the driver and
``docs/live-baselines-plan.md`` for how Track G relates to Tracks M/S.
"""

from scripts.gepa_tb2.adapter import SKILL_COMPONENT, TerminalBenchGEPAAdapter
from scripts.gepa_tb2.dataloader import carve_ratio_split, load_materialized_split

__all__ = [
    "TerminalBenchGEPAAdapter",
    "SKILL_COMPONENT",
    "load_materialized_split",
    "carve_ratio_split",
]
