"""
Track S -- SkillOpt Terminal-Bench 2.0 environment plugin.

Lives outside ``htir`` (and outside the installed ``skillopt`` package) for the
same reason Track M's capture driver does: rollouts shell out to ``harbor`` and
spend real API + Docker budget. SkillOpt's built-in env registry does not
include Terminal-Bench; this package is the missing ``EnvAdapter`` that the
thin driver ``scripts/skillopt_train_tb2.py`` hands to
``skillopt.engine.trainer.ReflACTTrainer``.
"""

from .adapter import TerminalBenchAdapter
from .dataloader import TerminalBenchDataLoader

__all__ = ["TerminalBenchAdapter", "TerminalBenchDataLoader"]
