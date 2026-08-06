"""SkillOpt ``EnvAdapter`` for Terminal-Bench 2.0 (Track S)."""

from __future__ import annotations

from skillopt.datasets.base import BatchSpec
from skillopt.envs.base import EnvAdapter

from .dataloader import TerminalBenchDataLoader
from .rollout import run_batch


class TerminalBenchAdapter(EnvAdapter):
    """
    Environment adapter that turns SkillOpt's training loop into live
    Terminal-Bench 2.0 harbor trials.

    Each rollout item is a TB2 task name; ``rollout`` shells out to
    ``harbor run -d terminal-bench@2.0 -i <task> --skill <candidate>`` and
    returns the verifier reward as SkillOpt's hard/soft score. Reflection
    uses SkillOpt's inherited default analyst prompts (no custom override
    needed for the pilot).
    """

    def __init__(
        self,
        split_dir: str = "",
        data_path: str = "",
        split_mode: str = "ratio",
        split_ratio: str = "2:1:7",
        split_seed: int = 42,
        split_output_dir: str = "",
        workers: int = 1,
        analyst_workers: int = 4,
        failure_only: bool = False,
        minibatch_size: int = 4,
        edit_budget: int = 4,
        seed: int = 42,
        limit: int = 0,
        max_completion_tokens: int = 4096,
        # TB2 / harbor-specific
        harbor_model: str = "openai/gpt-4o-mini",
        harbor_agent: str = "terminus-2",
        harbor_env: str = "docker",
        harbor_max_retries: int = 0,
        harbor_agent_kwargs: list[str] | None = None,
        dry_run: bool = False,
        inject_skill: bool = True,
    ) -> None:
        self.workers = int(workers)
        self.analyst_workers = int(analyst_workers)
        self.failure_only = bool(failure_only)
        self.minibatch_size = int(minibatch_size)
        self.edit_budget = int(edit_budget)
        self.max_completion_tokens = int(max_completion_tokens)
        self.harbor_model = harbor_model
        self.harbor_agent = harbor_agent
        self.harbor_env = harbor_env
        self.harbor_max_retries = int(harbor_max_retries)
        self.harbor_agent_kwargs = list(harbor_agent_kwargs or [])
        self.dry_run = bool(dry_run)
        self.inject_skill = bool(inject_skill)
        self.dataloader = TerminalBenchDataLoader(
            split_dir=split_dir,
            data_path=data_path,
            split_mode=split_mode,
            split_ratio=split_ratio,
            split_seed=split_seed,
            split_output_dir=split_output_dir,
            seed=seed,
            limit=limit,
        )

    def setup(self, cfg: dict) -> None:
        super().setup(cfg)
        # Allow flat cfg overrides without re-instantiating.
        for key, attr in (
            ("harbor_model", "harbor_model"),
            ("target_model", "harbor_model"),
            ("harbor_agent", "harbor_agent"),
            ("harbor_env", "harbor_env"),
            ("dry_run", "dry_run"),
        ):
            if key in cfg and cfg[key] is not None:
                setattr(self, attr, cfg[key])
        if "workers" in cfg and cfg["workers"] is not None:
            self.workers = int(cfg["workers"])
        self.dataloader.setup(cfg)

    def get_dataloader(self):
        return self.dataloader

    def build_env_from_batch(self, batch: BatchSpec, **kwargs):
        return list(batch.payload or [])

    def build_train_env(self, batch_size: int, seed: int, **kwargs):
        batch = self.dataloader.build_train_batch(batch_size=batch_size, seed=seed, **kwargs)
        return self.build_env_from_batch(batch, **kwargs)

    def build_eval_env(self, env_num: int, split: str, seed: int, **kwargs):
        batch = self.dataloader.build_eval_batch(
            env_num=env_num, split=split, seed=seed, **kwargs
        )
        return self.build_env_from_batch(batch, **kwargs)

    def rollout(
        self,
        env_manager,
        skill_content: str,
        out_dir: str,
        **kwargs,
    ) -> list[dict]:
        items: list[dict] = env_manager
        return run_batch(
            items=items,
            out_root=out_dir,
            skill_content=skill_content,
            model=self.harbor_model,
            agent=self.harbor_agent,
            env=self.harbor_env,
            workers=self.workers,
            max_retries=self.harbor_max_retries,
            agent_kwargs=self.harbor_agent_kwargs,
            dry_run=self.dry_run,
            inject_skill=self.inject_skill,
            diagnostic_mode=kwargs.get("diagnostic_mode", False),
        )

    def get_task_types(self) -> list[str]:
        return ["terminal_bench"]
