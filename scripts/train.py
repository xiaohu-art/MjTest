"""Trains the world model and the tracking policy."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import cast

import mjlab
import tyro
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.os import get_checkpoint_path
from mjlab.utils.torch import configure_torch_backends

from mj_test.envs.sim_env import SimEnv, SimEnvCfg
from mj_test.runner.runner import Runner, RunnerCfg


@dataclass
class TrainConfig:
    """Configuration of a training run.

    Attributes:
        env: Env configuration.
        runner: Runner configuration.
        num_envs: Number of parallel envs.
        device: Torch device.
        resume: Whether to load a checkpoint before training.
        load_run: Regex of the run directory under ``logs/<task>`` to resume from.
            The latest match is loaded.
        load_checkpoint: Regex of the checkpoint file in that run directory.
    """

    env: SimEnvCfg
    runner: RunnerCfg
    num_envs: int = 4096
    device: str = "cuda:0"
    resume: bool = False
    load_run: str = ".*"
    load_checkpoint: str = r"model\.pt"


def main() -> None:
    """Parses the task and its configuration, then trains."""
    task_id, remaining_args = tyro.cli(
        tyro.extras.literal_type_from_choices(list_tasks()),
        add_help=False,
        return_unknown_args=True,
        config=mjlab.TYRO_FLAGS,
    )
    cfg = tyro.cli(
        TrainConfig,
        args=remaining_args,
        default=TrainConfig(
            env=cast(SimEnvCfg, load_env_cfg(task_id)),
            runner=cast(RunnerCfg, load_rl_cfg(task_id)),
        ),
        config=mjlab.TYRO_FLAGS,
    )
    configure_torch_backends()
    log_root = Path("logs", task_id)
    resume_path = (
        get_checkpoint_path(log_root, cfg.load_run, cfg.load_checkpoint)
        if cfg.resume
        else None
    )
    cfg.env.scene.num_envs = cfg.num_envs
    cfg.env.seed = cfg.runner.seed
    env = SimEnv(cfg.env, device=cfg.device)
    log_dir = log_root / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Logging to {log_dir}")
    runner = (load_runner_cls(task_id) or Runner)(env, cfg.runner, log_dir)
    if resume_path is not None:
        print(f"[INFO] Resuming from {resume_path}")
        runner.load(resume_path)
    runner.run()
    env.close()


if __name__ == "__main__":
    main()
