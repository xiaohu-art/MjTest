"""Trains the world model and the tracking policy.

Usage:
    uv run scripts/train.py [--config <yaml>] [key.sub=value ...]
    uv run scripts/train.py --resume True --load-run <regex>
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import mjlab
import tyro
from mjlab.utils.os import get_checkpoint_path
from mjlab.utils.torch import configure_torch_backends
from omegaconf import OmegaConf

from mj_test.envs.sim_env import SimEnv
from mj_test.runner.runner import Runner

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"


@dataclass
class TrainArgs:
    config: Path = CONFIG_DIR / "train.yaml"
    """Config file."""
    resume: bool = False
    """Whether to load a checkpoint before training."""
    load_run: str = ".*"
    """Regex of the run under logs/<name> to resume. The latest match is loaded."""
    load_checkpoint: str = r"model.*\.pt"
    """Regex of the checkpoint file in that run. The latest match is loaded."""


def main() -> None:
    """Loads the config, then trains."""
    args, overrides = tyro.cli(
        TrainArgs, return_unknown_args=True, config=mjlab.TYRO_FLAGS
    )
    cfg = OmegaConf.load(args.config)
    OmegaConf.set_struct(cfg, True)  # Overrides may only set existing keys.
    cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))
    configure_torch_backends()
    log_root = Path("logs", cfg.name)
    resume_path = (
        get_checkpoint_path(log_root, args.load_run, args.load_checkpoint)
        if args.resume
        else None
    )
    env = SimEnv(cfg.env, device=cfg.device, seed=cfg.seed)
    log_dir = log_root / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    (log_dir / "params").mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, log_dir / "params" / "config.yaml")
    print(f"[INFO] Logging to {log_dir}")
    runner = Runner(env, cfg.runner, log_dir)
    if resume_path is not None:
        print(f"[INFO] Resuming from {resume_path}")
        runner.load(resume_path)
    runner.run()
    env.close()


if __name__ == "__main__":
    main()
