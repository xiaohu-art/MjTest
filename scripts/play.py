"""Plays a trained tracking policy in the simulator.

Usage:
    uv run scripts/play.py [--load-run <regex>] [--viewer native|viser] [--from-start True]
    uv run scripts/play.py env.num_envs=4
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import mjlab
import torch
import tyro
import yaml
from mjlab.utils.os import get_checkpoint_path
from mjlab.utils.torch import configure_torch_backends
from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer
from omegaconf import OmegaConf

from mj_test.envs.sim_env import SimEnv
from mj_test.runner.runner import Runner
from mj_test.viewer import ViewerEnv

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"
_NETWORKS = ("dynamics", "actor", "critic")


@dataclass
class PlayArgs:
    config: Path = CONFIG_DIR / "play.yaml"
    """Playback overrides, merged onto configs/train.yaml."""
    load_run: str = ".*"
    """Regex of the run under logs/<name> to play. The latest match is loaded."""
    load_checkpoint: str = r"model.*\.pt"
    """Regex of the checkpoint file in that run. The latest match is loaded."""
    viewer: Literal["auto", "native", "viser"] = "auto"
    """Viewer backend. ``auto`` picks native with a display, else viser."""
    from_start: bool = False
    """Whether every reset starts at the first bin of the motion."""


def saved_networks(params: Path) -> dict[str, Any]:
    """Returns the network configs that a run was trained with.

    Args:
        params: The run's ``params`` directory.
    """
    config = params / "config.yaml"
    if config.exists():
        runner = OmegaConf.load(config).runner
        return {name: OmegaConf.to_container(runner[name]) for name in _NETWORKS}
    # Runs from before the YAML configs saved only the runner config.
    saved = yaml.full_load((params / "runner.yaml").read_text())
    return {name: saved[name] for name in _NETWORKS}


def main() -> None:
    """Loads the config and the latest matching checkpoint, then plays it."""
    args, overrides = tyro.cli(
        PlayArgs, return_unknown_args=True, config=mjlab.TYRO_FLAGS
    )
    cfg = OmegaConf.load(CONFIG_DIR / "train.yaml")
    OmegaConf.set_struct(cfg, True)  # Overrides may only set existing keys.
    play = OmegaConf.load(args.config)
    cfg = OmegaConf.merge(cfg, play, OmegaConf.from_dotlist(overrides))
    configure_torch_backends()
    checkpoint = get_checkpoint_path(
        Path("logs", cfg.name), args.load_run, args.load_checkpoint
    )
    networks = saved_networks(checkpoint.parent / "params")
    cfg = OmegaConf.merge(cfg, {"runner": networks})
    print(f"[INFO] Loading {checkpoint}")

    env = SimEnv(cfg.env, device=cfg.device, seed=cfg.seed)
    runner = Runner(env, cfg.runner)
    runner.load(checkpoint, networks_only=True)
    policy = runner.get_inference_policy()

    if args.from_start:
        sampler = env.command.sampler
        probs = torch.zeros(sampler.bin_count, device=env.device)
        probs[0] = 1.0
        sampler.set_distribution(probs)
        env.reset()

    viewer = args.viewer
    if viewer == "auto":
        has_display = os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")
        viewer = "native" if has_display else "viser"
    viewer_env = ViewerEnv(env)
    if viewer == "native":
        NativeMujocoViewer(viewer_env, policy).run()
    else:
        ViserPlayViewer(viewer_env, policy).run()
    env.close()


if __name__ == "__main__":
    main()
