"""Plays a trained tracking policy in the simulator."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

import mjlab
import torch
import tyro
import yaml
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.os import get_checkpoint_path
from mjlab.utils.torch import configure_torch_backends
from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer

from mj_test.envs.sim_env import SimEnv, SimEnvCfg
from mj_test.policy.networks import ActorCfg, CriticCfg
from mj_test.runner.runner import Runner, RunnerCfg
from mj_test.viewer import ViewerEnv
from mj_test.wm.dynamics import DynamicsCfg


@dataclass
class PlayConfig:
    """Configuration of a playback run.

    Attributes:
        env: Env configuration.
        load_run: Regex of the run directory under ``logs/<task>``. The latest
            match is loaded.
        load_checkpoint: Regex of the checkpoint file in that run directory.
        num_envs: Number of parallel envs.
        device: Torch device.
        from_start: Whether every reset starts at the first bin of the motion.
        viewer: Viewer backend. ``auto`` picks native with a display, else viser.
    """

    env: SimEnvCfg
    load_run: str = ".*"
    load_checkpoint: str = r"model\.pt"
    num_envs: int = 16
    device: str = "cuda:0"
    from_start: bool = False
    viewer: Literal["auto", "native", "viser"] = "auto"


def main() -> None:
    """Parses the task and its configuration, then plays the latest checkpoint."""
    task_id, remaining_args = tyro.cli(
        tyro.extras.literal_type_from_choices(list_tasks()),
        add_help=False,
        return_unknown_args=True,
        config=mjlab.TYRO_FLAGS,
    )
    cfg = tyro.cli(
        PlayConfig,
        args=remaining_args,
        default=PlayConfig(env=cast(SimEnvCfg, load_env_cfg(task_id, play=True))),
        config=mjlab.TYRO_FLAGS,
    )
    configure_torch_backends()
    checkpoint = get_checkpoint_path(
        Path("logs", task_id), cfg.load_run, cfg.load_checkpoint
    )
    saved = yaml.full_load((checkpoint.parent / "params" / "runner.yaml").read_text())
    runner_cfg = cast(RunnerCfg, load_rl_cfg(task_id))
    runner_cfg.dynamics = DynamicsCfg(**saved["dynamics"])
    runner_cfg.actor = ActorCfg(**saved["actor"])
    runner_cfg.critic = CriticCfg(**saved["critic"])
    print(f"[INFO] Loading {checkpoint}")

    cfg.env.scene.num_envs = cfg.num_envs
    env = SimEnv(cfg.env, device=cfg.device)
    runner = (load_runner_cls(task_id) or Runner)(env, runner_cfg)
    runner.load(checkpoint, networks_only=True)
    policy = runner.get_inference_policy()

    if cfg.from_start:
        probs = torch.zeros(env.bin_count, device=env.device)
        probs[0] = 1.0
        env.set_sampling_distribution(probs)
        env.reset()

    viewer = cfg.viewer
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
