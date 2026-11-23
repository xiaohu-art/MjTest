"""Termination terms for the motion tracking task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mj_test.utils.math import projected_gravity

if TYPE_CHECKING:
    from mj_test.envs.env import Env
    from mj_test.envs.sim_env import SimEnv

__all__ = ["anchor_height_error", "anchor_tilt_error", "motion_end", "time_out"]


def time_out(env: SimEnv) -> torch.Tensor:
    """Whether the episode reached its time limit, shape (B,)."""
    return env.episode_length_buf >= env.max_episode_length


def motion_end(env: Env) -> torch.Tensor:
    """Whether the reference is on the last frame of its clip, shape (B,)."""
    return env.command.clip_end


def anchor_height_error(env: Env) -> torch.Tensor:
    """Computes the absolute anchor height error in meters."""
    cmd = env.command
    return torch.abs(cmd.anchor_pos_w[:, 2] - cmd.robot_anchor_pos_w[:, 2])


def anchor_tilt_error(env: Env) -> torch.Tensor:
    """Computes the anchor tilt error as the difference of the projected gravity z."""
    cmd = env.command
    ref_down_b = projected_gravity(cmd.anchor_quat_w)
    down_b = projected_gravity(cmd.robot_anchor_quat_w)
    return torch.abs(ref_down_b[:, 2] - down_b[:, 2])
