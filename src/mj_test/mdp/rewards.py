"""Reward terms for the motion tracking task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.utils.lab_api.math import quat_error_magnitude

if TYPE_CHECKING:
    from mj_test.envs.env import Env

__all__ = [
    "global_anchor_position_error_exp",
    "global_anchor_orientation_error_exp",
    "relative_body_position_error_exp",
    "relative_body_orientation_error_exp",
    "global_body_linear_velocity_error_exp",
    "global_body_angular_velocity_error_exp",
    "action_rate",
    "joint_limit_violation",
]


def global_anchor_position_error_exp(env: Env, std: float) -> torch.Tensor:
    """Computes the anchor position tracking reward."""
    error = torch.sum(
        torch.square(env.ref_bodies.pos[..., 0, :] - env.robot_bodies.pos[..., 0, :]),
        dim=-1,
    )
    return torch.exp(-error / std**2)


def global_anchor_orientation_error_exp(env: Env, std: float) -> torch.Tensor:
    """Computes the anchor orientation tracking reward."""
    error = (
        quat_error_magnitude(
            env.ref_bodies.quat[..., 0, :], env.robot_bodies.quat[..., 0, :]
        )
        ** 2
    )
    return torch.exp(-error / std**2)


def relative_body_position_error_exp(env: Env, std: float) -> torch.Tensor:
    """Computes the body position tracking reward, averaged over bodies."""
    error = torch.sum(torch.square(env.ref_bodies.pos - env.robot_bodies.pos), dim=-1)
    return torch.exp(-error.mean(-1) / std**2)


def relative_body_orientation_error_exp(env: Env, std: float) -> torch.Tensor:
    """Computes the body orientation tracking reward, averaged over bodies."""
    error = quat_error_magnitude(env.ref_bodies.quat, env.robot_bodies.quat) ** 2
    return torch.exp(-error.mean(-1) / std**2)


def global_body_linear_velocity_error_exp(env: Env, std: float) -> torch.Tensor:
    """Computes the body linear velocity tracking reward, averaged over bodies."""
    error = torch.sum(
        torch.square(env.ref_bodies.lin_vel - env.robot_bodies.lin_vel), dim=-1
    )
    return torch.exp(-error.mean(-1) / std**2)


def global_body_angular_velocity_error_exp(env: Env, std: float) -> torch.Tensor:
    """Computes the body angular velocity tracking reward, averaged over bodies."""
    error = torch.sum(
        torch.square(env.ref_bodies.ang_vel - env.robot_bodies.ang_vel), dim=-1
    )
    return torch.exp(-error.mean(-1) / std**2)


def action_rate(env: Env) -> torch.Tensor:
    """Computes the action rate penalty."""
    return torch.sum(torch.square(env.action - env.prev_action), dim=-1)


def joint_limit_violation(env: Env) -> torch.Tensor:
    """Computes the joint position violation of the soft limits, summed over joints."""
    joint_pos = env.robot_state.joint_pos
    lower, upper = env.joint_limits.unbind(-1)
    out_of_limits = -(joint_pos - lower).clip(max=0.0)
    out_of_limits += (joint_pos - upper).clip(min=0.0)
    return torch.sum(out_of_limits, dim=-1)
