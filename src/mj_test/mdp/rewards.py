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
    cmd = env.command
    error = torch.sum(torch.square(cmd.anchor_pos_w - cmd.robot_anchor_pos_w), dim=-1)
    return torch.exp(-error / std**2)


def global_anchor_orientation_error_exp(env: Env, std: float) -> torch.Tensor:
    """Computes the anchor orientation tracking reward."""
    cmd = env.command
    error = quat_error_magnitude(cmd.anchor_quat_w, cmd.robot_anchor_quat_w) ** 2
    return torch.exp(-error / std**2)


def relative_body_position_error_exp(env: Env, std: float) -> torch.Tensor:
    """Computes the body position tracking reward, averaged over bodies."""
    cmd = env.command
    error = torch.sum(torch.square(cmd.body_pos_w - cmd.robot_body_pos_w), dim=-1)
    return torch.exp(-error.mean(-1) / std**2)


def relative_body_orientation_error_exp(env: Env, std: float) -> torch.Tensor:
    """Computes the body orientation tracking reward, averaged over bodies."""
    cmd = env.command
    error = quat_error_magnitude(cmd.body_quat_w, cmd.robot_body_quat_w) ** 2
    return torch.exp(-error.mean(-1) / std**2)


def global_body_linear_velocity_error_exp(env: Env, std: float) -> torch.Tensor:
    """Computes the body linear velocity tracking reward, averaged over bodies."""
    cmd = env.command
    error = torch.sum(
        torch.square(cmd.body_lin_vel_w - cmd.robot_body_lin_vel_w), dim=-1
    )
    return torch.exp(-error.mean(-1) / std**2)


def global_body_angular_velocity_error_exp(env: Env, std: float) -> torch.Tensor:
    """Computes the body angular velocity tracking reward, averaged over bodies."""
    cmd = env.command
    error = torch.sum(
        torch.square(cmd.body_ang_vel_w - cmd.robot_body_ang_vel_w), dim=-1
    )
    return torch.exp(-error.mean(-1) / std**2)


def action_rate(env: Env) -> torch.Tensor:
    """Computes the action rate penalty."""
    actions = env.action_term
    return (actions.action - actions.prev_action).square().sum(dim=-1)


def joint_limit_violation(env: Env) -> torch.Tensor:
    """Computes the joint position violation of the soft limits, summed over joints."""
    robot = env.robot
    lower, upper = robot.soft_joint_pos_limits.unbind(-1)
    out_of_limits = -(robot.joint_pos - lower).clip(max=0.0)
    out_of_limits += (robot.joint_pos - upper).clip(min=0.0)
    return torch.sum(out_of_limits, dim=-1)
