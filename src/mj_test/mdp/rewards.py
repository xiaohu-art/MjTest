"""Reward terms for the motion tracking task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.utils.lab_api.math import quat_error_magnitude

from .commands import Transition

if TYPE_CHECKING:
    from collections.abc import Callable

    from mjlab.envs import ManagerBasedRlEnv

__all__ = [
    "transition_reward",
    "global_anchor_position_error_exp",
    "global_anchor_orientation_error_exp",
    "relative_body_position_error_exp",
    "relative_body_orientation_error_exp",
    "global_body_linear_velocity_error_exp",
    "global_body_angular_velocity_error_exp",
    "action_rate",
    "joint_limit_violation",
]


def transition_reward(
    env: ManagerBasedRlEnv, term: Callable[..., torch.Tensor], **params: float
) -> torch.Tensor:
    """Computes a reward term on the transition of the current env step.

    Args:
        env: The environment.
        term: Reward function of a ``Transition``.
        **params: Remaining keyword arguments of ``term``.

    Returns:
        The reward of shape (num_envs,).
    """
    return term(Transition.from_env(env), **params)


def global_anchor_position_error_exp(x: Transition, std: float) -> torch.Tensor:
    """Computes the anchor position tracking reward.

    Args:
        x: Transition of the step.
        std: Standard deviation of the exponential kernel.

    Returns:
        ``exp(-e / std**2)`` of shape (...), where ``e`` is the squared distance
        between the anchor positions.
    """
    error = torch.sum(
        torch.square(x.ref.pos[..., 0, :] - x.body.pos[..., 0, :]), dim=-1
    )
    return torch.exp(-error / std**2)


def global_anchor_orientation_error_exp(x: Transition, std: float) -> torch.Tensor:
    """Computes the anchor orientation tracking reward.

    Args:
        x: Transition of the step.
        std: Standard deviation of the exponential kernel.

    Returns:
        ``exp(-e / std**2)`` of shape (...), where ``e`` is the squared rotation
        angle between the anchor orientations.
    """
    error = quat_error_magnitude(x.ref.quat[..., 0, :], x.body.quat[..., 0, :]) ** 2
    return torch.exp(-error / std**2)


def relative_body_position_error_exp(x: Transition, std: float) -> torch.Tensor:
    """Computes the body position tracking reward.

    Args:
        x: Transition of the step.
        std: Standard deviation of the exponential kernel.

    Returns:
        ``exp(-e / std**2)`` of shape (...), where ``e`` is the squared distance
        between the body positions, averaged over bodies.
    """
    error = torch.sum(torch.square(x.ref.pos - x.body.pos), dim=-1)
    return torch.exp(-error.mean(-1) / std**2)


def relative_body_orientation_error_exp(x: Transition, std: float) -> torch.Tensor:
    """Computes the body orientation tracking reward.

    Args:
        x: Transition of the step.
        std: Standard deviation of the exponential kernel.

    Returns:
        ``exp(-e / std**2)`` of shape (...), where ``e`` is the squared rotation
        angle between the body orientations, averaged over bodies.
    """
    error = quat_error_magnitude(x.ref.quat, x.body.quat) ** 2
    return torch.exp(-error.mean(-1) / std**2)


def global_body_linear_velocity_error_exp(x: Transition, std: float) -> torch.Tensor:
    """Computes the body linear velocity tracking reward.

    Args:
        x: Transition of the step.
        std: Standard deviation of the exponential kernel.

    Returns:
        ``exp(-e / std**2)`` of shape (...), where ``e`` is the squared linear
        velocity error, averaged over bodies.
    """
    error = torch.sum(torch.square(x.ref.lin_vel - x.body.lin_vel), dim=-1)
    return torch.exp(-error.mean(-1) / std**2)


def global_body_angular_velocity_error_exp(x: Transition, std: float) -> torch.Tensor:
    """Computes the body angular velocity tracking reward.

    Args:
        x: Transition of the step.
        std: Standard deviation of the exponential kernel.

    Returns:
        ``exp(-e / std**2)`` of shape (...), where ``e`` is the squared angular
        velocity error, averaged over bodies.
    """
    error = torch.sum(torch.square(x.ref.ang_vel - x.body.ang_vel), dim=-1)
    return torch.exp(-error.mean(-1) / std**2)


def action_rate(x: Transition) -> torch.Tensor:
    """Computes the action rate penalty.

    Args:
        x: Transition of the step.

    Returns:
        The squared change of the raw actions, summed over actions, of shape (...).
    """
    return torch.sum(torch.square(x.action - x.prev_action), dim=-1)


def joint_limit_violation(x: Transition) -> torch.Tensor:
    """Computes the soft joint position limit penalty.

    Args:
        x: Transition of the step.

    Returns:
        The distance of the joint positions beyond their soft limits, summed over
        joints, of shape (...).
    """
    joint_pos = x.state.joint_pos
    lower, upper = x.joint_limits.unbind(-1)
    out_of_limits = -(joint_pos - lower).clip(max=0.0)
    out_of_limits += (joint_pos - upper).clip(min=0.0)
    return torch.sum(out_of_limits, dim=-1)
