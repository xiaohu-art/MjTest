"""Reward terms for the motion tracking task."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch

from mjlab.sensor import ContactSensor
from mjlab.utils.lab_api.math import quat_error_magnitude

from .commands import Motion

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

__all__ = [
    "motion_global_anchor_position_error_exp",
    "motion_global_anchor_orientation_error_exp",
    "motion_relative_body_position_error_exp",
    "motion_relative_body_orientation_error_exp",
    "motion_global_body_linear_velocity_error_exp",
    "motion_global_body_angular_velocity_error_exp",
    "self_collision_cost",
]


def _get_body_indexes(
    command: Motion, body_names: tuple[str, ...] | None
) -> list[int]:
    """Returns the indexes of the selected bodies among the tracked bodies.

    Args:
        command: The motion command term.
        body_names: Names of the bodies to select. None selects all tracked bodies.

    Returns:
        Indexes into ``command.cfg.body_names``.
    """
    return [
        i
        for i, name in enumerate(command.cfg.body_names)
        if (body_names is None) or (name in body_names)
    ]


def motion_global_anchor_position_error_exp(
    env: ManagerBasedRlEnv, command_name: str, std: float
) -> torch.Tensor:
    """Computes the anchor position tracking reward.

    Args:
        env: The environment.
        command_name: Name of the motion command term.
        std: Standard deviation of the exponential kernel.

    Returns:
        ``exp(-e / std**2)`` of shape (num_envs,), where ``e`` is the squared
        distance between the reference and robot anchor positions.
    """
    command = cast(Motion, env.command_manager.get_term(command_name))
    error = torch.sum(
        torch.square(command.anchor_pos_w - command.robot_anchor_pos_w), dim=-1
    )
    return torch.exp(-error / std**2)


def motion_global_anchor_orientation_error_exp(
    env: ManagerBasedRlEnv, command_name: str, std: float
) -> torch.Tensor:
    """Computes the anchor orientation tracking reward.

    Args:
        env: The environment.
        command_name: Name of the motion command term.
        std: Standard deviation of the exponential kernel.

    Returns:
        ``exp(-e / std**2)`` of shape (num_envs,), where ``e`` is the squared
        rotation angle between the reference and robot anchor orientations.
    """
    command = cast(Motion, env.command_manager.get_term(command_name))
    error = (
        quat_error_magnitude(command.anchor_quat_w, command.robot_anchor_quat_w) ** 2
    )
    return torch.exp(-error / std**2)


def motion_relative_body_position_error_exp(
    env: ManagerBasedRlEnv,
    command_name: str,
    std: float,
    body_names: tuple[str, ...] | None = None,
) -> torch.Tensor:
    """Computes the body position tracking reward.

    Args:
        env: The environment.
        command_name: Name of the motion command term.
        std: Standard deviation of the exponential kernel.
        body_names: Names of the bodies to include. None includes all tracked
            bodies.

    Returns:
        ``exp(-e / std**2)`` of shape (num_envs,), where ``e`` is the squared
        distance between the reference and robot body positions in the world
        frame, averaged over bodies.
    """
    command = cast(Motion, env.command_manager.get_term(command_name))
    body_indexes = _get_body_indexes(command, body_names)
    error = torch.sum(
        torch.square(
            command.body_pos_w[:, body_indexes]
            - command.robot_body_pos_w[:, body_indexes]
        ),
        dim=-1,
    )
    return torch.exp(-error.mean(-1) / std**2)


def motion_relative_body_orientation_error_exp(
    env: ManagerBasedRlEnv,
    command_name: str,
    std: float,
    body_names: tuple[str, ...] | None = None,
) -> torch.Tensor:
    """Computes the body orientation tracking reward.

    Args:
        env: The environment.
        command_name: Name of the motion command term.
        std: Standard deviation of the exponential kernel.
        body_names: Names of the bodies to include. None includes all tracked
            bodies.

    Returns:
        ``exp(-e / std**2)`` of shape (num_envs,), where ``e`` is the squared
        rotation angle between the reference and robot body orientations,
        averaged over bodies.
    """
    command = cast(Motion, env.command_manager.get_term(command_name))
    body_indexes = _get_body_indexes(command, body_names)
    error = (
        quat_error_magnitude(
            command.body_quat_w[:, body_indexes],
            command.robot_body_quat_w[:, body_indexes],
        )
        ** 2
    )
    return torch.exp(-error.mean(-1) / std**2)


def motion_global_body_linear_velocity_error_exp(
    env: ManagerBasedRlEnv,
    command_name: str,
    std: float,
    body_names: tuple[str, ...] | None = None,
) -> torch.Tensor:
    """Computes the body linear velocity tracking reward.

    Args:
        env: The environment.
        command_name: Name of the motion command term.
        std: Standard deviation of the exponential kernel.
        body_names: Names of the bodies to include. None includes all tracked
            bodies.

    Returns:
        ``exp(-e / std**2)`` of shape (num_envs,), where ``e`` is the squared
        world-frame linear velocity error, averaged over bodies.
    """
    command = cast(Motion, env.command_manager.get_term(command_name))
    body_indexes = _get_body_indexes(command, body_names)
    error = torch.sum(
        torch.square(
            command.body_lin_vel_w[:, body_indexes]
            - command.robot_body_lin_vel_w[:, body_indexes]
        ),
        dim=-1,
    )
    return torch.exp(-error.mean(-1) / std**2)


def motion_global_body_angular_velocity_error_exp(
    env: ManagerBasedRlEnv,
    command_name: str,
    std: float,
    body_names: tuple[str, ...] | None = None,
) -> torch.Tensor:
    """Computes the body angular velocity tracking reward.

    Args:
        env: The environment.
        command_name: Name of the motion command term.
        std: Standard deviation of the exponential kernel.
        body_names: Names of the bodies to include. None includes all tracked
            bodies.

    Returns:
        ``exp(-e / std**2)`` of shape (num_envs,), where ``e`` is the squared
        world-frame angular velocity error, averaged over bodies.
    """
    command = cast(Motion, env.command_manager.get_term(command_name))
    body_indexes = _get_body_indexes(command, body_names)
    error = torch.sum(
        torch.square(
            command.body_ang_vel_w[:, body_indexes]
            - command.robot_body_ang_vel_w[:, body_indexes]
        ),
        dim=-1,
    )
    return torch.exp(-error.mean(-1) / std**2)


def self_collision_cost(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    force_threshold: float = 10.0,
) -> torch.Tensor:
    """Computes the self-collision penalty.

    Args:
        env: The environment.
        sensor_name: Name of the self-collision contact sensor.
        force_threshold: Contact force norm, in newtons, above which a substep
            counts as colliding. Only used when the sensor keeps force history.

    Returns:
        Penalties of shape (num_envs,). With force history, the number of
        substeps whose contact force exceeds ``force_threshold``; otherwise, the
        current number of contacts.
    """
    sensor: ContactSensor = env.scene[sensor_name]
    data = sensor.data
    if data.force_history is not None:
        force_mag = torch.norm(data.force_history, dim=-1)  # [B, N, H]
        hit = (force_mag > force_threshold).any(dim=1)  # [B, H]
        return hit.sum(dim=-1).float()  # [B]
    assert data.found is not None
    return data.found.squeeze(-1)
