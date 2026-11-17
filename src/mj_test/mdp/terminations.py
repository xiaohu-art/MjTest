"""Termination terms for the motion tracking task."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch

from mjlab.utils.lab_api.math import quat_apply_inverse

from .commands import Motion
from .rewards import _get_body_indexes

if TYPE_CHECKING:
    from mjlab.entity import Entity
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.managers.scene_entity_config import SceneEntityCfg

__all__ = [
    "bad_anchor_pos",
    "bad_anchor_pos_z_only",
    "bad_anchor_ori",
    "bad_motion_body_pos",
    "bad_motion_body_pos_z_only",
]


def bad_anchor_pos(
    env: ManagerBasedRlEnv, command_name: str, threshold: float
) -> torch.Tensor:
    """Terminates when the anchor position error exceeds a threshold.

    Args:
        env: The environment.
        command_name: Name of the motion command term.
        threshold: Maximum allowed distance, in meters, between the reference and
            robot anchor positions.

    Returns:
        Termination flags of shape (num_envs,).
    """
    command = cast(Motion, env.command_manager.get_term(command_name))
    return (
        torch.norm(command.anchor_pos_w - command.robot_anchor_pos_w, dim=1) > threshold
    )


def bad_anchor_pos_z_only(
    env: ManagerBasedRlEnv, command_name: str, threshold: float
) -> torch.Tensor:
    """Terminates when the anchor height error exceeds a threshold.

    Args:
        env: The environment.
        command_name: Name of the motion command term.
        threshold: Maximum allowed height difference, in meters, between the
            reference and robot anchors.

    Returns:
        Termination flags of shape (num_envs,).
    """
    command = cast(Motion, env.command_manager.get_term(command_name))
    return (
        torch.abs(command.anchor_pos_w[:, -1] - command.robot_anchor_pos_w[:, -1])
        > threshold
    )


def bad_anchor_ori(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg,
    command_name: str,
    threshold: float,
) -> torch.Tensor:
    """Terminates when the anchor tilt error exceeds a threshold.

    Args:
        env: The environment.
        asset_cfg: Entity that provides the world gravity direction.
        command_name: Name of the motion command term.
        threshold: Maximum allowed difference between the z components of the
            gravity direction expressed in the reference and robot anchor frames.

    Returns:
        Termination flags of shape (num_envs,).
    """
    asset: Entity = env.scene[asset_cfg.name]
    command = cast(Motion, env.command_manager.get_term(command_name))
    motion_projected_gravity_b = quat_apply_inverse(
        command.anchor_quat_w, asset.data.gravity_vec_w
    )
    robot_projected_gravity_b = quat_apply_inverse(
        command.robot_anchor_quat_w, asset.data.gravity_vec_w
    )
    return (
        motion_projected_gravity_b[:, 2] - robot_projected_gravity_b[:, 2]
    ).abs() > threshold


def bad_motion_body_pos(
    env: ManagerBasedRlEnv,
    command_name: str,
    threshold: float,
    body_names: tuple[str, ...] | None = None,
) -> torch.Tensor:
    """Terminates when any selected body position error exceeds a threshold.

    Args:
        env: The environment.
        command_name: Name of the motion command term.
        threshold: Maximum allowed distance, in meters, between the reference
            and robot body positions.
        body_names: Names of the bodies to check. None checks all tracked bodies.

    Returns:
        Termination flags of shape (num_envs,).
    """
    command = cast(Motion, env.command_manager.get_term(command_name))
    body_indexes = _get_body_indexes(command, body_names)
    error = torch.norm(
        command.body_pos_w[:, body_indexes]
        - command.robot_body_pos_w[:, body_indexes],
        dim=-1,
    )
    return torch.any(error > threshold, dim=-1)


def bad_motion_body_pos_z_only(
    env: ManagerBasedRlEnv,
    command_name: str,
    threshold: float,
    body_names: tuple[str, ...] | None = None,
) -> torch.Tensor:
    """Terminates when any selected body height error exceeds a threshold.

    Args:
        env: The environment.
        command_name: Name of the motion command term.
        threshold: Maximum allowed height difference, in meters, between the
            reference and robot bodies.
        body_names: Names of the bodies to check. None checks all tracked bodies.

    Returns:
        Termination flags of shape (num_envs,).
    """
    command = cast(Motion, env.command_manager.get_term(command_name))
    body_indexes = _get_body_indexes(command, body_names)
    error = torch.abs(
        command.body_pos_w[:, body_indexes, -1]
        - command.robot_body_pos_w[:, body_indexes, -1]
    )
    return torch.any(error > threshold, dim=-1)
