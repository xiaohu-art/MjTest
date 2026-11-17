"""Observation terms for the motion tracking task."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch

from mjlab.utils.lab_api.math import matrix_from_quat, subtract_frame_transforms

from .commands import Motion

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

__all__ = [
    "motion_anchor_pos_b",
    "motion_anchor_ori_b",
    "robot_body_pos_b",
    "robot_body_ori_b",
]


def motion_anchor_pos_b(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    """Computes the reference anchor position in the robot anchor frame.

    Args:
        env: The environment.
        command_name: Name of the motion command term.

    Returns:
        Positions of shape (num_envs, 3).
    """
    command = cast(Motion, env.command_manager.get_term(command_name))
    pos, _ = subtract_frame_transforms(
        command.robot_anchor_pos_w,
        command.robot_anchor_quat_w,
        command.anchor_pos_w,
        command.anchor_quat_w,
    )
    return pos.view(env.num_envs, -1)


def motion_anchor_ori_b(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    """Computes the reference anchor orientation in the robot anchor frame.

    Args:
        env: The environment.
        command_name: Name of the motion command term.

    Returns:
        First two columns of the rotation matrix, flattened to shape (num_envs, 6).
    """
    command = cast(Motion, env.command_manager.get_term(command_name))
    _, ori = subtract_frame_transforms(
        command.robot_anchor_pos_w,
        command.robot_anchor_quat_w,
        command.anchor_pos_w,
        command.anchor_quat_w,
    )
    mat = matrix_from_quat(ori)
    return mat[..., :2].reshape(mat.shape[0], -1)


def robot_body_pos_b(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    """Computes the tracked body positions in the robot anchor frame.

    Args:
        env: The environment.
        command_name: Name of the motion command term.

    Returns:
        Positions flattened to shape (num_envs, num_bodies * 3).
    """
    command = cast(Motion, env.command_manager.get_term(command_name))
    num_bodies = len(command.cfg.body_names)
    pos_b, _ = subtract_frame_transforms(
        command.robot_anchor_pos_w[:, None, :].repeat(1, num_bodies, 1),
        command.robot_anchor_quat_w[:, None, :].repeat(1, num_bodies, 1),
        command.robot_body_pos_w,
        command.robot_body_quat_w,
    )
    return pos_b.view(env.num_envs, -1)


def robot_body_ori_b(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    """Computes the tracked body orientations in the robot anchor frame.

    Args:
        env: The environment.
        command_name: Name of the motion command term.

    Returns:
        First two columns of each body rotation matrix, flattened to shape
        (num_envs, num_bodies * 6).
    """
    command = cast(Motion, env.command_manager.get_term(command_name))
    num_bodies = len(command.cfg.body_names)
    _, ori_b = subtract_frame_transforms(
        command.robot_anchor_pos_w[:, None, :].repeat(1, num_bodies, 1),
        command.robot_anchor_quat_w[:, None, :].repeat(1, num_bodies, 1),
        command.robot_body_pos_w,
        command.robot_body_quat_w,
    )
    mat = matrix_from_quat(ori_b)
    return mat[..., :2].reshape(mat.shape[0], -1)
