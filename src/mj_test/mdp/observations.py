"""Observation terms for the motion tracking task."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import matrix_from_quat, subtract_frame_transforms

from .commands import MotionCommand, MotionLoader, RobotState

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

__all__ = [
    "NUM_GOAL_FRAMES",
    "GOAL_FRAME_STRIDE",
    "root_pos",
    "root_quat",
    "root_lin_vel",
    "root_ang_vel",
    "joint_pos",
    "joint_vel",
    "goal_window",
    "motion_goal",
]

DEFAULT_ASSET_CFG = SceneEntityCfg("robot")

NUM_GOAL_FRAMES = 10
GOAL_FRAME_STRIDE = 5


def root_pos(
    env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = DEFAULT_ASSET_CFG
) -> torch.Tensor:
    """Returns the root position from qpos, relative to the env origin.

    Args:
        env: The environment.
        asset_cfg: The robot entity.

    Returns:
        World-frame positions of shape (num_envs, 3).
    """
    state = RobotState.from_entity(env.scene[asset_cfg.name], env.scene.env_origins)
    return state.root_pos


def root_quat(
    env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = DEFAULT_ASSET_CFG
) -> torch.Tensor:
    """Returns the root orientation from qpos.

    Args:
        env: The environment.
        asset_cfg: The robot entity.

    Returns:
        World-frame wxyz quaternions of shape (num_envs, 4).
    """
    state = RobotState.from_entity(env.scene[asset_cfg.name], env.scene.env_origins)
    return state.root_quat


def root_lin_vel(
    env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = DEFAULT_ASSET_CFG
) -> torch.Tensor:
    """Returns the root linear velocity from qvel.

    Args:
        env: The environment.
        asset_cfg: The robot entity.

    Returns:
        World-frame linear velocities of shape (num_envs, 3).
    """
    state = RobotState.from_entity(env.scene[asset_cfg.name], env.scene.env_origins)
    return state.root_lin_vel


def root_ang_vel(
    env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = DEFAULT_ASSET_CFG
) -> torch.Tensor:
    """Returns the root angular velocity from qvel.

    Args:
        env: The environment.
        asset_cfg: The robot entity.

    Returns:
        World-frame angular velocities of shape (num_envs, 3).
    """
    state = RobotState.from_entity(env.scene[asset_cfg.name], env.scene.env_origins)
    return state.root_ang_vel


def joint_pos(
    env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = DEFAULT_ASSET_CFG
) -> torch.Tensor:
    """Returns the joint positions from qpos, without encoder bias.

    Args:
        env: The environment.
        asset_cfg: The robot entity.

    Returns:
        Joint positions of shape (num_envs, num_joints).
    """
    state = RobotState.from_entity(env.scene[asset_cfg.name], env.scene.env_origins)
    return state.joint_pos


def joint_vel(
    env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = DEFAULT_ASSET_CFG
) -> torch.Tensor:
    """Returns the joint velocities from qvel.

    Args:
        env: The environment.
        asset_cfg: The robot entity.

    Returns:
        Joint velocities of shape (num_envs, num_joints).
    """
    state = RobotState.from_entity(env.scene[asset_cfg.name], env.scene.env_origins)
    return state.joint_vel


def goal_window_indices(motion: MotionLoader, frames: torch.Tensor) -> torch.Tensor:
    """Computes the frame indices of the goal window.

    Args:
        motion: The reference motion.
        frames: Current frame of each env, shape (num_envs,).

    Returns:
        Frame indices of shape (num_envs, NUM_GOAL_FRAMES), spaced
        ``GOAL_FRAME_STRIDE`` apart and clamped to the last frame of each clip.
    """
    offsets = torch.arange(NUM_GOAL_FRAMES, device=frames.device) * GOAL_FRAME_STRIDE
    return torch.minimum(frames[:, None] + offsets, motion.last_frame(frames)[:, None])


def goal_window(
    motion: MotionLoader,
    frames: torch.Tensor,
    root_pos: torch.Tensor,
    root_quat: torch.Tensor,
) -> torch.Tensor:
    """Computes the goal over the strided window of reference frames.

    Args:
        motion: The reference motion.
        frames: Current frame of each env, shape (num_envs,).
        root_pos: Robot root positions of shape (num_envs, 3), without env origins.
        root_quat: Robot root wxyz quaternions of shape (num_envs, 4).

    Returns:
        Three blocks, each flattened over the window frames: the per-frame
        ``[joint_pos, joint_vel]``, the relative root positions, and the first two
        rotation matrix columns of the relative root orientations. The shape is
        (num_envs, NUM_GOAL_FRAMES * (2 * num_joints + 9)).
    """
    idx = goal_window_indices(motion, frames)
    num_envs, num_frames = idx.shape
    ref = motion.state[idx]
    joints = torch.cat([ref.joint_pos, ref.joint_vel], dim=-1)
    rel_pos, rel_quat = subtract_frame_transforms(
        root_pos[:, None].expand(-1, num_frames, -1).reshape(-1, 3),
        root_quat[:, None].expand(-1, num_frames, -1).reshape(-1, 4),
        ref.root_pos.reshape(-1, 3),
        ref.root_quat.reshape(-1, 4),
    )
    rel_ori = matrix_from_quat(rel_quat)[..., :2]
    return torch.cat(
        [
            joints.reshape(num_envs, -1),
            rel_pos.reshape(num_envs, -1),
            rel_ori.reshape(num_envs, -1),
        ],
        dim=-1,
    )


def motion_goal(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    """Computes the goal from the motion command and the robot root pose in qpos.

    Args:
        env: The environment.
        command_name: Name of the motion command term.

    Returns:
        The goal of shape (num_envs, NUM_GOAL_FRAMES * (2 * num_joints + 9)), laid
        out as in ``goal_window``.
    """
    command = cast(MotionCommand, env.command_manager.get_term(command_name))
    state = RobotState.from_entity(command.robot, env.scene.env_origins)
    return goal_window(
        command.motion, command.time_steps, state.root_pos, state.root_quat
    )
