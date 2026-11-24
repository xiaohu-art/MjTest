from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.utils.lab_api.math import (
    matrix_from_quat,
    quat_apply_inverse,
    quat_inv,
    quat_mul,
)

if TYPE_CHECKING:
    from mj_test.envs.env import Env

__all__ = [
    "base_ang_vel",
    "force_command",
    "joint_pos_rel",
    "joint_vel_rel",
    "last_action",
    "motion_body_error",
    "motion_goal",
    "projected_gravity",
    "robot_local_state",
    "sim_ctx",
]


def projected_gravity(env: Env) -> torch.Tensor:
    """Gravity direction in the root frame, shape (B, 3)."""
    return env.robot.projected_gravity_b


def base_ang_vel(env: Env) -> torch.Tensor:
    """Root angular velocity in the root frame, shape (B, 3)."""
    return env.robot.root_link_ang_vel_b


def force_command(env: Env, scale: float) -> torch.Tensor:
    """Force that each force body must exert, in the heading frame, divided by the
    fixed ``scale`` (N), (B, 3 * K)."""
    return env.command.force_cmd.flatten(1) / scale


def joint_pos_rel(env: Env) -> torch.Tensor:
    """Joint positions as the encoders read them, relative to the default, (B, J)."""
    robot = env.robot
    return robot.joint_pos_biased - robot.default_joint_pos


def joint_vel_rel(env: Env) -> torch.Tensor:
    """Joint velocities, shape (B, J)."""
    return env.robot.joint_vel


def last_action(env: Env) -> torch.Tensor:
    """Raw action of the last step, shape (B, A)."""
    return env.action_term.action


def motion_goal(env: Env) -> torch.Tensor:
    """Goal window of the motion command, shape (B, goal_dim)."""
    return env.command.goal()


def motion_body_error(env: Env) -> torch.Tensor:
    """Pose errors of the command's goal bodies over the goal window.
    shape (B, num_goal_frames * K * 9).
    """
    cmd = env.command
    idx = cmd.goal_body_idx
    # The window caches are without env origins.
    ref_pos = cmd._body_pos_w[:, :, idx] + env.env_origins[:, None, None]
    ref_quat = cmd._body_quat_w[:, :, idx]
    pos, quat = cmd.robot_body_pos_w[:, idx], cmd.robot_body_quat_w[:, idx]
    root_quat = env.robot.root_link_quat_w[:, None, None].expand_as(ref_quat)
    pos_error = quat_apply_inverse(root_quat, ref_pos - pos[:, None])
    rot_error = quat_mul(quat_inv(quat)[:, None].expand_as(ref_quat), ref_quat)
    rot_error = matrix_from_quat(rot_error)[..., :2]
    num_envs = ref_pos.shape[0]
    return torch.cat(
        [pos_error.reshape(num_envs, -1), rot_error.reshape(num_envs, -1)], dim=-1
    )


def robot_local_state(env: Env) -> torch.Tensor:
    robot = env.robot
    height = robot.root_link_pos_w[:, 2:3] - env.env_origins[:, 2:3]
    return torch.cat(
        [
            height,
            robot.projected_gravity_b,
            robot.root_link_lin_vel_b,
            robot.root_link_ang_vel_b,
            robot.joint_pos,
            robot.joint_vel,
        ],
        dim=-1,
    )


def sim_ctx(env: Env) -> torch.Tensor:
    """Normalized sim context, shape (B, C)."""
    return env.sim_ctx
