from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from mj_test.envs.env import Env

__all__ = [
    "base_ang_vel",
    "force_command",
    "joint_pos_rel",
    "joint_vel_rel",
    "last_action",
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


def force_command(env: Env) -> torch.Tensor:
    """Force that each force body must exert, in the heading frame, (B, 3 * K)."""
    return env.command.force_cmd.flatten(1)


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
