"""Generalized-coordinate representation of the robot state for the world model."""

from __future__ import annotations

import math

import torch
from mjlab.utils.lab_api.math import matrix_from_quat, quat_apply, quat_apply_inverse

from mj_test.mdp.motion import RobotState
from mj_test.utils.math import projected_gravity, quat_integrate


def local_state(state: RobotState) -> torch.Tensor:
    """Expresses the robot state in the root frame, without the root xy and yaw.

    Returns:
        ``[root z, gravity_b, lin_vel_b, ang_vel_b, joint_pos, joint_vel]`` of shape
        (..., 10 + 2 * J).
    """
    return torch.cat(
        [
            state.root_pos[..., 2:3],
            projected_gravity(state.root_quat),
            quat_apply_inverse(state.root_quat, state.root_lin_vel),
            quat_apply_inverse(state.root_quat, state.root_ang_vel),
            state.joint_pos,
            state.joint_vel,
        ],
        dim=-1,
    )


def proprio(
    state: RobotState,
    prev_action: torch.Tensor,
    encoder_bias: torch.Tensor,
    default_joint_pos: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Builds the noise-free proprioceptive terms that the onboard sensors measure.

    Args:
        state: Robot state.
        prev_action: Raw actions of the previous step, shape (..., A).
        encoder_bias: Joint encoder biases, shape (..., J).
        default_joint_pos: Default joint positions, shape (J,) or (..., J).

    Returns:
        The terms keyed by name: ``projected_gravity`` and ``base_ang_vel`` of shape
        (..., 3), ``joint_pos`` (``joint_pos + encoder_bias - default_joint_pos``) and
        ``joint_vel`` of shape (..., J), and ``actions`` of shape (..., A).
    """
    return {
        "projected_gravity": projected_gravity(state.root_quat),
        "base_ang_vel": quat_apply_inverse(state.root_quat, state.root_ang_vel),
        "joint_pos": state.joint_pos + encoder_bias - default_joint_pos,
        "joint_vel": state.joint_vel,
        "actions": prev_action,
    }


def actor_obs(proprio_obs: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
    """Builds the actor input ``[proprio_obs, goal]``."""
    return torch.cat([proprio_obs, goal], dim=-1)


def critic_obs(
    state: RobotState, goal: torch.Tensor, sim_ctx: torch.Tensor
) -> torch.Tensor:
    """Builds the critic input ``[local_state, goal, sim_ctx]``."""
    return torch.cat([local_state(state), goal, sim_ctx], dim=-1)


def delta_dim(state: RobotState) -> int:
    """Returns the size ``6 + J`` of the delta that ``integrate`` consumes."""
    return 3 + 3 + state.joint_pos.shape[-1]


def integrate(
    state: RobotState, delta: torch.Tensor, dt: float, c: float
) -> RobotState:
    """Advances the robot state by one env step with a semi-implicit update.

    Args:
        state: Robot state before the step.
        delta: Body-frame root linear and angular velocity deltas and joint
            velocity deltas, shape (..., 6 + J).
        dt: Env step duration in seconds.
        c: Weight of the velocity delta in the pose update.
    """
    dv = quat_apply(state.root_quat, delta[..., 0:3])
    dw = quat_apply(state.root_quat, delta[..., 3:6])
    dvj = delta[..., 6:]
    return RobotState(
        root_pos=state.root_pos + (state.root_lin_vel + c * dv) * dt,
        root_quat=quat_integrate(state.root_quat, state.root_ang_vel + c * dw, dt),
        joint_pos=state.joint_pos + (state.joint_vel + c * dvj) * dt,
        root_lin_vel=state.root_lin_vel + dv,
        root_ang_vel=state.root_ang_vel + dw,
        joint_vel=state.joint_vel + dvj,
        batch_size=state.batch_size,
    )


def loss_blocks(state: RobotState) -> dict[str, torch.Tensor]:
    """Splits the robot state into the blocks of the world model loss.

    Returns:
        The root position (..., 3), the flattened root rotation matrix times
        ``sqrt(0.5)`` (..., 9), the joint positions (..., J) and the joint
        velocities (..., J), keyed by block name.
    """
    return {
        "root_pos": state.root_pos,
        "root_ori": matrix_from_quat(state.root_quat).flatten(-2, -1) * math.sqrt(0.5),
        "joint_pos": state.joint_pos,
        "joint_vel": state.joint_vel,
    }
