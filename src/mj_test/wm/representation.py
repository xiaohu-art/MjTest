"""How the world model advances and compares generalized coordinates."""

from __future__ import annotations

import math

import torch
from mjlab.utils.lab_api.math import matrix_from_quat, quat_apply, quat_apply_inverse

from mj_test.utils.forward_kinematics import GenCoords
from mj_test.utils.math import projected_gravity, quat_integrate


def local_state(state: GenCoords) -> torch.Tensor:
    """The state in the root frame, without root xy and yaw, the input of the
    dynamics, shape (..., 10 + 2 * J)."""
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


def delta_dim(state: GenCoords) -> int:
    """Returns the size ``6 + J`` of the delta that ``integrate`` consumes."""
    return 3 + 3 + state.joint_pos.shape[-1]


def integrate(state: GenCoords, delta: torch.Tensor, dt: float, c: float) -> GenCoords:
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
    return GenCoords(
        root_pos=state.root_pos + (state.root_lin_vel + c * dv) * dt,
        root_quat=quat_integrate(state.root_quat, state.root_ang_vel + c * dw, dt),
        joint_pos=state.joint_pos + (state.joint_vel + c * dvj) * dt,
        root_lin_vel=state.root_lin_vel + dv,
        root_ang_vel=state.root_ang_vel + dw,
        joint_vel=state.joint_vel + dvj,
        batch_size=state.batch_size,
    )


def loss_blocks(state: GenCoords) -> dict[str, torch.Tensor]:
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
