"""Goal observation of the motion tracking task."""

from __future__ import annotations

import torch
from mjlab.utils.lab_api.math import matrix_from_quat, subtract_frame_transforms

from .motion import MotionLoader

__all__ = ["NUM_GOAL_FRAMES", "GOAL_FRAME_STRIDE", "goal_window"]

NUM_GOAL_FRAMES = 10
GOAL_FRAME_STRIDE = 5


def goal_window_indices(motion: MotionLoader, frames: torch.Tensor) -> torch.Tensor:
    """Computes the goal window frames, strided and clamped to the end of each clip."""
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
    ref = motion.states[idx]
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
