from __future__ import annotations

import torch
from mjlab.utils.lab_api.math import quat_apply_inverse, quat_mul


def projected_gravity(quat: torch.Tensor) -> torch.Tensor:
    """Expresses the world down direction in the frames of the quaternions.

    Args:
        quat: wxyz quaternions of shape (..., 4).

    Returns:
        Unit down vectors of shape (..., 3).
    """
    down_w = torch.zeros_like(quat[..., :3])
    down_w[..., 2] = -1.0
    return quat_apply_inverse(quat, down_w)


def quat_integrate(
    quat: torch.Tensor, ang_vel: torch.Tensor, dt: float
) -> torch.Tensor:
    """Integrate quaternion by angular velocity over dt (axis-angle method).

    Args:
      quat: (..., 4) current quaternion (w, x, y, z)
      ang_vel: (..., 3) angular velocity in world frame
      dt: time step

    Returns:
      (..., 4) next quaternion, normalized
    """
    omega = ang_vel * dt
    norm_sq = (omega * omega).sum(dim=-1, keepdim=True)
    norm_safe = torch.sqrt(norm_sq + 1e-8)
    half_angle = norm_safe / 2
    axis = omega / norm_safe
    delta_q = torch.cat([torch.cos(half_angle), torch.sin(half_angle) * axis], dim=-1)
    q_next = quat_mul(delta_q, quat)
    return q_next / torch.norm(q_next, dim=-1, keepdim=True)
