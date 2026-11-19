"""Termination errors for the motion tracking task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mj_test.utils.math import projected_gravity

if TYPE_CHECKING:
    from mj_test.envs.env import Env

__all__ = ["anchor_height_error", "anchor_tilt_error"]


def anchor_height_error(env: Env) -> torch.Tensor:
    """Computes the absolute anchor height error in meters."""
    return torch.abs(env.ref_bodies.pos[..., 0, 2] - env.robot_bodies.pos[..., 0, 2])


def anchor_tilt_error(env: Env) -> torch.Tensor:
    """Computes the anchor tilt error as the difference of the projected gravity z."""
    ref_down_b = projected_gravity(env.ref_bodies.quat[..., 0, :])
    down_b = projected_gravity(env.robot_bodies.quat[..., 0, :])
    return torch.abs(ref_down_b[..., 2] - down_b[..., 2])
