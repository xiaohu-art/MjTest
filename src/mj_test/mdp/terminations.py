"""Termination terms for the motion tracking task."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch

from mj_test.utils.math import projected_gravity

from .commands import MotionCommand, Transition

if TYPE_CHECKING:
    from collections.abc import Callable

    from mjlab.envs import ManagerBasedRlEnv

__all__ = [
    "transition_termination",
    "anchor_height_error",
    "anchor_tilt_error",
    "motion_end",
]


def transition_termination(
    env: ManagerBasedRlEnv,
    error: Callable[[Transition], torch.Tensor],
    threshold: float,
) -> torch.Tensor:
    """Terminates when an error of the current env step exceeds a threshold.

    Args:
        env: The environment.
        error: Error function of a ``Transition``.
        threshold: Largest allowed error.

    Returns:
        Termination flags of shape (num_envs,).
    """
    return error(Transition.from_env(env)) > threshold


def anchor_height_error(x: Transition) -> torch.Tensor:
    """Computes the anchor height error.

    Args:
        x: Transition of the step.

    Returns:
        Absolute height difference, in meters, between the reference and robot
        anchors, of shape (...).
    """
    return torch.abs(x.ref.pos[..., 0, 2] - x.body.pos[..., 0, 2])


def anchor_tilt_error(x: Transition) -> torch.Tensor:
    """Computes the anchor tilt error.

    Args:
        x: Transition of the step.

    Returns:
        Absolute difference of shape (...) between the z components of the world
        down direction expressed in the reference and robot anchor frames.
    """
    ref_down_b = projected_gravity(x.ref.quat[..., 0, :])
    down_b = projected_gravity(x.body.quat[..., 0, :])
    return torch.abs(ref_down_b[..., 2] - down_b[..., 2])


def motion_end(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    """Ends the episode on the last frame of its clip.

    Args:
        env: The environment.
        command_name: Name of the motion command term.

    Returns:
        Episode end flags of shape (num_envs,).
    """
    command = cast(MotionCommand, env.command_manager.get_term(command_name))
    frames = command.time_steps
    return frames == command.motion.last_frame(frames)
