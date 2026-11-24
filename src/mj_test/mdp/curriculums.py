"""Curriculum terms, called by ``SimEnv._reset_idx`` before the resets, as mjlab's
``CurriculumManager.compute``."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch

if TYPE_CHECKING:
    from mj_test.envs.sim_env import SimEnv
    from mj_test.mdp.command import MotionCommand

__all__ = ["force_levels"]


class force_levels:
    def __init__(
        self,
        env: SimEnv,
        initial_max: float,
        step: float,
        interval: int,
        promote_ratio: float,
    ) -> None:
        self.command = cast("MotionCommand", env.command)
        self.command.force_max = float(initial_max)
        self._return = torch.zeros((), device=env.device)
        self._length = torch.zeros((), device=env.device)
        self._next_check = int(interval)
        self._ratio: float | None = None

    def __call__(
        self,
        env: SimEnv,
        env_ids: torch.Tensor,
        initial_max: float,
        step: float,
        interval: int,
        promote_ratio: float,
    ) -> dict[str, float]:
        del initial_max
        command = self.command
        # The ended episodes, before the reset clears them.
        self._return += env.stats.episode_return[env_ids].sum()
        self._length += env.episode_length_buf[env_ids].sum()
        if env.common_step_counter >= self._next_check:
            self._next_check = env.common_step_counter + int(interval)
            length = float(self._length)
            if length > 0:
                self._ratio = float(self._return) / length / env.max_step_reward
                if self._ratio >= promote_ratio:
                    command.force_max = min(
                        command.force_max + float(step),
                        command.force_magnitude_range[1],
                    )
            self._return.zero_()
            self._length.zero_()
        state = {"force_max": command.force_max}
        if self._ratio is not None:
            state["reward_ratio"] = self._ratio
        return state
