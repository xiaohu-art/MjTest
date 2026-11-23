"""Where simulated episodes start on the motion timeline."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import torch
from mjlab.utils.lab_api.math import sample_uniform
from omegaconf import DictConfig

if TYPE_CHECKING:
    from mj_test.mdp.command import MotionLibrary


class ReturnSampler:
    """Measures the return from each motion bin and turns it into a reset distribution.

    Args:
        cfg: The ``env.command.sampler.returns`` section of the config.
        bin_count: Number of reset sampling bins.
        num_envs: Number of envs.
        max_step_reward: Largest reward of one env step.
        device: Device of the buffers.
    """

    def __init__(
        self,
        cfg: DictConfig,
        bin_count: int,
        num_envs: int,
        max_step_reward: float,
        device: str = "cpu",
    ) -> None:
        if not 0.0 < cfg.gamma < 1.0:
            raise ValueError(f"gamma must be in (0, 1), got {cfg.gamma}.")
        if cfg.explore + cfg.keep >= 1.0:
            raise ValueError("explore + keep must be below 1.")
        self.cfg = cfg
        self.bin_count = bin_count
        self.v_max = max_step_reward / (1.0 - cfg.gamma)
        self.value = torch.full((bin_count,), self.v_max, device=device)
        self.count = torch.zeros(bin_count, device=device)
        self.effort = torch.zeros(bin_count, dtype=torch.float64, device=device)
        shape = (num_envs, cfg.horizon)
        self._bins = torch.zeros(shape, dtype=torch.long, device=device)
        self._next_bins = torch.zeros(shape, dtype=torch.long, device=device)
        self._rewards = torch.zeros(shape, device=device)
        self._terminated = torch.zeros(shape, dtype=torch.bool, device=device)
        self._clip_end = torch.zeros(shape, dtype=torch.bool, device=device)
        self._timed_out = torch.zeros(shape, dtype=torch.bool, device=device)
        self._step = 0

    def record(
        self,
        bins: torch.Tensor,
        next_bins: torch.Tensor,
        rewards: torch.Tensor,
        terminated: torch.Tensor,
        clip_end: torch.Tensor,
        timed_out: torch.Tensor,
    ) -> torch.Tensor | None:
        """Buffers one env step and updates the distribution every ``horizon`` steps.

        Args:
            bins: Bin of the reference frame of each env before the step, shape
                (num_envs,).
            next_bins: Bin of the reference frame after the step, shape (num_envs,).
            rewards: Rewards of the step, shape (num_envs,).
            terminated: Whether the episode ended in a fall, shape (num_envs,).
            clip_end: Whether the episode reached the last frame of its clip, shape
                (num_envs,).
            timed_out: Whether the episode timed out, shape (num_envs,).

        Returns:
            The reset distribution of shape (bin_count,) after an update, otherwise
            None.
        """
        col = self._step
        self._bins[:, col] = bins
        self._next_bins[:, col] = next_bins
        self._rewards[:, col] = rewards
        self._terminated[:, col] = terminated
        self._clip_end[:, col] = clip_end
        self._timed_out[:, col] = timed_out
        self._step += 1
        if self._step < self.cfg.horizon:
            return None
        self._step = 0
        self._update()
        return self._distribution()

    def state_dict(self) -> dict[str, torch.Tensor]:
        """Returns the ``value``, ``count`` and ``effort`` of every bin."""
        return {"value": self.value, "count": self.count, "effort": self.effort}

    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        """Restores the bin statistics in place."""
        self.value.copy_(state["value"])
        self.count.copy_(state["count"])
        self.effort.copy_(state["effort"])

    def _dones(self) -> torch.Tensor:
        """Returns the episode end flags of shape (num_envs, horizon)."""
        return self._terminated | self._clip_end | self._timed_out

    def _returns(self) -> torch.Tensor:
        """Computes the discounted return of each buffered step."""
        gamma = self.cfg.gamma
        rewards = self._rewards
        next_value = torch.where(
            self._terminated,
            0.0,
            torch.where(self._clip_end, self.value.mean(), self.value[self._next_bins]),
        )
        dones = self._dones()
        returns = torch.empty_like(rewards)
        next_return = next_value[:, -1]
        for t in reversed(range(self.cfg.horizon)):
            next_return = torch.where(dones[:, t], next_value[:, t], next_return)
            returns[:, t] = rewards[:, t] + gamma * next_return
            next_return = returns[:, t]
        return returns

    def _update(self) -> None:
        """Folds the buffered returns into the bin values."""
        returns = self._returns()
        dones = self._dones()
        num_envs, horizon = dones.shape
        episode = torch.cumsum(dones.long(), dim=1) - dones.long()
        envs = torch.arange(num_envs, device=dones.device).unsqueeze(1)
        key = (envs * horizon + episode) * self.bin_count + self._bins
        keys, inverse, visits = torch.unique(
            key.reshape(-1), return_inverse=True, return_counts=True
        )
        samples = torch.zeros(keys.numel(), device=returns.device)
        samples.scatter_add_(0, inverse, returns.reshape(-1))
        samples /= visits
        bins = keys % self.bin_count
        new = torch.zeros_like(self.count).scatter_add_(
            0, bins, torch.ones_like(samples)
        )
        total = torch.zeros_like(self.value).scatter_add_(0, bins, samples)
        self.count = self.count.clamp(max=self.cfg.memory) + new
        self.value = torch.where(
            new > 0, self.value + (total - new * self.value) / self.count, self.value
        )

    def _distribution(self) -> torch.Tensor:
        """Computes the reset distribution over the bins from the bin values."""
        cfg = self.cfg
        focus = torch.softmax(-cfg.steepness * self.value / self.v_max, dim=0)
        self.effort += focus.double().square()
        keep = (self.effort / self.effort.sum()).float()
        explore = cfg.explore / self.bin_count
        return explore + (1.0 - cfg.explore - cfg.keep) * focus + cfg.keep * keep


class ResetSampler:
    """Draws reset frames from bins over the concatenated motion timeline.

    Args:
        cfg: The ``env.command.sampler`` section of the config.
        motion: The reference motion.
        dt: Env step duration in seconds.
        num_envs: Number of envs.
        max_step_reward: Largest reward of one env step.
        device: Device of the buffers.
    """

    def __init__(
        self,
        cfg: DictConfig,
        motion: MotionLibrary,
        dt: float,
        num_envs: int,
        max_step_reward: float,
        device: str,
    ) -> None:
        self.motion = motion
        self.device = device
        self.bin_count = int(motion.time_step_total // (cfg.bin_s / dt)) + 1
        self.probs = torch.full((self.bin_count,), 1.0 / self.bin_count, device=device)
        self._returns = (
            ReturnSampler(
                cfg.returns, self.bin_count, num_envs, max_step_reward, device
            )
            if cfg.mode == "return"
            else None
        )

    def bins(self, frames: torch.Tensor) -> torch.Tensor:
        """Maps frames to their bins."""
        bins = frames * self.bin_count // self.motion.time_step_total
        return bins.clamp(max=self.bin_count - 1)

    def sample(self, num: int) -> torch.Tensor:
        """Draws reset frames, never the last frame of a clip, shape (num,)."""
        bins = torch.multinomial(self.probs, num, replacement=True)
        total, count = self.motion.time_step_total, self.bin_count
        lo = (bins * total + count - 1) // count
        hi = ((bins + 1) * total + count - 1) // count
        offset = sample_uniform(0.0, 1.0, (num,), device=self.device) * (hi - lo)
        frames = lo + offset.long()
        return torch.where(frames == self.motion.last_frame(frames), frames - 1, frames)

    def set_distribution(self, probs: torch.Tensor) -> None:
        """Sets the bin weights, ignoring invalid inputs.

        Args:
            probs: Unnormalized nonnegative bin weights of shape (bin_count,).
        """
        probs = probs.detach().to(self.device, dtype=torch.float).clamp_min(0.0)
        total = probs.sum()
        if probs.numel() != self.bin_count or not torch.isfinite(total) or total <= 0:
            return
        self.probs = probs / total

    def record(
        self,
        frames: torch.Tensor,
        rewards: torch.Tensor,
        terminated: torch.Tensor,
        clip_end: torch.Tensor,
        timed_out: torch.Tensor,
    ) -> None:
        """Feeds one env step to the return estimate in ``return`` mode.

        Args:
            frames: Frame that each env tracked in the step, shape (num_envs,).
            rewards: Rewards of the step, shape (num_envs,).
            terminated: Whether the episode ended in a fall, shape (num_envs,).
            clip_end: Whether the episode reached the last frame of its clip, shape
                (num_envs,).
            timed_out: Whether the episode timed out, shape (num_envs,).
        """
        if self._returns is None:
            return
        probs = self._returns.record(
            self.bins(frames),
            self.bins(frames + 1),
            rewards,
            terminated,
            clip_end,
            timed_out,
        )
        if probs is not None:
            self.set_distribution(probs)

    def log(self) -> dict[str, float]:
        """Returns the normalized entropy and the mode of the bin weights."""
        probs = self.probs
        entropy = -(probs * (probs + 1e-12).log()).sum()
        top_prob, top_bin = probs.max(dim=0)
        return {
            "Metrics/motion/sampling_entropy": float(entropy)
            / math.log(max(self.bin_count, 2)),
            "Metrics/motion/sampling_top1_prob": float(top_prob),
            "Metrics/motion/sampling_top1_bin": float(top_bin) / self.bin_count,
        }

    def state_dict(self) -> dict[str, Any]:
        """Returns the return estimate (None in ``uniform`` mode) and the weights."""
        returns = None if self._returns is None else self._returns.state_dict()
        return {"return_sampler": returns, "sampling_probs": self.probs}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restores a ``state_dict`` in ``return`` mode; ``uniform`` keeps its
        weights."""
        returns = state["return_sampler"]
        if self._returns is None or returns is None:
            return
        self._returns.load_state_dict(returns)
        self.set_distribution(state["sampling_probs"])
