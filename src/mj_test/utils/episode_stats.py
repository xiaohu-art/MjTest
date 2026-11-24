"""Episode and tracking statistics of the simulated envs, summed on the device."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from mj_test.envs.env import Env, RewardTerm


class EpisodeStats:
    def __init__(
        self,
        rewards: dict[str, RewardTerm],
        reasons: tuple[str, ...],
        metrics: tuple[str, ...],
        num_envs: int,
        dt: float,
        episode_length_s: float,
        device: str,
    ) -> None:
        self.rewards = rewards
        self.dt = dt
        self.episode_length_s = episode_length_s
        self.episode_return = torch.zeros(num_envs, device=device)
        self._episode_sums = {
            name: torch.zeros(num_envs, device=device) for name in rewards
        }
        keys = (
            "steps",
            "episodes",
            "Episode/return",
            "Episode/length",
            *(f"Episode_Reward/{name}" for name in rewards),
            *(f"Episode_Termination/{name}" for name in reasons),
            *metrics,
        )
        self._sums = {key: torch.zeros((), device=device) for key in keys}

    @torch.compile
    def record(
        self,
        env: Env,
        reward: torch.Tensor,
        terms: dict[str, torch.Tensor],
        episode_length: torch.Tensor,
        reasons: dict[str, torch.Tensor],
        ended: torch.Tensor,
    ) -> None:
        sums = self._sums
        ended = ended.float()
        self.episode_return += reward
        sums["steps"] += 1.0
        sums["episodes"] += ended.sum()
        sums["Episode/return"] += (self.episode_return * ended).sum()
        sums["Episode/length"] += (episode_length * ended).sum()
        for name, term in self.rewards.items():
            episode = self._episode_sums[name]
            episode += terms[name] * term.weight * self.dt
            per_second = (episode * ended).sum() / self.episode_length_s
            sums[f"Episode_Reward/{name}"] += per_second
        for name, flag in reasons.items():
            sums[f"Episode_Termination/{name}"] += (flag * ended).sum()
        for name, metric in env.command.metrics().items():
            sums[name] += metric.mean()

    def reset(self, env_ids: torch.Tensor) -> None:
        """Starts new episodes, after the curriculum has read the ended ones."""
        self.episode_return[env_ids] = 0.0
        for episode in self._episode_sums.values():
            episode[env_ids] = 0.0

    def pop(self) -> dict[str, float]:
        sums = {key: float(value) for key, value in self._sums.items()}
        for value in self._sums.values():
            value.zero_()
        steps, episodes = sums.pop("steps"), sums.pop("episodes")
        log = {}
        for key, value in sums.items():
            count = steps if key.startswith("Metrics/") else episodes
            if count > 0:
                log[key] = value / count
        return log
