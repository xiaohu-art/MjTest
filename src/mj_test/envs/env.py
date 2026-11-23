"""Base env."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

import torch
from mjlab.utils.noise import NoiseCfg, UniformNoiseCfg
from omegaconf import DictConfig
from tensordict import TensorDict

from mj_test import mdp

if TYPE_CHECKING:
    from mjlab.entity import EntityData

    from mj_test.envs.imagination_env import ImaginedEntityData
    from mj_test.mdp.action import JointPositionAction


@dataclass
class RewardTerm:
    func: Callable[..., torch.Tensor]
    weight: float
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class ObservationTerm:
    func: Callable[..., torch.Tensor]
    params: dict[str, Any] = field(default_factory=dict)
    noise: NoiseCfg | None = None


@dataclass
class TerminationTerm:
    func: Callable[..., torch.Tensor]
    params: dict[str, Any] = field(default_factory=dict)
    threshold: float | None = None
    time_out: bool = False


def observation_group(cfg: DictConfig) -> dict[str, ObservationTerm]:
    noisy = cfg.get("enable_noise", False)
    terms = {}
    for name, term in cfg.terms.items():
        noise = term.get("noise") if noisy else None
        terms[name] = ObservationTerm(
            getattr(mdp, term.func),
            dict(term.get("params", {})),
            None if noise is None else UniformNoiseCfg(n_min=noise[0], n_max=noise[1]),
        )
    return terms


def termination_term(name: str, cfg: DictConfig) -> TerminationTerm:
    time_out = bool(cfg.get("time_out", False))
    threshold = cfg.get("threshold")
    if time_out == (threshold is not None):
        raise ValueError(
            f"Termination '{name}' needs either a threshold or time_out: true."
        )
    return TerminationTerm(
        getattr(mdp, cfg.func),
        dict(cfg.get("params", {})),
        None if threshold is None else float(threshold),
        time_out,
    )


class Command(Protocol):
    def reset(self, env_ids: torch.Tensor) -> None: ...

    def update_command(self) -> None: ...

    def like(self, env: Env) -> Command: ...

    def log(self) -> dict[str, float]: ...

    def state_dict(self) -> dict[str, Any]: ...

    def load_state_dict(self, state: dict[str, Any]) -> None: ...

    def metrics(self) -> dict[str, torch.Tensor]: ...


class Env(ABC):
    robot: EntityData | ImaginedEntityData
    sim_ctx: torch.Tensor  # Normalized sim context, (B, C).
    env_origins: torch.Tensor  # (B, 3).
    command: Command
    action_term: JointPositionAction
    num_envs: int

    def __init__(self, cfg: DictConfig, dt: float) -> None:
        self.cfg = cfg
        self.dt = dt
        self.rewards = {
            name: RewardTerm(
                getattr(mdp, term.func), term.weight, dict(term.get("params", {}))
            )
            for name, term in cfg.rewards.items()
        }
        self.terminations = {
            name: termination_term(name, term)
            for name, term in cfg.terminations.items()
        }
        self.observation_groups = {
            group: observation_group(group_cfg)
            for group, group_cfg in cfg.observations.items()
        }
        self.max_step_reward = dt * sum(
            term.weight for term in self.rewards.values() if term.weight > 0.0
        )

    @abstractmethod
    def step(
        self, action: torch.Tensor
    ) -> tuple[TensorDict, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """Advances every env by one step.

        Args:
            action: Raw actions of shape (B, A).
        """

    @torch.compile
    def observations(self) -> TensorDict:
        """
        Each observation group concatenated in order.
        Batch size (B,).
        """
        groups = {}
        for group, terms in self.observation_groups.items():
            values = []
            for term in terms.values():
                value = term.func(self, **term.params)
                values.append(value if term.noise is None else term.noise.apply(value))
            groups[group] = torch.cat(values, dim=-1)
        return TensorDict(groups, batch_size=[self.num_envs])

    @torch.compile
    def _post_step(
        self,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """Evaluates the reward terms and the error terms of the terminations."""
        terms = {}
        reward = self.env_origins.new_zeros(self.num_envs)
        for name, term in self.rewards.items():
            value = term.func(self, **term.params)
            terms[name] = value
            weighted = value * term.weight * self.dt
            reward = reward + torch.nan_to_num(
                weighted, nan=0.0, posinf=0.0, neginf=0.0
            )

        errors = {
            name: term.func(self, **term.params) / term.threshold
            for name, term in self.terminations.items()
            if term.threshold is not None
        }
        return reward, terms, errors

    def _time_outs(self) -> dict[str, torch.Tensor]:
        """Evaluates the time out terms of the terminations, keyed by term name."""
        return {
            name: term.func(self, **term.params)
            for name, term in self.terminations.items()
            if term.time_out
        }
