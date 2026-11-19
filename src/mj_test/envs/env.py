"""Motion tracking env shared by the simulator and the imagination."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import torch
from mjlab.utils.noise import NoiseCfg
from tensordict import TensorDict

from mj_test.mdp.motion import BodyState, MotionLoader, RobotState
from mj_test.mdp.observations import goal_window
from mj_test.wm.representation import actor_obs, critic_obs, proprio


@dataclass
class RewardTerm:
    """One weighted reward term.

    Attributes:
        func: Reward of the env's current step, ``func(env, **params)`` of shape
            (B,).
        weight: Weight of the term.
        params: Remaining keyword arguments of ``func``.
    """

    func: Callable[..., torch.Tensor]
    weight: float
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class TerminationTerm:
    """One termination condition.

    Attributes:
        error: Error of the env's current step, ``error(env)`` of shape (B,).
        threshold: Largest allowed error.
    """

    error: Callable[..., torch.Tensor]
    threshold: float


@dataclass(kw_only=True)
class EnvCfg:
    """Configuration of the tracking MDP, shared by the simulator and the imagination.

    Attributes:
        motion_file: Path of one reference clip, or paths of several clips.
        body_names: Tracked bodies, anchor first.
        rewards: Reward terms, keyed by name.
        terminations: Termination conditions, keyed by name.
        proprio_noise: Noise of the proprioceptive terms, keyed by the term names of
            ``representation.proprio``.
    """

    motion_file: str | list[str]
    body_names: tuple[str, ...]
    rewards: dict[str, RewardTerm]
    terminations: dict[str, TerminationTerm]
    proprio_noise: dict[str, NoiseCfg] = field(default_factory=dict)


class Env(ABC):
    """Motion tracking env whose robot state advances by a simulator or a world model.

    Args:
        cfg: MDP configuration.
        motion: Reference motion.
        dt: Env step duration in seconds.
        default_joint_pos: Default joint positions, shape (J,).
        joint_limits: Lower and upper soft joint position limits, shape (J, 2).
    """

    robot_state: RobotState
    frames: torch.Tensor
    action: torch.Tensor
    prev_action: torch.Tensor
    sim_ctx: torch.Tensor
    encoder_bias: torch.Tensor
    robot_bodies: BodyState
    ref_bodies: BodyState

    def __init__(
        self,
        cfg: EnvCfg,
        motion: MotionLoader,
        dt: float,
        default_joint_pos: torch.Tensor,
        joint_limits: torch.Tensor,
    ) -> None:
        self.cfg = cfg
        self.motion = motion
        self.dt = dt
        self.default_joint_pos = default_joint_pos
        self.joint_limits = joint_limits
        self.max_step_reward = dt * sum(
            term.weight for term in cfg.rewards.values() if term.weight > 0.0
        )

    @abstractmethod
    def step(
        self, action: torch.Tensor
    ) -> tuple[TensorDict, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """Advances every env by one step.

        Args:
            action: Raw actions of shape (B, A).
        """

    def goal(self) -> torch.Tensor:
        """Computes the goal of the current state.

        Returns:
            The goal window of shape (B, goal_dim).
        """
        return goal_window(
            self.motion,
            self.frames,
            self.robot_state.root_pos,
            self.robot_state.root_quat,
        )

    @torch.compile
    def observations(self) -> TensorDict:
        """Builds the observations of the current state.

        Returns:
            The noisy ``actor`` observations ``[proprio, goal]`` and the ``critic``
            observations, of batch size (B,).
        """
        terms = proprio(
            self.robot_state, self.action, self.encoder_bias, self.default_joint_pos
        )
        for name, noise in self.cfg.proprio_noise.items():
            terms[name] = noise.apply(terms[name])
        goal = self.goal()
        return TensorDict(
            {
                "actor": actor_obs(torch.cat(list(terms.values()), dim=-1), goal),
                "critic": critic_obs(self.robot_state, goal, self.sim_ctx),
            },
            batch_size=self.frames.shape,
        )

    @torch.compile
    def _update(
        self, next_state: RobotState, action: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """Records the step to ``next_state`` under ``action`` and evaluates it.

        Args:
            next_state: Robot state after the step.
            action: Raw actions of the step, shape (B, A).
        """
        self.prev_action, self.action = self.action, action
        self.robot_state = next_state
        self.robot_bodies = self.motion.body_state(next_state)
        self.ref_bodies = self.motion.bodies[self.frames]
        rewards = self.cfg.rewards.items()
        terms = {name: term.func(self, **term.params) for name, term in rewards}
        reward = sum(
            torch.nan_to_num(terms[name] * term.weight * self.dt, 0.0, 0.0, 0.0)
            for name, term in rewards
        )
        errors = {
            name: term.error(self) / term.threshold
            for name, term in self.cfg.terminations.items()
        }
        self.frames = torch.minimum(
            self.frames + 1, self.motion.last_frame(self.frames)
        )
        return reward, terms, errors
