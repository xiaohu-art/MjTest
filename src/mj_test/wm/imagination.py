"""Differentiable stand-in for the tracking env, driven by the world model."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch
from mjlab.utils.noise import NoiseModelCfg
from tensordict import TensorDict

from mj_test import mdp
from mj_test.mdp.commands import MotionCommand, RobotState, Transition
from mj_test.wm.representation import actor_obs, critic_obs, proprio

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

    from mj_test.wm.dynamics import Dynamics
    from mj_test.wm.sim_context import SimContext


class ImaginationEnv:
    """Rolls out imagined transitions with the world model.

    Args:
        env: The tracking env, read once for its reward and termination terms.
        dynamics: The world model.
        sim_context: The sim context of the env, which normalizes ``sim_ctx``.
        termination_softness: Width of the sigmoid that turns the termination
            thresholds into a termination probability.
    """

    def __init__(
        self,
        env: ManagerBasedRlEnv,
        dynamics: Dynamics,
        sim_context: SimContext,
        termination_softness: float = 0.05,
    ) -> None:
        if termination_softness <= 0.0:
            raise ValueError(
                f"termination_softness must be positive, got {termination_softness}."
            )
        command = cast(MotionCommand, env.command_manager.get_term("motion"))
        self.dynamics = dynamics
        self.sim_context = sim_context
        self.motion = command.motion
        self.dt = env.step_dt

        rewards = env.reward_manager
        self.reward_cfgs = {
            name: rewards.get_term_cfg(name) for name in rewards.active_terms
        }
        terminations = env.termination_manager
        self.termination_cfgs = {
            name: cfg
            for name in terminations.active_terms
            if not (cfg := terminations.get_term_cfg(name)).time_out
        }
        for name, cfg in self.reward_cfgs.items():
            if cfg.func is not mdp.transition_reward:
                raise ValueError(
                    f"The reward term {name!r} is not a transition_reward."
                )
        for name, cfg in self.termination_cfgs.items():
            if cfg.func is not mdp.transition_termination:
                raise ValueError(
                    f"The termination term {name!r} is not a transition_termination."
                )
        observations = env.observation_manager
        self.proprio_names = list(observations.active_terms["proprio"])
        self.proprio_noise = {}
        for name in self.proprio_names:
            cfg = observations.get_term_cfg("proprio", name)
            if (
                isinstance(cfg.noise, NoiseModelCfg)
                or cfg.clip is not None
                or cfg.scale is not None
                or cfg.delay_max_lag > 0
                or cfg.history_length > 0
            ):
                raise ValueError(
                    f"The proprio term {name!r} uses a noise model, clipping, "
                    "scaling, delay or history, which the imagination does not mirror."
                )
            if cfg.noise is not None:
                self.proprio_noise[name] = cfg.noise
        self.max_step_reward = self.dt * sum(
            cfg.weight for cfg in self.reward_cfgs.values() if cfg.weight > 0.0
        )
        self.termination_softness = termination_softness
        self.soft_joint_pos_limits = command.robot.data.soft_joint_pos_limits[0].clone()
        self.default_joint_pos = command.robot.data.default_joint_pos[0].clone()

    def reset(self, batch: TensorDict, horizon: int) -> TensorDict:
        """Starts imagined rollouts of ``horizon`` steps from real transitions.

        Args:
            batch: Start transitions of batch size (B,), holding the ``RobotState``
                fields, ``prev_action``, ``frame`` and ``sim_ctx``.
            horizon: Number of steps of the rollouts.

        Returns:
            The ``actor`` and ``critic`` observations, of batch size (B,).
        """
        self.dynamics.eval().requires_grad_(False)
        names = [field.name for field in RobotState.fields()]
        self.state = RobotState.from_tensordict(batch.select(*names)).detach()
        self.prev_action = batch["prev_action"].detach()
        self.sim_ctx = batch["sim_ctx"].detach()
        self.encoder_bias = self.sim_context.denormalize(self.sim_ctx, "encoder_bias")
        self.frames = self._frames(batch["frame"], horizon)
        self.reference = self.motion.body[self.frames[:-1]]
        self.t = 0
        self.goal = mdp.goal_window(
            self.motion, self.frames[0], self.state.root_pos, self.state.root_quat
        )
        return self._observations()

    @torch.compile
    def step(
        self, action: torch.Tensor
    ) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        """Advances the imagined rollouts by one env step.

        Args:
            action: Raw actions of shape (B, action_dim).

        Returns:
            The ``actor`` and ``critic`` observations of the next state, the reward
            of shape (B,), the termination probability of shape (B,), and the
            unweighted reward terms keyed by reward term name.
        """
        next_state = self.dynamics(self.state, action, self.sim_ctx)
        batch_size = next_state.batch_size
        transition = Transition(
            state=next_state,
            body=self.motion.body_state(next_state),
            ref=self.reference[self.t],
            action=action,
            prev_action=self.prev_action,
            joint_limits=self.soft_joint_pos_limits.expand(*batch_size, -1, -1),
            batch_size=batch_size,
        )
        terms = self._reward_terms(transition)
        reward = sum(
            value * self.reward_cfgs[name].weight * self.dt
            for name, value in terms.items()
        )
        term_prob = self._termination_prob(transition)

        self.state, self.prev_action = next_state, action
        self.t += 1
        self.goal = mdp.goal_window(
            self.motion, self.frames[self.t], next_state.root_pos, next_state.root_quat
        )
        return self._observations(), reward, term_prob, terms

    def _reward_terms(self, transition: Transition) -> dict[str, torch.Tensor]:
        """Computes the unweighted reward terms of the env on an imagined step.

        Args:
            transition: Imagined transition of the step.

        Returns:
            The value of each term with shape (B,), keyed by reward term name.
        """
        terms = {}
        for name, cfg in self.reward_cfgs.items():
            params = dict(cfg.params)
            terms[name] = params.pop("term")(transition, **params)
        return terms

    def _termination_prob(self, transition: Transition) -> torch.Tensor:
        """Computes the soft termination of the env on an imagined step.

        Args:
            transition: Imagined transition of the step.

        Returns:
            The termination probability of shape (B,), a sigmoid of the largest
            error relative to its threshold.
        """
        normalized_errors = [
            cfg.params["error"](transition) / cfg.params["threshold"]
            for cfg in self.termination_cfgs.values()
        ]
        normalized_error = torch.stack(normalized_errors).amax(0)
        return torch.sigmoid((normalized_error - 1.0) / self.termination_softness)

    def _frames(self, frame: torch.Tensor, horizon: int) -> torch.Tensor:
        """Computes the reference frames of the rollouts.

        Args:
            frame: Start frames of shape (B,).
            horizon: Number of steps of the rollouts.

        Returns:
            Frames of shape (horizon + 1, B), advancing by one per step and clamped
            to the last frame of each clip.
        """
        steps = torch.arange(horizon + 1, device=frame.device)
        return torch.minimum(frame + steps[:, None], self.motion.last_frame(frame))

    def _proprio(self) -> torch.Tensor:
        """Builds the noisy proprioceptive observation of the current state.

        Returns:
            The proprioceptive observation of shape (B, proprio_dim), with the term
            order and noise of the env's ``proprio`` observation group.
        """
        terms = proprio(
            self.state, self.prev_action, self.encoder_bias, self.default_joint_pos
        )
        for name, noise in self.proprio_noise.items():
            terms[name] = noise.apply(terms[name])
        return torch.cat([terms[name] for name in self.proprio_names], dim=-1)

    def _observations(self) -> TensorDict:
        """Builds the observations of the current state.

        Returns:
            The ``actor`` and ``critic`` observations, of batch size (B,).
        """
        return TensorDict(
            {
                "actor": actor_obs(self._proprio(), self.goal),
                "critic": critic_obs(self.state, self.goal, self.sim_ctx),
            },
            batch_size=self.frames.shape[1:],
        )
