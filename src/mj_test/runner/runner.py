"""Training loop over real collection, world model updates and policy updates."""

from __future__ import annotations

import datetime
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import torch
from omegaconf import DictConfig
from tensordict import TensorDict
from torch.utils.tensorboard import SummaryWriter

from mj_test.envs.imagination_env import ImaginationEnv
from mj_test.envs.sim_env import SimEnv
from mj_test.mdp.command import MotionCommand
from mj_test.policy.learner import ActorCriticPolicyLearner
from mj_test.policy.networks import Actor, Critic
from mj_test.runner.replay_buffer import ReplayBuffer
from mj_test.utils.forward_kinematics import GenCoords, KinematicModel
from mj_test.wm.dynamics import Dynamics
from mj_test.wm.learner import WorldModelLearner
from mj_test.wm.representation import delta_dim, local_state


class Ratio:
    def __init__(self, ratio: float) -> None:
        self._ratio = ratio
        self._credit = 0.0

    def __call__(self) -> int:
        self._credit += self._ratio
        repeats = int(self._credit)
        self._credit -= repeats
        return repeats


class Runner:
    """Trains the world model and the policy on transitions collected from the env."""

    def __init__(
        self, env: SimEnv, cfg: DictConfig, log_dir: Path | None = None
    ) -> None:
        self.env = env
        self.cfg = cfg
        self.log_dir = log_dir
        self.iteration = 0
        self.device = env.device

        self.obs, _ = env.reset()
        gen = GenCoords.from_entity(env.entity, env.env_origins)
        state_dim = local_state(gen).shape[-1]
        goal_dim = self.obs["goal"].shape[-1]
        proprio_dim = self.obs["proprio"].shape[-1]
        sim_ctx_dim = env.sim_context.dim
        action_dim = env.action_term.dim
        decimation = env.decimation

        command = cast(MotionCommand, env.command)
        self.dynamics = Dynamics(
            state_dim,
            sim_ctx_dim,
            delta_dim(gen),
            env.action_term,
            KinematicModel.from_entity(
                env.sim.mj_model, 
                env.entity, 
                command.force_body_names
            ),
            dt=env.dt,
            substep_coeff=(decimation + 1) / (2 * decimation),
            hidden_dims=tuple(cfg.dynamics.hidden_dims),
        ).to(self.device)
        self.actor = Actor(
            proprio_dim, goal_dim, action_dim, tuple(cfg.actor.hidden_dims)
        ).to(self.device)
        self.critic = Critic(
            self.obs["critic"].shape[-1], tuple(cfg.critic.hidden_dims)
        ).to(self.device)

        done = torch.zeros(env.num_envs, dtype=torch.bool, device=self.device)
        action = env.action_term.action
        row = TensorDict(
            {
                "gen": gen,
                "sim_ctx": env.sim_ctx,
                "time_steps": command.time_steps,
                "force_cmd": command.force_cmd,
                "force": command.external_force_w,
                "prev_action": action,
                "action": action,
                "terminated": done,
                "truncated": done,
            },
            batch_size=[env.num_envs],
        )
        self.buffer = ReplayBuffer(row, cfg.buffer_size, self.device)
        self.wm_learner = WorldModelLearner(self.dynamics, cfg.wm)
        self.imagination = ImaginationEnv(env, self.dynamics, cfg.termination_softness)
        self.policy_learner = ActorCriticPolicyLearner(
            self.actor, self.critic, self.imagination, cfg.policy
        )

    def run(self) -> None:
        """Runs ``max_iterations`` more iterations of collection and updates."""
        if self.log_dir is None:
            raise ValueError("Runner.run needs a log_dir.")
        cfg = self.cfg
        wm = cfg.wm
        should_train = Ratio(
            self.env.num_envs
            * cfg.replay_ratio
            / (wm.num_batches * wm.batch_size * wm.horizon)
        )
        writer = SummaryWriter(self.log_dir)
        losses: dict[str, torch.Tensor] = {}
        timer = {"collect": 0.0, "world_model": 0.0, "policy": 0.0}
        first = self.iteration
        last = first + cfg.max_iterations
        run_start = self._now()
        for it in range(first, last):
            start = self._now()
            self.collect()
            timer["collect"] += self._now() - start
            for _ in range(should_train()):
                if self.buffer.num_transitions < wm.horizon:
                    continue
                start = self._now()
                losses |= self.wm_learner.update(
                    self.buffer.sample(wm.horizon, wm.batch_size)
                    for _ in range(wm.num_batches)
                )
                timer["world_model"] += self._now() - start
                start = self._now()
                losses |= self.policy_learner.update(
                    self.buffer.sample(1, cfg.policy.batch_size).squeeze(1)
                    for _ in range(cfg.policy.num_batches)
                )
                timer["policy"] += self._now() - start
            self.iteration = it + 1
            if self.iteration % cfg.log_interval == 0:
                elapsed = self._now() - run_start
                self._log(writer, first, last, elapsed, losses, timer)
                timer = dict.fromkeys(timer, 0.0)
            if self.iteration % cfg.save_interval == 0:
                self.save()
        self.save()
        writer.close()

    @torch.no_grad()
    def collect(self) -> None:
        """Steps the env once with the noisy actor and stores the transition."""
        env = self.env
        gen = GenCoords.from_entity(env.entity, env.env_origins)
        sim_ctx = env.sim_ctx
        command = cast(MotionCommand, env.command)
        time_steps = command.time_steps
        force_cmd = command.force_cmd
        force = command.external_force_w  # Applied in the simulation this step.
        prev_action = env.action_term.action
        obs = self.obs
        action = self.actor(obs)
        action = action + self.cfg.action_noise * torch.randn_like(action)

        self.obs, _, terminated, truncated, _ = env.step(action)
        self.buffer.insert(
            TensorDict(
                {
                    "gen": gen,
                    "sim_ctx": sim_ctx,
                    "time_steps": time_steps,
                    "force_cmd": force_cmd,
                    "force": force,
                    "prev_action": prev_action,
                    "action": action,
                    "terminated": terminated & ~truncated,  # Falls only.
                    "truncated": truncated,
                },
                batch_size=[env.num_envs],
            )
        )
        self.wm_learner.update_normalization(gen, action)
        self.actor.update_normalization(obs)

    def save(self) -> None:
        """Saves the training state to ``model_<iteration>.pt`` in the log
        directory."""
        if self.log_dir is None:
            raise ValueError("Runner.save needs a log_dir.")
        checkpoint = {
            "iteration": self.iteration,
            "metadata": self._metadata(),
            "dynamics": self.dynamics.state_dict(),
            "sim_context": self.env.sim_context.state_dict(),
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "optimizers": {
                name: [optimizer.state_dict() for optimizer in optimizers]
                for name, optimizers in self._optimizers().items()
            },
            **self.env.command.state_dict(),
        }
        path = self.log_dir / f"model_{self.iteration}.pt"
        torch.save(checkpoint, path.with_suffix(".tmp"))
        path.with_suffix(".tmp").replace(path)

    def load(self, path: Path, networks_only: bool = False) -> None:
        """Loads a checkpoint written by ``save``.

        Args:
            path: Path of the checkpoint.
            networks_only: Whether to load only the networks and their normalizers,
                keeping the current optimizers, reset distribution and iteration.
        """
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        for key, value in self._metadata().items():
            saved = checkpoint["metadata"][key]
            if saved != value:
                raise ValueError(
                    f"{path} was trained with {key} {saved}, but the env has {value}."
                )
        self.dynamics.load_state_dict(self._dynamics_state(path, checkpoint))
        self.env.sim_context.load_state_dict(checkpoint["sim_context"])
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])
        if networks_only:
            return
        for name, optimizers in self._optimizers().items():
            states = checkpoint["optimizers"][name]
            for optimizer, state in zip(optimizers, states, strict=True):
                optimizer.load_state_dict(state)
        self.env.command.load_state_dict(checkpoint)
        self.iteration = checkpoint["iteration"]

    def get_inference_policy(self) -> Callable[[TensorDict], torch.Tensor]:
        """Returns the deterministic actor as a function of the env observations."""
        self.actor.eval()

        @torch.no_grad()
        def policy(obs: TensorDict) -> torch.Tensor:
            return self.actor(obs)

        return policy

    def _optimizers(self) -> dict[str, list[torch.optim.Optimizer]]:
        """Returns the world model, actor and critic optimizers, keyed by network."""
        return {
            "world_model": self.wm_learner.optimizers,
            "actor": self.policy_learner.actor_optimizers,
            "critic": self.policy_learner.critic_optimizers,
        }

    def _metadata(self) -> dict[str, list[str]]:
        """Returns the joint, action, tracked body and sim context names of the
        networks."""
        command = cast(MotionCommand, self.env.command)
        return {
            "joint_names": list(self.env.entity.joint_names),
            "action_names": list(self.dynamics.action_term.names),
            "body_names": list(command.body_names),
            "sim_context": [spec.name for spec in self.env.sim_context.specs],
        }

    def _dynamics_state(
        self, path: Path, checkpoint: dict[str, Any]
    ) -> dict[str, torch.Tensor]:
        """Returns the saved dynamics state without the action map of old checkpoints.

        Checkpoints written before ``JointPositionAction`` kept the action scale and offset
        in the dynamics. They must match the env's, which the dynamics now uses.
        """
        state = dict(checkpoint["dynamics"])
        targets = self.dynamics.action_term
        for key, value in (
            ("action_scale", targets.scale),
            ("action_offset", targets.offset),
        ):
            saved = state.pop(key, None)
            if saved is not None and not torch.allclose(saved, value):
                raise ValueError(f"{path} was trained with another {key} than the env.")
        return state

    def _log(
        self,
        writer: SummaryWriter,
        first: int,
        last: int,
        elapsed: float,
        losses: dict[str, torch.Tensor],
        timer: dict[str, float],
    ) -> None:
        """Writes the scalars of the last log interval to TensorBoard and the console.

        Args:
            writer: The TensorBoard writer.
            first: Iteration at which the current run started.
            last: Iteration at which the current run stops.
            elapsed: Wall time of the current run so far, in seconds.
            losses: Latest world model and policy statistics.
            timer: Seconds spent per phase since the last log.
        """
        scalars = {name: float(value) for name, value in losses.items()}
        scalars |= self.env.pop_log()
        interval_time = sum(timer.values())
        scalars["Perf/steps_per_second"] = (
            self.cfg.log_interval * self.env.num_envs / interval_time
        )
        scalars |= {f"Perf/{name}_time": value for name, value in timer.items()}

        it = self.iteration
        for name, value in scalars.items():
            writer.add_scalar(name, value, it)
        width = max(len(name) for name in scalars) + 2
        lines = [f"{f' Iteration {it}/{last} ':#^{width + 12}}"]
        lines += [
            f"{name:>{width}} {value:.4f}" for name, value in sorted(scalars.items())
        ]
        eta = elapsed / (it - first) * (last - it)
        lines += [
            "-" * (width + 12),
            f"{'Iteration time':>{width}} {interval_time / self.cfg.log_interval:.2f}s",
            f"{'Time elapsed':>{width}} {datetime.timedelta(seconds=int(elapsed))}",
            f"{'ETA':>{width}} {datetime.timedelta(seconds=int(eta))}",
        ]
        print("\n".join(lines))

    def _now(self) -> float:
        """Returns the wall time in seconds after the pending device work finishes."""
        if torch.device(self.device).type == "cuda":
            torch.cuda.synchronize(self.device)
        return time.perf_counter()
