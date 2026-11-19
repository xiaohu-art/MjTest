"""Training loop over real collection, world model updates and policy updates."""

from __future__ import annotations

import datetime
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal, cast

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.utils.os import dump_yaml
from tensordict import TensorDict
from torch.utils.tensorboard import SummaryWriter

from mj_test.mdp.commands import MotionCommand, RobotState
from mj_test.policy.learner import ActorCriticPolicyLearner, ActorCriticPolicyLearnerCfg
from mj_test.policy.networks import Actor, ActorCfg, Critic, CriticCfg
from mj_test.runner.replay_buffer import FieldSpec, ReplayBuffer
from mj_test.runner.return_sampler import ReturnSampler, ReturnSamplerCfg
from mj_test.wm.dynamics import Dynamics, DynamicsCfg
from mj_test.wm.imagination import ImaginationEnv
from mj_test.wm.learner import WorldModelLearner, WorldModelLearnerCfg
from mj_test.wm.representation import actor_obs, delta_dim, local_state
from mj_test.wm.sim_context import SimContext


@dataclass
class RunnerCfg:
    """Hyperparameters of the training loop."""

    seed: int = 42
    max_iterations: int = 1_000_000
    replay_ratio: int = 16
    buffer_size: int = 1000
    action_noise: float = 0.1
    sampling: Literal["uniform", "return"] = "return"
    termination_softness: float = 0.05
    log_interval: int = 10
    save_interval: int = 1000
    dynamics: DynamicsCfg = field(default_factory=DynamicsCfg)
    actor: ActorCfg = field(default_factory=ActorCfg)
    critic: CriticCfg = field(default_factory=CriticCfg)
    wm: WorldModelLearnerCfg = field(default_factory=WorldModelLearnerCfg)
    policy: ActorCriticPolicyLearnerCfg = field(
        default_factory=ActorCriticPolicyLearnerCfg
    )
    return_sampler: ReturnSamplerCfg = field(default_factory=ReturnSamplerCfg)


class Ratio:
    """Converts a fractional number of updates per step into whole updates.

    Args:
        ratio: Updates per step.
    """

    def __init__(self, ratio: float) -> None:
        self._ratio = ratio
        self._credit = 0.0

    def __call__(self) -> int:
        """Advances by one step and returns the number of updates due.

        Returns:
            The whole part of the accumulated updates, which is spent.
        """
        self._credit += self._ratio
        repeats = int(self._credit)
        self._credit -= repeats
        return repeats


class Runner:
    """Trains the world model and the policy on transitions collected from the env.

    Args:
        env: The tracking env.
        cfg: Runner hyperparameters.
        log_dir: Directory of the logs and checkpoints. None for a runner that only
            loads checkpoints.
    """

    def __init__(
        self, env: ManagerBasedRlEnv, cfg: RunnerCfg, log_dir: Path | None = None
    ) -> None:
        self.env = env
        self.cfg = cfg
        self.log_dir = log_dir
        self.iteration = 0
        self.device = env.device
        self.command = cast(MotionCommand, env.command_manager.get_term("motion"))
        self.sim_context = SimContext.from_env(env)

        self.obs, _ = env.reset()
        state = RobotState.from_dict(self.obs["state"], batch_dims=1)
        state_dim = local_state(state).shape[-1]
        goal_dim = self.obs["goal"].shape[-1]
        proprio_dim = self.obs["proprio"].shape[-1]
        sim_ctx_dim = self.sim_context.dim
        action_dim = env.action_manager.total_action_dim
        action_term = env.action_manager.get_term(env.action_manager.active_terms[0])
        action_shape = (env.num_envs, action_dim)
        scale = torch.as_tensor(action_term.scale, device=self.device).expand(
            action_shape
        )[0]
        offset = torch.as_tensor(action_term.offset, device=self.device).expand(
            action_shape
        )[0]
        decimation = env.cfg.decimation

        self.dynamics = Dynamics(
            state_dim,
            action_dim,
            sim_ctx_dim,
            delta_dim(state),
            scale.float(),
            offset.float(),
            dt=env.step_dt,
            substep_coeff=(decimation + 1) / (2 * decimation),
            **asdict(cfg.dynamics),
        ).to(self.device)
        self.actor = Actor(proprio_dim, goal_dim, action_dim, **asdict(cfg.actor)).to(
            self.device
        )
        self.critic = Critic(
            state_dim + goal_dim + sim_ctx_dim, **asdict(cfg.critic)
        ).to(self.device)

        schema = {
            name: FieldSpec(value.shape[-1]) for name, value in state.to_dict().items()
        } | {
            "action": FieldSpec(action_dim),
            "prev_action": FieldSpec(action_dim),
            "frame": FieldSpec(dtype=torch.long),
            "sim_ctx": FieldSpec(sim_ctx_dim),
            "terminated": FieldSpec(dtype=torch.bool),
            "truncated": FieldSpec(dtype=torch.bool),
        }
        self.buffer = ReplayBuffer(schema, cfg.buffer_size, env.num_envs, self.device)
        self.wm_learner = WorldModelLearner(self.dynamics, cfg.wm)
        self.imagination = ImaginationEnv(
            env, self.dynamics, self.sim_context, cfg.termination_softness
        )
        self.policy_learner = ActorCriticPolicyLearner(
            self.actor, self.critic, self.imagination, cfg.policy
        )
        self.sampler = (
            ReturnSampler(
                cfg.return_sampler,
                self.command.bin_count,
                env.num_envs,
                self.imagination.max_step_reward,
                self.device,
            )
            if cfg.sampling == "return"
            else None
        )

        self._episode_return = torch.zeros(env.num_envs, device=self.device)
        self._episode_length = torch.zeros(env.num_envs, device=self.device)
        self._finished_return = torch.zeros((), device=self.device)
        self._finished_length = torch.zeros((), device=self.device)
        self._finished_count = torch.zeros((), device=self.device)
        self._episode_logs: list[dict[str, float | torch.Tensor]] = []

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
        state = RobotState.from_dict(self.obs["state"], batch_dims=1).clone()
        frame = self.command.time_steps.clone()
        prev_action = env.action_manager.action.clone()
        sim_ctx = self.sim_context.read(env)
        obs = actor_obs(self.obs["proprio"], self.obs["goal"])
        action = self.actor(obs)
        action = action + self.cfg.action_noise * torch.randn_like(action)

        self.obs, reward, terminated, truncated, extras = env.step(action)
        fell = terminated & ~truncated
        self.buffer.insert(
            TensorDict(
                {
                    **state.to_dict(),
                    "action": action,
                    "prev_action": prev_action,
                    "frame": frame,
                    "sim_ctx": sim_ctx,
                    "terminated": fell,
                    "truncated": truncated,
                },
                batch_size=[env.num_envs],
            )
        )
        self.wm_learner.update_normalization(state, action)
        self.actor.update_normalization(obs)

        if self.sampler is not None:
            clip_end = env.termination_manager.get_term("motion_end")
            probs = self.sampler.record(
                self.command.frame_bins(frame),
                self.command.frame_bins(frame + 1),
                reward,
                fell,
                clip_end,
                truncated & ~clip_end,
            )
            if probs is not None:
                self.command.set_sampling_distribution(probs)
        self._track_episodes(reward, terminated | truncated, extras)

    def save(self) -> None:
        """Saves the configuration and the training state to the log directory."""
        if self.log_dir is None:
            raise ValueError("Runner.save needs a log_dir.")
        dump_yaml(self.log_dir / "params" / "runner.yaml", asdict(self.cfg))
        sampler = None if self.sampler is None else self.sampler.state_dict()
        checkpoint = {
            "iteration": self.iteration,
            "metadata": self._metadata(),
            "dynamics": self.dynamics.state_dict(),
            "sim_context": self.sim_context.state_dict(),
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "optimizers": {
                name: [optimizer.state_dict() for optimizer in optimizers]
                for name, optimizers in self._optimizers().items()
            },
            "return_sampler": sampler,
            "sampling_probs": self.command.sampling_probs,
        }
        path = self.log_dir / "model.pt"
        torch.save(checkpoint, path.with_suffix(".tmp"))
        path.with_suffix(".tmp").replace(path)

    def load(self, path: Path, networks_only: bool = False) -> None:
        """Loads a checkpoint written by ``save``.

        Args:
            path: Path of the checkpoint.
            networks_only: Whether to load only the networks and their normalizers,
                keeping the current optimizers, return sampler and iteration.
        """
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        for key, value in self._metadata().items():
            saved = checkpoint["metadata"][key]
            if saved != value:
                raise ValueError(
                    f"{path} was trained with {key} {saved}, but the env has {value}."
                )
        self.dynamics.load_state_dict(checkpoint["dynamics"])
        self.sim_context.load_state_dict(checkpoint["sim_context"])
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])
        if networks_only:
            return
        for name, optimizers in self._optimizers().items():
            states = checkpoint["optimizers"][name]
            for optimizer, state in zip(optimizers, states, strict=True):
                optimizer.load_state_dict(state)
        if self.sampler is not None and checkpoint["return_sampler"] is not None:
            self.sampler.load_state_dict(checkpoint["return_sampler"])
            self.command.set_sampling_distribution(checkpoint["sampling_probs"])
        self.iteration = checkpoint["iteration"]

    def get_inference_policy(self) -> Callable[[dict], torch.Tensor]:
        """Returns the deterministic actor as a function of the env observations.

        Returns:
            A function from the env observations to raw actions of shape
            (num_envs, action_dim).
        """
        self.actor.eval()

        @torch.no_grad()
        def policy(obs: dict) -> torch.Tensor:
            return self.actor(actor_obs(obs["proprio"], obs["goal"]))

        return policy

    def _optimizers(self) -> dict[str, list[torch.optim.Optimizer]]:
        """Returns the optimizers of the learners.

        Returns:
            The world model, actor and critic optimizers, keyed by network.
        """
        return {
            "world_model": self.wm_learner.optimizers,
            "actor": self.policy_learner.actor_optimizers,
            "critic": self.policy_learner.critic_optimizers,
        }

    def _metadata(self) -> dict[str, list[str]]:
        """Returns the env layout that the network inputs and outputs depend on.

        Returns:
            The joint, action target, tracked body and sim context spec names, in
            column order.
        """
        manager = self.env.action_manager
        action_term = manager.get_term(manager.active_terms[0])
        return {
            "joint_names": list(self.command.robot.joint_names),
            "action_names": list(action_term.target_names),
            "body_names": list(self.command.cfg.body_names),
            "sim_context": [spec.name for spec in self.sim_context.specs],
        }

    def _track_episodes(
        self, reward: torch.Tensor, done: torch.Tensor, extras: dict
    ) -> None:
        """Accumulates the returns and lengths of the episodes that end on the device.

        Args:
            reward: Rewards of the step, shape (num_envs,).
            done: Whether the env reset after the step, shape (num_envs,).
            extras: Extras returned by the env step.
        """
        if extras.get("log"):
            self._episode_logs.append(extras["log"])
        self._episode_return += reward
        self._episode_length += 1
        ended = done.float()
        self._finished_return += (self._episode_return * ended).sum()
        self._finished_length += (self._episode_length * ended).sum()
        self._finished_count += ended.sum()
        self._episode_return *= 1.0 - ended
        self._episode_length *= 1.0 - ended

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
        count = float(self._finished_count)
        if count > 0:
            scalars["Episode/return"] = float(self._finished_return) / count
            scalars["Episode/length"] = float(self._finished_length) / count
        for total in (
            self._finished_return,
            self._finished_length,
            self._finished_count,
        ):
            total.zero_()
        for key in dict.fromkeys(k for log in self._episode_logs for k in log):
            values = [float(log[key]) for log in self._episode_logs if key in log]
            scalars[key] = sum(values) / len(values)
        self._episode_logs.clear()
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
        """Returns the wall time after the pending device work finishes.

        Returns:
            Seconds since an arbitrary point.
        """
        if torch.device(self.device).type == "cuda":
            torch.cuda.synchronize(self.device)
        return time.perf_counter()
