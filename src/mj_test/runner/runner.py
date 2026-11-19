"""Training loop over real collection, world model updates and policy updates."""

from __future__ import annotations

import datetime
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

import torch
from mjlab.utils.os import dump_yaml
from tensordict import TensorDict
from torch.utils.tensorboard import SummaryWriter

from mj_test.envs.imagination_env import ImaginationEnv
from mj_test.envs.sim_env import SimEnv
from mj_test.policy.learner import ActorCriticPolicyLearner, ActorCriticPolicyLearnerCfg
from mj_test.policy.networks import Actor, ActorCfg, Critic, CriticCfg
from mj_test.runner.replay_buffer import FieldSpec, ReplayBuffer
from mj_test.runner.return_sampler import ReturnSampler, ReturnSamplerCfg
from mj_test.wm.dynamics import Dynamics, DynamicsCfg
from mj_test.wm.learner import WorldModelLearner, WorldModelLearnerCfg
from mj_test.wm.representation import delta_dim, local_state


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
    """Converts a fractional number of updates per step into whole updates."""

    def __init__(self, ratio: float) -> None:
        self._ratio = ratio
        self._credit = 0.0

    def __call__(self) -> int:
        """Advances by one step and returns the whole number of updates due."""
        self._credit += self._ratio
        repeats = int(self._credit)
        self._credit -= repeats
        return repeats


class Runner:
    """Trains the world model and the policy on transitions collected from the env.

    Args:
        env: The simulated tracking env.
        cfg: Runner hyperparameters.
        log_dir: Directory of the logs and checkpoints. None for a runner that only
            loads checkpoints.
    """

    def __init__(
        self, env: SimEnv, cfg: RunnerCfg, log_dir: Path | None = None
    ) -> None:
        self.env = env
        self.cfg = cfg
        self.log_dir = log_dir
        self.iteration = 0
        self.device = env.device
        self.sim_context = env.sim_context

        self.obs, _ = env.reset()
        state = env.robot_state
        state_dim = local_state(state).shape[-1]
        goal_dim = env.goal().shape[-1]
        proprio_dim = self.obs["actor"].shape[-1] - goal_dim
        sim_ctx_dim = self.sim_context.dim
        action_term = env.action_term
        action_dim = action_term.action_dim
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
            dt=env.dt,
            substep_coeff=(decimation + 1) / (2 * decimation),
            **asdict(cfg.dynamics),
        ).to(self.device)
        self.actor = Actor(proprio_dim, goal_dim, action_dim, **asdict(cfg.actor)).to(
            self.device
        )
        self.critic = Critic(self.obs["critic"].shape[-1], **asdict(cfg.critic)).to(
            self.device
        )

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
        self.imagination = ImaginationEnv(env, self.dynamics, cfg.termination_softness)
        self.policy_learner = ActorCriticPolicyLearner(
            self.actor, self.critic, self.imagination, cfg.policy
        )
        self.sampler = (
            ReturnSampler(
                cfg.return_sampler,
                env.bin_count,
                env.num_envs,
                self.imagination.max_step_reward,
                self.device,
            )
            if cfg.sampling == "return"
            else None
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
        state = env.robot_state.clone()
        frame = env.frames.clone()
        prev_action = env.action.clone()
        obs = self.obs["actor"]
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
                    "sim_ctx": env.sim_ctx,
                    "terminated": fell,
                    "truncated": truncated,
                },
                batch_size=[env.num_envs],
            )
        )
        self.wm_learner.update_normalization(state, action)
        self.actor.update_normalization(obs)

        if self.sampler is not None:
            clip_end = extras["clip_end"]
            probs = self.sampler.record(
                env.frame_bins(frame),
                env.frame_bins(frame + 1),
                reward,
                fell,
                clip_end,
                truncated & ~clip_end,
            )
            if probs is not None:
                env.set_sampling_distribution(probs)

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
            "sampling_probs": self.env.sampling_probs,
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
            self.env.set_sampling_distribution(checkpoint["sampling_probs"])
        self.iteration = checkpoint["iteration"]

    def get_inference_policy(self) -> Callable[[dict], torch.Tensor]:
        """Returns the deterministic actor as a function of the env observations."""
        self.actor.eval()

        @torch.no_grad()
        def policy(obs: dict) -> torch.Tensor:
            return self.actor(obs["actor"])

        return policy

    def _optimizers(self) -> dict[str, list[torch.optim.Optimizer]]:
        """Returns the world model, actor and critic optimizers, keyed by network."""
        return {
            "world_model": self.wm_learner.optimizers,
            "actor": self.policy_learner.actor_optimizers,
            "critic": self.policy_learner.critic_optimizers,
        }

    def _metadata(self) -> dict[str, list[str]]:
        """Returns the joint, action, body and sim context names of the networks."""
        return {
            "joint_names": list(self.env.robot.joint_names),
            "action_names": list(self.env.action_term.target_names),
            "body_names": list(self.env.cfg.body_names),
            "sim_context": [spec.name for spec in self.sim_context.specs],
        }

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
