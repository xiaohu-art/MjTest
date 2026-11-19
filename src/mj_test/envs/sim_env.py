"""Motion tracking env on the MuJoCo Warp simulator."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.event_manager import EventManager, EventTermCfg
from mjlab.scene import Scene
from mjlab.scene.scene import SceneCfg
from mjlab.sim import SimulationCfg
from mjlab.sim.sim import Simulation
from mjlab.utils.lab_api.math import (
    quat_error_magnitude,
    quat_from_euler_xyz,
    quat_mul,
    sample_uniform,
)
from mjlab.utils.random import seed_rng
from mjlab.viewer.viewer_config import ViewerConfig
from tensordict import TensorDict

from mj_test.envs.env import Env, EnvCfg
from mj_test.mdp.motion import MotionLoader, RobotState
from mj_test.utils.forward_kinematics import ForwardKinematics
from mj_test.wm.sim_context import SimContext

_METRICS = (
    "error_anchor_pos",
    "error_anchor_rot",
    "error_anchor_lin_vel",
    "error_anchor_ang_vel",
    "error_body_pos",
    "error_body_rot",
    "error_body_lin_vel",
    "error_body_ang_vel",
    "error_joint_pos",
    "error_joint_vel",
)


@dataclass(kw_only=True)
class SimEnvCfg(EnvCfg):
    """Configuration of the simulated tracking env."""

    scene: SceneCfg
    sim: SimulationCfg
    action: JointPositionActionCfg
    entity_name: str = "robot"
    decimation: int = 4
    episode_length_s: float = 10.0
    events: dict[str, EventTermCfg] = field(default_factory=dict)
    pose_range: dict[str, tuple[float, float]] = field(default_factory=dict)
    velocity_range: dict[str, tuple[float, float]] = field(default_factory=dict)
    joint_position_range: tuple[float, float] = (0.0, 0.0)
    viewer: ViewerConfig = field(default_factory=ViewerConfig)
    seed: int | None = None


class SimEnv(Env):
    """Motion tracking env whose robot state advances by the MuJoCo Warp simulator.

    Args:
        cfg: Env configuration.
        device: Torch device of the simulation.
    """

    cfg: SimEnvCfg

    def __init__(self, cfg: SimEnvCfg, device: str) -> None:
        if cfg.seed is not None:
            seed_rng(cfg.seed)
        self.scene = Scene(cfg.scene, device=device)
        self.sim = Simulation(
            num_envs=self.scene.num_envs,
            cfg=cfg.sim,
            spec=self.scene.spec,
            variant_info=self.scene.collect_variant_info(),
            device=device,
        )
        self.scene.initialize(
            mj_model=self.sim.mj_model, model=self.sim.model, data=self.sim.data
        )
        self.sim.add_data_listener(self.scene.invalidate_sensor_caches)
        if self.scene.sensor_context is not None:
            self.sim.set_sensor_context(self.scene.sensor_context)
        self.robot = self.scene[cfg.entity_name]
        self.event_manager = EventManager(cfg.events, self)
        self.sim.expand_model_fields(self.event_manager.domain_randomization_fields)
        self.action_term = cfg.action.build(self)

        body_ids = self.robot.find_bodies(cfg.body_names, preserve_order=True)[0]
        fk = ForwardKinematics(
            self.sim.mj_model,
            self.robot.indexing.body_ids,
            self.robot.indexing.joint_q_adr,
            torch.tensor(body_ids, dtype=torch.long, device=device),
            device=device,
        )
        data = self.robot.data
        super().__init__(
            cfg,
            MotionLoader(cfg.motion_file, fk, device=device),
            dt=cfg.sim.mujoco.timestep * cfg.decimation,
            default_joint_pos=data.default_joint_pos[0].clone(),
            joint_limits=data.soft_joint_pos_limits[0].clone(),
        )
        self.event_manager.apply(mode="startup")
        self.sim_context = SimContext.from_env(self)
        self.sim_ctx = self.sim_context.read(self)
        self.encoder_bias = data.encoder_bias

        n = self.num_envs
        self.robot_state = RobotState.from_entity(self.robot, self.scene.env_origins)
        self.frames = torch.zeros(n, dtype=torch.long, device=device)
        self.action = torch.zeros(n, self.action_term.action_dim, device=device)
        self.prev_action = torch.zeros_like(self.action)
        self.episode_length = torch.zeros(n, dtype=torch.long, device=device)
        self.max_episode_length = math.ceil(cfg.episode_length_s / self.dt)
        self.bin_count = int(self.motion.time_step_total // (0.2 / self.dt)) + 1
        self.sampling_probs = torch.full(
            (self.bin_count,), 1.0 / self.bin_count, device=device
        )
        self._pose_ranges = torch.tensor(
            [
                cfg.pose_range.get(axis, (0.0, 0.0))
                for axis in ("x", "y", "z", "roll", "pitch", "yaw")
            ],
            device=device,
        )
        self._velocity_ranges = torch.tensor(
            [
                cfg.velocity_range.get(axis, (0.0, 0.0))
                for axis in ("x", "y", "z", "roll", "pitch", "yaw")
            ],
            device=device,
        )
        self._episode_return = torch.zeros(n, device=device)
        self._episode_sums = {
            name: torch.zeros(n, device=device) for name in cfg.rewards
        }
        reasons = ("time_out", "motion_end", *cfg.terminations)
        log_keys = (
            "steps",
            "episodes",
            "Episode/return",
            "Episode/length",
            *(f"Episode_Reward/{name}" for name in cfg.rewards),
            *(f"Episode_Termination/{name}" for name in reasons),
            *(f"Metrics/motion/{name}" for name in _METRICS),
        )
        self._log = {key: torch.zeros((), device=device) for key in log_keys}
        self.reset()

    @property
    def num_envs(self) -> int:
        """Number of simulated envs."""
        return self.scene.num_envs

    @property
    def device(self) -> str:
        """Torch device of the simulation."""
        return self.sim.device

    def step(
        self, action: torch.Tensor
    ) -> tuple[TensorDict, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """Advances every env by one step and resets the envs whose episode ended.

        Args:
            action: Raw actions of shape (num_envs, A).
        """
        self.action_term.process_actions(action)
        for _ in range(self.cfg.decimation):
            self.action_term.apply_actions()
            self.scene.write_data_to_sim()
            self.sim.step()
            self.scene.update(dt=self.cfg.sim.mujoco.timestep)
        frame = self.frames
        clip_end = frame == self.motion.last_frame(frame)
        next_state = RobotState.from_entity(self.robot, self.scene.env_origins)
        reward, terms, errors = self._update(next_state, action)
        self.episode_length += 1
        terminated = torch.stack(list(errors.values())).amax(0) > 1.0
        truncated = clip_end | (self.episode_length >= self.max_episode_length)
        self._record(frame, reward, terms, errors, clip_end, terminated, truncated)
        ended = (terminated | truncated).nonzero().squeeze(-1)
        if len(ended) > 0:
            self._reset_idx(ended)
        extras = {"clip_end": clip_end, "terms": terms}
        return self.observations(), reward, terminated, truncated, extras

    def reset(self, env_ids: torch.Tensor | None = None) -> tuple[TensorDict, dict]:
        """Resets envs to sampled reference states with noise.

        Args:
            env_ids: Envs to reset, or None for every env.
        """
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        self._reset_idx(env_ids)
        return self.observations(), {}

    def reset_to_frame(self, env_ids: torch.Tensor, frame: int) -> None:
        """Places envs exactly on the reference state of one frame.

        Args:
            env_ids: Envs to place.
            frame: Reference frame.
        """
        frames = torch.full_like(env_ids, frame)
        ref_bodies = self.motion.bodies[frames]
        ref_state = self.motion.states[frames]
        self._write_state(
            env_ids,
            frames,
            ref_bodies.pos[:, 0] + self.scene.env_origins[env_ids],
            ref_bodies.quat[:, 0],
            ref_bodies.lin_vel[:, 0],
            ref_bodies.ang_vel[:, 0],
            ref_state.joint_pos,
            ref_state.joint_vel,
        )

    def frame_bins(self, frames: torch.Tensor) -> torch.Tensor:
        """Maps frames to their reset sampling bins."""
        bins = frames * self.bin_count // self.motion.time_step_total
        return bins.clamp(max=self.bin_count - 1)

    def sample_frames(self, num: int) -> torch.Tensor:
        """Draws reset frames from the sampling distribution."""
        bins = torch.multinomial(self.sampling_probs, num, replacement=True)
        total, count = self.motion.time_step_total, self.bin_count
        lo = (bins * total + count - 1) // count
        hi = ((bins + 1) * total + count - 1) // count
        offset = sample_uniform(0.0, 1.0, (num,), device=self.device) * (hi - lo)
        frames = lo + offset.long()
        return torch.where(frames == self.motion.last_frame(frames), frames - 1, frames)

    def set_sampling_distribution(self, probs: torch.Tensor) -> None:
        """Sets the reset sampling distribution, ignoring invalid inputs.

        Args:
            probs: Unnormalized nonnegative bin weights of shape (bin_count,).
        """
        probs = probs.detach().to(self.device, dtype=torch.float).clamp_min(0.0)
        total = probs.sum()
        if probs.numel() != self.bin_count or not torch.isfinite(total) or total <= 0:
            return
        self.sampling_probs = probs / total

    def pop_log(self) -> dict[str, float]:
        """Returns the statistics accumulated since the last call and clears them.

        Returns:
            Per-episode means of the return, length, weighted reward terms per
            second and share of each termination reason over the episodes that
            ended, per-step means of the tracking errors, and the entropy and mode
            of the reset sampling distribution, keyed by log name.
        """
        sums = {key: float(value) for key, value in self._log.items()}
        for value in self._log.values():
            value.zero_()
        steps, episodes = sums.pop("steps"), sums.pop("episodes")
        log = {}
        for key, value in sums.items():
            count = steps if key.startswith("Metrics/") else episodes
            if count > 0:
                log[key] = value / count
        probs = self.sampling_probs
        entropy = -(probs * (probs + 1e-12).log()).sum()
        top_prob, top_bin = probs.max(dim=0)
        log["Metrics/motion/sampling_entropy"] = float(entropy) / math.log(
            max(self.bin_count, 2)
        )
        log["Metrics/motion/sampling_top1_prob"] = float(top_prob)
        log["Metrics/motion/sampling_top1_bin"] = float(top_bin) / self.bin_count
        return log

    def close(self) -> None:
        """Releases the env."""

    def _reset_idx(self, env_ids: torch.Tensor) -> None:
        """Resets envs to sampled reference states with noise.

        Args:
            env_ids: Envs to reset, a non-empty index tensor.
        """
        self.sim.reset(env_ids)
        self.scene.reset(env_ids)
        num = len(env_ids)
        frames = self.sample_frames(num)
        ref_bodies = self.motion.bodies[frames]
        ref_state = self.motion.states[frames]
        pose = sample_uniform(
            self._pose_ranges[:, 0], self._pose_ranges[:, 1], (num, 6), self.device
        )
        velocity = sample_uniform(
            self._velocity_ranges[:, 0],
            self._velocity_ranges[:, 1],
            (num, 6),
            self.device,
        )
        lower, upper = self.cfg.joint_position_range
        joint_noise = sample_uniform(
            lower, upper, ref_state.joint_pos.shape, self.device
        )
        root_quat = quat_mul(
            quat_from_euler_xyz(pose[:, 3], pose[:, 4], pose[:, 5]),
            ref_bodies.quat[:, 0],
        )
        self._write_state(
            env_ids,
            frames,
            ref_bodies.pos[:, 0] + self.scene.env_origins[env_ids] + pose[:, :3],
            root_quat,
            ref_bodies.lin_vel[:, 0] + velocity[:, :3],
            ref_bodies.ang_vel[:, 0] + velocity[:, 3:],
            ref_state.joint_pos + joint_noise,
            ref_state.joint_vel,
        )

    def _write_state(
        self,
        env_ids: torch.Tensor,
        frames: torch.Tensor,
        root_pos: torch.Tensor,
        root_quat: torch.Tensor,
        root_lin_vel: torch.Tensor,
        root_ang_vel: torch.Tensor,
        joint_pos: torch.Tensor,
        joint_vel: torch.Tensor,
    ) -> None:
        """Writes a robot state to the simulation and starts new episodes there.

        Args:
            env_ids: Envs to write, shape (n,).
            frames: Reference frames of the written states, shape (n,).
            root_pos: World-frame root positions, shape (n, 3).
            root_quat: Root wxyz quaternions, shape (n, 4).
            root_lin_vel: World-frame root linear velocities, shape (n, 3).
            root_ang_vel: World-frame root angular velocities, shape (n, 3).
            joint_pos: Joint positions, clipped to the soft limits, shape (n, J).
            joint_vel: Joint velocities, shape (n, J).
        """
        limits = self.robot.data.soft_joint_pos_limits[env_ids]
        joint_pos = torch.clip(joint_pos, limits[..., 0], limits[..., 1])
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
        root_state = torch.cat([root_pos, root_quat, root_lin_vel, root_ang_vel], -1)
        self.robot.write_root_state_to_sim(root_state, env_ids=env_ids)
        self.robot.reset(env_ids=env_ids)
        self.scene.write_data_to_sim(env_ids=env_ids)
        self.frames = self.frames.index_copy(0, env_ids, frames + 1)
        self.action = self.action.index_fill(0, env_ids, 0.0)
        self.episode_length[env_ids] = 0
        self.robot_state = RobotState.from_entity(self.robot, self.scene.env_origins)

    @torch.compile
    def _record(
        self,
        frame: torch.Tensor,
        reward: torch.Tensor,
        terms: dict[str, torch.Tensor],
        errors: dict[str, torch.Tensor],
        clip_end: torch.Tensor,
        terminated: torch.Tensor,
        truncated: torch.Tensor,
    ) -> None:
        """Accumulates the statistics of a step and of the episodes it ended."""
        log = self._log
        ended = (terminated | truncated).float()
        self._episode_return += reward
        log["steps"] += 1.0
        log["episodes"] += ended.sum()
        log["Episode/return"] += (self._episode_return * ended).sum()
        log["Episode/length"] += (self.episode_length * ended).sum()
        self._episode_return *= 1.0 - ended
        for name, term in self.cfg.rewards.items():
            sums = self._episode_sums[name]
            sums += terms[name] * term.weight * self.dt
            episode_reward = (sums * ended).sum() / self.cfg.episode_length_s
            log[f"Episode_Reward/{name}"] += episode_reward
            sums *= 1.0 - ended
        reasons = {
            "time_out": self.episode_length >= self.max_episode_length,
            "motion_end": clip_end,
            **{name: error > 1.0 for name, error in errors.items()},
        }
        for name, flag in reasons.items():
            log[f"Episode_Termination/{name}"] += (flag * ended).sum()
        ref, robot, ref_state = (
            self.ref_bodies,
            self.robot_bodies,
            self.motion.states[frame],
        )
        metrics = {
            "error_anchor_pos": torch.norm(ref.pos[:, 0] - robot.pos[:, 0], dim=-1),
            "error_anchor_rot": quat_error_magnitude(ref.quat[:, 0], robot.quat[:, 0]),
            "error_anchor_lin_vel": torch.norm(
                ref.lin_vel[:, 0] - robot.lin_vel[:, 0], dim=-1
            ),
            "error_anchor_ang_vel": torch.norm(
                ref.ang_vel[:, 0] - robot.ang_vel[:, 0], dim=-1
            ),
            "error_body_pos": torch.norm(ref.pos - robot.pos, dim=-1).mean(-1),
            "error_body_rot": quat_error_magnitude(ref.quat, robot.quat).mean(-1),
            "error_body_lin_vel": torch.norm(ref.lin_vel - robot.lin_vel, dim=-1).mean(
                -1
            ),
            "error_body_ang_vel": torch.norm(ref.ang_vel - robot.ang_vel, dim=-1).mean(
                -1
            ),
            "error_joint_pos": torch.norm(
                ref_state.joint_pos - self.robot_state.joint_pos, dim=-1
            ),
            "error_joint_vel": torch.norm(
                ref_state.joint_vel - self.robot_state.joint_vel, dim=-1
            ),
        }
        for name, value in metrics.items():
            log[f"Metrics/motion/{name}"] += value.mean()
