from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import torch
from mjlab.utils.lab_api.math import (
    matrix_from_quat,
    quat_error_magnitude,
    quat_from_euler_xyz,
    quat_mul,
    sample_uniform,
    subtract_frame_transforms,
)
from omegaconf import DictConfig

from mj_test.utils.forward_kinematics import GenCoords, KinematicModel
from mj_test.utils.reset_sampler import ResetSampler

if TYPE_CHECKING:
    from mj_test.envs.env import Env
    from mj_test.envs.sim_env import SimEnv

__all__ = ["MotionCommand", "MotionLibrary"]


class MotionLibrary:
    """Reference clips."""

    def __init__(
        self, motion_file: str, kinematics: KinematicModel, device: str = "cpu"
    ) -> None:
        with np.load(motion_file) as data:
            states = torch.from_numpy(data["states"]).to(device)
            self.motion_ends = torch.from_numpy(data["motion_ends"]).to(device)
            self.motion_names: list[str] = data["motion_names"].tolist()
            self.fps = float(data["fps"])
        num_joints = kinematics.default_joint_pos.shape[0]
        self.time_step_total = states.shape[0]
        root_pos, root_quat, root_lin_vel, root_ang_vel, joint_pos, joint_vel = (
            states.split([3, 4, 3, 3, num_joints, num_joints], dim=-1)
        )
        self.gen = GenCoords(
            root_pos=root_pos,
            root_quat=root_quat,
            joint_pos=joint_pos,
            root_lin_vel=root_lin_vel,
            root_ang_vel=root_ang_vel,
            joint_vel=joint_vel,
            batch_size=[self.time_step_total],
        )
        self.kinematics = kinematics

    def body_state(
        self, frames: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        bodies = self.kinematics.bodies(self.gen[frames])
        return bodies.pos, bodies.quat, bodies.lin_vel, bodies.ang_vel

    def last_frame(self, frames: torch.Tensor) -> torch.Tensor:
        clip = torch.searchsorted(self.motion_ends, frames, right=True)
        return self.motion_ends[clip] - 1


class MotionCommand:
    def __init__(self, cfg: DictConfig, env: SimEnv) -> None:
        self.cfg = cfg
        self.env: Env = env
        self.entity = env.entity
        self.body_names = tuple(cfg.body_names)  # Anchor first.
        body_ids = self.entity.find_bodies(self.body_names, preserve_order=True)[0]
        self.body_ids = torch.tensor(body_ids, dtype=torch.long, device=env.device)
        self.anchor_id = body_ids[0]

        kinematics = KinematicModel.from_entity(
            env.sim.mj_model, self.entity, self.body_names
        )
        self.motion = MotionLibrary(cfg.motion_file, kinematics, device=env.device)
        self.num_goal_frames = int(cfg.num_goal_frames)
        self.goal_frame_stride = int(cfg.goal_frame_stride)
        self._goal_offsets = (
            torch.arange(self.num_goal_frames, device=env.device)
            * self.goal_frame_stride
        )
        n = env.num_envs
        self.set_time_steps(torch.zeros(n, dtype=torch.long, device=env.device))
        self.sampler: ResetSampler | None = ResetSampler(
            cfg.sampler, self.motion, env.dt, n, env.max_step_reward, env.device
        )

    def like(self, env: Env) -> MotionCommand:
        command = copy.copy(self)
        command.env = env
        command.sampler = None
        return command

    def set_time_steps(self, time_steps: torch.Tensor) -> None:
        self.time_steps = time_steps
        # The tracked bodies' reference states at ``time_steps``, without env origins.
        (
            self._body_pos_w,
            self._body_quat_w,
            self._body_lin_vel_w,
            self._body_ang_vel_w,
        ) = self.motion.body_state(time_steps)

    # Reference at the current frames.

    @property
    def joint_pos(self) -> torch.Tensor:
        return self.motion.gen.joint_pos[self.time_steps]

    @property
    def joint_vel(self) -> torch.Tensor:
        return self.motion.gen.joint_vel[self.time_steps]

    @property
    def body_pos_w(self) -> torch.Tensor:
        return self._body_pos_w + self.env.env_origins[:, None]

    @property
    def body_quat_w(self) -> torch.Tensor:
        return self._body_quat_w

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        return self._body_lin_vel_w

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        return self._body_ang_vel_w

    @property
    def anchor_pos_w(self) -> torch.Tensor:
        return self._body_pos_w[:, 0] + self.env.env_origins

    @property
    def anchor_quat_w(self) -> torch.Tensor:
        return self._body_quat_w[:, 0]

    @property
    def anchor_lin_vel_w(self) -> torch.Tensor:
        return self._body_lin_vel_w[:, 0]

    @property
    def anchor_ang_vel_w(self) -> torch.Tensor:
        return self._body_ang_vel_w[:, 0]

    @property
    def clip_end(self) -> torch.Tensor:
        return self.time_steps == self.motion.last_frame(self.time_steps)

    # Robot at the tracked bodies.

    @property
    def robot_joint_pos(self) -> torch.Tensor:
        return self.env.robot.joint_pos

    @property
    def robot_joint_vel(self) -> torch.Tensor:
        return self.env.robot.joint_vel

    @property
    def robot_body_pos_w(self) -> torch.Tensor:
        return self.env.robot.body_link_pos_w[:, self.body_ids]

    @property
    def robot_body_quat_w(self) -> torch.Tensor:
        return self.env.robot.body_link_quat_w[:, self.body_ids]

    @property
    def robot_body_lin_vel_w(self) -> torch.Tensor:
        return self.env.robot.body_link_lin_vel_w[:, self.body_ids]

    @property
    def robot_body_ang_vel_w(self) -> torch.Tensor:
        return self.env.robot.body_link_ang_vel_w[:, self.body_ids]

    @property
    def robot_anchor_pos_w(self) -> torch.Tensor:
        return self.env.robot.body_link_pos_w[:, self.anchor_id]

    @property
    def robot_anchor_quat_w(self) -> torch.Tensor:
        return self.env.robot.body_link_quat_w[:, self.anchor_id]

    @property
    def robot_anchor_lin_vel_w(self) -> torch.Tensor:
        return self.env.robot.body_link_lin_vel_w[:, self.anchor_id]

    @property
    def robot_anchor_ang_vel_w(self) -> torch.Tensor:
        return self.env.robot.body_link_ang_vel_w[:, self.anchor_id]

    def goal(self) -> torch.Tensor:
        """The goal over ``num_goal_frames`` reference frames at stride
        ``goal_frame_stride``, clamped to the clip end.

        Returns:
            shape (B, num_goal_frames * (2 * J + 9)).
        """
        time_steps = self.time_steps
        last = self.motion.last_frame(time_steps)
        idx = torch.minimum(time_steps[:, None] + self._goal_offsets, last[:, None])
        num_envs, num_frames = idx.shape
        ref = self.motion.gen[idx]
        robot = self.env.robot
        root_pos = robot.root_link_pos_w - self.env.env_origins
        root_quat = robot.root_link_quat_w
        rel_pos, rel_quat = subtract_frame_transforms(
            root_pos[:, None].expand(-1, num_frames, -1).reshape(-1, 3),
            root_quat[:, None].expand(-1, num_frames, -1).reshape(-1, 4),
            ref.root_pos.reshape(-1, 3),
            ref.root_quat.reshape(-1, 4),
        )
        rel_ori = matrix_from_quat(rel_quat)[..., :2]
        joints = torch.cat([ref.joint_pos, ref.joint_vel], dim=-1)
        return torch.cat(
            [
                joints.reshape(num_envs, -1),
                rel_pos.reshape(num_envs, -1),
                rel_ori.reshape(num_envs, -1),
            ],
            dim=-1,
        )

    def metrics(self) -> dict[str, torch.Tensor]:
        """Tracking errors of the current step, (B,) each, keyed by log name, as
        mjlab's ``_update_metrics``."""
        anchor_quat, robot_anchor_quat = self.anchor_quat_w, self.robot_anchor_quat_w
        errors = {
            "anchor_pos": torch.norm(
                self.anchor_pos_w - self.robot_anchor_pos_w, dim=-1
            ),
            "anchor_rot": quat_error_magnitude(anchor_quat, robot_anchor_quat),
            "anchor_lin_vel": torch.norm(
                self.anchor_lin_vel_w - self.robot_anchor_lin_vel_w, dim=-1
            ),
            "anchor_ang_vel": torch.norm(
                self.anchor_ang_vel_w - self.robot_anchor_ang_vel_w, dim=-1
            ),
            "body_pos": torch.norm(
                self.body_pos_w - self.robot_body_pos_w, dim=-1
            ).mean(-1),
            "body_rot": quat_error_magnitude(
                self.body_quat_w, self.robot_body_quat_w
            ).mean(-1),
            "body_lin_vel": torch.norm(
                self.body_lin_vel_w - self.robot_body_lin_vel_w, dim=-1
            ).mean(-1),
            "body_ang_vel": torch.norm(
                self.body_ang_vel_w - self.robot_body_ang_vel_w, dim=-1
            ).mean(-1),
            "joint_pos": torch.norm(self.joint_pos - self.robot_joint_pos, dim=-1),
            "joint_vel": torch.norm(self.joint_vel - self.robot_joint_vel, dim=-1),
        }
        return {f"Metrics/motion/error_{name}": v for name, v in errors.items()}

    def update_command(self) -> None:
        time_steps = self.time_steps
        if self.sampler is not None:
            env = cast("SimEnv", self.env)
            truncated = env.reset_time_outs
            clip_end = self.clip_end & truncated
            self.sampler.record(
                time_steps,
                env.reward_buf,
                env.reset_terminated & ~truncated,
                clip_end,
                truncated & ~clip_end,
            )
        self.set_time_steps(
            torch.minimum(time_steps + 1, self.motion.last_frame(time_steps))
        )

    def reset(self, env_ids: torch.Tensor) -> None:
        assert self.sampler is not None, "Only the simulated env's command resets."
        frames = self.sampler.sample(len(env_ids))
        # Every env, so the forward kinematics keeps one batch size.
        self.set_time_steps(self.time_steps.index_copy(0, env_ids, frames + 1))
        ref = self.motion.gen[frames]

        root_pos = ref.root_pos + self.env.env_origins[env_ids]
        root_ori = ref.root_quat
        root_lin_vel = ref.root_lin_vel
        root_ang_vel = ref.root_ang_vel

        pose_range = self.cfg.pose_range or {}
        range_list = [
            pose_range.get(key, (0.0, 0.0))
            for key in ["x", "y", "z", "roll", "pitch", "yaw"]
        ]
        ranges = torch.tensor(range_list, device=self.env.device)
        rand_samples = sample_uniform(
            ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.env.device
        )
        root_pos = root_pos + rand_samples[:, 0:3]
        orientations_delta = quat_from_euler_xyz(
            rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5]
        )
        root_ori = quat_mul(orientations_delta, root_ori)

        velocity_range = self.cfg.velocity_range or {}
        range_list = [
            velocity_range.get(key, (0.0, 0.0))
            for key in ["x", "y", "z", "roll", "pitch", "yaw"]
        ]
        ranges = torch.tensor(range_list, device=self.env.device)
        rand_samples = sample_uniform(
            ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.env.device
        )
        root_lin_vel = root_lin_vel + rand_samples[:, :3]
        root_ang_vel = root_ang_vel + rand_samples[:, 3:]

        joint_pos = ref.joint_pos
        joint_vel = ref.joint_vel
        joint_pos = joint_pos + sample_uniform(
            lower=self.cfg.joint_position_range[0],
            upper=self.cfg.joint_position_range[1],
            size=joint_pos.shape,
            device=joint_pos.device,
        )

        self._write_reference_state_to_sim(
            env_ids,
            root_pos,
            root_ori,
            root_lin_vel,
            root_ang_vel,
            joint_pos,
            joint_vel,
        )

    def reset_to_frame(self, env_ids: torch.Tensor, frame: int) -> None:
        """Places envs exactly on the reference state of one frame, without the
        reset noise."""
        frames = torch.full_like(env_ids, frame)
        self.set_time_steps(self.time_steps.index_copy(0, env_ids, frames + 1))
        ref = self.motion.gen[frames]
        self._write_reference_state_to_sim(
            env_ids,
            ref.root_pos + self.env.env_origins[env_ids],
            ref.root_quat,
            ref.root_lin_vel,
            ref.root_ang_vel,
            ref.joint_pos,
            ref.joint_vel,
        )

    def log(self) -> dict[str, float]:
        return {} if self.sampler is None else self.sampler.log()

    def state_dict(self) -> dict[str, Any]:
        return {} if self.sampler is None else self.sampler.state_dict()

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if self.sampler is not None:
            self.sampler.load_state_dict(state)

    def _write_reference_state_to_sim(
        self,
        env_ids: torch.Tensor,
        root_pos: torch.Tensor,
        root_ori: torch.Tensor,
        root_lin_vel: torch.Tensor,
        root_ang_vel: torch.Tensor,
        joint_pos: torch.Tensor,
        joint_vel: torch.Tensor,
    ) -> None:
        """Clip joint positions and write root + joint state to sim."""
        soft_limits = self.entity.data.soft_joint_pos_limits[env_ids]
        joint_pos = torch.clip(joint_pos, soft_limits[:, :, 0], soft_limits[:, :, 1])
        self.entity.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

        root_state = torch.cat([root_pos, root_ori, root_lin_vel, root_ang_vel], dim=-1)
        self.entity.write_root_state_to_sim(root_state, env_ids=env_ids)
        self.entity.reset(env_ids=env_ids)
