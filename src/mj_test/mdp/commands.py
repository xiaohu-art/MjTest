from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, cast

import numpy as np
import torch
from mjlab.managers import CommandTerm, CommandTermCfg
from mjlab.utils.lab_api.math import (
    matrix_from_quat,
    quat_apply,
    quat_error_magnitude,
    quat_from_euler_xyz,
    quat_mul,
    sample_uniform,
)
from mjlab.viewer.debug_visualizer import DebugVisualizer
from tensordict import tensorclass

from mj_test.utils.forward_kinematics import ForwardKinematics

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Any

    import viser
    from mjlab.entity import Entity
    from mjlab.envs import ManagerBasedRlEnv


__all__ = [
    "BodyState",
    "MotionCommand",
    "MotionCommandCfg",
    "MotionLoader",
    "RobotState",
    "Transition",
]

_DESIRED_FRAME_COLORS = ((1.0, 0.5, 0.5), (0.5, 1.0, 0.5), (0.5, 0.5, 1.0))
_FK_CHUNK = 65536


@tensorclass
class RobotState:
    """Generalized coordinates and velocities of the robot.

    Attributes:
        root_pos: Root positions relative to the env origin, shape (..., 3).
        root_quat: Root wxyz quaternions, shape (..., 4).
        joint_pos: Joint positions, shape (..., J).
        root_lin_vel: Root linear velocities in the world frame, shape (..., 3).
        root_ang_vel: Root angular velocities in the world frame, shape (..., 3).
        joint_vel: Joint velocities, shape (..., J).
    """

    root_pos: torch.Tensor
    root_quat: torch.Tensor
    joint_pos: torch.Tensor
    root_lin_vel: torch.Tensor
    root_ang_vel: torch.Tensor
    joint_vel: torch.Tensor

    @classmethod
    def from_entity(cls, robot: Entity, env_origins: torch.Tensor) -> RobotState:
        """Reads the robot state from qpos and qvel.

        Args:
            robot: The floating-base robot entity.
            env_origins: Env origins of shape (num_envs, 3).

        Returns:
            The robot state of every env.
        """
        qpos, qvel = robot.data.data.qpos, robot.data.data.qvel
        root_q = robot.indexing.free_joint_q_adr
        root_v = robot.indexing.free_joint_v_adr
        root_quat = qpos[:, root_q[3:7]]
        return cls(
            root_pos=qpos[:, root_q[0:3]] - env_origins,
            root_quat=root_quat,
            joint_pos=robot.data.joint_pos,
            root_lin_vel=qvel[:, root_v[0:3]],
            root_ang_vel=quat_apply(root_quat, qvel[:, root_v[3:6]]),
            joint_vel=robot.data.joint_vel,
            batch_size=root_quat.shape[:1],
        )


@tensorclass
class BodyState:
    """Poses and velocities of the tracked bodies, anchor first.

    Attributes:
        pos: Positions relative to the env origin, shape (..., B, 3).
        quat: World-frame wxyz quaternions, shape (..., B, 4).
        lin_vel: World-frame linear velocities, shape (..., B, 3).
        ang_vel: World-frame angular velocities, shape (..., B, 3).
    """

    pos: torch.Tensor
    quat: torch.Tensor
    lin_vel: torch.Tensor
    ang_vel: torch.Tensor


@tensorclass
class Transition:
    """Quantities of one env step that the reward and termination terms read.

    Attributes:
        state: Robot state after the step.
        body: Tracked bodies of ``state``.
        ref: Reference bodies at the frame of the step.
        action: Raw actions of the step, shape (..., A).
        prev_action: Raw actions of the previous step, shape (..., A).
        joint_limits: Lower and upper soft joint position limits, shape (..., J, 2).
    """

    state: RobotState
    body: BodyState
    ref: BodyState
    action: torch.Tensor
    prev_action: torch.Tensor
    joint_limits: torch.Tensor

    @classmethod
    @torch.compile
    def from_env(cls, env: ManagerBasedRlEnv) -> Transition:
        """Reads the transition of the current env step from the sim.

        Args:
            env: The environment, after the physics substeps of the step.

        Returns:
            The transition of every env.
        """
        command = cast(MotionCommand, env.command_manager.get_term("motion"))
        state = RobotState.from_entity(command.robot, env.scene.env_origins)
        return cls(
            state=state,
            body=command.motion.body_state(state),
            ref=command.motion.body[command.time_steps],
            action=env.action_manager.action,
            prev_action=env.action_manager.prev_action,
            joint_limits=command.robot.data.soft_joint_pos_limits,
            batch_size=state.batch_size,
        )


class MotionLoader:
    """Loads reference motion clips and concatenates them along the frame axis.

    Args:
        motion_file: Path of one clip, or paths of several clips.
        fk: Forward kinematics of the tracked bodies.
        device: Device of the loaded tensors.

    Attributes:
        state: Reference robot state of every frame, batch size (time_step_total,).
        body: Reference tracked bodies of every frame, batch size
            (time_step_total,).
        fk: Forward kinematics of the tracked bodies.
        motion_ends: End frame (exclusive) of each clip, shape (num_clips,).
        time_step_total: Total number of frames.
    """

    def __init__(
        self,
        motion_file: str | list[str],
        fk: ForwardKinematics,
        device: str = "cpu",
    ) -> None:
        motion_files = [motion_file] if isinstance(motion_file, str) else motion_file
        clips: list[dict[str, np.ndarray]] = []
        for path in motion_files:
            with np.load(path) as data:
                clips.append(dict(data))

        def cat(arrays: list[np.ndarray]) -> torch.Tensor:
            return torch.from_numpy(np.concatenate(arrays)).to(
                device=device, dtype=torch.float32
            )

        joint_pos = cat([clip["joint_pos"] for clip in clips])
        self.time_step_total = joint_pos.shape[0]
        self.state = RobotState(
            root_pos=cat([clip["body_pos_w"][:, 0] for clip in clips]),
            root_quat=cat([clip["body_quat_w"][:, 0] for clip in clips]),
            joint_pos=joint_pos,
            root_lin_vel=cat([clip["body_lin_vel_w"][:, 0] for clip in clips]),
            root_ang_vel=cat([clip["body_ang_vel_w"][:, 0] for clip in clips]),
            joint_vel=cat([clip["joint_vel"] for clip in clips]),
            batch_size=[self.time_step_total],
        )
        self.fk = fk
        with torch.no_grad():
            chunks = self.state.split(_FK_CHUNK)
            self.body = torch.cat([self.body_state(chunk) for chunk in chunks])

        self.motion_ends = torch.as_tensor(
            np.cumsum([clip["joint_pos"].shape[0] for clip in clips]),
            dtype=torch.long,
            device=device,
        )

    def body_state(self, state: RobotState) -> BodyState:
        """Computes the tracked bodies of a robot state with forward kinematics.

        Args:
            state: Robot state.

        Returns:
            The tracked bodies of ``state``.
        """
        pos, quat, lin_vel, ang_vel = self.fk(
            state.root_pos,
            state.root_quat,
            state.joint_pos,
            state.root_lin_vel,
            state.root_ang_vel,
            state.joint_vel,
        )
        return BodyState(
            pos=pos,
            quat=quat,
            lin_vel=lin_vel,
            ang_vel=ang_vel,
            batch_size=state.batch_size,
        )

    def last_frame(self, frames: torch.Tensor) -> torch.Tensor:
        """Returns the last frame of the clip of each frame.

        Args:
            frames: Frame indices of any shape, below ``time_step_total``.

        Returns:
            The last frame indices, of the same shape.
        """
        clip = torch.searchsorted(self.motion_ends, frames, right=True)
        return self.motion_ends[clip] - 1


class MotionCommand(CommandTerm):
    cfg: MotionCommandCfg
    _env: ManagerBasedRlEnv

    def __init__(self, cfg: MotionCommandCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)

        self.robot: Entity = env.scene[cfg.entity_name]
        self.robot_anchor_body_index = self.robot.body_names.index(
            self.cfg.anchor_body_name
        )
        self.motion_anchor_body_index = self.cfg.body_names.index(
            self.cfg.anchor_body_name
        )
        if self.motion_anchor_body_index != 0:
            raise ValueError("The anchor body must be the first tracked body.")
        self.body_indexes = torch.tensor(
            self.robot.find_bodies(self.cfg.body_names, preserve_order=True)[0],
            dtype=torch.long,
            device=self.device,
        )

        fk = ForwardKinematics(
            env.sim.mj_model,
            self.robot.indexing.body_ids,
            self.robot.indexing.joint_q_adr,
            self.body_indexes,
            device=self.device,
        )
        self.motion = MotionLoader(self.cfg.motion_file, fk, device=self.device)
        self.time_steps = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )

        self.bin_count = int(self.motion.time_step_total // (0.2 / env.step_dt)) + 1
        self.sampling_probs = torch.full(
            (self.bin_count,), 1.0 / self.bin_count, device=self.device
        )
        axes = ("x", "y", "z", "roll", "pitch", "yaw")
        self._pose_ranges = torch.tensor(
            [cfg.pose_range.get(axis, (0.0, 0.0)) for axis in axes], device=self.device
        )
        self._velocity_ranges = torch.tensor(
            [cfg.velocity_range.get(axis, (0.0, 0.0)) for axis in axes],
            device=self.device,
        )

        self.metrics["error_anchor_pos"] = torch.zeros(
            self.num_envs, device=self.device
        )
        self.metrics["error_anchor_rot"] = torch.zeros(
            self.num_envs, device=self.device
        )
        self.metrics["error_anchor_lin_vel"] = torch.zeros(
            self.num_envs, device=self.device
        )
        self.metrics["error_anchor_ang_vel"] = torch.zeros(
            self.num_envs, device=self.device
        )
        self.metrics["error_body_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_body_rot"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_joint_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_joint_vel"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["sampling_entropy"] = torch.zeros(
            self.num_envs, device=self.device
        )
        self.metrics["sampling_top1_prob"] = torch.zeros(
            self.num_envs, device=self.device
        )
        self.metrics["sampling_top1_bin"] = torch.zeros(
            self.num_envs, device=self.device
        )

        self._ghost_model = None
        self._ghost_color = np.array(cfg.viz.ghost_color, dtype=np.float32)

    @property
    def command(self) -> torch.Tensor:
        return torch.cat([self.joint_pos, self.joint_vel], dim=1)

    @property
    def joint_pos(self) -> torch.Tensor:
        return self.motion.state.joint_pos[self.time_steps]

    @property
    def joint_vel(self) -> torch.Tensor:
        return self.motion.state.joint_vel[self.time_steps]

    @property
    def body_pos_w(self) -> torch.Tensor:
        return (
            self.motion.body.pos[self.time_steps]
            + self._env.scene.env_origins[:, None, :]
        )

    @property
    def body_quat_w(self) -> torch.Tensor:
        return self.motion.body.quat[self.time_steps]

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        return self.motion.body.lin_vel[self.time_steps]

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        return self.motion.body.ang_vel[self.time_steps]

    @property
    def anchor_pos_w(self) -> torch.Tensor:
        return (
            self.motion.body.pos[self.time_steps, self.motion_anchor_body_index]
            + self._env.scene.env_origins
        )

    @property
    def anchor_quat_w(self) -> torch.Tensor:
        return self.motion.body.quat[self.time_steps, self.motion_anchor_body_index]

    @property
    def anchor_lin_vel_w(self) -> torch.Tensor:
        return self.motion.body.lin_vel[self.time_steps, self.motion_anchor_body_index]

    @property
    def anchor_ang_vel_w(self) -> torch.Tensor:
        return self.motion.body.ang_vel[self.time_steps, self.motion_anchor_body_index]

    @property
    def robot_joint_pos(self) -> torch.Tensor:
        return self.robot.data.joint_pos

    @property
    def robot_joint_vel(self) -> torch.Tensor:
        return self.robot.data.joint_vel

    @property
    def robot_body_pos_w(self) -> torch.Tensor:
        return self.robot.data.body_link_pos_w[:, self.body_indexes]

    @property
    def robot_body_quat_w(self) -> torch.Tensor:
        return self.robot.data.body_link_quat_w[:, self.body_indexes]

    @property
    def robot_body_lin_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_link_lin_vel_w[:, self.body_indexes]

    @property
    def robot_body_ang_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_link_ang_vel_w[:, self.body_indexes]

    @property
    def robot_anchor_pos_w(self) -> torch.Tensor:
        return self.robot.data.body_link_pos_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_quat_w(self) -> torch.Tensor:
        return self.robot.data.body_link_quat_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_lin_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_link_lin_vel_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_ang_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_link_ang_vel_w[:, self.robot_anchor_body_index]

    def _update_metrics(self):
        self.metrics["error_anchor_pos"] = torch.norm(
            self.anchor_pos_w - self.robot_anchor_pos_w, dim=-1
        )
        self.metrics["error_anchor_rot"] = quat_error_magnitude(
            self.anchor_quat_w, self.robot_anchor_quat_w
        )
        self.metrics["error_anchor_lin_vel"] = torch.norm(
            self.anchor_lin_vel_w - self.robot_anchor_lin_vel_w, dim=-1
        )
        self.metrics["error_anchor_ang_vel"] = torch.norm(
            self.anchor_ang_vel_w - self.robot_anchor_ang_vel_w, dim=-1
        )

        self.metrics["error_body_pos"] = torch.norm(
            self.body_pos_w - self.robot_body_pos_w, dim=-1
        ).mean(dim=-1)
        self.metrics["error_body_rot"] = quat_error_magnitude(
            self.body_quat_w, self.robot_body_quat_w
        ).mean(dim=-1)

        self.metrics["error_body_lin_vel"] = torch.norm(
            self.body_lin_vel_w - self.robot_body_lin_vel_w, dim=-1
        ).mean(dim=-1)
        self.metrics["error_body_ang_vel"] = torch.norm(
            self.body_ang_vel_w - self.robot_body_ang_vel_w, dim=-1
        ).mean(dim=-1)

        self.metrics["error_joint_pos"] = torch.norm(
            self.joint_pos - self.robot_joint_pos, dim=-1
        )
        self.metrics["error_joint_vel"] = torch.norm(
            self.joint_vel - self.robot_joint_vel, dim=-1
        )

    def frame_bins(self, frames: torch.Tensor) -> torch.Tensor:
        bins = frames * self.bin_count // self.motion.time_step_total
        return bins.clamp(max=self.bin_count - 1)

    def sample_frames(self, num: int) -> torch.Tensor:
        """Draws start frames from the sampling distribution.

        Args:
            num: Number of frames.

        Returns:
            Frames of shape (num,), uniform over the frames that ``frame_bins`` maps
            to each drawn bin, with the last frame of each clip moved one back.
        """
        probs = self.sampling_probs
        bins = torch.multinomial(probs, num, replacement=True)
        total, count = self.motion.time_step_total, self.bin_count
        lo = (bins * total + count - 1) // count
        hi = ((bins + 1) * total + count - 1) // count
        offset = sample_uniform(0.0, 1.0, (num,), device=self.device) * (hi - lo)
        frames = lo + offset.long()
        frames = torch.where(
            frames == self.motion.last_frame(frames), frames - 1, frames
        )

        entropy = -(probs * (probs + 1e-12).log()).sum()
        self.metrics["sampling_entropy"][:] = (
            entropy / math.log(count) if count > 1 else 1.0
        )
        pmax, imax = probs.max(dim=0)
        self.metrics["sampling_top1_prob"][:] = pmax
        self.metrics["sampling_top1_bin"][:] = imax.float() / count
        return frames

    def set_sampling_distribution(self, probs: torch.Tensor) -> None:
        """Sets the distribution of the start bins, ignoring invalid inputs.

        Args:
            probs: Unnormalized nonnegative bin weights of shape (bin_count,).
        """
        probs = probs.detach().to(self.device, dtype=torch.float).clamp_min(0.0)
        total = probs.sum()
        if probs.numel() != self.bin_count or not torch.isfinite(total) or total <= 0:
            return
        self.sampling_probs = probs / total

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
        soft_limits = self.robot.data.soft_joint_pos_limits[env_ids]
        joint_pos = torch.clip(joint_pos, soft_limits[:, :, 0], soft_limits[:, :, 1])
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

        root_state = torch.cat([root_pos, root_ori, root_lin_vel, root_ang_vel], dim=-1)
        self.robot.write_root_state_to_sim(root_state, env_ids=env_ids)
        self.robot.reset(env_ids=env_ids)

    def _resample_command(self, env_ids: torch.Tensor):
        self.time_steps[env_ids] = self.sample_frames(len(env_ids))

        root_pos = self.body_pos_w[env_ids, 0].clone()
        root_ori = self.body_quat_w[env_ids, 0].clone()
        root_lin_vel = self.body_lin_vel_w[env_ids, 0].clone()
        root_ang_vel = self.body_ang_vel_w[env_ids, 0].clone()

        ranges = self._pose_ranges
        rand_samples = sample_uniform(
            ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device
        )
        root_pos += rand_samples[:, 0:3]
        orientations_delta = quat_from_euler_xyz(
            rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5]
        )
        root_ori = quat_mul(orientations_delta, root_ori)
        ranges = self._velocity_ranges
        rand_samples = sample_uniform(
            ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device
        )
        root_lin_vel += rand_samples[:, :3]
        root_ang_vel += rand_samples[:, 3:]

        joint_pos = self.joint_pos[env_ids].clone()
        joint_vel = self.joint_vel[env_ids]

        joint_pos += sample_uniform(
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

    def _update_command(self, env_ids: torch.Tensor | None = None):
        if env_ids is None:
            self.time_steps += 1
        else:
            self.time_steps[env_ids] += 1

    def _debug_vis_impl(self, visualizer: DebugVisualizer) -> None:
        """Draw ghost robot or frames based on visualization mode."""
        env_indices = visualizer.get_env_indices(self.num_envs)
        if not env_indices:
            return

        if self.cfg.viz.mode == "ghost":
            if self._ghost_model is None:
                # Build a ghost model with only visual geoms visible. Collision geoms (nonzero
                # contype/conaffinity) get alpha=0 so the viewer's alpha filter excludes them.
                self._ghost_model = copy.deepcopy(self._env.sim.mj_model)
                for gi in range(self._ghost_model.ngeom):
                    if (
                        self._ghost_model.geom_contype[gi] != 0
                        or self._ghost_model.geom_conaffinity[gi] != 0
                    ):
                        self._ghost_model.geom_rgba[gi, 3] = 0
                    else:
                        self._ghost_model.geom_rgba[gi] = self._ghost_color

            entity: Entity = self._env.scene[self.cfg.entity_name]
            indexing = entity.indexing
            free_joint_q_adr = indexing.free_joint_q_adr.cpu().numpy()
            joint_q_adr = indexing.joint_q_adr.cpu().numpy()

            for batch in env_indices:
                qpos = np.zeros(self._env.sim.mj_model.nq)
                qpos[free_joint_q_adr[0:3]] = self.body_pos_w[batch, 0].cpu().numpy()
                qpos[free_joint_q_adr[3:7]] = self.body_quat_w[batch, 0].cpu().numpy()
                qpos[joint_q_adr] = self.joint_pos[batch].cpu().numpy()

                visualizer.add_ghost_mesh(
                    qpos,
                    model=self._ghost_model,
                    label=f"ghost_{batch}",
                )

        elif self.cfg.viz.mode == "frames":
            for batch in env_indices:
                desired_body_pos = self.body_pos_w[batch].cpu().numpy()
                desired_body_quat = self.body_quat_w[batch]
                desired_body_rotm = matrix_from_quat(desired_body_quat).cpu().numpy()

                current_body_pos = self.robot_body_pos_w[batch].cpu().numpy()
                current_body_quat = self.robot_body_quat_w[batch]
                current_body_rotm = matrix_from_quat(current_body_quat).cpu().numpy()

                for i, body_name in enumerate(self.cfg.body_names):
                    visualizer.add_frame(
                        position=desired_body_pos[i],
                        rotation_matrix=desired_body_rotm[i],
                        scale=0.08,
                        label=f"desired_{body_name}_{batch}",
                        axis_colors=_DESIRED_FRAME_COLORS,
                    )
                    visualizer.add_frame(
                        position=current_body_pos[i],
                        rotation_matrix=current_body_rotm[i],
                        scale=0.12,
                        label=f"current_{body_name}_{batch}",
                    )

                desired_anchor_pos = self.anchor_pos_w[batch].cpu().numpy()
                desired_anchor_quat = self.anchor_quat_w[batch]
                desired_rotation_matrix = (
                    matrix_from_quat(desired_anchor_quat).cpu().numpy()
                )
                visualizer.add_frame(
                    position=desired_anchor_pos,
                    rotation_matrix=desired_rotation_matrix,
                    scale=0.1,
                    label=f"desired_anchor_{batch}",
                    axis_colors=_DESIRED_FRAME_COLORS,
                )

                current_anchor_pos = self.robot_anchor_pos_w[batch].cpu().numpy()
                current_anchor_quat = self.robot_anchor_quat_w[batch]
                current_rotation_matrix = (
                    matrix_from_quat(current_anchor_quat).cpu().numpy()
                )
                visualizer.add_frame(
                    position=current_anchor_pos,
                    rotation_matrix=current_rotation_matrix,
                    scale=0.15,
                    label=f"current_anchor_{batch}",
                )

    def create_gui(
        self,
        name: str,
        server: viser.ViserServer,
        get_env_idx: Callable[[], int],
        on_change: Callable[[], None] | None = None,
        request_action: Callable[[str, Any], None] | None = None,
    ) -> None:
        """Create motion scrubber controls in the Viser viewer."""
        max_frame = int(self.motion.time_step_total) - 1

        with server.gui.add_folder(name.capitalize()):
            scrubber = server.gui.add_slider(
                "Frame",
                min=0,
                max=max_frame,
                step=1,
                initial_value=0,
            )

            @scrubber.on_update
            def _(_) -> None:
                idx = get_env_idx()
                self.time_steps[idx] = int(scrubber.value)
                if on_change is not None:
                    on_change()

            all_envs_cb = server.gui.add_checkbox("All envs", initial_value=True)
            start_btn = server.gui.add_button("Start Here")

            @start_btn.on_click
            def _(_) -> None:
                if request_action is not None:
                    request_action(
                        "CUSTOM",
                        {"type": "gui_reset", "all_envs": all_envs_cb.value},
                    )

        self._scrubber_handles = (scrubber, all_envs_cb, start_btn)
        self._set_scrubber_disabled(True)

    def _set_scrubber_disabled(self, disabled: bool) -> None:
        """Enable or disable the motion scrubber GUI controls."""
        for handle in self._scrubber_handles:
            handle.disabled = disabled

    def on_viewer_pause(self, paused: bool) -> None:
        if hasattr(self, "_scrubber_handles"):
            self._set_scrubber_disabled(not paused)

    def apply_gui_reset(self, env_ids: torch.Tensor) -> bool:
        if not hasattr(self, "_scrubber_handles"):
            return False
        frame = int(self._scrubber_handles[0].value)
        self.reset_to_frame(env_ids, frame)
        self._env.sim.forward()
        return True

    def reset_to_frame(self, env_ids: torch.Tensor, frame: int) -> None:
        """Reset to exact reference state at a specific frame.

        Like ``_resample_command`` but deterministic: no random
        perturbations to pose, velocity, or joint positions.
        """
        self.time_steps[env_ids] = frame
        self._write_reference_state_to_sim(
            env_ids,
            self.body_pos_w[env_ids, 0],
            self.body_quat_w[env_ids, 0],
            self.body_lin_vel_w[env_ids, 0],
            self.body_ang_vel_w[env_ids, 0],
            self.joint_pos[env_ids],
            self.joint_vel[env_ids],
        )


@dataclass(kw_only=True)
class MotionCommandCfg(CommandTermCfg):
    motion_file: str | list[str]
    anchor_body_name: str
    body_names: tuple[str, ...]
    entity_name: str
    pose_range: dict[str, tuple[float, float]] = field(default_factory=dict)
    velocity_range: dict[str, tuple[float, float]] = field(default_factory=dict)
    joint_position_range: tuple[float, float] = (-0.52, 0.52)

    @dataclass
    class VizCfg:
        mode: Literal["ghost", "frames"] = "ghost"
        ghost_color: tuple[float, float, float, float] = (0.5, 0.7, 0.5, 0.5)

    viz: VizCfg = field(default_factory=VizCfg)

    def build(self, env: ManagerBasedRlEnv) -> MotionCommand:
        return MotionCommand(self, env)
