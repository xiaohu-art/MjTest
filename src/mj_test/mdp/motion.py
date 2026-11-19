"""Robot and body state types and the reference motion of the tracking task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
from mjlab.utils.lab_api.math import quat_apply
from tensordict import tensorclass

from mj_test.utils.forward_kinematics import ForwardKinematics

if TYPE_CHECKING:
    from mjlab.entity import Entity

__all__ = ["BodyState", "MotionLoader", "RobotState"]

_FK_CHUNK = 65536


@tensorclass
class RobotState:
    """Generalized coordinates and velocities of the robot."""

    root_pos: torch.Tensor
    root_quat: torch.Tensor
    joint_pos: torch.Tensor
    root_lin_vel: torch.Tensor
    root_ang_vel: torch.Tensor
    joint_vel: torch.Tensor

    @classmethod
    def from_entity(cls, robot: Entity, env_origins: torch.Tensor) -> RobotState:
        """Reads the robot state from qpos and qvel."""
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
    """Poses and velocities of the tracked bodies, anchor first."""

    pos: torch.Tensor
    quat: torch.Tensor
    lin_vel: torch.Tensor
    ang_vel: torch.Tensor


class MotionLoader:
    """Loads reference motion clips and concatenates them along the frame axis."""

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
        self.states = RobotState(
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
            chunks = self.states.split(_FK_CHUNK)
            self.bodies = torch.cat([self.body_state(chunk) for chunk in chunks])

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
