from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import mujoco
import numpy as np
import torch
from mjlab.utils.lab_api.math import quat_apply
from tensordict import tensorclass

if TYPE_CHECKING:
    from mjlab.entity import Entity

Vec = tuple[torch.Tensor, ...]


@tensorclass
class GenCoords:
    """Generalized coordinates and velocities: the free-joint root and the joints."""

    root_pos: torch.Tensor
    root_quat: torch.Tensor
    joint_pos: torch.Tensor
    root_lin_vel: torch.Tensor
    root_ang_vel: torch.Tensor
    joint_vel: torch.Tensor

    @classmethod
    def from_entity(cls, entity: Entity, env_origins: torch.Tensor) -> GenCoords:
        """Reads an mjlab entity's coordinates from qpos and qvel, which are
        current without a ``sim.forward()``.

        Args:
            entity: The floating-base entity.
            env_origins: World position of each env's origin, shape (B, 3).
        """
        data = entity.data
        qpos, qvel = data.data.qpos, data.data.qvel
        root_q = entity.indexing.free_joint_q_adr
        root_v = entity.indexing.free_joint_v_adr
        root_quat = qpos[:, root_q[3:7]]
        return cls(
            root_pos=qpos[:, root_q[0:3]] - env_origins,
            root_quat=root_quat,
            joint_pos=data.joint_pos,
            root_lin_vel=qvel[:, root_v[0:3]],
            root_ang_vel=quat_apply(root_quat, qvel[:, root_v[3:6]]),
            joint_vel=data.joint_vel,
            batch_size=root_quat.shape[:1],
        )


@tensorclass
class MaxCoords:
    """World-frame poses and velocities of K bodies, without env origins."""

    pos: torch.Tensor
    quat: torch.Tensor
    lin_vel: torch.Tensor
    ang_vel: torch.Tensor


class ForwardKinematics:
    def __init__(
        self,
        mj_model: mujoco.MjModel,
        body_ids: torch.Tensor,
        joint_q_adr: torch.Tensor,
        body_rows: torch.Tensor,
        dtype: torch.dtype = torch.float32,
        device: str = "cpu",
    ):
        rows = [int(body_ids[row]) for row in body_rows]
        col_of = {int(adr): col for col, adr in enumerate(joint_q_adr)}
        tree = set()
        for body in rows:
            while body and body not in tree:
                tree.add(body)
                body = int(mj_model.body_parentid[body])
        order = sorted(tree)  # MuJoCo numbers parents before children

        free, hinge = mujoco.mjtJoint.mjJNT_FREE, mujoco.mjtJoint.mjJNT_HINGE
        zero, identity = np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0])
        # (parent, joint column, qpos0, axis, anchor, offset pos, offset quat); a step
        # without a hinge reads column J, a zero pad after the J joints. Step 0, the
        # root, is the identity.
        fixed = (len(col_of), 0.0, zero, zero)
        steps: list[tuple] = [(0, *fixed, zero, identity)]
        step_of = {order[0]: 0}
        for body in order:
            parent, first = (
                int(mj_model.body_parentid[body]),
                int(mj_model.body_jntadr[body]),
            )
            joints = range(first, first + int(mj_model.body_jntnum[body]))
            kinds = [mj_model.jnt_type[j] for j in joints]
            root = body == int(body_ids[0])
            if (parent == 0) != root or kinds != (
                [free] if root else [hinge] * len(kinds)
            ):
                raise ValueError(
                    f"body {mj_model.body(body).name}: needs the entity's first body as its "
                    "free-joint root under the world, with only hinges below it"
                )
            if root:
                continue
            adrs = [int(mj_model.jnt_qposadr[j]) for j in joints]
            hinges = [
                (
                    col_of[a],
                    mj_model.qpos0[a],
                    mj_model.jnt_axis[j],
                    mj_model.jnt_pos[j],
                )
                for j, a in zip(joints, adrs, strict=True)
            ]
            step = step_of[parent]
            offset = mj_model.body_pos[body], mj_model.body_quat[body]
            for joint in hinges or [fixed]:
                steps.append((step, *joint, *offset))
                step, offset = len(steps) - 1, (zero, identity)
            step_of[body] = step

        # Each returned body's path, root first, front-padded with the root step to
        # one length.
        paths = []
        for body in rows:
            step, path = step_of[body], []
            while step:
                path.append(step)
                step = steps[step][0]
            paths.append(path[::-1])
        depth = max(map(len, paths))
        paths = [[0] * (depth - len(path)) + path for path in paths]
        # Per path step, with o its offset quaternion: the hinge anchor and axis in the
        # parent frame, the anchor in the step frame, o, and o * (0, axis); the step's
        # quaternion is cos(h) o + sin(h) o * (0, axis) for h = (qpos - qpos0) / 2.
        consts = np.zeros((depth, len(rows), 17))
        for r, path in enumerate(paths):
            for d, step in enumerate(path):
                _, _, _, axis, anchor, offset_pos, offset_quat = steps[step]
                parent_anchor, parent_axis, turned = (
                    np.zeros(3),
                    np.zeros(3),
                    np.zeros(4),
                )
                mujoco.mju_rotVecQuat(parent_anchor, anchor, offset_quat)
                mujoco.mju_rotVecQuat(parent_axis, axis, offset_quat)
                mujoco.mju_mulQuat(turned, offset_quat, np.concatenate([[0.0], axis]))
                consts[d, r] = np.concatenate(
                    [
                        offset_pos + parent_anchor,
                        parent_axis,
                        anchor,
                        offset_quat,
                        turned,
                    ]
                )
        self.consts = torch.as_tensor(consts, device=device, dtype=dtype)
        self.com_offset = torch.as_tensor(
            mj_model.body_ipos[rows], device=device, dtype=dtype
        )
        self.num_joints = len(col_of)
        by_step = [[steps[s] for s in path] for path in zip(*paths, strict=True)]
        self.col = torch.tensor([[s[1] for s in d] for d in by_step], device=device)
        self.qpos0 = torch.tensor(
            [[s[2] for s in d] for d in by_step], device=device, dtype=dtype
        )
        names = {
            int(adr): mj_model.joint(j).name.split("/")[-1]
            for j, adr in enumerate(mj_model.jnt_qposadr)
        }
        self.joint_names = [names[adr] for adr in col_of if adr in names]

    def __call__(
        self,
        root_pos: torch.Tensor,
        root_quat: torch.Tensor,
        joint_pos: torch.Tensor,
        root_lin_vel: torch.Tensor,
        root_ang_vel: torch.Tensor,
        joint_vel: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Positions ``(..., R, 3)``, wxyz quaternions ``(..., R, 4)``, linear and
        angular velocities ``(..., R, 3)`` of the ``R`` bodies, in the world frame,
        from the root link's world pose and velocity and the joints ``(..., J)``."""
        joint_pos, joint_vel = (
            torch.nn.functional.pad(x, (0, 1)) for x in (joint_pos, joint_vel)
        )
        pos, quat, hinges = self._chain(joint_pos)
        zero = root_pos.new_zeros(())
        lin: Vec = (zero, zero, zero)
        ang: Vec = (zero, zero, zero)
        for col, anchor, axis in hinges:
            twist = tuple(a * joint_vel[..., col] for a in axis)
            ang = _add(ang, twist)
            lin = _add(lin, _cross(anchor, twist))

        root = _unit(root_quat.unsqueeze(-2).unbind(-1))
        pos = _rotate(root, pos)
        ang = _add(root_ang_vel.unsqueeze(-2).unbind(-1), _rotate(root, ang))
        lin = _add(root_lin_vel.unsqueeze(-2).unbind(-1), _rotate(root, lin))
        lin = _add(lin, _cross(ang, pos))
        pos = _add(root_pos.unsqueeze(-2).unbind(-1), pos)
        state = torch.stack([*pos, *_mul(root, quat), *lin, *ang], dim=-1)
        pos_w, quat_w, lin_vel_w, ang_vel_w = state.split((3, 4, 3, 3), dim=-1)
        return pos_w, quat_w, lin_vel_w, ang_vel_w

    def generalized_force(
        self, root_quat: torch.Tensor, joint_pos: torch.Tensor, force_w: torch.Tensor
    ) -> torch.Tensor:
        """``J(q)ᵀ F`` of world-frame forces ``(..., R, 3)`` at the CoMs of the R
        bodies, where MuJoCo applies ``xfrc_applied``.

        Returns:
            The root force and the root torque about the root origin, both in the
            root frame, then the joint torques, summed over the bodies, (..., 6 + J).
        """
        pos, quat, hinges = self._chain(torch.nn.functional.pad(joint_pos, (0, 1)))
        point = _add(pos, _rotate(quat, self.com_offset.unbind(-1)))
        force = _rotate(
            _conj(_unit(root_quat.unsqueeze(-2).unbind(-1))), force_w.unbind(-1)
        )
        # A hinge's torque is its axis times the moment of the force about its anchor.
        torque = torch.stack(
            [
                _dot(axis, _cross(_sub(point, anchor), force))
                for _, anchor, axis in hinges
            ],
            dim=-2,
        )
        joint = torque.new_zeros(*torque.shape[:-2], self.num_joints + 1).index_add(
            -1, self.col.flatten(), torque.flatten(-2)
        )
        return torch.cat(
            [
                torch.stack(force, dim=-1).sum(-2),
                torch.stack(_cross(point, force), dim=-1).sum(-2),
                joint[..., :-1],
            ],
            dim=-1,
        )

    def _chain(
        self, joint_pos: torch.Tensor
    ) -> tuple[Vec, Vec, list[tuple[torch.Tensor, Vec, Vec]]]:
        """Walks the paths in the root frame."""
        zero, one = joint_pos.new_zeros(()), joint_pos.new_ones(())
        pos: Vec = (zero, zero, zero)
        quat: Vec = (one, zero, zero, zero)
        hinges = []
        for col, qpos0, consts in zip(self.col, self.qpos0, self.consts, strict=True):
            c = consts.unbind(-1)
            half = (joint_pos[..., col] - qpos0) / 2
            # The hinge anchor and axis in the root frame, from the parent's pose.
            anchor = _add(pos, _rotate(quat, c[:3]))
            axis = _rotate(quat, c[3:6])
            turn = tuple(
                half.cos() * o + half.sin() * t
                for o, t in zip(c[9:13], c[13:], strict=True)
            )
            quat = _mul(quat, turn)
            pos = _sub(anchor, _rotate(quat, c[6:9]))
            hinges.append((col, anchor, axis))
        return pos, quat, hinges


class KinematicModel:
    """Forward kinematics of a floating-base robot.

    Args:
        fk: Forward kinematics pass whose output bodies are ``body_names``.
        body_names: Output bodies.
        default_joint_pos: Default joint positions, shape (J,).
        joint_limits: Lower and upper soft joint position limits, shape (J, 2).
        effort_limits: Actuator torque limit of each joint, shape (J,).
        mass: Total mass of the robot in kg.
    """

    def __init__(
        self,
        fk: ForwardKinematics,
        body_names: tuple[str, ...],
        default_joint_pos: torch.Tensor,
        joint_limits: torch.Tensor,
        effort_limits: torch.Tensor,
        mass: float,
    ) -> None:
        self.fk = fk
        self.body_names = body_names
        self.default_joint_pos = default_joint_pos
        self.joint_limits = joint_limits
        self.effort_limits = effort_limits
        self.mass = mass

    @classmethod
    def from_entity(
        cls,
        mj_model: mujoco.MjModel,
        entity: Entity,
        body_names: Sequence[str] | None = None,
    ) -> KinematicModel:
        """Builds the forward kinematics of bodies of an mjlab entity.

        Args:
            mj_model: The compiled model of the scene.
            entity: The floating-base entity.
            body_names: Output bodies in order, or None for every body in entity
                order, as in ``entity.data.body_link_pose_w``.
        """
        data = entity.data
        indexing = entity.indexing
        if body_names is None:
            body_names = entity.body_names
            rows = list(range(len(body_names)))
        else:
            rows = entity.find_bodies(tuple(body_names), preserve_order=True)[0]
        fk = ForwardKinematics(
            mj_model,
            indexing.body_ids,
            indexing.joint_q_adr,
            torch.tensor(rows, dtype=torch.long),
            device=data.device,
        )
        return cls(
            fk,
            tuple(body_names),
            default_joint_pos=data.default_joint_pos[0].clone(),
            joint_limits=data.soft_joint_pos_limits[0].clone(),
            effort_limits=_effort_limits(mj_model, entity),
            mass=float(mj_model.body_subtreemass[int(indexing.body_ids[0])]),
        )

    @torch.compile
    def bodies(self, gen: GenCoords) -> MaxCoords:
        pos, quat, lin_vel, ang_vel = self.fk(
            gen.root_pos,
            gen.root_quat,
            gen.joint_pos,
            gen.root_lin_vel,
            gen.root_ang_vel,
            gen.joint_vel,
        )
        return MaxCoords(
            pos=pos,
            quat=quat,
            lin_vel=lin_vel,
            ang_vel=ang_vel,
            batch_size=(*gen.batch_size, len(self.body_names)),
        )

    @torch.compile
    def generalized_force(self, gen: GenCoords, force_w: torch.Tensor) -> torch.Tensor:
        """``J(q)ᵀ F`` of world-frame forces ``(..., K, 3)`` at the CoMs of the K
        bodies: the root force and torque in the root frame, then the joint torques,
        (..., 6 + J)."""
        return self.fk.generalized_force(gen.root_quat, gen.joint_pos, force_w)


def _effort_limits(mj_model: mujoco.MjModel, entity: Entity) -> torch.Tensor:
    """The torque limit of each entity joint, from the joint's actuator force range
    or its actuators' force ranges, (J,)."""
    actuated: dict[int, float] = {}
    for a in range(mj_model.nu):
        if (
            mj_model.actuator_trntype[a] == mujoco.mjtTrn.mjTRN_JOINT
            and mj_model.actuator_forcelimited[a]
        ):
            joint = int(mj_model.actuator_trnid[a, 0])
            limit = float(np.abs(mj_model.actuator_forcerange[a]).max())
            actuated[joint] = max(actuated.get(joint, 0.0), limit)
    limits = []
    for joint, name in zip(
        entity.indexing.joint_ids.tolist(), entity.joint_names, strict=True
    ):
        if mj_model.jnt_actfrclimited[joint]:
            limits.append(float(np.abs(mj_model.jnt_actfrcrange[joint]).max()))
        elif joint in actuated:
            limits.append(actuated[joint])
        else:
            raise ValueError(f"joint {name} has no actuator torque limit")
    return torch.tensor(limits, device=entity.data.device)


def _add(a: Vec, b: Vec) -> Vec:
    return tuple(x + y for x, y in zip(a, b, strict=True))


def _sub(a: Vec, b: Vec) -> Vec:
    return tuple(x - y for x, y in zip(a, b, strict=True))


def _cross(a: Vec, b: Vec) -> Vec:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _dot(a: Vec, b: Vec) -> torch.Tensor:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _unit(q: Vec) -> Vec:
    w, x, y, z = q
    norm = (w * w + x * x + y * y + z * z).sqrt()
    return (w / norm, x / norm, y / norm, z / norm)


def _conj(q: Vec) -> Vec:
    return (q[0], -q[1], -q[2], -q[3])


def _rotate(q: Vec, v: Vec) -> Vec:
    """``v`` turned by the unit wxyz quaternion ``q``."""
    t = tuple(2 * x for x in _cross(q[1:], v))
    return _add(_add(v, tuple(q[0] * x for x in t)), _cross(q[1:], t))


def _mul(a: Vec, b: Vec) -> Vec:
    """The Hamilton product of wxyz quaternions."""
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return (
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    )
