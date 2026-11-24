"""The MDP on the world model."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch
from mjlab.utils.lab_api.math import quat_apply_inverse
from tensordict import TensorDict, tensorclass

from mj_test.envs.env import Env
from mj_test.mdp.command import MotionCommand
from mj_test.utils.forward_kinematics import GenCoords, KinematicModel, MaxCoords
from mj_test.utils.math import projected_gravity

if TYPE_CHECKING:
    from mj_test.envs.sim_env import SimEnv
    from mj_test.wm.dynamics import Dynamics


@tensorclass
class ImaginedEntityData:
    gen: GenCoords
    bodies: MaxCoords  # (B, NB).
    encoder_bias: torch.Tensor  # (B, J).
    default_joint_pos: torch.Tensor  # (B, J).
    soft_joint_pos_limits: torch.Tensor  # (B, J, 2).

    @property
    def root_link_pos_w(self) -> torch.Tensor:
        return self.gen.root_pos

    @property
    def root_link_quat_w(self) -> torch.Tensor:
        return self.gen.root_quat

    @property
    def root_link_lin_vel_w(self) -> torch.Tensor:
        return self.gen.root_lin_vel

    @property
    def root_link_ang_vel_w(self) -> torch.Tensor:
        return self.gen.root_ang_vel

    @property
    def root_link_lin_vel_b(self) -> torch.Tensor:
        return quat_apply_inverse(self.gen.root_quat, self.gen.root_lin_vel)

    @property
    def root_link_ang_vel_b(self) -> torch.Tensor:
        return quat_apply_inverse(self.gen.root_quat, self.gen.root_ang_vel)

    @property
    def projected_gravity_b(self) -> torch.Tensor:
        return projected_gravity(self.gen.root_quat)

    @property
    def joint_pos(self) -> torch.Tensor:
        return self.gen.joint_pos

    @property
    def joint_pos_biased(self) -> torch.Tensor:
        return self.gen.joint_pos + self.encoder_bias

    @property
    def joint_vel(self) -> torch.Tensor:
        return self.gen.joint_vel

    @property
    def body_link_pos_w(self) -> torch.Tensor:
        return self.bodies.pos

    @property
    def body_link_quat_w(self) -> torch.Tensor:
        return self.bodies.quat

    @property
    def body_link_lin_vel_w(self) -> torch.Tensor:
        return self.bodies.lin_vel

    @property
    def body_link_ang_vel_w(self) -> torch.Tensor:
        return self.bodies.ang_vel


class ImaginationEnv(Env):
    """Differentiable stand-in for the simulated env, driven by the world model."""

    def __init__(
        self, env: SimEnv, dynamics: Dynamics, termination_softness: float = 0.05
    ) -> None:
        super().__init__(env.cfg, env.dt)
        self.dynamics = dynamics
        self.kinematics = KinematicModel.from_entity(env.sim.mj_model, env.entity)
        self.sim_context = env.sim_context
        self.command = cast(MotionCommand, env.command).like(self)
        self.action_term = env.action_term.like()
        self.termination_softness = termination_softness

    @property
    def num_envs(self) -> int:
        return self.robot.batch_size[0]

    def reset(
        self,
        gen: GenCoords,
        sim_ctx: torch.Tensor,
        time_steps: torch.Tensor,
        force_cmd: torch.Tensor,
        prev_action: torch.Tensor,
    ) -> TensorDict:
        """Starts imagined rollouts from real transitions."""
        self.dynamics.eval().requires_grad_(False)
        gen = gen.detach()
        self.sim_ctx = sim_ctx.detach()
        self.env_origins = torch.zeros_like(gen.root_pos)
        self.robot = self._forward(gen)
        self.command.set_time_steps(time_steps)
        self.command.force_cmd = force_cmd
        self.action_term.action = prev_action.detach()
        self.action_term.prev_action = torch.zeros_like(prev_action)
        return self.observations()

    @torch.compile
    def step(
        self, action: torch.Tensor
    ) -> tuple[TensorDict, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """Advances the imagined rollouts by one step."""
        gen = self.dynamics(
            self.robot.gen, 
            action, 
            self.sim_ctx, 
            self.command.external_force_w
        )
        self.robot = self._forward(gen)
        self.action_term.process(action)
        reward, terms, errors = self._post_step()
        self.command.update_command()
        error = torch.stack(list(errors.values())).amax(0)
        terminated = torch.sigmoid((error - 1.0) / self.termination_softness)
        truncated = torch.zeros_like(terminated)
        return self.observations(), reward, terminated, truncated, {"terms": terms}

    def _forward(self, gen: GenCoords) -> ImaginedEntityData:
        """The imagination's ``sim.forward()``: the robot data at ``gen``.

        The encoder bias comes from ``sim_ctx``, so the ``encoder_bias`` event
        must be enabled.
        """
        num = gen.batch_size[0]
        kinematics = self.kinematics
        return ImaginedEntityData(
            gen=gen,
            bodies=kinematics.bodies(gen),
            encoder_bias=self.sim_context.denormalize(self.sim_ctx, "encoder_bias"),
            default_joint_pos=kinematics.default_joint_pos.expand(num, -1),
            soft_joint_pos_limits=kinematics.joint_limits.expand(num, -1, -1),
            batch_size=gen.batch_size,
        )
