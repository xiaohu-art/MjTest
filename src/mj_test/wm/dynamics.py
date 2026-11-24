"""Learned dynamics that advance the robot state by one env step."""

from __future__ import annotations

import torch
import torch.nn as nn

from mj_test.mdp.action import JointPositionAction
from mj_test.utils.forward_kinematics import GenCoords, KinematicModel
from mj_test.utils.mlp import build_mlp
from mj_test.utils.normalization import EmpiricalNormalization
from mj_test.wm.representation import integrate, local_state


class Dynamics(nn.Module):
    """Differentiable forward model that advances a robot state by one env step."""

    def __init__(
        self,
        state_dim: int,
        sim_ctx_dim: int,
        delta_dim: int,
        action_term: JointPositionAction,
        force_kinematics: KinematicModel,
        hidden_dims: tuple[int, ...],
        dt: float,
        substep_coeff: float,
    ) -> None:
        super().__init__()
        self.dt = dt
        self.substep_coeff = substep_coeff
        self.action_term = action_term
        self.force_kinematics = force_kinematics

        weight = force_kinematics.mass * 9.81
        effort = force_kinematics.effort_limits
        self.register_buffer(
            "force_scale",
            torch.cat(
                [
                    effort.new_full((3,), weight),  # root force
                    effort.new_full((3,), weight),  # root torque
                    effort,                         # joint torque
                ]
            ),
            persistent=False,
        )

        action_dim = action_term.dim
        self.state_normalizer = EmpiricalNormalization(state_dim)
        self.action_normalizer = EmpiricalNormalization(action_dim)

        self.network = build_mlp(
            state_dim + action_dim + sim_ctx_dim + self.force_scale.numel(),
            hidden_dims,
            delta_dim,
        )

    def forward(
        self,
        state: GenCoords,
        action: torch.Tensor,
        sim_ctx: torch.Tensor,
        force_w: torch.Tensor,
    ) -> GenCoords:
        state_norm = self.state_normalizer(local_state(state))
        action_norm = self.action_normalizer(self.action_term(action))
        force = self.force_kinematics.generalized_force(state, force_w)
        delta = self.network(
            torch.cat(
                [state_norm, action_norm, sim_ctx, force / self.force_scale], dim=-1
            )
        )
        return integrate(state, delta, self.dt, self.substep_coeff)
