"""Learned dynamics that advance the robot state by one env step."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from mj_test.mdp.motion import RobotState
from mj_test.utils.mlp import build_mlp
from mj_test.utils.normalization import EmpiricalNormalization
from mj_test.wm.representation import integrate, local_state


@dataclass
class DynamicsCfg:
    """Hyperparameters of the dynamics network."""

    hidden_dims: tuple[int, ...] = (2048, 2048, 2048)


class Dynamics(nn.Module):
    """Differentiable forward model that advances a robot state by one env step.

    Args:
        state_dim: Size of ``local_state``.
        action_dim: Number of action dimensions.
        sim_ctx_dim: Size of the sim context.
        delta_dim: Size of the delta that ``integrate`` consumes.
        action_scale: Per-joint action scale, shape (action_dim,).
        action_offset: Per-joint action offset, shape (action_dim,).
        hidden_dims: Hidden layer widths of the MLP.
        dt: Env step duration in seconds.
        substep_coeff: Weight of the delta in the pose update of ``integrate``.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        sim_ctx_dim: int,
        delta_dim: int,
        action_scale: torch.Tensor,
        action_offset: torch.Tensor,
        hidden_dims: tuple[int, ...],
        dt: float,
        substep_coeff: float,
    ) -> None:
        super().__init__()
        self.dt = dt
        self.substep_coeff = substep_coeff

        self.register_buffer("action_scale", action_scale.clone())
        self.register_buffer("action_offset", action_offset.clone())

        self.state_normalizer = EmpiricalNormalization(state_dim)
        self.action_normalizer = EmpiricalNormalization(action_dim)

        self.network = build_mlp(
            state_dim + action_dim + sim_ctx_dim, hidden_dims, delta_dim
        )

    def joint_pos_target(self, action: torch.Tensor) -> torch.Tensor:
        """Maps raw actions to the joint position targets of the PD controllers."""
        return action * self.action_scale + self.action_offset

    def forward(
        self,
        state: RobotState,
        action: torch.Tensor,
        sim_ctx: torch.Tensor,
    ) -> RobotState:
        """Predicts the robot state after one env step."""
        state_norm = self.state_normalizer(local_state(state))
        action_norm = self.action_normalizer(self.joint_pos_target(action))
        delta = self.network(torch.cat([state_norm, action_norm, sim_ctx], dim=-1))
        return integrate(state, delta, self.dt, self.substep_coeff)
