"""Motion tracking env whose robot state advances by the world model."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from tensordict import TensorDict

from mj_test.envs.env import Env
from mj_test.mdp.motion import RobotState

if TYPE_CHECKING:
    from mj_test.envs.sim_env import SimEnv
    from mj_test.wm.dynamics import Dynamics


class ImaginationEnv(Env):
    """Differentiable stand-in for the simulated env, driven by the world model."""

    def __init__(
        self, env: SimEnv, dynamics: Dynamics, termination_softness: float = 0.05
    ) -> None:
        if termination_softness <= 0.0:
            raise ValueError(
                f"termination_softness must be positive, got {termination_softness}."
            )
        super().__init__(
            env.cfg, env.motion, env.dt, env.default_joint_pos, env.joint_limits
        )
        self.dynamics = dynamics
        self.sim_context = env.sim_context
        self.termination_softness = termination_softness

    def reset(self, batch: TensorDict) -> TensorDict:
        """Starts imagined rollouts from real transitions."""
        self.dynamics.eval().requires_grad_(False)
        names = [field.name for field in RobotState.fields()]
        self.robot_state = RobotState.from_tensordict(batch.select(*names)).detach()
        self.action = batch["prev_action"].detach()
        self.frames = batch["frame"]
        self.sim_ctx = batch["sim_ctx"].detach()
        self.encoder_bias = self.sim_context.denormalize(self.sim_ctx, "encoder_bias")
        return self.observations()

    @torch.compile
    def step(
        self, action: torch.Tensor
    ) -> tuple[TensorDict, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """Advances the imagined rollouts by one step."""
        next_state = self.dynamics(self.robot_state, action, self.sim_ctx)
        reward, terms, errors = self._update(next_state, action)
        error = torch.stack(list(errors.values())).amax(0)
        terminated = torch.sigmoid((error - 1.0) / self.termination_softness)
        truncated = torch.zeros_like(terminated)
        return self.observations(), reward, terminated, truncated, {"terms": terms}
