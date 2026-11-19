"""Supervised training of the world model."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

import torch
from tensordict import TensorDict

from mj_test.mdp.commands import RobotState
from mj_test.utils.optim import CapturedUpdate, build_optimizers, optimize_step
from mj_test.wm.dynamics import Dynamics
from mj_test.wm.representation import local_state, loss_blocks


@dataclass
class WorldModelLearnerCfg:
    """Hyperparameters of the world model learner.

    Attributes:
        lr: Learning rate of both optimizers.
        num_batches: Gradient steps per update.
        batch_size: Windows per gradient step.
        horizon: Frames per window; the rollout predicts the last horizon - 1.
        max_grad_norm: Gradient norm clipping threshold.
        nominal_errors: Acceptable error of each loss block. Each block is
            weighted by 1 / nominal_error**2 (Bryson's rule).
    """

    lr: float = 2e-4
    num_batches: int = 10
    batch_size: int = 2048
    horizon: int = 8
    max_grad_norm: float = 1.0
    nominal_errors: dict[str, float] = field(
        default_factory=lambda: {
            "root_pos": 0.01,
            "root_ori": 0.02,
            "joint_pos": 0.1,
            "joint_vel": 3.14,
        }
    )


class WorldModelLearner:
    """Fits the dynamics to real windows with an autoregressive rollout loss.

    Args:
        dynamics: The dynamics model to train.
        cfg: Learner hyperparameters.
    """

    def __init__(self, dynamics: Dynamics, cfg: WorldModelLearnerCfg) -> None:
        self.dynamics = dynamics
        self.cfg = cfg
        self._params = list(dynamics.parameters())
        device = self._params[0].device
        self.optimizers = build_optimizers([dynamics], cfg.lr, device=device)
        self._step = CapturedUpdate(self._update_batch)

    def update_normalization(self, states: RobotState, actions: torch.Tensor) -> None:
        """Updates the dynamics normalizers with freshly collected data.

        Args:
            states: Collected robot states with any leading shape.
            actions: Raw actions with the same leading shape, (..., action_dim).
        """
        x = local_state(states)
        a = self.dynamics.joint_pos_target(actions)
        normalizers = (self.dynamics.state_normalizer, self.dynamics.action_normalizer)
        for normalizer in normalizers:
            normalizer.train()
        self.dynamics.state_normalizer.update(x.reshape(-1, x.shape[-1]))
        self.dynamics.action_normalizer.update(a.reshape(-1, a.shape[-1]))
        for normalizer in normalizers:
            normalizer.eval()

    def update(self, batches: Iterable[TensorDict]) -> dict[str, torch.Tensor]:
        """Runs one gradient step per batch of windows.

        Args:
            batches: Windows of batch size (batch_size, horizon) holding the
                ``RobotState`` fields, ``action`` and ``sim_ctx``.

        Returns:
            Total loss and per-block MSE, averaged over the batches and keyed
            ``WorldModel/<name>``.
        """
        self.dynamics.train()
        for param in self._params:
            param.requires_grad_(True)
        logs: dict[str, list[torch.Tensor]] = {}
        for td in batches:
            for name, value in self._step(td).items():
                logs.setdefault(name, []).append(value)
        return {f"WorldModel/{name}": torch.stack(v).mean() for name, v in logs.items()}

    def _update_batch(self, td: TensorDict) -> dict[str, torch.Tensor]:
        """Runs one gradient step on one batch of windows.

        Args:
            td: Windows of batch size (batch_size, horizon).

        Returns:
            Total loss under ``loss`` and the MSE of each loss block.
        """
        names = [field.name for field in RobotState.fields()]
        states = RobotState.from_tensordict(td.select(*names))
        pred = self._rollout(states, td["action"], td["sim_ctx"][:, 0])
        target = states[:, 1:]
        loss, block_mse = self._losses(pred, target)
        optimize_step(self.optimizers, loss, self._params, self.cfg.max_grad_norm)
        return {"loss": loss.detach(), **{k: v.detach() for k, v in block_mse.items()}}

    @torch.compile
    def _rollout(
        self, states: RobotState, actions: torch.Tensor, sim_ctx: torch.Tensor
    ) -> RobotState:
        """Rolls the dynamics out from the first frame with the window's actions.

        Args:
            states: Real states of shape (batch_size, horizon, ...).
            actions: Raw actions of shape (batch_size, horizon, action_dim).
            sim_ctx: Sim context of shape (batch_size, sim_ctx_dim).

        Returns:
            Predicted states of shape (batch_size, horizon - 1, ...).
        """
        state = states[:, 0]
        preds: list[RobotState] = []
        for k in range(actions.shape[1] - 1):
            state = self.dynamics(state, actions[:, k], sim_ctx)
            preds.append(state)
        return torch.stack(preds, dim=1)

    @torch.compile
    def _losses(
        self, pred: RobotState, target: RobotState
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Computes the Bryson-weighted loss over the loss blocks.

        Args:
            pred: Predicted states.
            target: Real states of the same shape.

        Returns:
            The total loss, and the per-element MSE of each loss block.
        """
        pred_blocks = loss_blocks(pred)
        target_blocks = loss_blocks(target)
        loss = 0.0
        total_width = 0
        block_mse = {}
        for name, pred_block in pred_blocks.items():
            sq_err = (pred_block - target_blocks[name]).square()
            nominal = self.cfg.nominal_errors[name]
            loss = loss + sq_err.sum(-1).mean() / nominal**2
            total_width += pred_block.shape[-1]
            block_mse[name] = sq_err.mean()
        return loss / total_width, block_mse
