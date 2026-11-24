"""Supervised training of the world model."""

from __future__ import annotations

from collections.abc import Iterable

import torch
from omegaconf import DictConfig
from tensordict import TensorDict

from mj_test.utils.forward_kinematics import GenCoords
from mj_test.utils.optim import CapturedUpdate, build_optimizers, optimize_step
from mj_test.wm.dynamics import Dynamics
from mj_test.wm.representation import local_state, loss_blocks


class WorldModelLearner:
    """Fits the dynamics to real windows with an autoregressive rollout loss."""

    def __init__(self, dynamics: Dynamics, cfg: DictConfig) -> None:
        self.dynamics = dynamics
        self.cfg = cfg
        self.nominal_errors = {name: float(v) for name, v in cfg.nominal_errors.items()}
        self._params = list(dynamics.parameters())
        device = self._params[0].device
        self.optimizers = build_optimizers([dynamics], cfg.lr, device=device)
        self._step = CapturedUpdate(self._update_batch)

    def update_normalization(self, states: GenCoords, actions: torch.Tensor) -> None:
        """Updates the dynamics normalizers with collected states and raw actions."""
        x = local_state(states)
        a = self.dynamics.action_term(actions)
        normalizers = (self.dynamics.state_normalizer, self.dynamics.action_normalizer)
        for normalizer in normalizers:
            normalizer.train()
        self.dynamics.state_normalizer.update(x.reshape(-1, x.shape[-1]))
        self.dynamics.action_normalizer.update(a.reshape(-1, a.shape[-1]))
        for normalizer in normalizers:
            normalizer.eval()

    def update(self, batches: Iterable[TensorDict]) -> dict[str, torch.Tensor]:
        """Runs one gradient step per batch of windows."""
        self.dynamics.train()
        for param in self._params:
            param.requires_grad_(True)
        logs: dict[str, list[torch.Tensor]] = {}
        for td in batches:
            for name, value in self._step(td).items():
                logs.setdefault(name, []).append(value)
        return {f"WorldModel/{name}": torch.stack(v).mean() for name, v in logs.items()}

    def _update_batch(self, td: TensorDict) -> dict[str, torch.Tensor]:
        """Runs one gradient step on one batch of windows."""
        states = td["gen"]
        pred = self._rollout(states, td["action"], td["sim_ctx"][:, 0], td["force"])
        target = states[:, 1:]
        loss, block_mse = self._losses(pred, target)
        optimize_step(self.optimizers, loss, self._params, self.cfg.max_grad_norm)
        return {"loss": loss.detach(), **{k: v.detach() for k, v in block_mse.items()}}

    @torch.compile
    def _rollout(
        self,
        states: GenCoords,
        actions: torch.Tensor,
        sim_ctx: torch.Tensor,
        forces: torch.Tensor,
    ) -> GenCoords:
        """Rolls the dynamics out from the first frame with the window's actions.

        Args:
            states: Real states of shape (batch_size, horizon, ...).
            actions: Raw actions of shape (batch_size, horizon, action_dim).
            sim_ctx: Sim context of shape (batch_size, sim_ctx_dim).
            forces: World-frame external forces of shape (batch_size, horizon, K, 3).

        Returns:
            Predicted states of shape (batch_size, horizon - 1, ...).
        """
        state = states[:, 0]
        preds: list[GenCoords] = []
        for k in range(actions.shape[1] - 1):
            state = self.dynamics(state, actions[:, k], sim_ctx, forces[:, k])
            preds.append(state)
        return torch.stack(preds, dim=1)

    @torch.compile
    def _losses(
        self, pred: GenCoords, target: GenCoords
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Computes the Bryson-weighted loss and the MSE of each loss block."""
        pred_blocks = loss_blocks(pred)
        target_blocks = loss_blocks(target)
        loss = 0.0
        total_width = 0
        block_mse = {}
        for name, pred_block in pred_blocks.items():
            sq_err = (pred_block - target_blocks[name]).square()
            nominal = self.nominal_errors[name]
            loss = loss + sq_err.sum(-1).mean() / nominal**2
            total_width += pred_block.shape[-1]
            block_mse[name] = sq_err.mean()
        return loss / total_width, block_mse
