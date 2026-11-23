"""Actor and critic networks."""

from __future__ import annotations

import torch
from tensordict import TensorDict
from torch import nn

from mj_test.utils.mlp import IFSQ, build_mlp
from mj_test.utils.normalization import EmpiricalNormalization

FSQ_SCALARS = 64
FSQ_LEVELS = 33


class Actor(nn.Module):
    """Deterministic actor with a goal encoder and a proprio-conditioned decoder.

    It reads the ``proprio`` and ``goal`` observation groups.
    """

    def __init__(
        self,
        proprio_dim: int,
        goal_dim: int,
        action_dim: int,
        hidden_dims: tuple[int, ...],
    ) -> None:
        super().__init__()
        self.proprio_normalizer = EmpiricalNormalization(proprio_dim)
        self.goal_normalizer = EmpiricalNormalization(goal_dim)
        self.encoder = build_mlp(goal_dim, hidden_dims, FSQ_SCALARS)
        self.quantizer = IFSQ(FSQ_LEVELS)
        self.decoder = build_mlp(FSQ_SCALARS + proprio_dim, hidden_dims, action_dim)

    def forward(self, obs: TensorDict) -> torch.Tensor:
        """Computes the raw actions from the ``proprio`` and ``goal`` groups."""
        latent = self.quantizer(self.encoder(self.goal_normalizer(obs["goal"])))
        proprio = self.proprio_normalizer(obs["proprio"])
        return self.decoder(torch.cat([latent, proprio], dim=-1))

    def update_normalization(self, obs: TensorDict) -> None:
        """Updates the input normalizers with real observations."""
        normalizers = (self.proprio_normalizer, self.goal_normalizer)
        for normalizer, x in zip(
            normalizers, (obs["proprio"], obs["goal"]), strict=True
        ):
            normalizer.train()
            normalizer.update(x.reshape(-1, x.shape[-1]))
            normalizer.eval()


class Critic(nn.Module):
    """Value function over critic observations."""

    def __init__(self, obs_dim: int, hidden_dims: tuple[int, ...]) -> None:
        super().__init__()
        self.network = build_mlp(obs_dim, hidden_dims, 1)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Computes the values of shape (..., 1)."""
        return self.network(obs)
