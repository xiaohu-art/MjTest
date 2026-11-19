"""Actor and critic networks."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from mj_test.utils.mlp import build_mlp
from mj_test.utils.normalization import EmpiricalNormalization


@dataclass
class ActorCfg:
    """Hyperparameters of the actor network.

    Attributes:
        latent_dim: Size of the tanh-bounded goal latent.
        hidden_dims: Hidden layer widths of the encoder and the decoder.
    """

    latent_dim: int = 32
    hidden_dims: tuple[int, ...] = (2048, 1024, 512, 256)


@dataclass
class CriticCfg:
    """Hyperparameters of the critic network.

    Attributes:
        hidden_dims: Hidden layer widths.
    """

    hidden_dims: tuple[int, ...] = (256, 128)


class Actor(nn.Module):
    """Deterministic actor with a goal encoder and a proprio-conditioned decoder.

    Args:
        proprio_dim: Size of the proprioceptive observation.
        goal_dim: Size of the goal.
        action_dim: Number of action dimensions.
        latent_dim: Size of the tanh-bounded goal latent.
        hidden_dims: Hidden layer widths of the encoder and the decoder.
    """

    def __init__(
        self,
        proprio_dim: int,
        goal_dim: int,
        action_dim: int,
        latent_dim: int,
        hidden_dims: tuple[int, ...],
    ) -> None:
        super().__init__()
        self.proprio_dim = proprio_dim
        self.goal_dim = goal_dim
        self.proprio_normalizer = EmpiricalNormalization(proprio_dim)
        self.goal_normalizer = EmpiricalNormalization(goal_dim)
        self.encoder = build_mlp(goal_dim, hidden_dims, latent_dim)
        self.decoder = build_mlp(latent_dim + proprio_dim, hidden_dims, action_dim)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Computes the raw actions.

        Args:
            obs: Actor observations ``[proprio, goal]`` of shape
                (..., proprio_dim + goal_dim).

        Returns:
            Raw actions of shape (..., action_dim).
        """
        proprio, goal = obs.split([self.proprio_dim, self.goal_dim], dim=-1)
        latent = torch.tanh(self.encoder(self.goal_normalizer(goal)))
        return self.decoder(torch.cat([latent, self.proprio_normalizer(proprio)], -1))

    def update_normalization(self, obs: torch.Tensor) -> None:
        """Updates the input normalizers with real actor observations.

        Args:
            obs: Actor observations of shape (..., proprio_dim + goal_dim).
        """
        proprio, goal = obs.split([self.proprio_dim, self.goal_dim], dim=-1)
        normalizers = (self.proprio_normalizer, self.goal_normalizer)
        for normalizer, x in zip(normalizers, (proprio, goal), strict=True):
            normalizer.train()
            normalizer.update(x.reshape(-1, x.shape[-1]))
            normalizer.eval()


class Critic(nn.Module):
    """Value function over critic observations.

    Args:
        obs_dim: Size of the critic observations.
        hidden_dims: Hidden layer widths.
    """

    def __init__(self, obs_dim: int, hidden_dims: tuple[int, ...]) -> None:
        super().__init__()
        self.network = build_mlp(obs_dim, hidden_dims, 1)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Computes the values.

        Args:
            obs: Critic observations of shape (..., obs_dim).

        Returns:
            Values of shape (..., 1).
        """
        return self.network(obs)
