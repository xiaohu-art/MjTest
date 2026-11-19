"""Multilayer perceptron builder and the iFSQ latent quantizer."""

from __future__ import annotations

import torch
from torch import nn


def build_mlp(
    input_dim: int,
    hidden_dims: tuple[int, ...],
    output_dim: int,
) -> nn.Sequential:
    """Build an MLP: ``input_dim → (Linear → LN → GELU)* → Linear(output_dim)``."""
    layers: list[nn.Module] = []
    curr = input_dim
    for h in hidden_dims:
        layers += [nn.Linear(curr, h), nn.LayerNorm(h), nn.GELU()]
        curr = h
    layers.append(nn.Linear(curr, output_dim))
    return nn.Sequential(*layers)


class IFSQ(nn.Module):
    """Finite scalar quantization with the iFSQ bound ``2 sigmoid(1.6 x) - 1``.

    Args:
        levels: Number of uniformly spaced values in [-1, 1] that each scalar is
            rounded to.
    """

    def __init__(self, levels: int) -> None:
        super().__init__()
        self.levels = levels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Quantizes each scalar, with straight-through gradients of the bound."""
        bounded = 2.0 * torch.sigmoid(1.6 * x) - 1.0
        steps = self.levels - 1
        quantized = torch.round((bounded + 1.0) * steps / 2.0) * 2.0 / steps - 1.0
        return bounded + (quantized - bounded).detach()
