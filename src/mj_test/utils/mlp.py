"""Multilayer perceptron builder."""

from __future__ import annotations

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
