"""Optimizer construction, the shared gradient step and its CUDA graph replay."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import torch
from tensordict import TensorDict
from torch import nn


def build_optimizers(
    modules: Sequence[nn.Module],
    lr: float,
    betas: tuple[float, float] = (0.9, 0.999),
    device: str | torch.device = "cpu",
) -> list[torch.optim.Optimizer]:
    """Builds Muon for hidden Linear weights and Adam for all other parameters.

    Args:
        modules: Modules whose parameters are optimized.
        lr: Learning rate of both optimizers.
        betas: Adam betas.
        device: Device of Adam's learning rate tensor.

    Returns:
        Adam, followed by Muon when any hidden weight exists.
    """
    hidden = [
        linear.weight
        for module in modules
        for seq in module.modules()
        if isinstance(seq, nn.Sequential)
        for linear in [m for m in seq if isinstance(m, nn.Linear)][1:-1]
    ]
    hidden_ids = {id(p) for p in hidden}
    rest = [p for m in modules for p in m.parameters() if id(p) not in hidden_ids]
    adam = torch.optim.Adam(
        rest,
        lr=torch.tensor(lr, device=device),
        betas=betas,
        fused=True,
        capturable=True,
    )
    if not hidden:
        return [adam]
    return [adam, torch.optim.Muon(hidden, lr=lr, adjust_lr_fn="match_rms_adamw")]


def optimize_step(
    optimizers: Sequence[torch.optim.Optimizer],
    loss: torch.Tensor,
    params: Sequence[torch.Tensor],
    max_norm: float,
    clip_value: float = 0.0,
) -> torch.Tensor:
    """Runs zero_grad, backward, gradient clipping and one optimizer step.

    Args:
        optimizers: Optimizers that own ``params``.
        loss: Scalar loss to minimize.
        params: Parameters whose gradients are clipped.
        max_norm: Gradient norm clipping threshold.
        clip_value: Elementwise gradient clipping threshold; 0 disables it.

    Returns:
        The gradient norm before clipping.
    """
    for optimizer in optimizers:
        optimizer.zero_grad(set_to_none=True)
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(params, max_norm=max_norm)
    if clip_value > 0:
        torch.nn.utils.clip_grad_value_(params, clip_value)
    for optimizer in optimizers:
        optimizer.step()
    return grad_norm


class CapturedUpdate:
    """Runs an update eagerly for its first calls, then replays it as a CUDA graph.

    Args:
        fn: Update from a TensorDict of inputs to a dict of statistics. It must not
            sync with the host, must keep its shapes, and must change the tensors
            it reads besides its inputs only in place.
        warmup: Number of eager calls before the capture.
    """

    def __init__(
        self,
        fn: Callable[[TensorDict], dict[str, torch.Tensor]],
        warmup: int = 3,
    ) -> None:
        self._fn = fn
        self._warmup = warmup
        self._inputs: TensorDict | None = None
        self._outputs: dict[str, torch.Tensor] = {}
        self._graph: torch.cuda.CUDAGraph | None = None
        self._stream: torch.cuda.Stream | None = None

    def __call__(self, inputs: TensorDict) -> dict[str, torch.Tensor]:
        """Runs the update on ``inputs``.

        Args:
            inputs: Inputs of the update, with the same keys and shapes on every
                call.

        Returns:
            The statistics of the update.
        """
        if inputs.device is None or inputs.device.type != "cuda":
            return self._fn(inputs)
        if self._inputs is None:
            self._inputs = inputs.clone()
            self._stream = torch.cuda.Stream(inputs.device)
        else:
            self._inputs.update_(inputs)
        if self._warmup > 0:
            self._warmup -= 1
            self._stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(self._stream):
                outputs = self._fn(self._inputs)
            torch.cuda.current_stream().wait_stream(self._stream)
            return outputs
        if self._graph is None:
            self._graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self._graph, stream=self._stream):
                self._outputs = self._fn(self._inputs)
        self._graph.replay()
        return {name: value.clone() for name, value in self._outputs.items()}
