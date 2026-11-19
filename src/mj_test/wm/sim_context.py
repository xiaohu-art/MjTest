"""Per-env simulation context that conditions the world model."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import torch
from torch import nn

if TYPE_CHECKING:
    from mjlab.entity import Entity
    from mjlab.envs import ManagerBasedRlEnv

_DEGENERATE_HALF_WIDTH = 1e-3


def _mean_scale(lo: float, hi: float) -> tuple[float, float]:
    """Computes the normalization of one randomization range.

    Args:
        lo: Lower bound of the range.
        hi: Upper bound of the range.

    Returns:
        The range center and half-width.
    """
    if hi < lo:
        raise ValueError(f"Empty range: lo={lo}, hi={hi}.")
    half = (hi - lo) / 2.0
    return (lo + hi) / 2.0, half if half > _DEGENERATE_HALF_WIDTH else 1.0


def _event_params(env: ManagerBasedRlEnv, name: str) -> dict[str, Any]:
    """Returns the parameters of a domain randomization event.

    Args:
        env: The environment.
        name: Name of the event term.

    Returns:
        The event's parameter dictionary.
    """
    event_cfg = env.cfg.events.get(name)
    if event_cfg is None:
        raise RuntimeError(f"No event '{name}' in env.cfg.events.")
    return event_cfg.params


def _operation_name(params: dict[str, Any], default: str) -> str:
    """Returns the name of a domain randomization event's operation.

    Args:
        params: The event's parameter dictionary.
        default: Operation the event function uses when none is given.

    Returns:
        The operation name, such as ``"abs"`` or ``"add"``.
    """
    operation = params.get("operation", default)
    return operation if isinstance(operation, str) else operation.name


def _global_id(asset: Entity, kind: str, local_id: int) -> int:
    """Converts an entity-local body or geom index to its MuJoCo model id.

    Args:
        asset: The entity that owns the element.
        kind: Either ``"body"`` or ``"geom"``.
        local_id: Index of the element within the entity.

    Returns:
        The element's id in the MuJoCo model.
    """
    ids = asset.indexing.body_ids if kind == "body" else asset.indexing.geom_ids
    return int(ids[local_id])


class SimContextSpec(ABC):
    """One randomized simulation parameter exposed to the world model.

    Attributes:
        name: Name of the domain randomization event that sets the parameter.
        dim: Number of context columns the parameter occupies.
    """

    name: str
    dim: int

    @abstractmethod
    def bind(self, env: ManagerBasedRlEnv) -> list[tuple[float, float]]:
        """Resolves model indices and returns the randomization range per column.

        Args:
            env: The environment.

        Returns:
            One ``(lo, hi)`` range per context column.
        """

    @abstractmethod
    def read(self, env: ManagerBasedRlEnv) -> torch.Tensor:
        """Reads the raw per-env values.

        Args:
            env: The environment.

        Returns:
            Raw values of shape (num_envs, dim).
        """


class FootFrictionSpec(SimContextSpec):
    """Tangential foot friction, read from the first matching geom."""

    name = "foot_friction"
    dim = 1

    def __init__(self) -> None:
        self._geom_id: int | None = None

    def bind(self, env: ManagerBasedRlEnv) -> list[tuple[float, float]]:
        params = _event_params(env, self.name)
        if _operation_name(params, "abs") != "abs":
            raise ValueError(
                f"'{self.name}' must use operation='abs' so that its ranges bound "
                "the friction values."
            )
        asset_cfg = params["asset_cfg"]
        asset = env.scene[asset_cfg.name]
        local_ids, _ = asset.find_geoms(asset_cfg.geom_names)
        if not local_ids:
            raise ValueError(
                f"No geoms in '{asset_cfg.name}' match {asset_cfg.geom_names!r}."
            )
        self._geom_id = _global_id(asset, "geom", local_ids[0])
        ranges = params["ranges"]
        return [ranges[0] if isinstance(ranges, dict) else ranges]

    def read(self, env: ManagerBasedRlEnv) -> torch.Tensor:
        if self._geom_id is None:
            raise RuntimeError("FootFrictionSpec.read() called before bind().")
        return env.sim.model.geom_friction[:, self._geom_id : self._geom_id + 1, 0]


class CoMOffsetSpec(SimContextSpec):
    """Center-of-mass offset of one body from its unrandomized position."""

    name = "base_com"
    dim = 3

    def __init__(self) -> None:
        self._body_id: int | None = None
        self._origin: torch.Tensor | None = None

    def bind(self, env: ManagerBasedRlEnv) -> list[tuple[float, float]]:
        params = _event_params(env, self.name)
        asset_cfg = params["asset_cfg"]
        if len(asset_cfg.body_names) != 1:
            raise ValueError(
                f"'{self.name}' must name exactly one body, got "
                f"{asset_cfg.body_names!r}."
            )
        asset = env.scene[asset_cfg.name]
        local_id = asset.body_names.index(asset_cfg.body_names[0])
        self._body_id = _global_id(asset, "body", local_id)

        operation = _operation_name(params, "add")
        default_ipos = env.sim.get_default_field("body_ipos")[self._body_id]
        if operation == "add":
            self._origin = default_ipos
        elif operation == "abs":
            self._origin = torch.zeros_like(default_ipos)
        else:
            raise ValueError(
                f"'{self.name}' must use operation='add' or 'abs', got {operation!r}."
            )

        ranges = params["ranges"]
        if isinstance(ranges, dict):
            return [ranges.get(axis, (0.0, 0.0)) for axis in range(self.dim)]
        return [ranges] * self.dim

    def read(self, env: ManagerBasedRlEnv) -> torch.Tensor:
        if self._body_id is None or self._origin is None:
            raise RuntimeError("CoMOffsetSpec.read() called before bind().")
        return env.sim.model.body_ipos[:, self._body_id, :] - self._origin


class EncoderBiasSpec(SimContextSpec):
    """Per-joint encoder bias."""

    name = "encoder_bias"
    dim = 0

    def __init__(self) -> None:
        self._asset_name: str | None = None

    def bind(self, env: ManagerBasedRlEnv) -> list[tuple[float, float]]:
        params = _event_params(env, self.name)
        self._asset_name = params["asset_cfg"].name
        self.dim = env.scene[self._asset_name].num_joints
        return [params["bias_range"]] * self.dim

    def read(self, env: ManagerBasedRlEnv) -> torch.Tensor:
        if self._asset_name is None:
            raise RuntimeError("EncoderBiasSpec.read() called before bind().")
        return env.scene[self._asset_name].data.encoder_bias


class SimContext(nn.Module):
    """Normalized per-env simulation context, with each column in [-1, 1].

    Args:
        specs: The bound parameter specs, in column order.
        ranges: One ``(lo, hi)`` range per context column.
    """

    def __init__(
        self, specs: list[SimContextSpec], ranges: list[tuple[float, float]]
    ) -> None:
        super().__init__()
        self.specs = list(specs)
        stats = [_mean_scale(lo, hi) for lo, hi in ranges]
        self.register_buffer("mean", torch.tensor([m for m, _ in stats]))
        self.register_buffer("scale", torch.tensor([s for _, s in stats]))

    @classmethod
    def from_env(cls, env: ManagerBasedRlEnv) -> SimContext:
        """Builds and binds the specs of the env's active randomization events.

        Args:
            env: The environment.

        Returns:
            The simulation context on the env's device.
        """
        candidates: tuple[type[SimContextSpec], ...] = (
            FootFrictionSpec,
            CoMOffsetSpec,
            EncoderBiasSpec,
        )
        specs = [c() for c in candidates if c.name in env.cfg.events]
        if not specs:
            raise RuntimeError(
                "No randomization event in env.cfg.events has a sim context spec. "
                "Enable at least one of foot_friction, base_com and encoder_bias."
            )
        ranges = [r for spec in specs for r in spec.bind(env)]
        return cls(specs, ranges).to(env.device)

    @property
    def dim(self) -> int:
        """Number of context columns."""
        return sum(spec.dim for spec in self.specs)

    def read(self, env: ManagerBasedRlEnv) -> torch.Tensor:
        """Reads the normalized context of every env.

        Args:
            env: The environment.

        Returns:
            Normalized context of shape (num_envs, dim), in spec order.
        """
        raw = torch.cat([spec.read(env) for spec in self.specs], dim=-1)
        return (raw - self.mean) / self.scale

    def denormalize(self, sim_ctx: torch.Tensor, name: str) -> torch.Tensor:
        """Recovers the raw values of one spec from a normalized context.

        Args:
            sim_ctx: Normalized context of shape (..., dim).
            name: Name of the spec.

        Returns:
            Raw values of shape (..., spec.dim).
        """
        start = 0
        for spec in self.specs:
            if spec.name == name:
                columns = slice(start, start + spec.dim)
                return sim_ctx[..., columns] * self.scale[columns] + self.mean[columns]
            start += spec.dim
        raise KeyError(f"No sim context spec named {name!r}.")
