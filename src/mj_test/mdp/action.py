"""The joint position action term of the policy."""

from __future__ import annotations

import copy
from pkgutil import resolve_name
from typing import TYPE_CHECKING

import torch
from mjlab.utils.lab_api.string import resolve_matching_names_values
from omegaconf import DictConfig

if TYPE_CHECKING:
    from mjlab.entity import Entity

__all__ = ["JointPositionAction"]


class JointPositionAction:
    """Raw actions to PD joint position targets, ``raw * scale + offset``."""

    def __init__(self, cfg: DictConfig, entity: Entity) -> None:
        self.entity = entity
        device = entity.data.device
        joint_ids, self.names = entity.find_joints_by_actuator_names(
            tuple(cfg.actuator_names)
        )
        self.joint_ids = torch.tensor(joint_ids, dtype=torch.long, device=device)

        scale = cfg.scale
        if isinstance(scale, str):
            scale = resolve_name(scale)
        self.scale = torch.ones(self.dim, device=device)
        if isinstance(scale, (int, float)):
            self.scale.fill_(scale)
        else:
            ids, _, values = resolve_matching_names_values(dict(scale), self.names)
            self.scale[ids] = torch.tensor(values, device=device)
        self.offset = torch.zeros(self.dim, device=device)
        if cfg.use_default_offset:
            self.offset = entity.data.default_joint_pos[0, self.joint_ids].clone()

        num_envs = entity.data.joint_pos.shape[0]
        self.action = torch.zeros(num_envs, self.dim, device=device)
        self.prev_action = torch.zeros_like(self.action)

    @property
    def dim(self) -> int:
        return len(self.names)

    def __call__(self, action: torch.Tensor) -> torch.Tensor:
        return action * self.scale + self.offset

    def like(self) -> JointPositionAction:
        return copy.copy(self)

    def process(self, action: torch.Tensor) -> None:
        self.prev_action, self.action = self.action, action

    def apply(self) -> None:
        encoder_bias = self.entity.data.encoder_bias[:, self.joint_ids]
        target = self(self.action) - encoder_bias
        self.entity.set_joint_position_target(target, joint_ids=self.joint_ids)

    def reset(self, env_ids: torch.Tensor) -> None:
        self.action = self.action.index_fill(0, env_ids, 0.0)
        self.prev_action = self.prev_action.index_fill(0, env_ids, 0.0)
