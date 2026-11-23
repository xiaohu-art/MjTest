"""A simplified direct RL env on mjlab's MuJoCo Warp simulation."""

from __future__ import annotations

import importlib
import math
from dataclasses import dataclass
from typing import Any

import mujoco_warp as mjwarp
import torch
from mjlab.entity import EntityData
from mjlab.envs.mdp import dr
from mjlab.managers.event_manager import EventManager, EventTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.scene import Scene
from mjlab.scene.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.sim.sim import Simulation
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.random import seed_rng
from mjlab.viewer.viewer_config import ViewerConfig
from omegaconf import DictConfig, OmegaConf
from tensordict import TensorDict

from mj_test.envs.env import Env
from mj_test.mdp.action import JointPositionAction
from mj_test.utils.episode_stats import EpisodeStats
from mj_test.wm.sim_context import SimContext


def _import(path: str) -> Any:
    """Imports the object at ``module:attribute``."""
    module, _, attribute = path.partition(":")
    return getattr(importlib.import_module(module), attribute)


@dataclass(kw_only=True)
class QuietSimulationCfg(SimulationCfg):
    """Simulation config that silences the solver iteration overflow warnings."""

    def apply_wp_opt(self, wp_opt: mjwarp.Option) -> None:
        """Applies the options and clears the iteration overflow warning flags."""
        super().apply_wp_opt(wp_opt)
        wp_opt.warn_overflow &= ~int(
            mjwarp.OverflowType.ITERATIONS | mjwarp.OverflowType.LS_ITERATIONS
        )


class SimEnv(Env):
    """Direct RL env."""

    def __init__(self, cfg: DictConfig, device: str, seed: int | None = None) -> None:
        if seed is not None:
            seed_rng(seed)
        scene_cfg = SceneCfg(
            terrain=TerrainEntityCfg(terrain_type=cfg.scene.terrain),
            entities={cfg.scene.entity_name: _import(cfg.scene.robot)()},
            num_envs=cfg.num_envs,
        )
        self.scene = Scene(scene_cfg, device=device)

        sim_cfg = QuietSimulationCfg(
            nconmax=cfg.sim.nconmax,
            njmax=cfg.sim.njmax,
            mujoco=MujocoCfg(
                timestep=cfg.sim.timestep,
                iterations=cfg.sim.iterations,
                ls_iterations=cfg.sim.ls_iterations,
            ),
        )
        self.sim = Simulation(
            num_envs=self.scene.num_envs,
            cfg=sim_cfg,
            spec=self.scene.spec,
            variant_info=self.scene.collect_variant_info(),
            device=device,
        )
        self.scene.initialize(
            mj_model=self.sim.mj_model, model=self.sim.model, data=self.sim.data
        )
        self.sim.add_data_listener(self.scene.invalidate_sensor_caches)
        if self.scene.sensor_context is not None:
            self.sim.set_sensor_context(self.scene.sensor_context)

        self.decimation = cfg.decimation
        self.physics_dt = cfg.sim.timestep
        self.entity = self.scene[cfg.scene.entity_name]

        self.events: dict[str, EventTermCfg] = {}
        for name, event in cfg.events.items():
            params = OmegaConf.to_object(event.params) if "params" in event else {}
            ranges = params.get("ranges")
            if isinstance(ranges, list):
                params["ranges"] = tuple(ranges)
            elif isinstance(ranges, dict):
                params["ranges"] = {
                    key: tuple(bounds) for key, bounds in ranges.items()
                }
            if "asset_cfg" in params:
                params["asset_cfg"] = SceneEntityCfg(**params["asset_cfg"])
            self.events[name] = EventTermCfg(
                mode=event.mode, func=getattr(dr, event.func), params=params
            )
        self.event_manager = EventManager(self.events, self)
        self.sim.expand_model_fields(self.event_manager.domain_randomization_fields)
        self.event_manager.apply(mode="startup")

        self.action_term = JointPositionAction(cfg.action, self.entity)

        viewer = OmegaConf.to_object(cfg.viewer)
        for key in ("lookat", "geom_group", "site_group"):
            if key in viewer:
                viewer[key] = tuple(viewer[key])
        self.viewer_cfg = ViewerConfig(**viewer)

        super().__init__(cfg, dt=self.physics_dt * self.decimation)

        self.sim_context = SimContext.from_env(self)

        n = self.num_envs
        self.episode_length_buf = torch.zeros(n, dtype=torch.long, device=device)
        self.max_episode_length = math.ceil(cfg.episode_length_s / self.dt)
        self.reward_buf = torch.zeros(n, device=device)
        self.reset_terminated = torch.zeros(n, dtype=torch.bool, device=device)
        self.reset_time_outs = torch.zeros(n, dtype=torch.bool, device=device)

        self.command = _import(cfg.command.class_type)(cfg.command, self)
        self.reset()
        self.stats = EpisodeStats(
            self.rewards,
            tuple(self.terminations),
            tuple(self.command.metrics()),
            n,
            self.dt,
            cfg.episode_length_s,
            device,
        )

    @property
    def num_envs(self) -> int:
        return self.scene.num_envs

    @property
    def device(self) -> str:
        return self.sim.device

    @property
    def robot(self) -> EntityData:
        return self.entity.data

    @property
    def sim_ctx(self) -> torch.Tensor:
        return self.sim_context.read(self)

    @property
    def env_origins(self) -> torch.Tensor:
        return self.scene.env_origins

    def step(
        self, action: torch.Tensor
    ) -> tuple[TensorDict, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """Advances every env by one step and resets the envs whose episode ended.

        Args:
            action: Raw actions of shape (num_envs, A).
        """
        self.action_term.process(action)
        for _ in range(self.decimation):
            self.action_term.apply()
            self.scene.write_data_to_sim()
            self.sim.step()
            self.scene.update(dt=self.physics_dt)
        self.sim.forward()  # The rewards and terminations read the step's end.
        self.episode_length_buf += 1

        self.reward_buf, terms, errors = self._post_step()
        time_outs = self._time_outs()
        reasons = {
            name: time_outs[name] if term.time_out else errors[name] > 1.0
            for name, term in self.terminations.items()
        }
        self.reset_terminated = _any(
            [errors[name] > 1.0 for name in errors], self.episode_length_buf
        )
        self.reset_time_outs = _any(list(time_outs.values()), self.episode_length_buf)
        ended = self.reset_terminated | self.reset_time_outs
        self.stats.record(
            self, self.reward_buf, terms, self.episode_length_buf, reasons, ended
        )
        self.command.update_command()

        env_ids = ended.nonzero().squeeze(-1)
        if len(env_ids) > 0:
            self._reset_idx(env_ids)
            self.sim.forward()  # The observations read the reset states.
        return (
            self.observations(),
            self.reward_buf,
            self.reset_terminated,
            self.reset_time_outs,
            {"terms": terms},
        )

    def reset(self, env_ids: torch.Tensor | None = None) -> tuple[TensorDict, dict]:
        """Resets envs and starts their next episodes.

        Args:
            env_ids: Envs to reset, or None for every env.
        """
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        self._reset_idx(env_ids)
        self.sim.forward()
        return self.observations(), {}

    def pop_log(self) -> dict[str, float]:
        return self.stats.pop() | self.command.log()

    def close(self) -> None:
        """Releases the env."""

    def _reset_idx(self, env_ids: torch.Tensor) -> None:
        self.sim.reset(env_ids)
        self.scene.reset(env_ids)
        self.action_term.reset(env_ids)
        self.command.reset(env_ids)
        self.scene.write_data_to_sim(env_ids=env_ids)
        self.episode_length_buf[env_ids] = 0


def _any(flags: list[torch.Tensor], like: torch.Tensor) -> torch.Tensor:
    """Elementwise OR of boolean flags, False everywhere without any."""
    out = torch.zeros_like(like, dtype=torch.bool)
    for flag in flags:
        out = out | flag
    return out
