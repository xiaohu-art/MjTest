"""Adapter that runs the simulated tracking env in the mjlab viewers."""

from __future__ import annotations

import copy
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import torch
from mjlab.utils.lab_api.math import matrix_from_quat
from tensordict import TensorDict

if TYPE_CHECKING:
    import viser
    from mjlab.viewer.debug_visualizer import DebugVisualizer

    from mj_test.envs.sim_env import SimEnv

_DESIRED_FRAME_COLORS = ((1.0, 0.5, 0.5), (0.5, 1.0, 0.5), (0.5, 0.5, 1.0))


class _RewardTerms:
    """Weighted reward terms of the last step, in the form the viewer plots."""

    def __init__(self, env: SimEnv) -> None:
        self.env = env
        self.terms = {
            name: torch.zeros(env.num_envs, device=env.device)
            for name in env.cfg.rewards
        }

    def get_active_iterable_terms(
        self, env_idx: int
    ) -> Sequence[tuple[str, Sequence[float]]]:
        """Returns ``(name, [value])`` pairs of the weighted reward terms of one env."""
        rewards = self.env.cfg.rewards
        return [
            (name, [float(self.terms[name][env_idx]) * term.weight])
            for name, term in rewards.items()
        ]

    def get_visualizable_terms(self) -> list[tuple[str, Any]]:
        """Returns no reward terms with debug visualizations."""
        return []


class _NoTerms:
    """Empty metrics for the viewer."""

    def get_active_iterable_terms(
        self, env_idx: int
    ) -> Sequence[tuple[str, Sequence[float]]]:
        """Returns no terms."""
        del env_idx
        return []


class _MotionReference:
    """Reference motion drawing and frame controls, in the form of a command manager.

    Args:
        env: The simulated env.
        mode: Draw the reference as a ghost robot or as body frames.
        ghost_color: RGBA color of the ghost robot.
    """

    active_terms = ["motion"]

    def __init__(
        self,
        env: SimEnv,
        mode: Literal["ghost", "frames"],
        ghost_color: tuple[float, float, float, float],
    ) -> None:
        self.env = env
        self.mode = mode
        self.ghost_color = np.array(ghost_color, dtype=np.float32)
        self.enabled = True
        self.body_ids = torch.tensor(
            env.robot.find_bodies(env.cfg.body_names, preserve_order=True)[0],
            device=env.device,
        )
        self._ghost_model = None
        self._scrubber_handles: tuple[Any, ...] = ()

    def debug_vis(self, visualizer: DebugVisualizer) -> None:
        """Draws the reference of the frame that each shown env tracks."""
        env_indices = visualizer.get_env_indices(self.env.num_envs)
        if not self.enabled or not env_indices:
            return
        env = self.env
        ref_bodies = env.motion.bodies[env.frames]
        ref_pos = (ref_bodies.pos + env.scene.env_origins[:, None]).cpu().numpy()
        if self.mode == "ghost":
            model = self._ghost()
            indexing = env.robot.indexing
            root_q = indexing.free_joint_q_adr.cpu().numpy()
            joint_q = indexing.joint_q_adr.cpu().numpy()
            ref_quat = ref_bodies.quat[:, 0].cpu().numpy()
            ref_joint_pos = env.motion.states.joint_pos[env.frames].cpu().numpy()
            for i in env_indices:
                qpos = np.zeros(env.sim.mj_model.nq)
                qpos[root_q[0:3]] = ref_pos[i, 0]
                qpos[root_q[3:7]] = ref_quat[i]
                qpos[joint_q] = ref_joint_pos[i]
                visualizer.add_ghost_mesh(qpos, model=model, label=f"ghost_{i}")
            return
        ref_rot = matrix_from_quat(ref_bodies.quat).cpu().numpy()
        data = env.robot.data
        robot_pos = data.body_link_pos_w[:, self.body_ids].cpu().numpy()
        robot_rot = matrix_from_quat(data.body_link_quat_w[:, self.body_ids])
        robot_rot = robot_rot.cpu().numpy()
        for i in env_indices:
            for j, name in enumerate(env.cfg.body_names):
                visualizer.add_frame(
                    position=ref_pos[i, j],
                    rotation_matrix=ref_rot[i, j],
                    scale=0.08,
                    label=f"desired_{name}_{i}",
                    axis_colors=_DESIRED_FRAME_COLORS,
                )
                visualizer.add_frame(
                    position=robot_pos[i, j],
                    rotation_matrix=robot_rot[i, j],
                    scale=0.12,
                    label=f"current_{name}_{i}",
                )

    def create_gui(
        self,
        server: viser.ViserServer,
        get_env_idx: Callable[[], int],
        on_change: Callable[[], None] | None = None,
        request_action: Callable[[str, Any], None] | None = None,
    ) -> None:
        """Creates the motion frame scrubber in the Viser viewer."""
        max_frame = int(self.env.motion.time_step_total) - 1
        with server.gui.add_folder("Motion"):
            scrubber = server.gui.add_slider(
                "Frame", min=0, max=max_frame, step=1, initial_value=0
            )

            @scrubber.on_update
            def _(_) -> None:
                self.env.frames[get_env_idx()] = int(scrubber.value)
                if on_change is not None:
                    on_change()

            all_envs = server.gui.add_checkbox("All envs", initial_value=True)
            start = server.gui.add_button("Start Here")

            @start.on_click
            def _(_) -> None:
                if request_action is not None:
                    request_action(
                        "CUSTOM", {"type": "gui_reset", "all_envs": all_envs.value}
                    )

        self._scrubber_handles = (scrubber, all_envs, start)
        self.on_viewer_pause(False)

    def create_debug_vis_gui(
        self, server: viser.ViserServer, on_change: Callable[[], None] | None = None
    ) -> None:
        """Adds the checkbox that shows or hides the reference."""
        checkbox = server.gui.add_checkbox("Motion", initial_value=self.enabled)

        @checkbox.on_update
        def _(_) -> None:
            self.enabled = checkbox.value
            if on_change is not None:
                on_change()

    def on_viewer_pause(self, paused: bool) -> None:
        """Enables the scrubber while the viewer is paused."""
        for handle in self._scrubber_handles:
            handle.disabled = not paused

    def apply_gui_reset(self, env_ids: torch.Tensor) -> bool:
        """Places envs on the reference state of the scrubber frame."""
        if not self._scrubber_handles:
            return False
        self.env.reset_to_frame(env_ids, int(self._scrubber_handles[0].value))
        self.env.sim.forward()
        return True

    def _ghost(self):
        """Returns a copy of the MuJoCo model with transparent collision geoms."""
        if self._ghost_model is None:
            model = copy.deepcopy(self.env.sim.mj_model)
            for gi in range(model.ngeom):
                if model.geom_contype[gi] != 0 or model.geom_conaffinity[gi] != 0:
                    model.geom_rgba[gi, 3] = 0
                else:
                    model.geom_rgba[gi] = self.ghost_color
            self._ghost_model = model
        return self._ghost_model


class ViewerEnv:
    """Exposes a simulated tracking env through the interface of the mjlab viewers.

    Args:
        env: The simulated env.
        mode: Draw the reference as a ghost robot or as body frames.
        ghost_color: RGBA color of the ghost robot.
    """

    def __init__(
        self,
        env: SimEnv,
        mode: Literal["ghost", "frames"] = "ghost",
        ghost_color: tuple[float, float, float, float] = (0.5, 0.7, 0.5, 0.5),
    ) -> None:
        self.env = env
        self.cfg = env.cfg
        self.reward_manager = _RewardTerms(env)
        self.metrics_manager = _NoTerms()
        self.command_manager = _MotionReference(env, mode, ghost_color)
        env.sim.forward()

    @property
    def unwrapped(self) -> ViewerEnv:
        """The adapter itself."""
        return self

    @property
    def num_envs(self) -> int:
        """Number of simulated envs."""
        return self.env.num_envs

    @property
    def device(self) -> str:
        """Torch device of the simulation."""
        return self.env.device

    @property
    def sim(self):
        """The simulation."""
        return self.env.sim

    @property
    def scene(self):
        """The scene."""
        return self.env.scene

    @property
    def step_dt(self) -> float:
        """Env step duration in seconds."""
        return self.env.dt

    def get_observations(self) -> TensorDict:
        """Builds the ``actor`` and ``critic`` observations of the current state."""
        return self.env.observations()

    def step(
        self, action: torch.Tensor
    ) -> tuple[TensorDict, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """Steps the env and refreshes the body poses that the viewer draws."""
        result = self.env.step(action)
        self.reward_manager.terms = result[4]["terms"]
        self.env.sim.forward()
        return result

    def reset(self, env_ids: torch.Tensor | None = None) -> tuple[TensorDict, dict]:
        """Resets envs and refreshes the body poses that the viewer draws."""
        result = self.env.reset(env_ids)
        self.env.sim.forward()
        return result

    def update_visualizers(self, visualizer: DebugVisualizer) -> None:
        """Draws the reference motion."""
        self.command_manager.debug_vis(visualizer)

    def close(self) -> None:
        """Releases the env."""
        self.env.close()
