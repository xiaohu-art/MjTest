"""Env configuration of the G1 motion tracking task."""

import glob
from dataclasses import dataclass

import mujoco_warp as mjwarp
from mjlab.asset_zoo.robots import G1_ACTION_SCALE, get_g1_robot_cfg
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise

from mj_test import mdp
from mj_test.envs.env import RewardTerm, TerminationTerm
from mj_test.envs.sim_env import SimEnvCfg


@dataclass(kw_only=True)
class QuietSimulationCfg(SimulationCfg):
    """Simulation config that silences the solver iteration overflow warnings."""

    def apply_wp_opt(self, wp_opt: mjwarp.Option) -> None:
        """Applies the options and clears the iteration overflow warning flags."""
        super().apply_wp_opt(wp_opt)
        wp_opt.warn_overflow &= ~int(
            mjwarp.OverflowType.ITERATIONS | mjwarp.OverflowType.LS_ITERATIONS
        )


MOTION_FILE = sorted(glob.glob("/data/motion_data/datasets/lafan1/*.npz"))

VELOCITY_RANGE = {
    "x": (-0.5, 0.5),
    "y": (-0.5, 0.5),
    "z": (-0.2, 0.2),
    "roll": (-0.52, 0.52),
    "pitch": (-0.52, 0.52),
    "yaw": (-0.78, 0.78),
}


def make_env_cfg(play: bool = False) -> SimEnvCfg:
    """Builds the env configuration.

    Args:
        play: Whether to build the playback variant, with near-infinite episodes,
            no observation noise and no reset randomization of the root pose and
            velocity.
    """
    rewards = {
        "motion_global_root_pos": RewardTerm(
            mdp.global_anchor_position_error_exp, 0.5, {"std": 0.3}
        ),
        "motion_global_root_ori": RewardTerm(
            mdp.global_anchor_orientation_error_exp, 0.5, {"std": 0.4}
        ),
        "motion_body_pos": RewardTerm(
            mdp.relative_body_position_error_exp, 1.0, {"std": 0.3}
        ),
        "motion_body_ori": RewardTerm(
            mdp.relative_body_orientation_error_exp, 1.0, {"std": 0.4}
        ),
        "motion_body_lin_vel": RewardTerm(
            mdp.global_body_linear_velocity_error_exp, 1.0, {"std": 1.0}
        ),
        "motion_body_ang_vel": RewardTerm(
            mdp.global_body_angular_velocity_error_exp, 1.0, {"std": 3.14}
        ),
        "action_rate_l2": RewardTerm(mdp.action_rate, -1e-1),
        "joint_limit": RewardTerm(mdp.joint_limit_violation, -2.0),
    }

    terminations = {
        "anchor_pos": TerminationTerm(mdp.anchor_height_error, 0.25),
        "anchor_ori": TerminationTerm(mdp.anchor_tilt_error, 0.8),
    }

    proprio_noise = {
        "projected_gravity": Unoise(n_min=-0.05, n_max=0.05),
        "base_ang_vel": Unoise(n_min=-0.2, n_max=0.2),
        "joint_pos": Unoise(n_min=-0.01, n_max=0.01),
        "joint_vel": Unoise(n_min=-0.5, n_max=0.5),
    }

    events = {
        "base_com": EventTermCfg(
            mode="startup",
            func=dr.body_com_offset,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=("torso_link",)),
                "operation": "add",
                "ranges": {
                    0: (-0.025, 0.025),
                    1: (-0.05, 0.05),
                    2: (-0.05, 0.05),
                },
            },
        ),
        "encoder_bias": EventTermCfg(
            mode="startup",
            func=dr.encoder_bias,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "bias_range": (-0.01, 0.01),
            },
        ),
        "foot_friction": EventTermCfg(
            mode="startup",
            func=dr.geom_friction,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot", geom_names=r"^(left|right)_foot[1-7]_collision$"
                ),
                "operation": "abs",
                "ranges": (0.3, 1.2),
                "shared_random": True,  # All foot geoms share the same friction.
            },
        ),
    }

    cfg = SimEnvCfg(
        motion_file=MOTION_FILE,
        body_names=(
            "pelvis",
            "left_hip_roll_link",
            "left_knee_link",
            "left_ankle_roll_link",
            "right_hip_roll_link",
            "right_knee_link",
            "right_ankle_roll_link",
            "torso_link",
            "left_shoulder_roll_link",
            "left_elbow_link",
            "left_wrist_yaw_link",
            "right_shoulder_roll_link",
            "right_elbow_link",
            "right_wrist_yaw_link",
        ),
        rewards=rewards,
        terminations=terminations,
        proprio_noise=proprio_noise,
        scene=SceneCfg(
            terrain=TerrainEntityCfg(terrain_type="plane"),
            entities={"robot": get_g1_robot_cfg()},
            num_envs=1,
        ),
        sim=QuietSimulationCfg(
            nconmax=35,
            njmax=250,
            mujoco=MujocoCfg(timestep=0.005, iterations=10, ls_iterations=20),
        ),
        action=JointPositionActionCfg(
            entity_name="robot",
            actuator_names=(".*",),
            scale=G1_ACTION_SCALE,
            use_default_offset=True,
        ),
        decimation=4,
        episode_length_s=10.0,
        events=events,
        pose_range={
            "x": (-0.05, 0.05),
            "y": (-0.05, 0.05),
            "z": (-0.01, 0.01),
            "roll": (-0.1, 0.1),
            "pitch": (-0.1, 0.1),
            "yaw": (-0.2, 0.2),
        },
        velocity_range=VELOCITY_RANGE,
        joint_position_range=(-0.1, 0.1),
    )

    if play:
        cfg.episode_length_s = int(1e9)
        cfg.proprio_noise = {}
        cfg.pose_range = {}
        cfg.velocity_range = {}

    return cfg
