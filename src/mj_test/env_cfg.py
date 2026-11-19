"""Env configuration of the G1 motion tracking task."""

import glob
from dataclasses import dataclass

import mujoco_warp as mjwarp
from mjlab.asset_zoo.robots import G1_ACTION_SCALE, get_g1_robot_cfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise

from mj_test import mdp


@dataclass(kw_only=True)
class QuietSimulationCfg(SimulationCfg):
    """Simulation config that silences the solver iteration overflow warnings."""

    def apply_wp_opt(self, wp_opt: mjwarp.Option) -> None:
        """Applies the options and clears the iteration overflow warning flags.

        Args:
            wp_opt: The MuJoCo Warp options to modify in place.
        """
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


def make_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Builds the env configuration.

    Args:
        play: Whether to build the playback variant, with near-infinite episodes
            and no reset randomization of the root pose and velocity.

    Returns:
        The env configuration.
    """
    motion = mdp.MotionCommandCfg(
        entity_name="robot",
        resampling_time_range=(1.0e9, 1.0e9),
        debug_vis=True,
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
        motion_file=MOTION_FILE,
        anchor_body_name="pelvis",
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
    )

    ##
    # Observations
    ##

    proprio_terms = {
        "projected_gravity": ObservationTermCfg(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        ),
        "base_ang_vel": ObservationTermCfg(
            func=mdp.builtin_sensor,
            params={"sensor_name": "robot/imu_ang_vel"},
            noise=Unoise(n_min=-0.2, n_max=0.2),
        ),
        "joint_pos": ObservationTermCfg(
            func=mdp.joint_pos_rel,
            params={"biased": True},
            noise=Unoise(n_min=-0.01, n_max=0.01),
        ),
        "joint_vel": ObservationTermCfg(
            func=mdp.joint_vel_rel,
            noise=Unoise(n_min=-0.5, n_max=0.5),
        ),
        "actions": ObservationTermCfg(func=mdp.last_action),
    }

    state_terms = {
        "root_pos": ObservationTermCfg(func=mdp.root_pos),
        "root_quat": ObservationTermCfg(func=mdp.root_quat),
        "joint_pos": ObservationTermCfg(func=mdp.joint_pos),
        "root_lin_vel": ObservationTermCfg(func=mdp.root_lin_vel),
        "root_ang_vel": ObservationTermCfg(func=mdp.root_ang_vel),
        "joint_vel": ObservationTermCfg(func=mdp.joint_vel),
    }

    goal_terms = {
        "goal": ObservationTermCfg(
            func=mdp.motion_goal, params={"command_name": "motion"}
        ),
    }

    observations = {
        "proprio": ObservationGroupCfg(
            terms=proprio_terms,
            concatenate_terms=True,
            enable_corruption=True,
        ),
        "state": ObservationGroupCfg(
            terms=state_terms,
            concatenate_terms=False,
            enable_corruption=False,
        ),
        "goal": ObservationGroupCfg(
            terms=goal_terms, concatenate_terms=True, enable_corruption=False
        ),
    }

    ##
    # Actions
    ##

    actions = {
        "joint_pos": JointPositionActionCfg(
            entity_name="robot",
            actuator_names=(".*",),
            scale=G1_ACTION_SCALE,
            use_default_offset=True,
        )
    }

    ##
    # Rewards
    ##

    rewards = {
        "motion_global_root_pos": RewardTermCfg(
            func=mdp.transition_reward,
            weight=0.5,
            params={"term": mdp.global_anchor_position_error_exp, "std": 0.3},
        ),
        "motion_global_root_ori": RewardTermCfg(
            func=mdp.transition_reward,
            weight=0.5,
            params={"term": mdp.global_anchor_orientation_error_exp, "std": 0.4},
        ),
        "motion_body_pos": RewardTermCfg(
            func=mdp.transition_reward,
            weight=1.0,
            params={"term": mdp.relative_body_position_error_exp, "std": 0.3},
        ),
        "motion_body_ori": RewardTermCfg(
            func=mdp.transition_reward,
            weight=1.0,
            params={"term": mdp.relative_body_orientation_error_exp, "std": 0.4},
        ),
        "motion_body_lin_vel": RewardTermCfg(
            func=mdp.transition_reward,
            weight=1.0,
            params={"term": mdp.global_body_linear_velocity_error_exp, "std": 1.0},
        ),
        "motion_body_ang_vel": RewardTermCfg(
            func=mdp.transition_reward,
            weight=1.0,
            params={"term": mdp.global_body_angular_velocity_error_exp, "std": 3.14},
        ),
        "action_rate_l2": RewardTermCfg(
            func=mdp.transition_reward,
            weight=-1e-1,
            params={"term": mdp.action_rate},
        ),
        "joint_limit": RewardTermCfg(
            func=mdp.transition_reward,
            weight=-2.0,
            params={"term": mdp.joint_limit_violation},
        ),
    }

    ##
    # Events
    ##

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

    ##
    # Terminations
    ##

    terminations = {
        "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
        "motion_end": TerminationTermCfg(
            func=mdp.motion_end, time_out=True, params={"command_name": "motion"}
        ),
        "anchor_pos": TerminationTermCfg(
            func=mdp.transition_termination,
            params={"error": mdp.anchor_height_error, "threshold": 0.25},
        ),
        "anchor_ori": TerminationTermCfg(
            func=mdp.transition_termination,
            params={"error": mdp.anchor_tilt_error, "threshold": 0.8},
        ),
    }

    cfg = ManagerBasedRlEnvCfg(
        scene=SceneCfg(
            terrain=TerrainEntityCfg(terrain_type="plane"),
            entities={"robot": get_g1_robot_cfg()},
            num_envs=1,
        ),
        commands={"motion": motion},
        observations=observations,
        actions=actions,
        events=events,
        rewards=rewards,
        terminations=terminations,
        # sim=SimulationCfg(
        sim=QuietSimulationCfg(
            nconmax=35,
            njmax=250,
            mujoco=MujocoCfg(timestep=0.005, iterations=10, ls_iterations=20),
        ),
        decimation=4,
        episode_length_s=10.0,
    )

    if play:
        cfg.episode_length_s = int(1e9)
        cfg.observations["proprio"].enable_corruption = False
        motion.pose_range = {}
        motion.velocity_range = {}

    return cfg
