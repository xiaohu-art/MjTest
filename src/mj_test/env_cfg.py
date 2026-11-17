from mjlab.asset_zoo.robots import G1_ACTION_SCALE, get_g1_robot_cfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.scene import SceneCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise

from mj_test import mdp


from dataclasses import dataclass

import mujoco_warp as mjwarp
from mjlab.sim import MujocoCfg, SimulationCfg


@dataclass(kw_only=True)
class QuietSimulationCfg(SimulationCfg):
    """不打印求解器迭代上限的警告,其余溢出警告保留。"""

    def apply_wp_opt(self, wp_opt: mjwarp.Option) -> None:
        super().apply_wp_opt(wp_opt)
        wp_opt.warn_overflow &= ~int(
            mjwarp.OverflowType.ITERATIONS | mjwarp.OverflowType.LS_ITERATIONS
        )


MOTION_FILE = "/data/motion_data/datasets/lafan1/walk1_subject1.npz"

VELOCITY_RANGE = {
    "x": (-0.5, 0.5),
    "y": (-0.5, 0.5),
    "z": (-0.2, 0.2),
    "roll": (-0.52, 0.52),
    "pitch": (-0.52, 0.52),
    "yaw": (-0.78, 0.78),
}


def make_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    motion = mdp.MotionCfg(
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

    actor_terms = {
        "command": ObservationTermCfg(
            func=mdp.generated_commands, params={"command_name": "motion"}
        ),
        "motion_anchor_pos_b": ObservationTermCfg(
            func=mdp.motion_anchor_pos_b,
            params={"command_name": "motion"},
            noise=Unoise(n_min=-0.25, n_max=0.25),
        ),
        "motion_anchor_ori_b": ObservationTermCfg(
            func=mdp.motion_anchor_ori_b,
            params={"command_name": "motion"},
            noise=Unoise(n_min=-0.05, n_max=0.05),
        ),
        "base_lin_vel": ObservationTermCfg(
            func=mdp.builtin_sensor,
            params={"sensor_name": "robot/imu_lin_vel"},
            noise=Unoise(n_min=-0.5, n_max=0.5),
        ),
        "base_ang_vel": ObservationTermCfg(
            func=mdp.builtin_sensor,
            params={"sensor_name": "robot/imu_ang_vel"},
            noise=Unoise(n_min=-0.2, n_max=0.2),
        ),
        "joint_pos": ObservationTermCfg(
            func=mdp.joint_pos_rel,
            noise=Unoise(n_min=-0.01, n_max=0.01),
            params={"biased": True},
        ),
        "joint_vel": ObservationTermCfg(
            func=mdp.joint_vel_rel, noise=Unoise(n_min=-0.5, n_max=0.5)
        ),
        "actions": ObservationTermCfg(func=mdp.last_action),
    }

    critic_terms = {
        "command": ObservationTermCfg(
            func=mdp.generated_commands, params={"command_name": "motion"}
        ),
        "motion_anchor_pos_b": ObservationTermCfg(
            func=mdp.motion_anchor_pos_b, params={"command_name": "motion"}
        ),
        "motion_anchor_ori_b": ObservationTermCfg(
            func=mdp.motion_anchor_ori_b, params={"command_name": "motion"}
        ),
        "body_pos": ObservationTermCfg(
            func=mdp.robot_body_pos_b, params={"command_name": "motion"}
        ),
        "body_ori": ObservationTermCfg(
            func=mdp.robot_body_ori_b, params={"command_name": "motion"}
        ),
        "base_lin_vel": ObservationTermCfg(
            func=mdp.builtin_sensor, params={"sensor_name": "robot/imu_lin_vel"}
        ),
        "base_ang_vel": ObservationTermCfg(
            func=mdp.builtin_sensor, params={"sensor_name": "robot/imu_ang_vel"}
        ),
        "joint_pos": ObservationTermCfg(func=mdp.joint_pos_rel),
        "joint_vel": ObservationTermCfg(func=mdp.joint_vel_rel),
        "actions": ObservationTermCfg(func=mdp.last_action),
    }

    observations = {
        "actor": ObservationGroupCfg(
            terms=actor_terms,
            concatenate_terms=True,
            enable_corruption=True,
        ),
        "critic": ObservationGroupCfg(
            terms=critic_terms,
            concatenate_terms=True,
            enable_corruption=False,
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
            func=mdp.motion_global_anchor_position_error_exp,
            weight=0.5,
            params={"command_name": "motion", "std": 0.3},
        ),
        "motion_global_root_ori": RewardTermCfg(
            func=mdp.motion_global_anchor_orientation_error_exp,
            weight=0.5,
            params={"command_name": "motion", "std": 0.4},
        ),
        "motion_body_pos": RewardTermCfg(
            func=mdp.motion_relative_body_position_error_exp,
            weight=1.0,
            params={"command_name": "motion", "std": 0.3},
        ),
        "motion_body_ori": RewardTermCfg(
            func=mdp.motion_relative_body_orientation_error_exp,
            weight=1.0,
            params={"command_name": "motion", "std": 0.4},
        ),
        "motion_body_lin_vel": RewardTermCfg(
            func=mdp.motion_global_body_linear_velocity_error_exp,
            weight=1.0,
            params={"command_name": "motion", "std": 1.0},
        ),
        "motion_body_ang_vel": RewardTermCfg(
            func=mdp.motion_global_body_angular_velocity_error_exp,
            weight=1.0,
            params={"command_name": "motion", "std": 3.14},
        ),
        "action_rate_l2": RewardTermCfg(func=mdp.action_rate_l2, weight=-1e-1),
        "joint_limit": RewardTermCfg(
            func=mdp.joint_pos_limits,
            weight=-10.0,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*",))},
        ),
        "self_collisions": RewardTermCfg(
            func=mdp.self_collision_cost,
            weight=-10.0,
            params={"sensor_name": "self_collision", "force_threshold": 10.0},
        ),
    }

    ##
    # Events
    ##

    events = {
        "push_robot": EventTermCfg(
            func=mdp.push_by_setting_velocity,
            mode="interval",
            interval_range_s=(1.0, 3.0),
            params={"velocity_range": VELOCITY_RANGE},
        ),
        "base_com": EventTermCfg(
            mode="startup",
            func=dr.body_com_offset,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot", body_names=("torso_link",)
                ),
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
                "asset_cfg": SceneEntityCfg("robot", geom_names=r"^(left|right)_foot[1-7]_collision$"),
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
        "anchor_pos": TerminationTermCfg(
            func=mdp.bad_anchor_pos_z_only,
            params={"command_name": "motion", "threshold": 0.25},
        ),
        "anchor_ori": TerminationTermCfg(
            func=mdp.bad_anchor_ori,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "command_name": "motion",
                "threshold": 0.8,
            },
        ),
        "ee_body_pos": TerminationTermCfg(
            func=mdp.bad_motion_body_pos_z_only,
            params={
                "command_name": "motion",
                "threshold": 0.25,
                "body_names": (
                    "left_ankle_roll_link",
                    "right_ankle_roll_link",
                    "left_wrist_yaw_link",
                    "right_wrist_yaw_link",
                )
            },
        ),
    }

    self_collision_cfg = ContactSensorCfg(
        name="self_collision",
        primary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
        secondary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
        fields=("found", "force"),
        reduce="none",
        num_slots=1,
        history_length=4,
    )

    return ManagerBasedRlEnvCfg(
        scene=SceneCfg(
            terrain=TerrainEntityCfg(terrain_type="plane"),
            entities={"robot": get_g1_robot_cfg()},
            sensors=(self_collision_cfg, ),
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
            mujoco=MujocoCfg(
                timestep=0.005,
                iterations=10,
                ls_iterations=20
            )
        ),
        decimation=4,
        episode_length_s=10.0,
    )