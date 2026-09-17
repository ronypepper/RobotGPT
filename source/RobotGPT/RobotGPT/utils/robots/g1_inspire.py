# Based on code from the Isaac Lab project:
# https://github.com/isaac-sim/IsaacLab
#
# Original work:
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# Modifications:
# Copyright (c) 2026 ronypepper.
#
# SPDX-License-Identifier: BSD-3-Clause

import logging

import numpy as np
from RobotGPT.tasks.manager_based.robotgpt_env_cfg import RobotGPTEnvCfg
from RobotGPT.utils.mdp.env_step_differential_ik_action import EnvStepDifferentialInverseKinematicsActionCfg

from RobotGPT.utils.mdp.inspire_hand_action import InspireHandActionCfg
import isaaclab.envs.mdp as mdp
from isaaclab.assets.articulation.articulation_cfg import ArticulationCfg
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.devices.openxr.openxr_device import XrCfg

try:
    import isaacteleop  # noqa: F401  -- pipeline builders need isaacteleop at runtime
    from isaaclab_teleop import IsaacTeleopCfg
    from RobotGPT.utils.teleop.build_teleop_pipeline import build_teleop_pipeline

    _TELEOP_AVAILABLE = True
except ImportError:
    _TELEOP_AVAILABLE = False
    logging.getLogger(__name__).warning("isaaclab_teleop is not installed. XR teleoperation features will be disabled.")

from isaaclab.sensors.camera.camera_cfg import CameraCfg
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

from isaaclab_assets.robots.unitree import G1_INSPIRE_FTP_CFG


G1_ARM_JOINTS = [
    ".*_shoulder_pitch_joint",
    ".*_shoulder_roll_joint",
    ".*_shoulder_yaw_joint",
    ".*_elbow_joint",
    ".*_wrist_yaw_joint",
    ".*_wrist_roll_joint",
    ".*_wrist_pitch_joint",
]
G1_LEFT_ARM_JOINTS = [s.replace(".*", "left") for s in G1_ARM_JOINTS]
G1_RIGHT_ARM_JOINTS = [s.replace(".*", "right") for s in G1_ARM_JOINTS]


INSPIRE_LEFT_FINGER_OBSERVATION_JOINTS = [
    "L_index_proximal_joint",
    "L_thumb_proximal_pitch_joint",
]
INSPIRE_RIGHT_FINGER_OBSERVATION_JOINTS = [
    "R_index_proximal_joint",
    "R_thumb_proximal_pitch_joint",
]


def setup_g1_inspire_joint_pos_env(env_cfg: RobotGPTEnvCfg):
    # Set Unitree G1 with inspire hands as robot
    env_cfg.scene.robot = G1_INSPIRE_FTP_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0, 0, 1.0),
            rot=(0.0, 0.0, 0.7071, 0.7071),
            joint_pos={
                # right-arm
                "right_shoulder_pitch_joint": 0.0,
                "right_shoulder_roll_joint": 0.0,
                "right_shoulder_yaw_joint": 0.0,
                "right_elbow_joint": 0.0,
                "right_wrist_yaw_joint": 0.0,
                "right_wrist_roll_joint": 0.0,
                "right_wrist_pitch_joint": 0.0,
                # left-arm
                "left_shoulder_pitch_joint": 0.0,
                "left_shoulder_roll_joint": 0.0,
                "left_shoulder_yaw_joint": 0.0,
                "left_elbow_joint": 0.0,
                "left_wrist_yaw_joint": 0.0,
                "left_wrist_roll_joint": 0.0,
                "left_wrist_pitch_joint": 0.0,
                # --
                "waist_.*": 0.0,
                ".*_hip_.*": 0.0,
                ".*_knee_.*": 0.0,
                ".*_ankle_.*": 0.0,
                # -- left/right hand
                ".*_thumb_.*": 0.0,
                ".*_index_.*": 0.0,
                ".*_middle_.*": 0.0,
                ".*_ring_.*": 0.0,
                ".*_pinky_.*": 0.0,
            },
            joint_vel={".*": 0.0},
        ),
    )
    env_cfg.scene.robot.init_state.pos = (0.0, 0.0, 1.0)

    # Intitialize right wirst camera
    env_cfg.scene.initialize_right_wrist_camera()

    # Configure default pose with vertically aligned gripper orientation
    # env_cfg.scene.robot.init_state.joint_pos = {
    #     "panda_joint1": 0.0444,
    #     "panda_joint2": -0.1894,
    #     "panda_joint3": -0.1107,
    #     "panda_joint4": -2.5148,
    #     "panda_joint5": 0.0044,
    #     "panda_joint6": 2.3775,
    #     "panda_joint7": 0.6952,
    #     "panda_finger_joint.*": 0.04,
    # }
    # env_cfg.scene.robot_2.init_state.joint_pos = {
    #     "panda_joint1": 0.0444,
    #     "panda_joint2": -0.1894,
    #     "panda_joint3": -0.1107,
    #     "panda_joint4": -2.5148,
    #     "panda_joint5": 0.0044,
    #     "panda_joint6": 2.3775,
    #     "panda_joint7": 0.6952,
    #     "panda_finger_joint.*": 0.04,
    # }

    # Set joint position actions for the specific robot type
    env_cfg.actions.arm_action = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=G1_LEFT_ARM_JOINTS, preserve_order=True, use_default_offset=False
    )
    env_cfg.actions.arm_action_2 = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=G1_RIGHT_ARM_JOINTS, preserve_order=True, use_default_offset=False
    )

    # Set gripper actions for the specific robot type
    env_cfg.actions.gripper_action = InspireHandActionCfg(
        asset_name="robot",
        left_hand=True
    )
    env_cfg.actions.gripper_action_2 = InspireHandActionCfg(
        asset_name="robot",
        left_hand=False
    )

    # Change table camera anchor to camera mounted in robot's head
    env_cfg.scene.table_cam.prim_path = "{ENV_REGEX_NS}/Robot/torso_link/d435_link/table_cam"
    # env_cfg.scene.table_cam.offset=CameraCfg.OffsetCfg(
    #     pos=(0.0565, -0.168, 0.0),
    #     rot=(0.68301, 0.18301, -0.68301, 0.18301),
    #     convention="opengl"
    # )

    # Set wrist camera anchors on robot
    env_cfg.scene.left_wrist_cam.prim_path = "{ENV_REGEX_NS}/Robot/left_wrist_yaw_link/L_hand_base_link/left_wrist_cam"
    env_cfg.scene.right_wrist_cam.prim_path = "{ENV_REGEX_NS}/Robot/right_wrist_yaw_link/R_hand_base_link/right_wrist_cam"

    # Set dual arm observation group
    env_cfg.observations.setup_dual_arm_observations(left_joint_names=G1_LEFT_ARM_JOINTS + INSPIRE_LEFT_FINGER_OBSERVATION_JOINTS,
                                                     right_joint_names=G1_RIGHT_ARM_JOINTS + INSPIRE_RIGHT_FINGER_OBSERVATION_JOINTS,
                                                     use_robot_2_for_right_arm=False)

    # Setup ee-markers
    env_cfg.scene.initialize_ee_marker(dual_arm=True)
    env_cfg.scene.ee_marker.prim_path = "{ENV_REGEX_NS}/Robot/left_wrist_yaw_link/L_hand_base_link/ee_marker_left"
    env_cfg.scene.ee_marker_2.prim_path = "{ENV_REGEX_NS}/Robot/right_wrist_yaw_link/R_hand_base_link/ee_marker_right"


def setup_g1_inspire_ik_abs_env(env_cfg: RobotGPTEnvCfg):
    setup_g1_inspire_joint_pos_env(env_cfg)

    # Set inverse kinematics actions for the specific robot type
    env_cfg.actions.arm_action = EnvStepDifferentialInverseKinematicsActionCfg(
        asset_name="robot",
        joint_names=G1_LEFT_ARM_JOINTS,
        body_name="L_hand_base_link",
        controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls"),
        # body_offset=EnvStepDifferentialInverseKinematicsActionCfg.OffsetCfg(pos=(0.0, 0.0, 0.107)),
        # world_offset=EnvStepDifferentialInverseKinematicsActionCfg.OffsetCfg(pos=(-0.3, -0.02, 0.0)),
        auto_world_offset=True
    )
    env_cfg.actions.arm_action_2 = EnvStepDifferentialInverseKinematicsActionCfg(
        asset_name="robot_2",
        joint_names=G1_RIGHT_ARM_JOINTS,
        body_name="R_hand_base_link",
        controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls"),
        # body_offset=EnvStepDifferentialInverseKinematicsActionCfg.OffsetCfg(pos=(0.0, 0.0, 0.107)),
        # world_offset=EnvStepDifferentialInverseKinematicsActionCfg.OffsetCfg(pos=(-0.3, -0.02, 0.0)),
        auto_world_offset=True
    )

    # Teleoperation configuration
    env_cfg.xr = XrCfg(
        anchor_pos=(1.3, 0, -1.0),
        anchor_rot=(0, 0, 0.70711, 0.70711),
    )
    if _TELEOP_AVAILABLE:
        env_cfg.isaac_teleop = IsaacTeleopCfg(
            pipeline_builder=lambda: build_teleop_pipeline(dual_arm=True)[0],
            # retargeters_to_tune=lambda: build_teleop_pipeline(dual_arm=True)[1],
            sim_device=env_cfg.sim.device,
            xr_cfg=env_cfg.xr,
        )


def process_observation_for_openpi_g1_inspire(obs: dict, prompt: str):
    # Pi0 models are trained for gripper positions in [0.0, 1.0], with 0.0 corresponding to fully open and 1.0 corresponding to fully closed.
    # The environment provides observations for the proximal index and the proximal thumb pitch finger joints, which are
    # actuated in the ranges [0.0, 0.7] and [0.0, 0.26], respectively (see inspire_hand_action.py).
    # A single gripper position in the Pi0 models' format is computed from these joints.
    # Proprioceptive state normalization is handled on the server side.
    left_joint_pos = obs["left_joint_pos"][:7]
    left_index_pitch = np.clip(obs["left_joint_pos"][7], 0.0, 0.7) / 0.7
    left_thumb_pitch = np.clip(obs["left_joint_pos"][8], 0.0, 0.26) / 0.26
    left_gripper_pos = (left_index_pitch + left_thumb_pitch) / 2

    right_joint_pos = obs["right_joint_pos"][:7]
    right_index_pitch = np.clip(obs["right_joint_pos"][7], 0.0, 0.7) / 0.7
    right_thumb_pitch = np.clip(obs["right_joint_pos"][8], 0.0, 0.26) / 0.26
    right_gripper_pos = (right_index_pitch + right_thumb_pitch) / 2

    joint_pos = np.concatenate((left_joint_pos, left_gripper_pos, right_joint_pos, right_gripper_pos))

    policy_server_obs = {
        "observation/table_img": obs["table_img"],
        "observation/left_wrist_img": obs["left_wrist_img"],
        "observation/right_wrist_img": obs["right_wrist_img"],
        "observation/joint_pos": joint_pos,
        "prompt": prompt,
    }
    return policy_server_obs


def process_openpi_action_g1_inspire(action: np.array):
    # No adjustment necessary - action format of Pi0 models and envirionment are identical.
    return action
