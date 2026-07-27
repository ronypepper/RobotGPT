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
from RobotGPT.tasks.manager_based.robotgpt_env_cfg import RobotGPTEnvCfg, RobotGPTObservationsCfg
from RobotGPT.utils.mdp.env_step_differential_ik_action import EnvStepDifferentialInverseKinematicsActionCfg

import isaaclab.envs.mdp as mdp
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.devices.openxr.openxr_device import XrCfg
from isaaclab.sensors import CameraCfg

try:
    import isaacteleop  # noqa: F401  -- pipeline builders need isaacteleop at runtime
    from isaaclab_teleop import IsaacTeleopCfg
    from RobotGPT.utils.teleop.build_teleop_pipeline import build_teleop_pipeline

    _TELEOP_AVAILABLE = True
except ImportError:
    _TELEOP_AVAILABLE = False
    logging.getLogger(__name__).warning("isaaclab_teleop is not installed. XR teleoperation features will be disabled.")

import RobotGPT.utils.robots.base_configurations.tiago_pro_cfg as tiago_pro_cfg


def setup_tiago_pro_joint_pos_env(env_cfg: RobotGPTEnvCfg, dual_arm: bool):
    # Adjust scene
    env_cfg.scene.table.init_state.pos = (0.6, 0, 0)
    env_cfg.scene.table.init_state.rot = (0, 0, -0.707, 0.707)
    env_cfg.scene.background.init_state.pos = (0, 0, -0.6)

    # Set Tiago Pro as robot
    env_cfg.scene.robot = tiago_pro_cfg.TIAGO_PRO_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    env_cfg.scene.robot.init_state.pos = (-0.3, 0.0, -0.6)

    print(tiago_pro_cfg.S_PLUS)
    print(tiago_pro_cfg.S_MINUS)
    print(tiago_pro_cfg.XS)
    print(tiago_pro_cfg.TORSO)

    # Configure default pose with vertically aligned gripper orientation
    env_cfg.scene.robot.init_state.joint_pos.update({
        "arm_left_1_joint": 170.77 / 180 * 3.14159,
        "arm_left_2_joint": -71.26 / 180 * 3.14159,
        "arm_left_3_joint": -74.02 / 180 * 3.14159,
        "arm_left_4_joint": -92.76 / 180 * 3.14159,
        "arm_left_5_joint": -131.82 / 180 * 3.14159,
        "arm_left_6_joint": 84.14 / 180 * 3.14159,
        "arm_left_7_joint": -56.23 / 180 * 3.14159,
    })
    if dual_arm:
        env_cfg.scene.robot.init_state.joint_pos.update({
            "arm_left_1_joint": 143.65 / 180 * 3.14159,
            "arm_right_1_joint": -143.65 / 180 * 3.14159,
            "arm_left_2_joint": -73.93 / 180 * 3.14159,
            "arm_right_2_joint": -73.93 / 180 * 3.14159,
            "arm_left_3_joint": -60.48 / 180 * 3.14159,
            "arm_right_3_joint": 60.48 / 180 * 3.14159,
            "arm_left_4_joint": -101.42 / 180 * 3.14159,
            "arm_right_4_joint": -101.42 / 180 * 3.14159,
            "arm_left_5_joint": -124.3 / 180 * 3.14159,
            "arm_right_5_joint": 124.3 / 180 * 3.14159,
            "arm_left_6_joint": 90.92 / 180 * 3.14159,
            "arm_right_6_joint": 90.92 / 180 * 3.14159,
            "arm_left_7_joint": -40.89 / 180 * 3.14159,
            "arm_right_7_joint": 40.89 / 180 * 3.14159,
        })

    # Set joint position actions for the specific robot type
    env_cfg.actions.arm_action = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=["arm_left_.*_joint"], preserve_order=True, use_default_offset=True
    )
    if dual_arm:
        env_cfg.actions.arm_action_2 = mdp.JointPositionActionCfg(
            asset_name="robot", joint_names=["arm_right_.*_joint"], preserve_order=True, use_default_offset=True
        )

    # Set gripper actions for the specific robot type
    env_cfg.actions.gripper_action = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["gripper_left_outer_finger_.*_joint"],
        scale=-0.45,
        offset=0.45,
        use_default_offset=False
    )
    if dual_arm:
        env_cfg.actions.gripper_action_2 = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=["gripper_right_outer_finger_.*_joint"],
            scale=-0.475,
            offset=0.475,
            use_default_offset=False
        )

    # Change table camera anchor to camera mounted in robot's head
    env_cfg.scene.table_cam.prim_path = "{ENV_REGEX_NS}/Robot/Geometry/base_footprint/torso_lift_link/head_1_link/head_2_link/table_cam"
    env_cfg.scene.table_cam.offset=CameraCfg.OffsetCfg(
        pos=(0.0565, -0.168, 0.0),
        rot=(0.68301, 0.18301, -0.68301, 0.18301),
        convention="opengl"
    )

    # Intitialize wrist cameras
    env_cfg.scene.left_wrist_cam.prim_path = "{ENV_REGEX_NS}/Robot/Geometry/base_footprint/torso_lift_link/arm_left_1_link/arm_left_2_link/arm_left_3_link/arm_left_4_link/arm_left_5_link/arm_left_6_link/arm_left_7_link/left_wrist_cam"
    if dual_arm:
        env_cfg.scene.initialize_right_wrist_camera()
        env_cfg.scene.right_wrist_cam.prim_path = "{ENV_REGEX_NS}/Robot/Geometry/base_footprint/torso_lift_link/arm_right_1_link/arm_right_2_link/arm_right_3_link/arm_right_4_link/arm_right_5_link/arm_right_6_link/arm_right_7_link/right_wrist_cam"

    # Set dual arm observation group (if using two arms)
    env_cfg.observations.setup_single_arm_observations(["arm_left_.*_joint"])
    if dual_arm:
        env_cfg.observations.policy = RobotGPTObservationsCfg.DualArmPolicyCfg()

    # Setup ee-markers
    env_cfg.scene.initialize_ee_marker(dual_arm=dual_arm)
    env_cfg.scene.ee_marker.prim_path = "{ENV_REGEX_NS}/Robot/Geometry/base_footprint/torso_lift_link/arm_left_1_link/arm_left_2_link/arm_left_3_link/arm_left_4_link/arm_left_5_link/arm_left_6_link/arm_left_7_link/ee_left/ee_marker_left"
    if dual_arm:
        env_cfg.scene.ee_marker_2.prim_path = "{ENV_REGEX_NS}/Robot/Geometry/base_footprint/torso_lift_link/arm_right_1_link/arm_right_2_link/arm_right_3_link/arm_right_4_link/arm_right_5_link/arm_right_6_link/arm_right_7_link/ee_right/ee_marker_right"


def setup_tiago_pro_ik_abs_env(env_cfg: RobotGPTEnvCfg, dual_arm: bool):
    setup_tiago_pro_joint_pos_env(env_cfg, dual_arm)

    # Set inverse kinematics actions for the specific robot type
    env_cfg.actions.arm_action = EnvStepDifferentialInverseKinematicsActionCfg(
        asset_name="robot",
        joint_names=["arm_left_.*_joint"],
        body_name="arm_left_7_link",
        controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls"),
        body_offset=EnvStepDifferentialInverseKinematicsActionCfg.OffsetCfg(pos=(0.0, 0.0, 0.107)),
        world_offset=EnvStepDifferentialInverseKinematicsActionCfg.OffsetCfg(pos=(-0.3, -0.02, 0.0)),
    )
    if dual_arm:
        env_cfg.actions.arm_action_2 = EnvStepDifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=["arm_right_.*_joint"],
            body_name="arm_right_7_link",
            controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls"),
            body_offset=EnvStepDifferentialInverseKinematicsActionCfg.OffsetCfg(pos=(0.0, 0.0, 0.107)),
            world_offset=EnvStepDifferentialInverseKinematicsActionCfg.OffsetCfg(pos=(-0.3, -0.02, 0.0)),
        )

    # Teleoperation configuration
    env_cfg.xr = XrCfg(
        anchor_pos=(1.3, 0, -1.0),
        anchor_rot=(0, 0, 0.70711, 0.70711),
    )
    if _TELEOP_AVAILABLE:
        env_cfg.isaac_teleop = IsaacTeleopCfg(
            pipeline_builder=lambda: build_teleop_pipeline(dual_arm=dual_arm)[0],
            # retargeters_to_tune=lambda: build_teleop_pipeline(dual_arm=dual_arm)[1],
            sim_device=env_cfg.sim.device,
            xr_cfg=env_cfg.xr,
        )


def process_observation_for_openpi_tiago_pro(obs: dict, prompt: str):
    # Pi0 models are trained for gripper positions in [0.0, 1.0], with 0.0 corresponding to fully open and 1.0 corresponding to fully closed.
    # Observations in the dataset are in [0.0, 0.95/0.9], with 0.0 corresponding to fully closed and 0.95/0.9 corresponding to fully open (right/left).
    # Therefore we adjust the gripper observations to fit the Pi0 models' format.
    # Proprioceptive state normalization is handled on the server side.
    if "joint_pos" in obs:
        # Single arm variant
        joint_pos = obs["joint_pos"][:8] # 7 joints + 1 gripper
        joint_pos[7] = joint_pos[7] / 0.9

        policy_server_obs = {
            "observation/table_img": obs["table_img"],
            "observation/wrist_img": obs["wrist_img"],
            "observation/joint_pos": joint_pos,
            "prompt": prompt,
        }
        return policy_server_obs
    else:
        # Dual arm variant
        left_joint_pos = obs["left_joint_pos"][:8] # 7 joints + 1 gripper
        left_joint_pos[7] = left_joint_pos[7] / 0.9

        right_joint_pos = obs["right_joint_pos"][:8] # 7 joints + 1 gripper
        right_joint_pos[7] = right_joint_pos[7] / 0.95

        joint_pos = np.concatenate((left_joint_pos, right_joint_pos))

        policy_server_obs = {
            "observation/table_img": obs["table_img"],
            "observation/left_wrist_img": obs["left_wrist_img"],
            "observation/right_wrist_img": obs["right_wrist_img"],
            "observation/joint_pos": joint_pos,
            "prompt": prompt,
        }
        return policy_server_obs


def process_openpi_action_tiago_pro(action: np.array):
    # Pi0 models are trained for gripper positions in [0.0, 1.0], with 0.0 corresponding to fully open and 1.0 corresponding to fully closed.
    # The environment expects gripper positions to be in [1.0, -1.0], with 1.0 corresponding to fully open and -1.0 corresponding to fully closed.
    # Therefore we adjust the gripper actions to fit the environment's format.
    # We also duplicate the gripper actions for the environment.
    if len(action) == 8:
        # Single arm variant
        gripper_action = (action[7] * -2) + 1
        return np.concatenate((action[:7], (gripper_action, gripper_action)))
    else:
        # Dual arm variant
        left_gripper_action = (action[7] * -2) + 1
        right_gripper_action = (action[15] * -2) + 1
        return np.concatenate((action[:7], (left_gripper_action, left_gripper_action),
                               action[8:15], (right_gripper_action, right_gripper_action)))
