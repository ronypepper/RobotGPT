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

import numpy as np
from RobotGPT.tasks.manager_based.robotgpt_env_cfg import RobotGPTEnvCfg
from RobotGPT.utils.mdp.env_step_differential_ik_action import EnvStepDifferentialInverseKinematicsActionCfg
from RobotGPT.utils.teleop.controller_gripper_retargeter import ControllerGripperRetargeterCfg
from RobotGPT.utils.teleop.controller_se3_abs_retargeter import ControllerSe3AbsRetargeterCfg
from RobotGPT.utils.teleop.openxr_device_with_record_controls import DemoRecorderOpenXRDeviceCfg

import isaaclab.envs.mdp as mdp
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.devices.device_base import DeviceBase, DevicesCfg
from isaaclab.devices.openxr.openxr_device import OpenXRDeviceCfg, XrCfg
from isaaclab.devices.openxr.retargeters.manipulator.gripper_retargeter import GripperRetargeterCfg
from isaaclab.devices.openxr.retargeters.manipulator.se3_abs_retargeter import Se3AbsRetargeterCfg

from isaaclab_assets.robots.franka import FRANKA_PANDA_HIGH_PD_CFG


def setup_franka_single_arm_joint_pos_env(env_cfg: RobotGPTEnvCfg):
    # Set Franka as robot
    env_cfg.scene.robot = FRANKA_PANDA_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    # Configure default pose with vertically aligned gripper orientation for better teleoperation
    env_cfg.scene.robot.init_state.joint_pos = {
        "panda_joint1": 0.0444,
        "panda_joint2": -0.1894,
        "panda_joint3": -0.1107,
        "panda_joint4": -2.5148,
        "panda_joint5": 0.0044,
        "panda_joint6": 2.3775,
        "panda_joint7": 0.6952,
        "panda_finger_joint.*": 0.04,
    }

    # Set joint position actions for the specific robot type
    env_cfg.actions.arm_action = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=["panda_joint.*"], preserve_order=True, use_default_offset=False
    )

    # Set gripper actions for the specific robot type
    env_cfg.actions.gripper_action = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["panda_finger.*"],
        scale=-0.04,
        offset=0.04,
        use_default_offset=False
    )

    # Setup ee-markers
    # env_cfg.scene.initialize_ee_marker(dual_arm=False)
    # env_cfg.scene.ee_marker.prim_path="{ENV_REGEX_NS}/Robot/panda_hand/ee_marker"

    # Set wrist camera anchor on robot
    env_cfg.scene.left_wrist_cam.prim_path = "{ENV_REGEX_NS}/Robot/panda_hand/left_wrist_cam"


def setup_franka_single_arm_ik_abs_env(env_cfg: RobotGPTEnvCfg):
    setup_franka_single_arm_joint_pos_env(env_cfg)

    # Set inverse kinematics actions for the specific robot type
    env_cfg.actions.arm_action = EnvStepDifferentialInverseKinematicsActionCfg(
        asset_name="robot",
        joint_names=["panda_joint.*"],
        body_name="panda_hand",
        controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls"),
        body_offset=EnvStepDifferentialInverseKinematicsActionCfg.OffsetCfg(pos=(0.0, 0.0, 0.107)),
    )

    # Teleoperation configuration
    env_cfg.xr = XrCfg(
        anchor_pos=(1.3, 0, -1.0),
        anchor_rot=(0, 0, 0.70711, 0.70711),
    )
    env_cfg.teleop_devices = DevicesCfg(
        devices={
            "handtracking": OpenXRDeviceCfg(
                retargeters=[
                    Se3AbsRetargeterCfg(
                        bound_hand=DeviceBase.TrackingTarget.HAND_RIGHT,
                        zero_out_xy_rotation=True,
                        use_wrist_rotation=False,
                        use_wrist_position=True,
                        sim_device=env_cfg.sim.device,
                    ),
                    GripperRetargeterCfg(
                        bound_hand=DeviceBase.TrackingTarget.HAND_RIGHT, sim_device=env_cfg.sim.device
                    ),
                    GripperRetargeterCfg(
                        bound_hand=DeviceBase.TrackingTarget.HAND_RIGHT, sim_device=env_cfg.sim.device
                    ),
                ],
                sim_device=env_cfg.sim.device,
                xr_cfg=env_cfg.xr,
            ),
            "motioncontroller": DemoRecorderOpenXRDeviceCfg(
                retargeters=[
                    ControllerSe3AbsRetargeterCfg(
                        bound_controller=DeviceBase.TrackingTarget.CONTROLLER_RIGHT,
                        pos_offset=(-0.3, 0.0, 0.0),
                        rot_offset=(0.0, -90.0, 90.0),
                        zero_out_xy_rotation=True,
                        sim_device=env_cfg.sim.device,
                    ),
                    ControllerGripperRetargeterCfg(
                        bound_controller=DeviceBase.TrackingTarget.CONTROLLER_RIGHT,
                        num_joints=2,
                        sim_device=env_cfg.sim.device
                    ),
                ],
                sim_device=env_cfg.sim.device,
                xr_cfg=env_cfg.xr,
            ),
        }
    )


def process_observation_for_openpi_franka_single_arm(obs: dict, prompt: str):
    # Pi0 models are trained for gripper positions in [0.0, 1.0], with 0.0 corresponding to fully open and 1.0 corresponding to fully closed.
    # Observations in the dataset are in [0.0, 0.04], with 0.0 corresponding to fully closed and 0.04 corresponding to fully open.
    # Therefore we adjust the gripper observation to fit the Pi0 models' format.
    # For received actions (later), we don't need to do the reverse transformation since the environment expects this format for the gripper action as well.
    # Proprioceptive state normalization is handled on the server side.
    joint_pos = obs["joint_pos"][:8] # 7 joints + 1 gripper
    joint_pos[7] = (joint_pos[7] - 0.04) / 0.04

    policy_server_obs = {
        "observation/table_img": obs["table_img"],
        "observation/wrist_img": obs["wrist_img"],
        "observation/joint_pos": joint_pos,
        "prompt": prompt,
    }
    return policy_server_obs


def process_openpi_action_franka_single_arm(action: np.array):
    # We duplicate the gripper action command here
    return np.append(action, action[-1])
