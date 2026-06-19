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

try:
    import isaacteleop  # noqa: F401  -- pipeline builders need isaacteleop at runtime
    from isaaclab_teleop import IsaacTeleopCfg
    from RobotGPT.utils.teleop.gripper_continuous_retargeter import (
        ContinuousGripperRetargeter,
        ContinuousGripperRetargeterConfig,
    )

    _TELEOP_AVAILABLE = True
except ImportError:
    _TELEOP_AVAILABLE = False
    logging.getLogger(__name__).warning("isaaclab_teleop is not installed. XR teleoperation features will be disabled.")

from isaaclab_assets.robots.franka import FRANKA_PANDA_HIGH_PD_CFG


def setup_franka_dual_arm_joint_pos_env(env_cfg: RobotGPTEnvCfg):
    # Set Franka as left arm robot (seen from behind the robot looking at workspace)
    env_cfg.scene.robot = FRANKA_PANDA_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    env_cfg.scene.robot.init_state.pos = (0.0, 0.2825, 0.0)

    # Set Franka as right arm robot (seen from behind the robot looking at workspace)
    env_cfg.scene.robot_2 = FRANKA_PANDA_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot_2")
    env_cfg.scene.robot_2.init_state.pos = (0.0, -0.2825, 0.0)

    # Intitialize right wirst camera
    env_cfg.scene.initialize_right_wrist_camera()

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
    env_cfg.scene.robot_2.init_state.joint_pos = {
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
    env_cfg.actions.arm_action_2 = mdp.JointPositionActionCfg(
        asset_name="robot_2", joint_names=["panda_joint.*"], preserve_order=True, use_default_offset=False
    )

    # Set gripper actions for the specific robot type
    env_cfg.actions.gripper_action = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["panda_finger.*"],
        scale=-0.04,
        offset=0.04,
        use_default_offset=False
    )
    env_cfg.actions.gripper_action_2 = mdp.JointPositionActionCfg(
        asset_name="robot_2",
        joint_names=["panda_finger.*"],
        scale=-0.04,
        offset=0.04,
        use_default_offset=False
    )

    # Setup ee-markers
    env_cfg.scene.initialize_ee_marker(dual_arm=True)
    env_cfg.scene.ee_marker.prim_path="{ENV_REGEX_NS}/Robot/panda_hand/ee_marker"
    env_cfg.scene.ee_marker_2.prim_path="{ENV_REGEX_NS}/Robot_2/panda_hand/ee_marker_2"

    # Set dual arm observation group
    env_cfg.observations.policy = RobotGPTObservationsCfg.DualArmPolicyCfg()

    # Set wrist camera anchors on robot
    env_cfg.scene.left_wrist_cam.prim_path = "{ENV_REGEX_NS}/Robot/panda_hand/left_wrist_cam"
    env_cfg.scene.right_wrist_cam.prim_path = "{ENV_REGEX_NS}/Robot_2/panda_hand/right_wrist_cam"


def _build_franka_dual_arm_teleop_pipeline():
    """Modified from IsaacLab/source/isaaclab_tasks/isaaclab_tasks/contrib/stack/config/franka/stack_ik_abs_env_cfg.py

    Build a IsaacTeleop retargeting pipeline for dual Franka arms (motion controllers).

    Creates an Se3AbsRetargeter for right-hand pose tracking and a ContinuousGripperRetargeter
    for right-hand gripper control, flattened into a single action tensor via
    TensorReorderer.

    Note: gripper values is returned twice for feeding a JointPositionActionCfg for the gripper.

    Returns:
        OutputCombiner with a single "action" output containing the flattened
        9D action tensor: [pos_x, pos_y, pos_z, quat_w, quat_x, quat_y, quat_z, gripper, gripper].
    """
    from isaacteleop.retargeters import (
        Se3AbsRetargeter,
        Se3RetargeterConfig,
        TensorReorderer,
    )
    from isaacteleop.retargeting_engine.deviceio_source_nodes import ControllersSource, HandsSource
    from isaacteleop.retargeting_engine.interface import OutputCombiner, ValueInput
    from isaacteleop.retargeting_engine.tensor_types import TransformMatrix

    # Create input sources (trackers are auto-discovered from pipeline)
    controllers = ControllersSource(name="controllers")
    hands = HandsSource(name="hands")

    # External input: world-to-anchor 4x4 transform matrix provided by IsaacTeleopDevice
    transform_input = ValueInput("world_T_anchor", TransformMatrix())

    # Apply the coordinate-frame transform to controller poses so that
    # downstream retargeters receive data in the simulation world frame.
    transformed_controllers = controllers.transformed(transform_input.output(ValueInput.VALUE))

    # SE3 Absolute Pose Retargeter (right hand)
    se3_right_cfg = Se3RetargeterConfig(
        input_device=ControllersSource.RIGHT,
        zero_out_xy_rotation=False,
        use_wrist_rotation=False,
        use_wrist_position=False,
        target_offset_x=0.0,
        target_offset_y=0.0,
        target_offset_z=0.0,
        target_offset_roll=45.0,
        target_offset_pitch=0.0,
        target_offset_yaw=80.0,
    )
    se3_right = Se3AbsRetargeter(se3_right_cfg, name="ee_pose_right")
    connected_se3_right = se3_right.connect(
        {
            ControllersSource.RIGHT: transformed_controllers.output(ControllersSource.RIGHT),
        }
    )

    # Gripper Retargeter (right hand)
    gripper_right_cfg = ContinuousGripperRetargeterConfig(hand_side="right")
    gripper_right = ContinuousGripperRetargeter(gripper_right_cfg, name="gripper_right")
    connected_gripper_right = gripper_right.connect(
        {
            ControllersSource.RIGHT: transformed_controllers.output(ControllersSource.RIGHT),
            HandsSource.RIGHT: hands.output(HandsSource.RIGHT),
        }
    )

    # SE3 Absolute Pose Retargeter (left hand)
    se3_left_cfg = Se3RetargeterConfig(
        input_device=ControllersSource.LEFT,
        zero_out_xy_rotation=False,
        use_wrist_rotation=False,
        use_wrist_position=False,
        target_offset_x=0.0,
        target_offset_y=0.0,
        target_offset_z=0.0,
        target_offset_roll=45.0,
        target_offset_pitch=0.0,
        target_offset_yaw=80.0,
    )
    se3_left = Se3AbsRetargeter(se3_left_cfg, name="ee_pose_left")
    connected_se3_left = se3_left.connect(
        {
            ControllersSource.LEFT: transformed_controllers.output(ControllersSource.LEFT),
        }
    )

    # Gripper Retargeter (left hand)
    gripper_left_cfg = ContinuousGripperRetargeterConfig(hand_side="left")
    gripper_left = ContinuousGripperRetargeter(gripper_left_cfg, name="gripper_left")
    connected_gripper_left = gripper_left.connect(
        {
            ControllersSource.LEFT: transformed_controllers.output(ControllersSource.LEFT),
            HandsSource.LEFT: hands.output(HandsSource.LEFT),
        }
    )

    # TensorReorderer to flatten into a single action vector
    # Se3AbsRetargeter outputs a 7D NDArray (pos xyz + quat xyzw)
    # GripperRetargeter outputs a single float (gripper command)
    ee_pose_elements = ["pos_x", "pos_y", "pos_z", "quat_x", "quat_y", "quat_z", "quat_w"]
    ee_pose_elements_right = [elem + "_right" for elem in ee_pose_elements]
    ee_pose_elements_left = [elem + "_left" for elem in ee_pose_elements]
    gripper_elements_right = ["gripper_value_right"]
    gripper_elements_left = ["gripper_value_left"]

    # We set the right hand retargeters first, as we want to control the left robot arm with it.
    # The first actions control the left robot arm, the latter actions control the right robot arm.
    # (The left & right robot arm as seen from behind the robot looking at the workspace)
    reorderer = TensorReorderer(
        input_config={
            "ee_pose_right": ee_pose_elements_right,
            "gripper_command_right": gripper_elements_right,
            "ee_pose_left": ee_pose_elements_left,
            "gripper_command_left": gripper_elements_left,
        },
        output_order=ee_pose_elements_right + gripper_elements_right + gripper_elements_right +
                     ee_pose_elements_left + gripper_elements_left + gripper_elements_left,
        name="action_reorderer",
        input_types={
            "ee_pose_right": "array", "gripper_command_right": "scalar",
            "ee_pose_left": "array", "gripper_command_left": "scalar"
        },
    )
    connected_reorderer = reorderer.connect(
        {
            "ee_pose_right": connected_se3_right.output("ee_pose_right"),
            "gripper_command_right": connected_gripper_right.output("gripper_command_right"),
            "ee_pose_left": connected_se3_left.output("ee_pose_left"),
            "gripper_command_left": connected_gripper_left.output("gripper_command_left"),
        }
    )

    pipeline = OutputCombiner({"action": connected_reorderer.output("output")})

    return pipeline, [se3_left, se3_right]


def setup_franka_dual_arm_ik_abs_env(env_cfg: RobotGPTEnvCfg):
    setup_franka_dual_arm_joint_pos_env(env_cfg)

    # Set inverse kinematics actions for the specific robot type
    env_cfg.actions.arm_action = EnvStepDifferentialInverseKinematicsActionCfg(
        asset_name="robot",
        joint_names=["panda_joint.*"],
        body_name="panda_hand",
        controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls"),
        body_offset=EnvStepDifferentialInverseKinematicsActionCfg.OffsetCfg(pos=(0.0, 0.0, 0.107)),
        world_offset=EnvStepDifferentialInverseKinematicsActionCfg.OffsetCfg(pos=(-0.3, -0.02, 0.0)),
    )
    env_cfg.actions.arm_action_2 = EnvStepDifferentialInverseKinematicsActionCfg(
        asset_name="robot_2",
        joint_names=["panda_joint.*"],
        body_name="panda_hand",
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
            pipeline_builder=lambda: _build_franka_dual_arm_teleop_pipeline()[0],
            # retargeters_to_tune=lambda: _build_franka_dual_arm_teleop_pipeline()[1],
            sim_device=env_cfg.sim.device,
            xr_cfg=env_cfg.xr,
        )


def process_observation_for_openpi_franka_dual_arm(obs: dict, prompt: str):
    # Pi0 models are trained for gripper positions in [0.0, 1.0], with 0.0 corresponding to fully open and 1.0 corresponding to fully closed.
    # Observations in the dataset are in [0.0, 0.04], with 0.0 corresponding to fully closed and 0.04 corresponding to fully open.
    # Therefore we adjust the gripper observation to fit the Pi0 models' format.
    # For received actions (later), we don't need to do the reverse transformation since the environment expects this format for the gripper action as well.
    # Proprioceptive state normalization is handled on the server side.
    left_joint_pos = obs["left_joint_pos"][:8] # 7 joints + 1 gripper
    left_joint_pos[7] = (left_joint_pos[7] - 0.04) / 0.04

    right_joint_pos = obs["right_joint_pos"][:8] # 7 joints + 1 gripper
    right_joint_pos[7] = (right_joint_pos[7] - 0.04) / 0.04

    joint_pos = np.concatenate((left_joint_pos, right_joint_pos))

    policy_server_obs = {
        "observation/table_img": obs["table_img"],
        "observation/left_wrist_img": obs["left_wrist_img"],
        "observation/right_wrist_img": obs["right_wrist_img"],
        "observation/joint_pos": joint_pos,
        "prompt": prompt,
    }
    return policy_server_obs


def process_openpi_action_franka_dual_arm(action: np.array):
    # We duplicate the gripper action command here
    return np.concatenate((action[:7], action[7], action[7], action[8:15], action[15], action[15]))
