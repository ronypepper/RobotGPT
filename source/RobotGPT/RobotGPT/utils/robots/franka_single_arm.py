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

from RobotGPT.utils.teleop.gripper_continuous_retargeter import ContinuousGripperRetargeter, ContinuousGripperRetargeterConfig
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

from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

from isaaclab_assets.robots.franka import FRANKA_PANDA_HIGH_PD_CFG


def setup_franka_single_arm_joint_pos_env(env_cfg: RobotGPTEnvCfg):
    # Set Franka as robot
    env_cfg.scene.robot = FRANKA_PANDA_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    env_cfg.scene.robot.spawn.usd_path = f"{ISAACLAB_NUCLEUS_DIR}/Robots/FrankaEmika/Legacy/panda_instanceable.usd"

    # Configure default pose with vertically aligned gripper orientation
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
        scale=0.02,
        offset=0.02,
        use_default_offset=False
    )

    # Adjust camera anchors and poses
    env_cfg.scene.table_cam.prim_path = "{ENV_REGEX_NS}/table_cam"
    env_cfg.scene.table_cam.offset = CameraCfg.OffsetCfg(
        pos=(0.0, 0.72, 0.6),
        rot=(0.09143, -0.47766, -0.83945, 0.24249),
        convention="opengl"
    )

    env_cfg.scene.left_wrist_cam.prim_path = "{ENV_REGEX_NS}/Robot/panda_hand/left_wrist_cam"
    env_cfg.scene.left_wrist_cam.offset = CameraCfg.OffsetCfg(
        pos=(0.1009906081856474, -2.2170453280873081e-7, 0.005195286872436311),
        rot=(0.68618, 0.68618, 0.17074, 0.17074), convention="opengl"
    )

    # Set single arm observation group
    env_cfg.observations.setup_single_arm_observations()

    # Setup ee-markers
    # env_cfg.scene.initialize_ee_marker(dual_arm=False)
    # env_cfg.scene.ee_marker.prim_path = "{ENV_REGEX_NS}/Robot/panda_hand/ee_marker"


def setup_franka_single_arm_ik_abs_env(env_cfg: RobotGPTEnvCfg):
    setup_franka_single_arm_joint_pos_env(env_cfg)

    # Set inverse kinematics actions for the specific robot type
    env_cfg.actions.arm_action = EnvStepDifferentialInverseKinematicsActionCfg(
        asset_name="robot",
        joint_names=["panda_joint.*"],
        body_name="panda_hand",
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
        pipeline, retargeters = build_franka_single_arm_teleop_pipeline()
        env_cfg.isaac_teleop = IsaacTeleopCfg(
            pipeline_builder=lambda: pipeline,
            # retargeters_to_tune=lambda: retargeters,
            sim_device=env_cfg.sim.device,
            xr_cfg=env_cfg.xr,
        )


def build_franka_single_arm_teleop_pipeline():
    """Based on IsaacLab/source/isaaclab_tasks/isaaclab_tasks/contrib/stack/config/franka/stack_ik_abs_env_cfg.py

    Build a IsaacTeleop retargeting pipeline for a single Franka robot and motion controllers.

    Creates Se3AbsRetargeter for hand pose tracking and ContinuousGripperRetargeter for hand gripper control,
    flattened into a single action tensor via a TensorReorderer.

    Returns:
        OutputCombiner with a single "action" output containing the flattened action tensor.
    """
    from isaacteleop.retargeters import (
        Se3AbsRetargeter,
        Se3RetargeterConfig,
        TensorReorderer,
    )
    from isaacteleop.retargeting_engine.deviceio_source_nodes import ControllersSource
    from isaacteleop.retargeting_engine.interface import OutputCombiner, ValueInput
    from isaacteleop.retargeting_engine.tensor_types import TransformMatrix

    # Create input sources (trackers are auto-discovered from pipeline)
    controllers = ControllersSource(name="controllers")

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
        }
    )

    # TensorReorderer to flatten into a single action vector
    # Se3AbsRetargeter outputs a 7D NDArray (pos xyz + quat xyzw)
    # GripperRetargeter outputs a single float (gripper command)
    ee_pose_elements = ["pos_x", "pos_y", "pos_z", "quat_x", "quat_y", "quat_z", "quat_w"]

    ee_pose_elements_right = [elem + "_right" for elem in ee_pose_elements]
    gripper_elements_right = ["gripper_value_right"]

    input_config = {"ee_pose_right": ee_pose_elements_right, "gripper_command_right": gripper_elements_right,}
    input_types = {"ee_pose_right": "array", "gripper_command_right": "scalar",}
    input_connections = {
            "ee_pose_right": connected_se3_right.output("ee_pose"),
            "gripper_command_right": connected_gripper_right.output("gripper_command"),
        }

    output_order = ee_pose_elements_right + gripper_elements_right + gripper_elements_right

    reorderer = TensorReorderer(
        input_config=input_config,
        output_order=output_order,
        name="action_reorderer",
        input_types=input_types,
    )
    connected_reorderer = reorderer.connect(input_connections)

    pipeline = OutputCombiner({"action": connected_reorderer.output("output")})

    return pipeline, [se3_right]


def process_observation_for_openpi_franka_single_arm(obs: dict, prompt: str):
    # Pi0 models are trained for gripper positions in [0.0, 1.0], with 0.0 corresponding to fully open and 1.0 corresponding to fully closed.
    # Observations from the environment are in [0.0, 0.04], with 0.0 corresponding to fully closed and 0.04 corresponding to fully open.
    # Therefore we adjust the gripper observation to fit the Pi0 models' format.
    # Proprioceptive state normalization is handled on the server side.
    joint_pos = obs["joint_pos"][:8] # 7 joints + 1 gripper
    joint_pos[7] = (joint_pos[7] - 0.04) / -0.04

    policy_server_obs = {
        "observation/table_img": obs["table_img"],
        "observation/wrist_img": obs["wrist_img"],
        "observation/joint_pos": joint_pos,
        "prompt": prompt,
    }
    return policy_server_obs


def process_openpi_action_franka_single_arm(action: np.array):
    # Pi0 models are trained for gripper positions in [0.0, 1.0], with 0.0 corresponding to fully open and 1.0 corresponding to fully closed.
    # The environment expects the action inputs for the gripper to be in [1.0, -1.0], with 1.0 corresponding to fully open and -1.0 corresponding to fully closed.
    # Therefore we adjust the gripper action to fit the environment's format.
    # We also duplicate the gripper action for the environment.
    gripper_action = (action[7] * -2) + 1
    return np.concatenate((action[:7], (gripper_action, gripper_action)))
