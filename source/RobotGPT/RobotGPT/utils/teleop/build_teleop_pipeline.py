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

try:
    import isaacteleop  # noqa: F401  -- pipeline builders need isaacteleop at runtime
    from RobotGPT.utils.teleop.gripper_continuous_retargeter import (
        ContinuousGripperRetargeter,
        ContinuousGripperRetargeterConfig,
    )

    _TELEOP_AVAILABLE = True
except ImportError:
    _TELEOP_AVAILABLE = False
    logging.getLogger(__name__).warning("isaaclab_teleop is not installed. XR teleoperation features will be disabled.")


def build_teleop_pipeline(dual_arm: bool):
    """Based on IsaacLab/source/isaaclab_tasks/isaaclab_tasks/contrib/stack/config/franka/stack_ik_abs_env_cfg.py

    Build a IsaacTeleop retargeting pipeline for two or one robot arms (motion controllers).

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

    if dual_arm:
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
    gripper_elements_right = ["gripper_value_right"]

    # We set the right hand retargeters first, as we want to control the left robot arm with it.
    # The first actions control the left robot arm, the latter actions control the right robot arm.
    # (The left & right robot arm as seen from behind the robot looking at the workspace)
    input_config = {"ee_pose_right": ee_pose_elements_right, "gripper_command_right": gripper_elements_right,}
    output_order = ee_pose_elements_right + gripper_elements_right + gripper_elements_right
    input_types = {"ee_pose_right": "array", "gripper_command_right": "scalar",}

    input_connections = {
            "ee_pose_right": connected_se3_right.output("ee_pose"),
            "gripper_command_right": connected_gripper_right.output("gripper_command"),
        }

    if dual_arm:
        ee_pose_elements_left = [elem + "_left" for elem in ee_pose_elements]
        gripper_elements_left = ["gripper_value_left"]

        input_config["ee_pose_left"] = ee_pose_elements_left
        input_config["gripper_command_left"] = gripper_elements_left
        output_order += ee_pose_elements_left + gripper_elements_left + gripper_elements_left
        input_types["ee_pose_left"] = "array"
        input_types["gripper_command_left"] = "scalar"

        input_connections["ee_pose_left"] = connected_se3_left.output("ee_pose")
        input_connections["gripper_command_left"] = connected_gripper_left.output("gripper_command")

    reorderer = TensorReorderer(
        input_config=input_config,
        output_order=output_order,
        name="action_reorderer",
        input_types=input_types,
    )
    connected_reorderer = reorderer.connect(input_connections)

    pipeline = OutputCombiner({"action": connected_reorderer.output("output")})

    return pipeline, [se3_right, se3_left] if dual_arm else [se3_right]
