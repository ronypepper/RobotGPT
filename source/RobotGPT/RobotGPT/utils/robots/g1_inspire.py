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
import tempfile

import numpy as np
from RobotGPT.tasks.manager_based.robotgpt_env_cfg import RobotGPTEnvCfg
from RobotGPT.utils.mdp.inspire_hand_action import InspireHandActionCfg
from RobotGPT.utils.mdp.pink_ik_action import PinkInverseKinematicsArmOnlyActionCfg
from RobotGPT.utils.teleop.gripper_continuous_retargeter import (
    ContinuousGripperRetargeter,
    ContinuousGripperRetargeterConfig,
)

import isaaclab.envs.mdp as mdp
from isaaclab.assets.articulation.articulation_cfg import ArticulationCfg
from isaaclab.assets.asset_base_cfg import AssetBaseCfg
from isaaclab.controllers.pink_ik.pink_ik_cfg import PinkIKControllerCfg
from isaaclab.controllers.pink_ik.pink_task_cfg import FrameTaskCfg, NullSpacePostureTaskCfg
from isaaclab.devices.openxr.openxr_device import XrCfg

try:
    import isaacteleop  # noqa: F401  -- pipeline builders need isaacteleop at runtime
    from isaaclab_teleop import IsaacTeleopCfg

    _TELEOP_AVAILABLE = True
except ImportError:
    _TELEOP_AVAILABLE = False
    logging.getLogger(__name__).warning("isaaclab_teleop is not installed. XR teleoperation features will be disabled.")

from isaaclab.sensors.camera.camera_cfg import CameraCfg

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


INSPIRE_JOINTS = [
    ".*_index_proximal_joint",  # These two must be the first for the gripper observation computation!
    ".*_thumb_proximal_pitch_joint",  # These two must be the first for the gripper observation computation!
    ".*_middle_proximal_joint",
    ".*_pinky_proximal_joint",
    ".*_ring_proximal_joint",
    ".*_index_intermediate_joint",
    ".*_middle_intermediate_joint",
    ".*_pinky_intermediate_joint",
    ".*_ring_intermediate_joint",
    ".*_thumb_intermediate_joint",
    ".*_thumb_distal_joint",
    ".*_thumb_proximal_yaw_joint",
]
INSPIRE_LEFT_JOINTS = [s.replace(".*", "L") for s in INSPIRE_JOINTS]
INSPIRE_RIGHT_JOINTS = [s.replace(".*", "R") for s in INSPIRE_JOINTS]


def setup_g1_inspire_joint_pos_env(env_cfg: RobotGPTEnvCfg):
    # Adjust scene
    env_cfg.scene.table.init_state = AssetBaseCfg.InitialStateCfg(pos=(0.78, 0, 0.0), rot=(0, 0, -0.707, 0.707))
    env_cfg.scene.background.init_state.pos = (0, 0, -0.65)

    # Set Unitree G1 with inspire hands as robot
    env_cfg.scene.robot = G1_INSPIRE_FTP_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.13, 0, 0.14),
            rot=(0.0, 0.0, 0.0, 1.0),
            joint_pos={
                # right-arm
                "right_shoulder_pitch_joint": 0.0,
                "right_shoulder_roll_joint": 0.0,#-30.0 / 180.0 * 3.14,
                "right_shoulder_yaw_joint": 0.0,#15.0 / 180.0 * 3.14,
                "right_elbow_joint": 0.0,#-10.0 / 180.0 * 3.14,
                "right_wrist_yaw_joint": 0.0,
                "right_wrist_roll_joint": 0.0,
                "right_wrist_pitch_joint": 0.0,
                # left-arm
                "left_shoulder_pitch_joint": 0.0,
                "left_shoulder_roll_joint": 0.0,#30.0 / 180.0 * 3.14,
                "left_shoulder_yaw_joint": 0.0,#-15.0 / 180.0 * 3.14,
                "left_elbow_joint": 0.0,#-10.0 / 180.0 * 3.14,
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

    # Intitialize right wirst camera
    env_cfg.scene.initialize_right_wrist_camera()

    # Adjust camera anchors and poses
    env_cfg.scene.table_cam.prim_path = "{ENV_REGEX_NS}/Robot/torso_link/d435_link/table_cam"
    env_cfg.scene.table_cam.offset = CameraCfg.OffsetCfg(
        pos=(0.0, 0.0, 0.0),
        rot=(0.5, -0.5, -0.5, 0.5), convention="opengl"
    )

    env_cfg.scene.left_wrist_cam.prim_path = "{ENV_REGEX_NS}/Robot/left_wrist_yaw_link/L_hand_base_link/left_wrist_cam"
    env_cfg.scene.left_wrist_cam.offset = CameraCfg.OffsetCfg(
        pos=(-0.01, -0.02, 0.07),
        rot=(0.13302, 0.63302, 0.75441, 0.11162), convention="opengl"
    )

    env_cfg.scene.right_wrist_cam.prim_path = "{ENV_REGEX_NS}/Robot/right_wrist_yaw_link/R_hand_base_link/right_wrist_cam"
    env_cfg.scene.right_wrist_cam.offset = CameraCfg.OffsetCfg(
        pos=(-0.01, -0.02, -0.07),
        rot=(-0.75441, 0.11162, -0.13302, 0.63302), convention="opengl"
    )

    # Set dual arm observation group
    env_cfg.observations.setup_dual_arm_observations(left_joint_names=G1_LEFT_ARM_JOINTS + INSPIRE_LEFT_JOINTS,
                                                     right_joint_names=G1_RIGHT_ARM_JOINTS + INSPIRE_RIGHT_JOINTS,
                                                     use_robot_2_for_right_arm=False)

    # Setup ee-markers
    # env_cfg.scene.initialize_ee_marker(dual_arm=True)
    # env_cfg.scene.ee_marker.prim_path = "{ENV_REGEX_NS}/Robot/left_wrist_yaw_link/L_hand_base_link/ee_marker_left"
    # env_cfg.scene.ee_marker_2.prim_path = "{ENV_REGEX_NS}/Robot/right_wrist_yaw_link/R_hand_base_link/ee_marker_right"


def setup_g1_inspire_ik_abs_env(env_cfg: RobotGPTEnvCfg):
    setup_g1_inspire_joint_pos_env(env_cfg)

    # Set inverse kinematics actions for the specific robot type
    env_cfg.actions.arm_action = PinkInverseKinematicsArmOnlyActionCfg(
        pink_controlled_joint_names=G1_ARM_JOINTS,
        hand_joint_names=[],
        target_eef_link_names={
            "left_wrist": "left_wrist_yaw_link",
            "right_wrist": "right_wrist_yaw_link",
        },
        # the robot in the sim scene we are controlling
        asset_name="robot",
        controller=PinkIKControllerCfg(
            articulation_name="robot",
            base_link_name="pelvis",
            num_hand_joints=0,
            show_ik_warnings=True,
            fail_on_joint_limit_violation=False,
            variable_input_tasks=[
                FrameTaskCfg(
                    frame="g1_29dof_rev_1_0_left_wrist_yaw_link",
                    position_cost=8.0,  # [cost] / [m]
                    orientation_cost=2.0,  # [cost] / [rad]
                    lm_damping=10,  # dampening for solver for step jumps
                    gain=0.5,
                ),
                FrameTaskCfg(
                    frame="g1_29dof_rev_1_0_right_wrist_yaw_link",
                    position_cost=8.0,  # [cost] / [m]
                    orientation_cost=2.0,  # [cost] / [rad]
                    lm_damping=10,  # dampening for solver for step jumps
                    gain=0.5,
                ),
                NullSpacePostureTaskCfg(
                    cost=0.5,
                    lm_damping=1,
                    controlled_frames=[
                        "g1_29dof_rev_1_0_left_wrist_yaw_link",
                        "g1_29dof_rev_1_0_right_wrist_yaw_link",
                    ],
                    controlled_joints=[
                        "left_shoulder_pitch_joint",
                        "left_shoulder_roll_joint",
                        "left_shoulder_yaw_joint",
                        "right_shoulder_pitch_joint",
                        "right_shoulder_roll_joint",
                        "right_shoulder_yaw_joint",
                        "waist_yaw_joint",
                        "waist_pitch_joint",
                        "waist_roll_joint",
                    ],
                    gain=0.3,
                ),
            ],
            fixed_input_tasks=[],
            usd_path=env_cfg.scene.robot.spawn.usd_path,
            urdf_output_dir=tempfile.gettempdir()
        ),
        enable_gravity_compensation=False,
    )
    env_cfg.actions.arm_action_2 = None

    # Teleoperation configuration
    env_cfg.xr = XrCfg(
        anchor_pos=(0.1, 0, -0.9),
        anchor_rot=(0, 0, -0.70711, 0.70711),
    )
    if _TELEOP_AVAILABLE:
        pipeline, retargeters = build_g1_inspire_teleop_pipeline()
        env_cfg.isaac_teleop = IsaacTeleopCfg(
            pipeline_builder=lambda: pipeline,
            # retargeters_to_tune=lambda: retargeters,
            sim_device=env_cfg.sim.device,
            xr_cfg=env_cfg.xr,
        )


def build_g1_inspire_teleop_pipeline():
    """Based on IsaacLab/source/isaaclab_tasks/isaaclab_tasks/contrib/stack/config/franka/stack_ik_abs_env_cfg.py

    Build a IsaacTeleop retargeting pipeline for a Unitree G1 robot with Inspire dexteroius hands and motion controllers.

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
        target_offset_x=0.03,
        target_offset_y=0.105,
        target_offset_z=0.082,
        target_offset_roll=-135.0,
        target_offset_pitch=0.0,
        target_offset_yaw=100.0,
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

    # SE3 Absolute Pose Retargeter (left hand)
    se3_left_cfg = Se3RetargeterConfig(
        input_device=ControllersSource.LEFT,
        zero_out_xy_rotation=False,
        use_wrist_rotation=False,
        use_wrist_position=False,
        target_offset_x=-0.045,
        target_offset_y=0.105,
        target_offset_z=0.075,
        target_offset_roll=-135.0,
        target_offset_pitch=0.0,
        target_offset_yaw=75.0,
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

    ee_pose_elements_left = [elem + "_left" for elem in ee_pose_elements]
    gripper_elements_left = ["gripper_value_left"]

    input_config["ee_pose_left"] = ee_pose_elements_left
    input_config["gripper_command_left"] = gripper_elements_left
    input_types["ee_pose_left"] = "array"
    input_types["gripper_command_left"] = "scalar"
    input_connections["ee_pose_left"] = connected_se3_left.output("ee_pose")
    input_connections["gripper_command_left"] = connected_gripper_left.output("gripper_command")

    output_order = ee_pose_elements_left + ee_pose_elements_right + gripper_elements_left + gripper_elements_right

    reorderer = TensorReorderer(
        input_config=input_config,
        output_order=output_order,
        name="action_reorderer",
        input_types=input_types,
    )
    connected_reorderer = reorderer.connect(input_connections)

    pipeline = OutputCombiner({"action": connected_reorderer.output("output")})

    return pipeline, [se3_right, se3_left]


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
    # Pi0 models are trained for gripper positions in [0.0, 1.0], with 0.0 corresponding to fully open and 1.0 corresponding to fully closed.
    # The environment expects the action inputs for the gripper to be in [1.0, -1.0], with 1.0 corresponding to fully open and -1.0 corresponding to fully closed.
    # Therefore we adjust the gripper action to fit the environment's format.
    left_gripper_action = (action[7] * -2) + 1
    right_gripper_action = (action[15] * -2) + 1
    return np.concatenate((action[:7], (left_gripper_action, ), action[8:15], (right_gripper_action, )))
