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

    # Change table camera anchor to camera mounted in robot's head
    env_cfg.scene.table_cam.offset=CameraCfg.OffsetCfg(
        pos=(0.0, 0.72, 0.6),
        rot=(0.09143, -0.47766, -0.83945, 0.24249),
        convention="opengl"
    )

    # Set wrist camera anchor on robot
    env_cfg.scene.left_wrist_cam.prim_path = "{ENV_REGEX_NS}/Robot/panda_hand/left_wrist_cam"

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
        env_cfg.isaac_teleop = IsaacTeleopCfg(
            pipeline_builder=lambda: build_teleop_pipeline(dual_arm=False)[0],
            # retargeters_to_tune=lambda: build_teleop_pipeline(dual_arm=False)[1],
            sim_device=env_cfg.sim.device,
            xr_cfg=env_cfg.xr,
        )


def process_observation_for_openpi_franka_single_arm(obs: dict, prompt: str):
    # Pi0 models are trained for gripper positions in [0.0, 1.0], with 0.0 corresponding to fully open and 1.0 corresponding to fully closed.
    # Observations in the dataset are in [0.0, 0.04], with 0.0 corresponding to fully closed and 0.04 corresponding to fully open.
    # Therefore we adjust the gripper observation to fit the Pi0 models' format.
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
    # Pi0 models are trained for gripper positions in [0.0, 1.0], with 0.0 corresponding to fully open and 1.0 corresponding to fully closed.
    # The environment expects gripper positions to be in [1.0, -1.0], with 1.0 corresponding to fully open and -1.0 corresponding to fully closed.
    # Therefore we adjust the gripper action to fit the environment's format.
    # We also duplicate the gripper action for the environment.
    gripper_action = (action[7] * -2) + 1
    return np.concatenate((action[:7], (gripper_action, gripper_action)))
