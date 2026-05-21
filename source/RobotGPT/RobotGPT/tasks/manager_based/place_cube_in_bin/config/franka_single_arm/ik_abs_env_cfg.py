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

import isaaclab.envs.mdp as mdp
from isaaclab.assets import DeformableObjectCfg
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.devices.device_base import DeviceBase, DevicesCfg
from isaaclab.devices.openxr.openxr_device import OpenXRDeviceCfg, XrCfg
from isaaclab.devices.openxr.retargeters.manipulator.gripper_retargeter import GripperRetargeterCfg
from isaaclab.devices.openxr.retargeters.manipulator.se3_abs_retargeter import Se3AbsRetargeterCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sim.spawners import UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

from RobotGPT.utils.mdp.env_step_differential_ik_action import EnvStepDifferentialInverseKinematicsActionCfg
from RobotGPT.utils.teleop.controller_gripper_retargeter import ControllerGripperRetargeterCfg
from RobotGPT.utils.teleop.controller_se3_abs_retargeter import ControllerSe3AbsRetargeterCfg
from RobotGPT.utils.teleop.openxr_device_with_record_controls import DemoRecorderOpenXRDeviceCfg

from . import joint_pos_env_cfg

##
# Pre-defined configs
##
from isaaclab_assets.robots.franka import FRANKA_PANDA_HIGH_PD_CFG  # isort: skip


@configclass
class FrankaSingleArmPlaceCubeInBinEnvCfg(joint_pos_env_cfg.FrankaSingleArmPlaceCubeInBinEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Set Franka as robot
        # We switch here to a stiffer PD controller for IK tracking to be better.
        self.scene.robot = FRANKA_PANDA_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        # Configure default pose with vertically aligned gripper orientation for better teleoperation
        self.scene.robot.init_state.joint_pos = {
            "panda_joint1": 0.0444,
            "panda_joint2": -0.1894,
            "panda_joint3": -0.1107,
            "panda_joint4": -2.5148,
            "panda_joint5": 0.0044,
            "panda_joint6": 2.3775,
            "panda_joint7": 0.6952,
            "panda_finger_joint.*": 0.04,
        }

        # Set actions for the specific robot type (franka)
        self.actions.arm_action = EnvStepDifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=["panda_joint.*"],
            body_name="panda_hand",
            controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls"),
            body_offset=EnvStepDifferentialInverseKinematicsActionCfg.OffsetCfg(pos=(0.0, 0.0, 0.107)),
        )
        self.actions.gripper_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=["panda_finger.*"],
            scale=-0.04,
            offset=0.04,
            use_default_offset=False
        )

        # Teleoperation
        self.xr: XrCfg = XrCfg(
            anchor_pos=(1.3, 0, -1.0),
            anchor_rot=(0.70711, 0, 0, 0.70711),
        )
        self.teleop_devices = DevicesCfg(
            devices={
                "handtracking": OpenXRDeviceCfg(
                    retargeters=[
                        Se3AbsRetargeterCfg(
                            bound_hand=DeviceBase.TrackingTarget.HAND_RIGHT,
                            zero_out_xy_rotation=True,
                            use_wrist_rotation=False,
                            use_wrist_position=True,
                            sim_device=self.sim.device,
                        ),
                        GripperRetargeterCfg(
                            bound_hand=DeviceBase.TrackingTarget.HAND_RIGHT, sim_device=self.sim.device
                        ),
                        GripperRetargeterCfg(
                            bound_hand=DeviceBase.TrackingTarget.HAND_RIGHT, sim_device=self.sim.device
                        ),
                    ],
                    sim_device=self.sim.device,
                    xr_cfg=self.xr,
                ),
                "motioncontroller": DemoRecorderOpenXRDeviceCfg(
                    retargeters=[
                        ControllerSe3AbsRetargeterCfg(
                            bound_controller=DeviceBase.TrackingTarget.CONTROLLER_RIGHT,
                            pos_offset=(-0.3, 0.0, 0.0),
                            rot_offset=(0.0, -90.0, 90.0),
                            zero_out_xy_rotation=True,
                            sim_device=self.sim.device,
                        ),
                        ControllerGripperRetargeterCfg(
                            bound_controller=DeviceBase.TrackingTarget.CONTROLLER_RIGHT,
                            num_joints=2,
                            sim_device=self.sim.device
                        ),
                    ],
                    sim_device=self.sim.device,
                    xr_cfg=self.xr,
                ),
            }
        )


# ##
# # Deformable object lift environment.
# ##


# @configclass
# class FrankaTeddyBearLiftEnvCfg(FrankaSingleArmPlaceCubeInBinEnvCfg):
#     def __post_init__(self):
#         # post init of parent
#         super().__post_init__()

#         self.scene.object = DeformableObjectCfg(
#             prim_path="{ENV_REGEX_NS}/Object",
#             init_state=DeformableObjectCfg.InitialStateCfg(pos=(0.5, 0, 0.05), rot=(0.707, 0, 0, 0.707)),
#             spawn=UsdFileCfg(
#                 usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Objects/Teddy_Bear/teddy_bear.usd",
#                 scale=(0.01, 0.01, 0.01),
#             ),
#         )

#         # Make the end effector less stiff to not hurt the poor teddy bear
#         self.scene.robot.actuators["panda_hand"].effort_limit_sim = 50.0
#         self.scene.robot.actuators["panda_hand"].stiffness = 40.0
#         self.scene.robot.actuators["panda_hand"].damping = 10.0

#         # Disable replicate physics as it doesn't work for deformable objects
#         # FIXME: This should be fixed by the PhysX replication system.
#         self.scene.replicate_physics = False

#         # Set events for the specific object type (deformable cube)
#         self.events.reset_object_position = EventTerm(
#             func=mdp.reset_nodal_state_uniform,
#             mode="reset",
#             params={
#                 "position_range": {"x": (-0.1, 0.1), "y": (-0.25, 0.25), "z": (0.0, 0.0)},
#                 "velocity_range": {},
#                 "asset_cfg": SceneEntityCfg("object"),
#             },
#         )

#         # Remove all the terms for the state machine demo
#         # TODO: Computing the root pose of deformable object from nodal positions is expensive.
#         #       We need to fix that part before enabling these terms for the training.
#         self.terminations.object_dropping = None
#         self.rewards.reaching_object = None
#         self.rewards.lifting_object = None
#         self.rewards.object_goal_tracking = None
#         self.rewards.object_goal_tracking_fine_grained = None
#         self.observations.policy.object_position = None
