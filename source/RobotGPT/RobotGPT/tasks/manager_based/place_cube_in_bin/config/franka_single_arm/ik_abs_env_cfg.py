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

from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.devices.device_base import DeviceBase, DevicesCfg
from isaaclab.devices.openxr.openxr_device import OpenXRDeviceCfg, XrCfg
from isaaclab.devices.openxr.retargeters.manipulator.gripper_retargeter import GripperRetargeterCfg
from isaaclab.devices.openxr.retargeters.manipulator.se3_abs_retargeter import Se3AbsRetargeterCfg
from isaaclab.utils import configclass

from RobotGPT.utils.mdp.env_step_differential_ik_action import EnvStepDifferentialInverseKinematicsActionCfg
from RobotGPT.utils.teleop.controller_gripper_retargeter import ControllerGripperRetargeterCfg
from RobotGPT.utils.teleop.controller_se3_abs_retargeter import ControllerSe3AbsRetargeterCfg
from RobotGPT.utils.teleop.openxr_device_with_record_controls import DemoRecorderOpenXRDeviceCfg

from . import joint_pos_env_cfg


@configclass
class FrankaSingleArmPlaceCubeInBinEnvCfg(joint_pos_env_cfg.FrankaSingleArmPlaceCubeInBinEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Set inverse kinematics actions for the specific robot type
        self.actions.arm_action = EnvStepDifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=["panda_joint.*"],
            body_name="panda_hand",
            controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls"),
            body_offset=EnvStepDifferentialInverseKinematicsActionCfg.OffsetCfg(pos=(0.0, 0.0, 0.107)),
        )

        # Teleoperation configuration
        self.xr: XrCfg = XrCfg(
            anchor_pos=(1.3, 0, -1.0),
            anchor_rot=(0, 0, 0.70711, 0.70711),
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
