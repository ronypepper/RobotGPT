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
from isaaclab.assets import RigidObjectCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from RobotGPT.tasks.manager_based.place_cube_in_bin.place_cube_in_bin_env_cfg import PlaceCubeInBinEnvCfg

##
# Pre-defined configs
##
from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip
from isaaclab_assets.robots.franka import FRANKA_PANDA_CFG  # isort: skip


@configclass
class FrankaSingleArmPlaceCubeInBinEnvCfg(PlaceCubeInBinEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Set Franka as robot
        self.scene.robot = FRANKA_PANDA_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
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
        self.actions.arm_action = mdp.JointVelocityActionCfg(
            asset_name="robot", joint_names=["panda_joint.*"]
        )
        self.actions.gripper_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=["panda_finger.*"],
            scale=-0.04,
            offset=0.04,
            use_default_offset=False
        )

        # self.scene.ee_marker.prim_path="{ENV_REGEX_NS}/Robot/panda_hand/ee_marker"

        # Set wrist camera anchor on robot
        self.scene.wrist_cam.prim_path = "{ENV_REGEX_NS}/Robot/panda_hand/wrist_cam"

        # # Listens to the required transforms
        # marker_cfg = FRAME_MARKER_CFG.copy()
        # marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        # marker_cfg.prim_path = "/Visuals/FrameTransformer"
        # self.scene.ee_frame = FrameTransformerCfg(
        #     prim_path="{ENV_REGEX_NS}/Robot/panda_link0",
        #     debug_vis=False,
        #     visualizer_cfg=marker_cfg,
        #     target_frames=[
        #         FrameTransformerCfg.FrameCfg(
        #             prim_path="{ENV_REGEX_NS}/Robot/panda_hand",
        #             name="end_effector",
        #             offset=OffsetCfg(
        #                 pos=(0.0, 0.0, 0.1034),
        #             ),
        #         ),
        #     ],
        # )
