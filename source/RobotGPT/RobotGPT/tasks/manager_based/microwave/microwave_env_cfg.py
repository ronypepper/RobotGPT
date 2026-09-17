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
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import MjcfFileCfg, UsdFileCfg
from isaaclab.utils import configclass

from RobotGPT.tasks.manager_based.robotgpt_env_cfg import (
    RobotGPTBaseSceneCfg,
    RobotGPTEnvCfg,
    RobotGPTEventCfg,
    RobotGPTTerminationsCfg,
)
from RobotGPT.utils.asset_root_path import ROBOTGPT_ASSETS_PATH

##
# Scene definition
##

@configclass
class MicrowaveSceneCfg(RobotGPTBaseSceneCfg):
    """Scene specification."""

    # props
    mug = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/mug",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.2, 0.0, 0.05), rot=(0, 0, 0, 1)),
        spawn=MjcfFileCfg(
            asset_path=f"{ROBOTGPT_ASSETS_PATH}/google_scanned_objects/mujoco_scanned_objects/models/Threshold_Porcelain_Coffee_Mug_All_Over_Bead_White/model.xml",
            usd_dir=f"{ROBOTGPT_ASSETS_PATH}/google_scanned_objects/usd_conversions/Threshold_Porcelain_Coffee_Mug_All_Over_Bead_White",
            scale=(0.8, 0.8, 0.8),
            rigid_props=RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                max_angular_velocity=1000.0,
                max_linear_velocity=1000.0,
                max_depenetration_velocity=5.0,
                disable_gravity=False,
            ),
        ),
    )

    microwave = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/microwave",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.5, 0.0, 0.2619372), rot=(0, 0, 0, 1)),
        spawn=UsdFileCfg(
            usd_path=f"{ROBOTGPT_ASSETS_PATH}/Lightwheel_Kitchen/Collected_KitchenRoom/Microwave017/Microwave017.usd",
            scale=(1.0, 1.0, 1.0),
            rigid_props=RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                max_angular_velocity=1000.0,
                max_linear_velocity=1000.0,
                max_depenetration_velocity=5.0,
                disable_gravity=False,
            ),
        ),
    )


##
# MDP settings
##


@configclass
class MicrowaveEventCfg(RobotGPTEventCfg):
    """Configuration for events."""

    pass
    # randomize_mug_position = EventTerm(
    #     func=mdp.reset_root_state_uniform,
    #     mode="reset",
    #     params={
    #         "pose_range": {"x": (-0.15, 0.15), "y": (-0.1, 0.2), "yaw": (-3.14, 3.14)},
    #         "velocity_range": {},
    #         "asset_cfg": SceneEntityCfg("mug"),
    #     },
    # )

    # randomize_microwave_position = EventTerm(
    #     func=mdp.reset_root_state_uniform,
    #     mode="reset",
    #     params={
    #         # "pose_range": {"x": (-0.15, 0.15), "y": (-0.2, 0.1), "yaw": (-3.14, 3.14)},
    #         "pose_range": {"x": (-0.15, 0.15), "y": (-0.2, 0.1), "yaw": (-3.14, 3.14), "roll": (-3.14, 3.14), "pitch": (-3.14, 3.14)},
    #         "velocity_range": {},
    #         "asset_cfg": SceneEntityCfg("microwave"),
    #     },
    # )


@configclass
class MicrowaveTerminationsCfg(RobotGPTTerminationsCfg):
    """Termination terms for the MDP."""

    pass


##
# Environment configuration
##


@configclass
class MicrowaveEnvCfg(RobotGPTEnvCfg):
    """Configuration for the microwave environment."""

    # Scene settings
    scene: MicrowaveSceneCfg = MicrowaveSceneCfg(num_envs=1, env_spacing=2.5, replicate_physics=False)

    # MDP settings
    events: MicrowaveEventCfg = MicrowaveEventCfg()
    terminations: MicrowaveTerminationsCfg = MicrowaveTerminationsCfg()

    # Prompt for the openpi policy.
    prompt: str = "Pick up the mug and open the microwave door. Then place the mug in the microwave and close the microwave door again"

    def __post_init__(self):
        """Post initialization."""
        super().__post_init__()
        # general settings
        self.episode_length_s = 60.0
