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
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from RobotGPT.tasks.manager_based.robotgpt_env_cfg import (
    RobotGPTBaseSceneCfg,
    RobotGPTEnvCfg,
    RobotGPTEventCfg,
    RobotGPTTerminationsCfg,
)
from RobotGPT.utils.mdp.object_in_container import object_in_container

##
# Scene definition
##


@configclass
class PlaceCubeInBinSceneCfg(RobotGPTBaseSceneCfg):
    """Scene specification."""

    # props
    cube = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/cube",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.5, 0.25, 0.05), rot=(0, 0, 0, 1)),
        spawn=UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/blue_block.usd",
            scale=(1.5, 1.5, 1.5),
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

    bin = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/bin",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.5, -0.25, 0.05), rot=(0, 0, 0, 1)),
        spawn=UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/KLT_Bin/small_KLT.usd",
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
class PlaceCubeInBinEventCfg(RobotGPTEventCfg):
    """Configuration for events."""

    randomize_bin_position = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.15, 0.15), "y": (-0.1, 0.2), "yaw": (-3.14, 3.14)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("bin"),
        },
    )

    randomize_cube_position = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.15, 0.15), "y": (-0.2, 0.1), "yaw": (-3.14, 3.14)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("cube"),
        },
    )


@configclass
class PlaceCubeInBinTerminationsCfg(RobotGPTTerminationsCfg):
    """Termination terms for the MDP."""

    success = DoneTerm(func=object_in_container, params={
            "object_cfg": SceneEntityCfg("cube"),
            "container_cfg": SceneEntityCfg("bin"),
            "container_halfsize_x": 9.971924 / 100,
            "container_halfsize_y": 14.951359 / 100,
            "container_halfsize_z": 13.640263 / 2 / 100,
        }
    )


##
# Environment configuration
##


@configclass
class PlaceCubeInBinEnvCfg(RobotGPTEnvCfg):
    """Configuration for the place cube in bin environment."""

    # Scene settings
    scene: PlaceCubeInBinSceneCfg = PlaceCubeInBinSceneCfg(num_envs=1, env_spacing=2.5, replicate_physics=False)

    # MDP settings
    events: PlaceCubeInBinEventCfg = PlaceCubeInBinEventCfg()
    terminations: PlaceCubeInBinTerminationsCfg = PlaceCubeInBinTerminationsCfg()

    # Prompt for the openpi policy.
    prompt: str = "Pick up the blue cube and drop it in the bin"

    def __post_init__(self):
        """Post initialization."""
        super().__post_init__()
        # general settings
        self.episode_length_s = 60.0
