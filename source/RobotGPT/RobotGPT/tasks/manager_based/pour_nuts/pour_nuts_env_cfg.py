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
import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.assets.rigid_object_collection.rigid_object_collection_cfg import RigidObjectCollectionCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

from RobotGPT.tasks.manager_based.robotgpt_env_cfg import (
    RobotGPTBaseSceneCfg,
    RobotGPTEnvCfg,
    RobotGPTEventCfg,
    RobotGPTTerminationsCfg,
)

from .mdp.reset_cup_with_nuts import reset_cup_with_nuts

##
# Scene definition
##

SOURCE_CUP_POSITION = (0.5, 0.15, 0.02)

NUT_OFFSETS = [
    (-0.005, 0.005, 0.005),
    (-0.005, -0.005, 0.005),
    (0.005, -0.005, 0.005),
    (0.005, 0.005, 0.005),
    (-0.005, 0.005, 0.015),
    (-0.005, -0.005, 0.015),
    (0.005, -0.005, 0.015),
    (0.005, 0.005, 0.015),
    (-0.005, 0.005, 0.025),
    (-0.005, -0.005, 0.025),
    (0.005, -0.005, 0.025),
    (0.005, 0.005, 0.025),
]

NUT_POSITIONS = [(offset[0] + SOURCE_CUP_POSITION[0],
                  offset[1] + SOURCE_CUP_POSITION[1],
                  offset[2] + SOURCE_CUP_POSITION[2]) for offset in NUT_OFFSETS]

@configclass
class PourNutsSceneCfg(RobotGPTBaseSceneCfg):
    """Scene specification."""

    # props
    source_cup = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/source_cup",
        init_state=RigidObjectCfg.InitialStateCfg(pos=SOURCE_CUP_POSITION, rot=(0, 0, 0, 1)),
        spawn=UsdFileCfg(
            usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Mimic/nut_pour_task/nut_pour_assets/sorting_bowl_yellow.usd",
            scale=(0.7, 0.7, 2.0),
            rigid_props=RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                max_angular_velocity=1000.0,
                max_linear_velocity=1000.0,
                max_depenetration_velocity=5.0,
                disable_gravity=False,
            ),
            visual_material=sim_utils.GlassMdlCfg(),
        ),
    )

    target_cup = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/target_cup",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.5, -0.15, 0.02), rot=(0, 0, 0, 1)),
        spawn=UsdFileCfg(
            usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Mimic/nut_pour_task/nut_pour_assets/sorting_bowl_yellow.usd",
            scale=(0.7, 0.7, 2.0),
            rigid_props=RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                max_angular_velocity=1000.0,
                max_linear_velocity=1000.0,
                max_depenetration_velocity=5.0,
                disable_gravity=False,
            ),
            visual_material=sim_utils.GlassMdlCfg(),
        ),
    )

    nuts = RigidObjectCollectionCfg(
        rigid_objects = {
            f"nut_{i}" : RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/nuts/" + f"nut_{i}",
                init_state=RigidObjectCfg.InitialStateCfg(pos=nut_position, rot=(0, 0, 0, 1)),
                spawn=UsdFileCfg(
                    usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Mimic/nut_pour_task/nut_pour_assets/factory_m16_nut_green.usd",
                    scale=(0.7, 0.7, 0.7),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                    collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.3, 0.3, 0.3), metallic=1.0),
                ),
            ) for i, nut_position in enumerate(NUT_POSITIONS)
        }
    )


##
# MDP settings
##


@configclass
class PourNutsEventCfg(RobotGPTEventCfg):
    """Configuration for events."""

    randomize_cup_with_nuts_poses = EventTerm(
        func=reset_cup_with_nuts,
        mode="reset",
        params={
            "cup_pose_range": {"x": (-0.05, 0.1), "y": (-0.05, 0.05)},
            "nuts_pose_range": {"x": (-0.005, 0.005), "y": (-0.005, 0.005),
                                "yaw": (-3.14, 3.14), "roll": (-3.14, 3.14), "pitch": (-3.14, 3.14)},
            "cup_cfg": SceneEntityCfg("source_cup"),
            "nuts_cfg": SceneEntityCfg("nuts"),
        },
    )

    randomize_target_cup_position = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.05, 0.1), "y": (-0.05, 0.05)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("target_cup"),
        },
    )


@configclass
class PourNutsTerminationsCfg(RobotGPTTerminationsCfg):
    """Termination terms for the MDP."""

    pass


##
# Environment configuration
##


@configclass
class PourNutsEnvCfg(RobotGPTEnvCfg):
    """Configuration for the pour nuts environment."""

    # Scene settings
    scene: PourNutsSceneCfg = PourNutsSceneCfg(num_envs=1, env_spacing=2.5, replicate_physics=False)

    # MDP settings
    events: PourNutsEventCfg = PourNutsEventCfg()
    terminations: PourNutsTerminationsCfg = PourNutsTerminationsCfg()

    # Prompt for the openpi policy.
    prompt: str = "Pour the nuts from one container into the other"

    def __post_init__(self):
        """Post initialization."""
        super().__post_init__()
        # general settings
        self.episode_length_s = 60.0

        # Enable translucency for cups
        self.sim.render.enable_translucency = True
