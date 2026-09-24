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

BOTTLE_POSITION = (0.45, 0.15, 0.02)

NUT_OFFSETS = [
    (0.0, 0.0, 0.005),
    (0.0, 0.0, 0.015),
    (0.0, 0.0, 0.025),
    (0.0, 0.0, 0.035),
    (0.0, 0.0, 0.045),
    (0.0, 0.0, 0.055),
]
# NUT_OFFSETS = [
#     (-0.005, 0.005, 0.005),
#     (-0.005, -0.005, 0.005),
#     (0.005, -0.005, 0.005),
#     (0.005, 0.005, 0.005),
#     (-0.005, 0.005, 0.015),
#     (-0.005, -0.005, 0.015),
#     (0.005, -0.005, 0.015),
#     (0.005, 0.005, 0.015),
#     (-0.005, 0.005, 0.025),
#     (-0.005, -0.005, 0.025),
#     (0.005, -0.005, 0.025),
#     (0.005, 0.005, 0.025),
# ]

NUT_POSITIONS = [(offset[0] + BOTTLE_POSITION[0],
                  offset[1] + BOTTLE_POSITION[1],
                  offset[2] + BOTTLE_POSITION[2]) for offset in NUT_OFFSETS]

@configclass
class PourNutsSceneCfg(RobotGPTBaseSceneCfg):
    """Scene specification."""

    # props
    bottle = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/bottle",
        init_state=RigidObjectCfg.InitialStateCfg(pos=BOTTLE_POSITION, rot=(0, 0, 0, 1)),
        spawn=UsdFileCfg(
            usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Mimic/nut_pour_task/nut_pour_assets/sorting_bowl_yellow.usd",
            scale=(0.5, 0.5, 2.5),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005),
            visual_material=sim_utils.GlassMdlCfg(),
        ),
    )

    bowl = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/bowl",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.45, -0.15, 0.02), rot=(0, 0, 0, 1)),
        spawn=UsdFileCfg(
            usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Mimic/nut_pour_task/nut_pour_assets/sorting_bowl_yellow.usd",
            scale=(2.0, 2.0, 1.5),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005),
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

    randomize_glass_with_nuts_poses = EventTerm(
        func=reset_cup_with_nuts,
        mode="reset",
        params={
            "cup_pose_range": {"x": (-0.05, 0.05), "y": (-0.05, 0.05)},
            "nuts_pose_range": {"x": (-0.003, 0.003), "y": (-0.003, 0.003),
                                "yaw": (-3.14, 3.14), "roll": (-3.14, 3.14), "pitch": (-3.14, 3.14)},
            "cup_cfg": SceneEntityCfg("bottle"),
            "nuts_cfg": SceneEntityCfg("nuts"),
        },
    )

    randomize_bowl_position = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.05, 0.05), "y": (-0.05, 0.05)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("bowl"),
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
    prompt: str = "Pick up the bottle and pour the nuts contained in it into the bowl"

    def __post_init__(self):
        """Post initialization."""
        super().__post_init__()
        # general settings
        self.episode_length_s = 60.0

        # Enable translucency for cups
        self.sim.render.enable_translucency = True
