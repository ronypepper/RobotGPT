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
from isaaclab.assets import RigidObjectCfg, RigidObjectCollectionCfg
from isaaclab.assets.asset_base_cfg import AssetBaseCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sim.schemas import CollisionPropertiesCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from RobotGPT.tasks.manager_based.robotgpt_env_cfg import (
    RobotGPTBaseSceneCfg,
    RobotGPTEnvCfg,
    RobotGPTEventCfg,
    RobotGPTTerminationsCfg,
)
from RobotGPT.utils.mdp.object_in_container import object_in_container
from RobotGPT.utils.mdp.reset_multiple_objects_randomly import reset_multiple_objects_randomly

##
# Scene definition
##

CUBE_NAMES = [
    "red_cube_1",
    "red_cube_2",
    "red_cube_3",
    "green_cube_1",
    "green_cube_2",
    "green_cube_3",
]

CUBE_COLORS = [
    (1.0, 0.0, 0.0),
    (1.0, 0.0, 0.0),
    (1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0),
    (0.0, 1.0, 0.0),
    (0.0, 1.0, 0.0),
]

CUBE_DEFAULT_POSITIONS = [
    (-0.25, 0.25, 0.05),
    (-0.25, 0.5, 0.05),
    (-0.25, 0.75, 0.05),
    (0.25, 0.25, 0.05),
    (0.25, 0.5, 0.05),
    (0.25, 0.75, 0.05),
]

@configclass
class SortCubesSceneCfg(RobotGPTBaseSceneCfg):
    """Scene specification."""

    # cubes
    cubes = RigidObjectCollectionCfg(
        rigid_objects = {
            cube_name : RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/cube/" + cube_name,
                init_state=RigidObjectCfg.InitialStateCfg(pos=cube_pos, rot=(0, 0, 0, 1)),
                spawn=sim_utils.CuboidCfg(
                    size=(0.03, 0.03, 0.03),
                    rigid_props=RigidBodyPropertiesCfg(
                        solver_position_iteration_count=16,
                        solver_velocity_iteration_count=1,
                        max_angular_velocity=1000.0,
                        max_linear_velocity=1000.0,
                        max_depenetration_velocity=5.0,
                        disable_gravity=False,
                    ),
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                    physics_material=sim_utils.RigidBodyMaterialCfg(),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=cube_color),
                ),
            ) for cube_name, cube_color, cube_pos in zip(CUBE_NAMES, CUBE_COLORS, CUBE_DEFAULT_POSITIONS)
        }
    )

    red_area = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/red_area",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.5, -0.3, -0.002), rot=(0, 0, 0.7071068, 0.7071068)),
        spawn=sim_utils.CuboidCfg(
            size=(0.21, 0.297, 0.001),  # A4 paper size
            collision_props=None,
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(1.0, 0.0, 0.0)
            ),
        ),
    )

    green_area = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/green_area",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.5, 0.3, -0.002), rot=(0, 0, 0.7071068, 0.7071068)),
        spawn=sim_utils.CuboidCfg(
            size=(0.21, 0.297, 0.001),  # A4 paper size
            collision_props=None,
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.0, 1.0, 0.0)
            ),
        ),
    )


##
# MDP settings
##


@configclass
class SortCubesEventCfg(RobotGPTEventCfg):
    """Configuration for events."""

    randomize_cube_positions = EventTerm(
        func=reset_multiple_objects_randomly,
        mode="reset",
        params={
            "pose_range": {"x": (-0.05, 0.05), "y": (-0.05, 0.05)},
            "init_positions": [
                (0.5, 0.0, 0.1),
                (0.55, -0.1, 0.1),
                (0.55, 0.1, 0.1),
                (0.45, -0.1, 0.1),
                (0.45, 0.1, 0.1),
            ],
            "num_inits_range": (2, 5),
            "hide_offset": (-4.0, 0.0, -1.0),
            "objects_cfg": SceneEntityCfg("cubes"),
        },
    )


@configclass
class SortCubesTerminationsCfg(RobotGPTTerminationsCfg):
    """Termination terms for the MDP."""

    pass


##
# Environment configuration
##


@configclass
class SortCubesEnvCfg(RobotGPTEnvCfg):
    """Configuration for the place cube in bin environment."""

    # Scene settings
    scene: SortCubesSceneCfg = SortCubesSceneCfg(num_envs=1, env_spacing=2.5, replicate_physics=False)

    # MDP settings
    events: SortCubesEventCfg = SortCubesEventCfg()
    terminations: SortCubesTerminationsCfg = SortCubesTerminationsCfg()

    # Prompt for the openpi policy.
    prompt: str = "Place all red cubes in the red area and all green cubes in the green area"

    def __post_init__(self):
        """Post initialization."""
        super().__post_init__()
        # general settings
        self.episode_length_s = 120.0
