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

from isaaclab_physx.sim.schemas import PhysxDeformableBodyPropertiesCfg
from isaaclab_physx.sim.spawners.materials import PhysxSurfaceDeformableBodyMaterialCfg

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import DeformableObjectCfg
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
class FoldTowelSceneCfg(RobotGPTBaseSceneCfg):
    """Scene specification."""

    # props
    towel = DeformableObjectCfg(
        prim_path="{ENV_REGEX_NS}/towel",
        init_state=DeformableObjectCfg.InitialStateCfg(pos=(0.4, 0.0, 0.05)),
        spawn=sim_utils.MeshRectangleCfg(
            size=(0.4, 0.5),
            resolution=(50, 50),
            deformable_props=PhysxDeformableBodyPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.95, 0.1)),
            physics_material=PhysxSurfaceDeformableBodyMaterialCfg(),
        ),
    )


##
# MDP settings
##


@configclass
class FoldTowelEventCfg(RobotGPTEventCfg):
    """Configuration for events."""

    # randomize_bin_position = EventTerm(
    #     func=mdp.reset_root_state_uniform,
    #     mode="reset",
    #     params={
    #         "pose_range": {"x": (-0.15, 0.15), "y": (-0.1, 0.2), "yaw": (-3.14, 3.14)},
    #         "velocity_range": {},
    #         "asset_cfg": SceneEntityCfg("bin"),
    #     },
    # )

    # randomize_cube_position = EventTerm(
    #     func=mdp.reset_root_state_uniform,
    #     mode="reset",
    #     params={
    #         "pose_range": {"x": (-0.15, 0.15), "y": (-0.2, 0.1), "yaw": (-3.14, 3.14)},
    #         "velocity_range": {},
    #         "asset_cfg": SceneEntityCfg("cube"),
    #     },
    # )


@configclass
class FoldTowelTerminationsCfg(RobotGPTTerminationsCfg):
    """Termination terms for the MDP."""

    # success = DoneTerm(func=object_in_container, params={
    #         "object_cfg": SceneEntityCfg("cube"),
    #         "container_cfg": SceneEntityCfg("bin"),
    #         "container_halfsize_x": 9.971924 / 100,
    #         "container_halfsize_y": 14.951359 / 100,
    #         "container_halfsize_z": 13.640263 / 2 / 100,
    #     }
    # )


##
# Environment configuration
##


@configclass
class FoldTowelEnvCfg(RobotGPTEnvCfg):
    """Configuration for the place cube in bin environment."""

    # Scene settings
    scene: FoldTowelSceneCfg = FoldTowelSceneCfg(num_envs=1, env_spacing=2.5, replicate_physics=False)

    # MDP settings
    events: FoldTowelEventCfg = FoldTowelEventCfg()
    terminations: FoldTowelTerminationsCfg = FoldTowelTerminationsCfg()

    # Prompt for the openpi policy.
    prompt: str = "Fold the towel two times"
