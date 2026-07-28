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

import numpy as np

import isaaclab.envs.mdp as mdp
from isaaclab.assets import RigidObjectCfg, RigidObjectCollectionCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import MjcfFileCfg, UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from RobotGPT.tasks.manager_based.robotgpt_env_cfg import (
    RobotGPTBaseSceneCfg,
    RobotGPTEnvCfg,
    RobotGPTEventCfg,
    RobotGPTTerminationsCfg,
)
from RobotGPT.utils.asset_root_path import ROBOTGPT_ASSETS_PATH
from RobotGPT.utils.mdp.object_in_container import object_in_container
from RobotGPT.utils.mdp.reset_multiple_objects_randomly import reset_multiple_objects_randomly

##
# Scene definition
##

GOOGLE_SCANNED_OBJECTS_ITEMS = [
    "Nescafe_Momento_Mocha_Specialty_Coffee_Mix_8_ct",
    "NESCAFE_NESCAFE_TC_STKS_DECAF_6_CT",
    "Threshold_Porcelain_Coffee_Mug_All_Over_Bead_White",
    "Marc_Anthony_Skip_Professional_Oil_of_Morocco_Conditioner_with_Argan_Oil",
    "Nestle_Nesquik_Chocolate_Powder_Flavored_Milk_Additive_109_Oz_Canister",
    "Nestle_Candy_19_oz_Butterfinger_Singles_116567",
    "Polar_Herring_Fillets_Smoked_Peppered_705_oz_total",
    "YumYum_D3_Liquid",
    "Cole_Hardware_Mug_Classic_Blue",
    "Shurtape_30_Day_Removal_UV_Delct_15",
    "Shurtape_Gaffers_Tape_Silver_2_x_60_yd",
    "Nestle_Nips_Hard_Candy_Peanut_Butter",
    "HeavyDuty_Flashlight",
    "Weston_No_22_Cajun_Jerky_Tonic_12_fl_oz_nLj64ZnGwDh",
    "Wilton_Pearlized_Sugar_Sprinkles_525_oz_Gold",
]

ITEMS_SCALES = [
    (0.6, 0.7, 0.7),
    (0.7, 0.9, 0.7),
    (0.8, 0.8, 0.8),
    (0.8, 0.8, 0.8),
    (0.8, 0.5, 0.8),
    (0.7, 1.0, 1.0),
    (0.8, 0.9, 0.7),
    (1.0, 1.0, 1.0),
    (0.7, 0.7, 0.7),
    (1.0, 1.0, 1.0),
    (0.8, 0.8, 0.8),
    (0.5, 0.9, 0.9),
    (0.7, 0.7, 0.7),
    (0.8, 0.8, 0.8),
    (0.9, 0.9, 0.9),
]

ITEMS_DEFAULT_POSITIONS = [
    (x, y, 0.05) for y in np.linspace(-0.5, 0.5, 5) for x in np.linspace(0.25, 0.75, 4)
]

@configclass
class PackItemsSceneCfg(RobotGPTBaseSceneCfg):
    """Scene specification."""

    # props
    items = RigidObjectCollectionCfg(
        rigid_objects = {
            item_name : RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/rigid_items/" + item_name,
                init_state=RigidObjectCfg.InitialStateCfg(pos=init_pos, rot=(0, 0, 0, 1)),
                spawn=MjcfFileCfg(
                    asset_path=f"{ROBOTGPT_ASSETS_PATH}/google_scanned_objects/mujoco_scanned_objects/models/{item_name}/model.xml",
                    usd_dir=f"{ROBOTGPT_ASSETS_PATH}/google_scanned_objects/usd_conversions/{item_name}",
                    scale=scale,
                    rigid_props=RigidBodyPropertiesCfg(
                        solver_position_iteration_count=16,
                        solver_velocity_iteration_count=1,
                        max_angular_velocity=1000.0,
                        max_linear_velocity=1000.0,
                        max_depenetration_velocity=5.0,
                        disable_gravity=False,
                    ),
                ),
            ) for item_name, init_pos, scale in zip(GOOGLE_SCANNED_OBJECTS_ITEMS, ITEMS_DEFAULT_POSITIONS, ITEMS_SCALES)
        }
    )

    bin = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/bin",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.45, -0.3, 0.1), rot=(0, 0, 0.7071068, 0.7071068)),
        spawn=UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/KLT_Bin/small_KLT.usd",
            scale=(1.1, 1.1, 1.1),
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
class PackItemsEventCfg(RobotGPTEventCfg):
    """Configuration for events."""

    randomize_bin_position = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.1, 0.1), "y": (-0.015, 0.025), "yaw": (-0.5, 0.5)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("bin"),
        },
    )

    randomize_item_positions = EventTerm(
        func=reset_multiple_objects_randomly,
        mode="reset",
        params={
            "pose_range": {"x": (-0.02, 0.02), "y": (-0.02, 0.02)},
            "init_positions": [
                (0.5, 0.125, 0.15),
                (0.55, 0.025, 0.15),
                (0.55, 0.225, 0.15),
                (0.45, 0.025, 0.15),
                (0.45, 0.225, 0.15),
            ],
            "num_inits_range": (3, 5),
            "hide_offset": (-4.0, 0.0, -1.0),
            "objects_cfg": SceneEntityCfg("items"),
        },
    )


@configclass
class PackItemsTerminationsCfg(RobotGPTTerminationsCfg):
    """Termination terms for the MDP."""

    pass


##
# Environment configuration
##


@configclass
class PackItemsEnvCfg(RobotGPTEnvCfg):
    """Configuration for the place cube in bin environment."""

    # Scene settings
    scene: PackItemsSceneCfg = PackItemsSceneCfg(num_envs=1, env_spacing=2.5, replicate_physics=False)

    # MDP settings
    events: PackItemsEventCfg = PackItemsEventCfg()
    terminations: PackItemsTerminationsCfg = PackItemsTerminationsCfg()

    # Prompt for the openpi policy.
    prompt: str = "Pick up the items on the table and drop them in the bin, one by one"

    def __post_init__(self):
        """Post initialization."""
        super().__post_init__()
        # general settings
        self.episode_length_s = 120.0
