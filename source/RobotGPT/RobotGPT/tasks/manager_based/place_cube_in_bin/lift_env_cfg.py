# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, DeformableObjectCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, NVIDIA_NUCLEUS_DIR

##
# Scene definition
##


@configclass
class ObjectTableSceneCfg(InteractiveSceneCfg):
    """Configuration for the lift scene with a robot and a object.
    This is the abstract base implementation, the exact scene is defined in the derived classes
    which need to set the target object, robot and end-effector frames
    """

    # robots: will be populated by agent env cfg
    robot: ArticulationCfg = MISSING

    # Table
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.5, 0, 1), rot=(0.707, 0, 0, 0.707)),
        spawn=UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd",
            scale=(2.0, 1.0, 1.0),
        )
    )

    # # plane
    # plane = AssetBaseCfg(
    #     prim_path="/World/GroundPlane",
    #     init_state=AssetBaseCfg.InitialStateCfg(pos=(0, 0, -1.05)),
    #     spawn=GroundPlaneCfg(),
    # )

    # usd environment
    # terrain = TerrainImporterCfg(
    #     prim_path="/World/ground",
    #     terrain_type="usd",
    #     usd_path=f"{ISAAC_NUCLEUS_DIR}/Environments/Simple_Warehouse/warehouse.usd",
    # )
    background = AssetBaseCfg(
        prim_path="/World/background",
        spawn=UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Environments/Simple_Warehouse/warehouse.usd",
        ),
    )

    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )

    # end-effector marker
    ee_marker = AssetBaseCfg(
        prim_path=MISSING,
        spawn=UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/frame_prim.usd",
            scale=(1.5, 1.5, 1.5),
        )
    )

    # wrist view camera + visualization
    wrist_cam = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/panda_hand/wrist_cam",
        update_period=0.0,
        height=200,
        width=200,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 100)
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(0.18108, 0.0, -0.16356),
            rot=(0.17074, 0.68618, 0.68618, 0.17074), convention="opengl"
        ),
        # offset=CameraCfg.OffsetCfg(
        #     pos=(0.13, 0.0, -0.15), rot=(-0.70614, 0.03701, 0.03701, -0.70614), convention="ros"
        # ),
    )
    wrist_cam_vis = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Robot/panda_hand/wrist_cam/wrist_cam_visualization",
        spawn=sim_utils.CuboidCfg(
            size=(0.15, 0.05, 0.05),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
        ),
    )

    # table view camera + visualization
    table_cam = CameraCfg(
        prim_path="{ENV_REGEX_NS}/table_cam",
        update_period=0.0,
        height=200,
        width=200,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 100)
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(-0.3069141847036739, 1.4860790429322785, 2.081719253805095),
            rot=(0.24249, 0.09143, -0.47766, -0.83945), convention="opengl"
        ),
    )
    table_cam_vis = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/table_cam/table_cam_visualization",
        spawn=sim_utils.CuboidCfg(
            size=(0.15, 0.05, 0.05),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
        ),
    )

    # props
    cube = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/cube",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.5, 0.25, 1.05), rot=(1, 0, 0, 0)),
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

    box = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/box",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.5, -0.25, 1.05), rot=(1, 0, 0, 0)),
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
class ActionsCfg:
    """Action specifications for the MDP."""

    # will be set by agent env cfg
    arm_action: mdp.JointPositionActionCfg | mdp.JointVelocityActionCfg | mdp.DifferentialInverseKinematicsActionCfg = MISSING
    gripper_action: mdp.JointPositionActionCfg | mdp.BinaryJointPositionActionCfg = MISSING


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        joint_pos = ObsTerm(func=mdp.joint_pos)
        table_img = ObsTerm(
            func=mdp.image, params={"sensor_cfg": SceneEntityCfg("table_cam"), "data_type": "rgb", "normalize": False}
        )
        wrist_img = ObsTerm(
            func=mdp.image, params={"sensor_cfg": SceneEntityCfg("wrist_cam"), "data_type": "rgb", "normalize": False}
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    # randomize_light = EventTerm(
    #     func=franka_stack_events.randomize_scene_lighting_domelight,
    #     mode="reset",
    #     params={
    #         "intensity_range": (1500.0, 10000.0),
    #         "color_variation": 0.4,
    #         "textures": [
    #             f"{NVIDIA_NUCLEUS_DIR}/Assets/Skies/Cloudy/abandoned_parking_4k.hdr",
    #             f"{NVIDIA_NUCLEUS_DIR}/Assets/Skies/Cloudy/evening_road_01_4k.hdr",
    #             f"{NVIDIA_NUCLEUS_DIR}/Assets/Skies/Cloudy/lakeside_4k.hdr",
    #             f"{NVIDIA_NUCLEUS_DIR}/Assets/Skies/Indoor/autoshop_01_4k.hdr",
    #             f"{NVIDIA_NUCLEUS_DIR}/Assets/Skies/Indoor/carpentry_shop_01_4k.hdr",
    #             f"{NVIDIA_NUCLEUS_DIR}/Assets/Skies/Indoor/hospital_room_4k.hdr",
    #             f"{NVIDIA_NUCLEUS_DIR}/Assets/Skies/Indoor/hotel_room_4k.hdr",
    #             f"{NVIDIA_NUCLEUS_DIR}/Assets/Skies/Indoor/old_bus_depot_4k.hdr",
    #             f"{NVIDIA_NUCLEUS_DIR}/Assets/Skies/Indoor/small_empty_house_4k.hdr",
    #             f"{NVIDIA_NUCLEUS_DIR}/Assets/Skies/Indoor/surgery_4k.hdr",
    #             f"{NVIDIA_NUCLEUS_DIR}/Assets/Skies/Studio/photo_studio_01_4k.hdr",
    #         ],
    #         "default_intensity": 3000.0,
    #         "default_color": (0.75, 0.75, 0.75),
    #         "default_texture": "",
    #     },
    # )

    # randomize_table_visual_material = EventTerm(
    #     func=franka_stack_events.randomize_visual_texture_material,
    #     mode="reset",
    #     params={
    #         "asset_cfg": SceneEntityCfg("table"),
    #         "textures": [
    #             f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Wood/Ash/Ash_BaseColor.png",
    #             f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Wood/Bamboo_Planks/Bamboo_Planks_BaseColor.png",
    #             f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Wood/Birch/Birch_BaseColor.png",
    #             f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Wood/Cherry/Cherry_BaseColor.png",
    #             f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Wood/Mahogany_Planks/Mahogany_Planks_BaseColor.png",
    #             f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Wood/Oak/Oak_BaseColor.png",
    #             f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Wood/Plywood/Plywood_BaseColor.png",
    #             f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Wood/Timber/Timber_BaseColor.png",
    #             f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Wood/Timber_Cladding/Timber_Cladding_BaseColor.png",
    #             f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Wood/Walnut_Planks/Walnut_Planks_BaseColor.png",
    #             f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Stone/Marble/Marble_BaseColor.png",
    #             f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Metals/Steel_Stainless/Steel_Stainless_BaseColor.png",
    #         ],
    #         "default_texture": (
    #             f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/Materials/Textures/DemoTable_TableBase_BaseColor.png"
    #         ),
    #     },
    # )

    # randomize_robot_arm_visual_texture = EventTerm(
    #     func=franka_stack_events.randomize_visual_texture_material,
    #     mode="reset",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot"),
    #         "textures": [
    #             f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Metals/Aluminum_Cast/Aluminum_Cast_BaseColor.png",
    #             f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Metals/Aluminum_Polished/Aluminum_Polished_BaseColor.png",
    #             f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Metals/Brass/Brass_BaseColor.png",
    #             f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Metals/Bronze/Bronze_BaseColor.png",
    #             f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Metals/Brushed_Antique_Copper/Brushed_Antique_Copper_BaseColor.png",
    #             f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Metals/Cast_Metal_Silver_Vein/Cast_Metal_Silver_Vein_BaseColor.png",
    #             f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Metals/Copper/Copper_BaseColor.png",
    #             f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Metals/Gold/Gold_BaseColor.png",
    #             f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Metals/Iron/Iron_BaseColor.png",
    #             f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Metals/RustedMetal/RustedMetal_BaseColor.png",
    #             f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Metals/Silver/Silver_BaseColor.png",
    #             f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Metals/Steel_Carbon/Steel_Carbon_BaseColor.png",
    #             f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Metals/Steel_Stainless/Steel_Stainless_BaseColor.png",
    #         ],
    #     },
    # )

    # reset_object_position = EventTerm(
    #     func=mdp.reset_root_state_uniform,
    #     mode="reset",
    #     params={
    #         "pose_range": {"x": (-0.1, 0.1), "y": (-0.25, 0.25), "z": (0.0, 0.0)},
    #         "velocity_range": {},
    #         "asset_cfg": SceneEntityCfg("object", body_names="Object"),
    #     },
    # )


##
# Environment configuration
##


@configclass
class LiftEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the lifting environment."""

    # Scene settings
    scene: ObjectTableSceneCfg = ObjectTableSceneCfg(num_envs=4096, env_spacing=2.5)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()

    # MDP settings
    events: EventCfg = EventCfg()

    # Unused managers
    commands = None
    rewards = None
    terminations = None
    curriculum = None

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 2
        self.episode_length_s = 10.0
        # simulation settings
        self.sim.dt = 0.01  # 100Hz
        self.sim.render_interval = self.decimation

        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625

        # Set settings for camera rendering
        self.num_rerenders_on_reset = 3
        self.sim.render.antialiasing_mode = "DLAA"  # Use DLAA for higher quality rendering
