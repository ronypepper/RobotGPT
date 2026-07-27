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

"""
Base configuration for all RobotGPT environments.

Defines basic scene, robot action/observations, lighting randomization, time-out termination and general simulation
settings.
Derivative configurations define specific task props, robots and optionally task completion criteria.
"""
from dataclasses import MISSING

# from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg
# from isaaclab_newton.renderers import NewtonWarpRendererCfg
from isaaclab_physx.physics import PhysxCfg

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg
from isaaclab.sim.schemas import CollisionPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, NVIDIA_NUCLEUS_DIR

from RobotGPT.utils.mdp.env_step_differential_ik_action import EnvStepDifferentialInverseKinematicsActionCfg
from RobotGPT.utils.mdp.image_converted_for_openpi import image_converted_for_openpi
from RobotGPT.utils.mdp.randomize_utils import randomize_scene_lighting_domelight

##
# Scene definition
##


@configclass
class RobotGPTBaseSceneCfg(InteractiveSceneCfg):
    """Scene specification."""

    #
    # Embodiment
    #

    # Robot
    robot: ArticulationCfg = MISSING

    # Second robot, used by bimanual setups with two individual robot arms (eg. bimanual franka setup)
    # In these cases, this is the right arm (seen from behind the robot looking at workspace)
    robot_2: ArticulationCfg | None = None

    # End-effector markers for debug visualization
    ee_marker : AssetBaseCfg | None = None
    ee_marker_2 : AssetBaseCfg | None = None

    # Helper for initializing end-effector markers
    def initialize_ee_marker(self, dual_arm: bool):
        self.ee_marker = AssetBaseCfg(
            prim_path=MISSING,
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0, 0.107), rot=(0, 0, 0, 1)),
            spawn=UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/frame_prim.usd",
                scale=(0.1, 0.1, 0.1),
            )
        )
        if dual_arm:
            self.ee_marker_2 = AssetBaseCfg(
                prim_path=MISSING,
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0, 0.107), rot=(0, 0, 0, 1)),
                spawn=UsdFileCfg(
                    usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/frame_prim.usd",
                    scale=(0.1, 0.1, 0.1),
                )
            )

    # Wrist view cameras
    left_wrist_cam = CameraCfg(
        prim_path=MISSING,
        update_period=0.0,
        height=480,
        width=640,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=2.1, focus_distance=28.0, horizontal_aperture=5.376, vertical_aperture=3.024
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(0.1009906081856474, -2.2170453280873081e-7, 0.005195286872436311),
            rot=(0.68618, 0.68618, 0.17074, 0.17074), convention="opengl"
        ),
        # renderer_cfg=NewtonWarpRendererCfg(create_default_light=False, enable_shadows=True)
    )

    right_wrist_cam : CameraCfg | None = None

    # Helper for initializing right wrist camera in dual arm setups
    def initialize_right_wrist_camera(self):
        self.right_wrist_cam = CameraCfg(
            prim_path=MISSING,
            update_period=0.0,
            height=480,
            width=640,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=2.1, focus_distance=28.0, horizontal_aperture=5.376, vertical_aperture=3.024
            ),
            offset=CameraCfg.OffsetCfg(
                pos=(0.1009906081856474, -2.2170453280873081e-7, 0.005195286872436311),
                rot=(0.68618, 0.68618, 0.17074, 0.17074), convention="opengl"
            ),
            # renderer_cfg=NewtonWarpRendererCfg(create_default_light=False, enable_shadows=True)
        )

    # Table view camera
    table_cam = CameraCfg(
        prim_path="{ENV_REGEX_NS}/table_cam",
        update_period=0.0,
        height=480,
        width=640,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=2.1, focus_distance=28.0, horizontal_aperture=5.376, vertical_aperture=3.024
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(-0.03890996200355793, 0.9736847657158553, 0.8084830725058005),
            rot=(0.09143, -0.47766, -0.83945, 0.24249), convention="opengl"
        ),
        # renderer_cfg=NewtonWarpRendererCfg(create_default_light=False, enable_shadows=True)
    )

    #
    # Scene
    #

    # Table
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.5, 0, 0), rot=(0, 0, 0.707, 0.707)),
        spawn=UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd",
            scale=(2.0, 1.0, 1.0)
        )
    )

    # Background
    background = AssetBaseCfg(
        prim_path="/World/background",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0, 0, -1)),
        spawn=UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Environments/Simple_Warehouse/warehouse.usd",
            collision_props=CollisionPropertiesCfg(
                collision_enabled=False,
            )
        ),
    )

    # Lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )


##
# MDP settings
##


@configclass
class RobotGPTActionsCfg:
    """Action specifications for the MDP."""

    # will be set by agent env cfg
    arm_action: mdp.JointPositionActionCfg | EnvStepDifferentialInverseKinematicsActionCfg = MISSING
    gripper_action: mdp.JointPositionActionCfg | mdp.BinaryJointPositionActionCfg = MISSING

    # second arm actions (used by bimanual franka setup)
    arm_action_2: mdp.JointPositionActionCfg | EnvStepDifferentialInverseKinematicsActionCfg | None = None
    gripper_action_2: mdp.JointPositionActionCfg | mdp.BinaryJointPositionActionCfg | None = None


@configclass
class RobotGPTObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class SingleArmPolicyCfg(ObsGroup):
        """Observations for policy group."""

        joint_pos = ObsTerm(func=mdp.joint_pos)
        table_img = ObsTerm(
            func=image_converted_for_openpi, params={"sensor_cfg": SceneEntityCfg("table_cam"),
                                                     "data_type": "rgb", "normalize": False}
        )
        wrist_img = ObsTerm(
            func=image_converted_for_openpi, params={"sensor_cfg": SceneEntityCfg("left_wrist_cam"),
                                                     "data_type": "rgb", "normalize": False}
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    @configclass
    class DualArmPolicyCfg(ObsGroup):
        """Observations for policy group."""

        left_joint_pos = ObsTerm(func=mdp.joint_pos, params={"asset_cfg": SceneEntityCfg("robot")})
        right_joint_pos = ObsTerm(func=mdp.joint_pos, params={"asset_cfg": SceneEntityCfg("robot")})
        left_wrist_img = ObsTerm(
            func=image_converted_for_openpi, params={"sensor_cfg": SceneEntityCfg("left_wrist_cam"),
                                                     "data_type": "rgb", "normalize": False}
        )
        right_wrist_img = ObsTerm(
            func=image_converted_for_openpi, params={"sensor_cfg": SceneEntityCfg("right_wrist_cam"),
                                                     "data_type": "rgb", "normalize": False}
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    # Helper functions for setting up the observations in a task's environment cfg
    def setup_single_arm_observations(self, joint_names: str | list[str] | None = None):
        self.policy = self.SingleArmPolicyCfg()
        self.policy.joint_pos.params = {"asset_cfg": SceneEntityCfg("robot", joint_names=joint_names)}

    def setup_dual_arm_observations(self, left_joint_names: str | list[str] | None = None,
                                    right_joint_names: str | list[str] | None = None,
                                    use_robot_2_for_right_arm: bool = False):
        self.policy = self.DualArmPolicyCfg()
        self.policy.left_joint_pos.params = {"asset_cfg": SceneEntityCfg("robot", joint_names=left_joint_names)}
        self.policy.right_joint_pos.params = {
            "asset_cfg": SceneEntityCfg("robot_2" if use_robot_2_for_right_arm else "robot", joint_names=right_joint_names)
        }

    # observation groups
    policy: SingleArmPolicyCfg | DualArmPolicyCfg = MISSING


@configclass
class RobotGPTEventCfg:
    """Configuration for events."""

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    randomize_light = EventTerm(
        func=randomize_scene_lighting_domelight,
        mode="reset",
        params={
            "intensity_range": (1500.0, 10000.0),
            "color_variation": 0.4,
            "textures": [
                f"{NVIDIA_NUCLEUS_DIR}/Assets/Skies/Cloudy/abandoned_parking_4k.hdr",
                f"{NVIDIA_NUCLEUS_DIR}/Assets/Skies/Cloudy/evening_road_01_4k.hdr",
                f"{NVIDIA_NUCLEUS_DIR}/Assets/Skies/Cloudy/lakeside_4k.hdr",
                f"{NVIDIA_NUCLEUS_DIR}/Assets/Skies/Indoor/autoshop_01_4k.hdr",
                f"{NVIDIA_NUCLEUS_DIR}/Assets/Skies/Indoor/carpentry_shop_01_4k.hdr",
                f"{NVIDIA_NUCLEUS_DIR}/Assets/Skies/Indoor/hospital_room_4k.hdr",
                f"{NVIDIA_NUCLEUS_DIR}/Assets/Skies/Indoor/hotel_room_4k.hdr",
                f"{NVIDIA_NUCLEUS_DIR}/Assets/Skies/Indoor/old_bus_depot_4k.hdr",
                f"{NVIDIA_NUCLEUS_DIR}/Assets/Skies/Indoor/small_empty_house_4k.hdr",
                f"{NVIDIA_NUCLEUS_DIR}/Assets/Skies/Indoor/surgery_4k.hdr",
                f"{NVIDIA_NUCLEUS_DIR}/Assets/Skies/Studio/photo_studio_01_4k.hdr",
            ],
            "default_intensity": 3000.0,
            "default_color": (0.75, 0.75, 0.75),
            "default_texture": "",
        },
    )


@configclass
class RobotGPTTerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)


##
# Environment configuration
##


@configclass
class RobotGPTEnvCfg(ManagerBasedRLEnvCfg):
    """Base configuration for RobotGPT environments."""

    # Scene settings
    scene: RobotGPTBaseSceneCfg = RobotGPTBaseSceneCfg(num_envs=1, env_spacing=2.5, replicate_physics=False)
    # Basic settings
    observations: RobotGPTObservationsCfg = RobotGPTObservationsCfg()
    actions: RobotGPTActionsCfg = RobotGPTActionsCfg()

    # MDP settings
    events: RobotGPTEventCfg = RobotGPTEventCfg()
    terminations: RobotGPTTerminationsCfg = RobotGPTTerminationsCfg()

    # Unused managers
    commands = None
    rewards = None
    curriculum = None

    # Prompt for the openpi policy.
    prompt: str = MISSING

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 3
        self.episode_length_s = 30.0

        # simulation settings
        self.sim.dt = 1 / 60.0
        self.sim.render_interval = self.decimation

        self.sim.physics = PhysxCfg(
            bounce_threshold_velocity=0.01,
            gpu_found_lost_aggregate_pairs_capacity=1024 * 1024 * 4,
            gpu_total_aggregate_pairs_capacity=16 * 1024,
            friction_correlation_distance=0.00625,
        )
        # self.sim.physics = NewtonCfg(
        #     solver_cfg=MJWarpSolverCfg(
        #         njmax=50,
        #         nconmax=20,
        #         cone="pyramidal",
        #         integrator="implicitfast",
        #         impratio=1,
        #     ),
        #     num_substeps=1,
        #     debug_mode=False,
        # )

        # Set settings for camera rendering
        self.num_rerenders_on_reset = 3
        self.sim.render.rendering_mode = 'balanced' # Reduce from 'quality' for better performance

        self.viewer.eye = (2.0, 1.8, 1.5)

    def get_ep_meta(self):
        ep_meta = dict()

        # Add basic episode metadata
        ep_meta["sim_args"] = {
            "dt": self.sim.dt,
            "decimation": self.decimation,
            "render_interval": self.sim.render_interval,
            "num_envs": self.scene.num_envs
        }

        # Add prompt
        ep_meta['prompt'] = self.prompt

        return ep_meta
