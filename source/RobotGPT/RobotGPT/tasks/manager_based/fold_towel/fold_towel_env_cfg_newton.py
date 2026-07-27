# # Based on code from the Isaac Lab project:
# # https://github.com/isaac-sim/IsaacLab
# #
# # Original work:
# # Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# # All rights reserved.
# #
# # Modifications:
# # Copyright (c) 2026 ronypepper.
# #
# # SPDX-License-Identifier: BSD-3-Clause

# from isaaclab_physx.sim.schemas import PhysxDeformableBodyPropertiesCfg
# from isaaclab_physx.sim.spawners.materials import PhysxSurfaceDeformableBodyMaterialCfg

# import isaaclab.envs.mdp as mdp
# import isaaclab.sim as sim_utils
# from isaaclab.assets import DeformableObjectCfg
# from isaaclab.managers import EventTermCfg as EventTerm
# from isaaclab.managers import SceneEntityCfg
# from isaaclab.managers import TerminationTermCfg as DoneTerm
# from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
# from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
# from isaaclab.utils import configclass
# from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
# from isaaclab_contrib.deformable.newton_manager_cfg import NewtonModelCfg
# from isaaclab_newton.sim.schemas import NewtonDeformableBodyPropertiesCfg
# from isaaclab_newton.sim.spawners.materials import NewtonSurfaceDeformableBodyMaterialCfg
# from isaaclab_tasks.core.lift.config.franka_soft.franka_soft_env_cfg import (
#     DeformableNewtonCfg,
#     coupled_mjwarp_vbd_solver_cfg,
# )

# from RobotGPT.tasks.manager_based.robotgpt_env_cfg import (
#     RobotGPTBaseSceneCfg,
#     RobotGPTEnvCfg,
#     RobotGPTEventCfg,
#     RobotGPTTerminationsCfg,
# )
# from RobotGPT.utils.mdp.object_in_container import object_in_container

# ##
# # Scene definition
# ##

# @configclass
# class FoldTowelSceneCfg(RobotGPTBaseSceneCfg):
#     """Scene specification."""

#     # props
#     # Matching the Newton example.
#     towel = DeformableObjectCfg(
#         prim_path="/World/envs/env_.*/Deformable",
#         init_state=DeformableObjectCfg.InitialStateCfg(pos=(0.4, 0.0, 0.2)),
#         spawn=sim_utils.MeshRectangleCfg(
#             size=(0.4, 0.5),
#             resolution=(40, 50),
#             deformable_props=NewtonDeformableBodyPropertiesCfg(),
#             visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.95, 0.85, 0.1)),
#             physics_material=NewtonSurfaceDeformableBodyMaterialCfg(
#                 density=50.0,
#                 particle_radius=0.005,
#                 tri_ke=5e2,
#                 tri_ka=5e2,
#                 tri_kd=1e-3,
#                 edge_ke=2.0,
#                 edge_kd=1e-3,
#             ),
#         ),
#     )


# ##
# # MDP settings
# ##

# ROBOT_SHAPE_MATERIAL_MU = 100.0
# ROBOT_SHAPE_MATERIAL_BODY_NAMES = ".*"

# @configclass
# class FoldTowelEventCfg(RobotGPTEventCfg):
#     """Configuration for events."""
#     robot_physics_material = EventTerm(
#         func=mdp.randomize_rigid_body_material,
#         mode="reset",
#         params={
#             "asset_cfg": SceneEntityCfg("robot", body_names=ROBOT_SHAPE_MATERIAL_BODY_NAMES),
#             "static_friction_range": (ROBOT_SHAPE_MATERIAL_MU, ROBOT_SHAPE_MATERIAL_MU),
#             "dynamic_friction_range": (ROBOT_SHAPE_MATERIAL_MU, ROBOT_SHAPE_MATERIAL_MU),
#             "restitution_range": (0.0, 0.0),
#             "num_buckets": 1,
#         },
#     )
#     robot_2_physics_material = EventTerm(
#         func=mdp.randomize_rigid_body_material,
#         mode="reset",
#         params={
#             "asset_cfg": SceneEntityCfg("robot_2", body_names=ROBOT_SHAPE_MATERIAL_BODY_NAMES),
#             "static_friction_range": (ROBOT_SHAPE_MATERIAL_MU, ROBOT_SHAPE_MATERIAL_MU),
#             "dynamic_friction_range": (ROBOT_SHAPE_MATERIAL_MU, ROBOT_SHAPE_MATERIAL_MU),
#             "restitution_range": (0.0, 0.0),
#             "num_buckets": 1,
#         },
#     )

#     # robot_2_physics_material = EventTerm(
#     #     func=mdp.randomize_rigid_body_material,
#     #     mode="startup",
#     #     params={
#     #         "asset_cfg": SceneEntityCfg("robot_2", body_names=ROBOT_SHAPE_MATERIAL_BODY_NAMES),
#     #         "static_friction_range": (ROBOT_SHAPE_MATERIAL_MU, ROBOT_SHAPE_MATERIAL_MU),
#     #         "dynamic_friction_range": (ROBOT_SHAPE_MATERIAL_MU, ROBOT_SHAPE_MATERIAL_MU),
#     #         "restitution_range": (0.0, 0.0),
#     #         "num_buckets": 1,
#     #     },
#     # )

#     # reset_deformable = EventTerm(
#     #     func=mdp.reset_nodal_state_uniform,
#     #     mode="reset",
#     #     params={
#     #         "position_range": {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (0.0, 0.0)},
#     #         "velocity_range": {},
#     #         "asset_cfg": SceneEntityCfg("deformable"),
#     #     },
#     # )

#     # randomize_bin_position = EventTerm(
#     #     func=mdp.reset_root_state_uniform,
#     #     mode="reset",
#     #     params={
#     #         "pose_range": {"x": (-0.15, 0.15), "y": (-0.1, 0.2), "yaw": (-3.14, 3.14)},
#     #         "velocity_range": {},
#     #         "asset_cfg": SceneEntityCfg("bin"),
#     #     },
#     # )

#     # randomize_cube_position = EventTerm(
#     #     func=mdp.reset_root_state_uniform,
#     #     mode="reset",
#     #     params={
#     #         "pose_range": {"x": (-0.15, 0.15), "y": (-0.2, 0.1), "yaw": (-3.14, 3.14)},
#     #         "velocity_range": {},
#     #         "asset_cfg": SceneEntityCfg("cube"),
#     #     },
#     # )


# @configclass
# class FoldTowelTerminationsCfg(RobotGPTTerminationsCfg):
#     """Termination terms for the MDP."""

#     # success = DoneTerm(func=object_in_container, params={
#     #         "object_cfg": SceneEntityCfg("cube"),
#     #         "container_cfg": SceneEntityCfg("bin"),
#     #         "container_halfsize_x": 9.971924 / 100,
#     #         "container_halfsize_y": 14.951359 / 100,
#     #         "container_halfsize_z": 13.640263 / 2 / 100,
#     #     }
#     # )


# ##
# # Environment configuration
# ##


# @configclass
# class FoldTowelEnvCfg(RobotGPTEnvCfg):
#     """Configuration for the place cube in bin environment."""

#     # Scene settings
#     scene: FoldTowelSceneCfg = FoldTowelSceneCfg(num_envs=1, env_spacing=2.5, replicate_physics=False)

#     # MDP settings
#     events: FoldTowelEventCfg = FoldTowelEventCfg()
#     terminations: FoldTowelTerminationsCfg = FoldTowelTerminationsCfg()

#     # Prompt for the openpi policy.
#     prompt: str = "Fold the towel two times"

#     def __post_init__(self):
#         """Post initialization."""
#         super().__post_init__()
#         # general settings
#         self.episode_length_s = 120.0

#         # physics settings for cloth
#         # Newton physics: MJWarp rigid + VBD soft, two-way coupled
#         # (matches newton/examples/softbody/example_softbody_franka.py)
#         self.sim.physics = DeformableNewtonCfg(
#             solver_cfg=coupled_mjwarp_vbd_solver_cfg(),
#             model_cfg=NewtonModelCfg(
#                 soft_contact_ke=1e3,
#                 soft_contact_kd=1e-5,
#                 soft_contact_mu=0.5,
#                 shape_material_ke=1e3,
#                 shape_material_kd=1e-5,
#                 shape_material_mu=1e-4,
#             ),
#             num_substeps=10,
#             use_cuda_graph=True,
#         )
