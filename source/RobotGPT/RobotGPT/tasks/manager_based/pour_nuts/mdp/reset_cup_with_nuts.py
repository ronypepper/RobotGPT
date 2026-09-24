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

from __future__ import annotations

import logging
import random
from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.assets import RigidObject, RigidObjectCollection
    from isaaclab.envs import ManagerBasedEnv

# import logger
logger = logging.getLogger(__name__)

def reset_cup_with_nuts(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    cup_pose_range: dict[str, tuple[float, float]],
    nuts_pose_range: dict[str, tuple[float, float]],
    cup_cfg: SceneEntityCfg = SceneEntityCfg("source_cup"),
    nuts_cfg: SceneEntityCfg = SceneEntityCfg("nuts"),
):
    # extract the used quantities (to enable type-hinting)
    cup: RigidObject = env.scene[cup_cfg.name]
    nuts: RigidObjectCollection = env.scene[nuts_cfg.name]

    # get default cup state
    cup_default_pose = cup.data.default_root_pose.torch[env_ids].clone()
    cup_default_vel = cup.data.default_root_vel.torch[env_ids].clone()

    # compute randomized cup poses
    range_list = [cup_pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z"]]
    ranges = torch.tensor(range_list, device=cup.device)
    cup_rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 3), device=cup.device)

    cup_default_pose[:, 0:3] += env.scene.env_origins[env_ids] + cup_rand_samples

    # set cup state into the physics simulation
    cup.write_root_pose_to_sim_index(root_pose=cup_default_pose, env_ids=env_ids)
    cup.write_root_velocity_to_sim_index(root_velocity=cup_default_vel, env_ids=env_ids)

    # get default nuts states
    nuts_default_poses = nuts.data.default_body_pose.torch[env_ids].clone()
    nuts_default_vels = nuts.data.default_body_vel.torch[env_ids].clone()

    # compute randomized nut poses
    num_nuts = nuts_default_poses.shape[1]
    range_list = [nuts_pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=cup.device)
    nuts_rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1],
                                                  (len(env_ids), num_nuts, 6), device=cup.device)

    nuts_default_poses[:, :, 0:3] += env.scene.env_origins[env_ids] + cup_rand_samples + nuts_rand_samples[:, :, 0:3]
    orientations_delta = math_utils.quat_from_euler_xyz(nuts_rand_samples[:, :, 3], nuts_rand_samples[:, :, 4],
                                                        nuts_rand_samples[:, :, 5])
    nuts_default_poses[:, :, 3:7] = math_utils.quat_mul(nuts_default_poses[:, :, 3:7], orientations_delta)

    # set nuts states into the physics simulation
    nuts.write_body_pose_to_sim_index(body_poses=nuts_default_poses, env_ids=env_ids)
    nuts.write_body_velocity_to_sim_index(body_velocities=nuts_default_vels, env_ids=env_ids)
