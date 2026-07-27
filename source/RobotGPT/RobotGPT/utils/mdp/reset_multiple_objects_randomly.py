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
    from isaaclab.assets import RigidObjectCollection
    from isaaclab.envs import ManagerBasedEnv

# import logger
logger = logging.getLogger(__name__)

def reset_multiple_objects_randomly(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    pose_range: dict[str, tuple[float, float]],
    init_positions: list[tuple[float, float, float]],
    num_inits_range: tuple[int, int],
    hide_offset: tuple[float, float, float],
    items_cfg: SceneEntityCfg = SceneEntityCfg("items"),
):
    # extract the used quantities (to enable type-hinting)
    items: RigidObjectCollection = env.scene[items_cfg.name]

    if num_inits_range[0] < 1 or num_inits_range[1] > len(init_positions) or num_inits_range[1] > items.num_bodies:
        raise ValueError("num_inits_range is invalid")

    # Select random objects
    num_objects = random.randint(*num_inits_range)
    object_ids = list(range(items.num_bodies))
    init_ids = random.sample(object_ids, num_objects)

    # get default body states
    default_pose = items.data.default_body_pose.torch[env_ids].clone()
    default_vel = items.data.default_body_vel.torch[env_ids].clone()

    # Set randomzied poses for selected objects
    range_list = [pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z"]]
    ranges = torch.tensor(range_list, device=items.device)
    rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), num_objects, 3), device=items.device)

    init_positions_torch = torch.tensor(init_positions, device=items.device)#[None, :, :]
    init_positions_torch = init_positions_torch[None, random.sample(range(len(init_positions)), num_objects), :]
    positions = init_positions_torch + env.scene.env_origins[env_ids][:, None, :] + rand_samples
    orientations = math_utils.random_orientation(len(env_ids), device=items.device)

    default_pose[env_ids, init_ids, :3] = positions
    default_pose[env_ids, init_ids, 3:] = orientations

    # Move not-selected objects away
    if num_objects < items.num_bodies:
        hide_ids = [i for i in object_ids if i not in init_ids]
        default_pose[env_ids, hide_ids, :3] += torch.tensor(hide_offset, device=items.device)

    # set into the physics simulation
    items.write_body_pose_to_sim_index(body_poses=default_pose, env_ids=env_ids)
    items.write_body_velocity_to_sim_index(body_velocities=default_vel, env_ids=env_ids)
