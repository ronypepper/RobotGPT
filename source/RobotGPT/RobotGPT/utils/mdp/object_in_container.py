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

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply_inverse

if TYPE_CHECKING:
    from isaaclab.assets import RigidObject
    from isaaclab.envs import ManagerBasedRLEnv

def object_in_container(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    container_cfg: SceneEntityCfg = SceneEntityCfg("container"),
    container_halfsize_x: float = 1.0,
    container_halfsize_y: float = 1.0,
    container_halfsize_z: float = 1.0,
) -> torch.Tensor:
    object: RigidObject = env.scene[object_cfg.name]
    container: RigidObject = env.scene[container_cfg.name]

    # Get position of object in container's frame
    object_pos_w = object.data.root_pos_w.torch
    container_pos_w = container.data.root_pos_w.torch
    container_quat_w = container.data.root_quat_w.torch
    local_pos = quat_apply_inverse(container_quat_w, object_pos_w - container_pos_w)

    is_object_in_container = (
        (local_pos[:, 0] > -container_halfsize_x) &
        (local_pos[:, 0] < container_halfsize_x) &
        (local_pos[:, 1] > -container_halfsize_y) &
        (local_pos[:, 1] < container_halfsize_y) &
        (local_pos[:, 2] > -container_halfsize_z) &
        (local_pos[:, 2] < container_halfsize_z)
    )

    object_lin_speed = torch.linalg.norm(object.data.root_lin_vel_b.torch, dim=1)
    object_ang_speed = torch.linalg.norm(object.data.root_ang_vel_b.torch, dim=1)
    is_object_still = (object_lin_speed < 0.001) & (object_ang_speed < 0.001)

    return is_object_in_container & is_object_still
