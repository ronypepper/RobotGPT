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

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

from openpi_client.image_tools import convert_to_uint8, resize_with_pad

from isaaclab.envs.mdp import image


def image_converted_for_openpi(
    env: ManagerBasedEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("tiled_camera"),
    data_type: str = "rgb",
    convert_perspective_to_orthogonal: bool = False,
    normalize: bool = True,
) -> torch.Tensor:
    """A wrapper around Isaac Lab's mdp.image that applies resizes and pads the image using openpi's image_tools.
    Can be used when recording demonstrations to convert data directly to openpi's expected format to save disk storage
    and speed up the demonstration saving process when running record_demos.py"""

    img_np = image(env, sensor_cfg, data_type, convert_perspective_to_orthogonal, normalize).numpy(force=True)
    return torch.from_numpy(convert_to_uint8(resize_with_pad(img_np, 224, 224)))
