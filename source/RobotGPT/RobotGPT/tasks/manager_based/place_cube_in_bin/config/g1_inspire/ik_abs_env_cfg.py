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

from isaaclab.utils import configclass

from RobotGPT.tasks.manager_based.place_cube_in_bin.place_cube_in_bin_env_cfg import PlaceCubeInBinEnvCfg
from RobotGPT.utils.robots.g1_inspire import setup_g1_inspire_ik_abs_env

from .env_adjustment import adjust_env_cfg


@configclass
class G1InspirePlaceCubeInBinEnvCfg(PlaceCubeInBinEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        adjust_env_cfg(self)

        setup_g1_inspire_ik_abs_env(self)
