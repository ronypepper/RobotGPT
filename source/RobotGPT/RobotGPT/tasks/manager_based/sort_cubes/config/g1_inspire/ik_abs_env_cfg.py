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

from RobotGPT.tasks.manager_based.sort_cubes.sort_cubes_env_cfg import SortCubesEnvCfg
from RobotGPT.utils.robots.g1_inspire import setup_g1_inspire_ik_abs_env


@configclass
class G1InspireSortCubesEnvCfg(SortCubesEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        setup_g1_inspire_ik_abs_env(self)
