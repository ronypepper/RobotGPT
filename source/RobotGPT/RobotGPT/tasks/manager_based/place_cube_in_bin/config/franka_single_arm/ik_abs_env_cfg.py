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
from RobotGPT.utils.robots.franka_single_arm import setup_franka_single_arm_ik_abs_env


@configclass
class FrankaSingleArmPlaceCubeInBinEnvCfg(PlaceCubeInBinEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        setup_franka_single_arm_ik_abs_env(self)
