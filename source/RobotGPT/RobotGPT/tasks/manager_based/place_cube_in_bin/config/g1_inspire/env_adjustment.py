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

from RobotGPT.tasks.manager_based.place_cube_in_bin.place_cube_in_bin_env_cfg import PlaceCubeInBinEnvCfg


def adjust_env_cfg(env_cfg: PlaceCubeInBinEnvCfg):
    env_cfg.scene.cube.init_state.pos = (0.5, -0.2, 0.05)
    env_cfg.events.randomize_cube_position.params["pose_range"] = {
        "x": (-0.05, 0.05), "y": (-0.1, 0.05), "yaw": (-3.14, 3.14), "roll": (-3.14, 3.14), "pitch": (-3.14, 3.14)
    }

    env_cfg.scene.bin.init_state.pos = (0.5, 0.1, 0.05)
    env_cfg.events.randomize_bin_position.params["pose_range"] = {
        "x": (-0.0, 0.1), "y": (-0.0, 0.05), "yaw": (-3.14, 3.14)
    }
