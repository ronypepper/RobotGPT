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

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

##
# Joint Position Control
##

gym.register(
    id="RobotGPT-Place-Cube-In-Bin-Franka-Single-Arm-Pos-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:FrankaSingleArmPlaceCubeInBinEnvCfg",
    },
    disable_env_checker=True,
)

##
# Joint Velocity Control
##

gym.register(
    id="RobotGPT-Place-Cube-In-Bin-Franka-Single-Arm-Vel-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_vel_env_cfg:FrankaSingleArmPlaceCubeInBinEnvCfg",
    },
    disable_env_checker=True,
)

##
# Inverse Kinematics - Absolute Pose Control
##

gym.register(
    id="RobotGPT-Place-Cube-In-Bin-Franka-Single-Arm-IK-Abs-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ik_abs_env_cfg:FrankaSingleArmPlaceCubeInBinEnvCfg",
    },
    disable_env_checker=True,
)

##
# Inverse Kinematics - Relative Pose Control
##

gym.register(
    id="RobotGPT-Place-Cube-In-Bin-Franka-Single-Arm-IK-Rel-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ik_rel_env_cfg:FrankaSingleArmPlaceCubeInBinEnvCfg",
    },
    disable_env_checker=True,
)
