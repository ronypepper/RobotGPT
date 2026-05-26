"""
Collects all custom configurations for easily adding them to openpi.

Copyright (c) 2026 ronypepper.

License: Apache 2.0
"""


def get_robotgpt_configs():
    import robotgpt.franka_single_arm_configs as franka_single_arm_configs

    all_cfgs = franka_single_arm_configs.franka_single_arm_configs

    return all_cfgs
