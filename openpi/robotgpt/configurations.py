"""
Collects all custom configurations for easily adding them to openpi.
"""


def get_robotgpt_configs():
    import robotgpt.franka_single_arm_configuration as franka_single_arm_configuration

    all_cfgs = franka_single_arm_configuration.franka_configs

    return all_cfgs
