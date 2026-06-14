"""
Collects all openpi interface funtions.

Copyright (c) 2026 ronypepper.

License: Apache 2.0
"""

from RobotGPT.utils.robots.franka_dual_arm import (
    process_observation_for_openpi_franka_dual_arm,
    process_openpi_action_franka_dual_arm,
)
from RobotGPT.utils.robots.franka_single_arm import (
    process_observation_for_openpi_franka_single_arm,
    process_openpi_action_franka_single_arm,
)

# Openpi observation/action interface functions for implemented robot types
OPENPI_INTERFACE_FCTS = {
    "franka_single_arm" : {"obs" : process_observation_for_openpi_franka_single_arm,
                           "act" : process_openpi_action_franka_single_arm},
    "franka_dual_arm" : {"obs" : process_observation_for_openpi_franka_dual_arm,
                         "act" : process_openpi_action_franka_dual_arm},
}

def get_openpi_interface_fcts(robot_type: str):
    if robot_type not in OPENPI_INTERFACE_FCTS:
        raise ValueError("robot_type not specified in OPENPI_INTERFACE_FCTS.")
    process_observation_for_openpi_fct = OPENPI_INTERFACE_FCTS[robot_type]["obs"]
    process_openpi_action_fct = OPENPI_INTERFACE_FCTS[robot_type]["act"]
    return process_observation_for_openpi_fct, process_openpi_action_fct
