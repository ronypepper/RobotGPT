"""
Collects all hdf5 dataset interface funtions.

Copyright (c) 2026 ronypepper.

License: Apache 2.0
"""

from robotgpt.hdf5_interfaces.franka_single_arm_hdf5_interfaces import (
    get_data_dimensions_franka_single_arm,
    process_hdf5_frame_franka_single_arm
)
from robotgpt.hdf5_interfaces.franka_dual_arm_hdf5_interfaces import (
    get_data_dimensions_franka_dual_arm,
    process_hdf5_frame_franka_dual_arm
)


# HDF5 dataset interface functions for implemented robot types
HDF5_DATASET_INTERFACE_FCTS = {
    "franka_single_arm" : {"data_dim" : get_data_dimensions_franka_single_arm,
                           "process_hdf5" : process_hdf5_frame_franka_single_arm},
    "franka_dual_arm" : {"obs" : get_data_dimensions_franka_dual_arm,
                         "act" : process_hdf5_frame_franka_dual_arm},
}

def get_hdf5_dataset_interface_fcts(robot_type: str):
    if robot_type not in HDF5_DATASET_INTERFACE_FCTS:
        raise ValueError("robot_type not specified in HDF5_DATASET_INTERFACE_FCTS.")
    get_data_dimensions_fct = HDF5_DATASET_INTERFACE_FCTS[robot_type]["data_dim"]
    process_hdf5_frame_fct = HDF5_DATASET_INTERFACE_FCTS[robot_type]["process_hdf5"]
    return get_data_dimensions_fct, process_hdf5_frame_fct
