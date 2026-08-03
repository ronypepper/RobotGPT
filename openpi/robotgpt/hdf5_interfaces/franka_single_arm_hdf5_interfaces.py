import h5py
import numpy as np


def get_data_dimensions_franka_single_arm():
    return {
        "actions": 8,
        "state": 8, # proprioceptive observation, i.e. joint & gripper positions
        "img_width": 224,
        "img_height": 224,
    }


def process_hdf5_frame_franka_single_arm(demo: h5py.Group, step: int) -> dict:
    # Pi0 models are trained for gripper positions in [0.0, 1.0], with 0.0 corresponding to fully open and 1.0 corresponding to fully closed.
    # Gripper observations and actions in the dataset are in [0.0, 0.04], with 0.0 corresponding to fully closed and 0.04 corresponding to fully open.
    # Therefore we adjust the gripper values to fit the Pi0 models' format.
    joint_pos_obs = demo["obs"]["joint_pos"][step][:8] # 7 joints + 1 gripper
    joint_pos_obs[7] = (joint_pos_obs[7] - 0.04) / -0.04

    joint_pos_actions = demo["processed_actions"][step][:8] # 7 joints + 1 gripper
    joint_pos_actions[7] = (joint_pos_actions[7] - 0.04) / -0.04
    return {
        "table_img": demo["obs"]["table_img"][step],
        "wrist_img": demo["obs"]["wrist_img"][step],
        "state": joint_pos_obs,
        "actions": joint_pos_actions,
    }
