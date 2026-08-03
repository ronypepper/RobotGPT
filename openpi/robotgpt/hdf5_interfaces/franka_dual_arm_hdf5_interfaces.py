import h5py
import numpy as np


def get_data_dimensions_franka_dual_arm():
    return {
        "actions": 16,
        "state": 16, # proprioceptive observation, i.e. joint & gripper positions
        "img_width": 224,
        "img_height": 224,
    }


def process_hdf5_frame_franka_dual_arm(demo: h5py.Group, step: int) -> dict:
    # Pi0 models are trained for gripper positions in [0.0, 1.0], with 0.0 corresponding to fully open and 1.0 corresponding to fully closed.
    # Gripper observations and actions in the dataset are in [0.0, 0.04], with 0.0 corresponding to fully closed and 0.04 corresponding to fully open.
    # Therefore we adjust the gripper values to fit the Pi0 models' format.
    left_joint_pos_obs = demo["obs"]["left_joint_pos"][step][:8] # 7 joints + 1 gripper
    left_joint_pos_obs[7] = (left_joint_pos_obs[7] - 0.04) / -0.04
    right_joint_pos_obs = demo["obs"]["right_joint_pos"][step][:8] # 7 joints + 1 gripper
    right_joint_pos_obs[7] = (right_joint_pos_obs[7] - 0.04) / -0.04
    observations = np.concatenate((left_joint_pos_obs, right_joint_pos_obs))

    left_joint_pos_actions = demo["processed_actions"][step][:8]
    left_joint_pos_actions[7] = (left_joint_pos_actions[7] - 0.04) / -0.04
    right_joint_pos_actions = demo["processed_actions"][step][9:17]
    right_joint_pos_actions[7] = (right_joint_pos_actions[7] - 0.04) / -0.04
    actions = np.concatenate((left_joint_pos_actions, right_joint_pos_actions))

    return {
        "table_img": demo["obs"]["table_img"][step],
        "left_wrist_img": demo["obs"]["left_wrist_img"][step],
        "right_wrist_img": demo["obs"]["right_wrist_img"][step],
        "state": observations,
        "actions": actions,
    }
