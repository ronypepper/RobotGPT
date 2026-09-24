"""
Hdf5 dataset interface funtions for dual franka arms.

Copyright (c) 2026 ronypepper.

License: Apache 2.0
"""

import h5py
import numpy as np


def get_data_dimensions_g1_inspire():
    return {
        "actions": 16,
        "state": 16, # proprioceptive observation, i.e. joint & gripper positions
        "img_width": 224,
        "img_height": 224,
    }


def process_hdf5_frame_g1_inspire(demo: h5py.Group, step: int) -> dict:
    # Pi0 models are trained for gripper positions in [0.0, 1.0], with 0.0 corresponding to fully open and 1.0 corresponding to fully closed.
    # The environment provides observations for the proximal index and the proximal thumb pitch finger joints, which are
    # actuated in the ranges [0.0, 0.7] and [0.0, 0.26], respectively (see inspire_hand_action.py).
    # A single gripper position in the Pi0 models' format is computed from these joints.
    left_joint_pos = demo["obs"]["left_joint_pos"][step][:7]
    left_index_pitch = np.clip(demo["obs"]["left_joint_pos"][step][7], 0.0, 0.7) / 0.7
    left_thumb_pitch = np.clip(demo["obs"]["left_joint_pos"][step][8], 0.0, 0.26) / 0.26
    left_gripper_pos = (left_index_pitch + left_thumb_pitch) / 2

    right_joint_pos = demo["obs"]["right_joint_pos"][step][:7]
    right_index_pitch = np.clip(demo["obs"]["right_joint_pos"][step][7], 0.0, 0.7) / 0.7
    right_thumb_pitch = np.clip(demo["obs"]["right_joint_pos"][step][8], 0.0, 0.26) / 0.26
    right_gripper_pos = (right_index_pitch + right_thumb_pitch) / 2

    observations = np.concatenate((left_joint_pos, (left_gripper_pos, ), right_joint_pos, (right_gripper_pos, )))

    left_joint_pos_actions = demo["processed_actions"][step][:7]
    left_gripper_action = (demo["actions"][step][14] - 1.0) * -0.5
    right_joint_pos_actions = demo["processed_actions"][step][7:14]
    right_gripper_action = (demo["actions"][step][15] - 1.0) * -0.5
    actions = np.concatenate((left_joint_pos_actions, left_gripper_action,
                              right_joint_pos_actions, right_gripper_action))

    return {
        "table_img": demo["obs"]["table_img"][step],
        "left_wrist_img": demo["obs"]["left_wrist_img"][step],
        "right_wrist_img": demo["obs"]["right_wrist_img"][step],
        "state": observations,
        "actions": actions,
    }
