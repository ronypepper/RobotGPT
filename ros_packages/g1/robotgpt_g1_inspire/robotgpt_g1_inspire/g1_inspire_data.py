"""
Various data concerning interfaces and hardware of the Unitree G1 robot and the Inspire 6DOF dexterous hand.

Part of the data is taken from:
https://github.com/unitreerobotics/xr_teleoperate
Licensed under the Apache License, Version 2.0.

License: Apache 2.0
"""

from enum import IntEnum


class G1JointIndex(IntEnum):
    LEFT_HIP_PITCH = 0
    LEFT_HIP_ROLL = 1
    LEFT_HIP_YAW = 2
    LEFT_KNEE = 3
    LEFT_ANKLE_PITCH = 4
    LEFT_ANKLE_ROLL = 5

    RIGHT_HIP_PITCH = 6
    RIGHT_HIP_ROLL = 7
    RIGHT_HIP_YAW = 8
    RIGHT_KNEE = 9
    RIGHT_ANKLE_PITCH = 10
    RIGHT_ANKLE_ROLL = 11

    WAIST_YAW = 12
    WAIST_ROLL = 13
    WAIST_PITCH = 14

    LEFT_SHOULDER_PITCH = 15
    LEFT_SHOULDER_ROLL = 16
    LEFT_SHOULDER_YAW = 17
    LEFT_ELBOW = 18
    LEFT_WRIST_ROLL = 19
    LEFT_WRIST_PITCH = 20
    LEFT_WRIST_YAW = 21

    RIGHT_SHOULDER_PITCH = 22
    RIGHT_SHOULDER_ROLL = 23
    RIGHT_SHOULDER_YAW = 24
    RIGHT_ELBOW = 25
    RIGHT_WRIST_ROLL = 26
    RIGHT_WRIST_PITCH = 27
    RIGHT_WRIST_YAW = 28

    CONTROL_WEIGHT = 29


G1_PROPRIETARY_JOINT_MAPPING = {
    G1JointIndex.LEFT_SHOULDER_PITCH : 10,
    G1JointIndex.LEFT_SHOULDER_ROLL : 28,
    G1JointIndex.LEFT_SHOULDER_YAW : 17,
    G1JointIndex.LEFT_ELBOW : 15,
    G1JointIndex.LEFT_WRIST_ROLL : 23,
    G1JointIndex.LEFT_WRIST_PITCH : 14,
    G1JointIndex.LEFT_WRIST_YAW : 21,

    G1JointIndex.RIGHT_SHOULDER_PITCH : 20,
    G1JointIndex.RIGHT_SHOULDER_ROLL : 13,
    G1JointIndex.RIGHT_SHOULDER_YAW : 16,
    G1JointIndex.RIGHT_ELBOW : 11,
    G1JointIndex.RIGHT_WRIST_ROLL : 12,
    G1JointIndex.RIGHT_WRIST_PITCH : 27,
    G1JointIndex.RIGHT_WRIST_YAW : 19,
}


G1_LEFT_ARM_INDICES = [
    G1JointIndex.LEFT_SHOULDER_PITCH,
    G1JointIndex.LEFT_SHOULDER_ROLL,
    G1JointIndex.LEFT_SHOULDER_YAW,
    G1JointIndex.LEFT_ELBOW,
    G1JointIndex.LEFT_WRIST_ROLL,
    G1JointIndex.LEFT_WRIST_PITCH,
    G1JointIndex.LEFT_WRIST_YAW,
]

G1_RIGHT_ARM_INDICES = [
    G1JointIndex.RIGHT_SHOULDER_PITCH,
    G1JointIndex.RIGHT_SHOULDER_ROLL,
    G1JointIndex.RIGHT_SHOULDER_YAW,
    G1JointIndex.RIGHT_ELBOW,
    G1JointIndex.RIGHT_WRIST_ROLL,
    G1JointIndex.RIGHT_WRIST_PITCH,
    G1JointIndex.RIGHT_WRIST_YAW,
]

G1_HIGH_STIFFNESS = 100.0
G1_HIGH_DAMPING = 5.0
G1_LOW_STIFFNESS = 20.0
G1_LOW_DAMPING = 1.0
G1_WRIST_STIFFNESS = 5.0
G1_WRIST_DAMPING = 1.0

# Dynamics are only defined for arm and waist joints, as we are publishing to the /armsdk topic and using G1's builtin motion 
# control, where only these joints are valid.
G1_DYNAMICS = {
    G1JointIndex.LEFT_SHOULDER_PITCH : (G1_LOW_STIFFNESS, G1_LOW_DAMPING),
    G1JointIndex.LEFT_SHOULDER_ROLL : (G1_LOW_STIFFNESS, G1_LOW_DAMPING),
    G1JointIndex.LEFT_SHOULDER_YAW : (G1_LOW_STIFFNESS, G1_LOW_DAMPING),
    G1JointIndex.LEFT_ELBOW : (G1_LOW_STIFFNESS, G1_LOW_DAMPING),
    G1JointIndex.LEFT_WRIST_ROLL : (G1_WRIST_STIFFNESS, G1_WRIST_DAMPING),
    G1JointIndex.LEFT_WRIST_PITCH : (G1_WRIST_STIFFNESS, G1_WRIST_DAMPING),
    G1JointIndex.LEFT_WRIST_YAW : (G1_WRIST_STIFFNESS, G1_WRIST_DAMPING),

    G1JointIndex.RIGHT_SHOULDER_PITCH : (G1_LOW_STIFFNESS, G1_LOW_DAMPING),
    G1JointIndex.RIGHT_SHOULDER_ROLL : (G1_LOW_STIFFNESS, G1_LOW_DAMPING),
    G1JointIndex.RIGHT_SHOULDER_YAW : (G1_LOW_STIFFNESS, G1_LOW_DAMPING),
    G1JointIndex.RIGHT_ELBOW : (G1_LOW_STIFFNESS, G1_LOW_DAMPING),
    G1JointIndex.RIGHT_WRIST_ROLL : (G1_WRIST_STIFFNESS, G1_WRIST_DAMPING),
    G1JointIndex.RIGHT_WRIST_PITCH : (G1_WRIST_STIFFNESS, G1_WRIST_DAMPING),
    G1JointIndex.RIGHT_WRIST_YAW : (G1_WRIST_STIFFNESS, G1_WRIST_DAMPING),

    G1JointIndex.WAIST_YAW : (G1_HIGH_STIFFNESS, G1_HIGH_DAMPING),
    G1JointIndex.WAIST_ROLL : (G1_HIGH_STIFFNESS, G1_HIGH_DAMPING),
    G1JointIndex.WAIST_PITCH : (G1_HIGH_STIFFNESS, G1_HIGH_DAMPING),
}

G1_CONTROL_FREQUENCY = 50.0
G1_CONTROL_DT = 1.0 / G1_CONTROL_FREQUENCY

G1_CONTROL_WEIGHT_ADJUST_DURATION = 2.0  # s
G1_MAX_CONTROL_WEIGHT = 0.5

G1_TELEOPERATION_MAX_JOINT_SPEED = 700.0 / 180.0 * 3.14  # rad/s

G1_MOVING_HOME_JOINT_SPEED = 350.0 / 180.0 * 3.14  # rad/s
G1_HOME_POSE_TOLERANCE = 15.0 / 180.0 * 3.14  # rad

class InspireJointIndex(IntEnum):
    RIGHT_PINKY = 0
    RIGHT_RING = 1
    RIGHT_MIDDLE = 2
    RIGHT_INDEX = 3
    RIGHT_THUMB_BEND = 4
    RIGHT_THUMB_ROTATION = 5
    LEFT_PINKY = 6
    LEFT_RING = 7
    LEFT_MIDDLE = 8
    LEFT_INDEX = 9
    LEFT_THUMB_BEND = 10
    LEFT_THUMB_ROTATION = 11


INSPIRE_LEFT_INDICES = [
    # The order matters here! (matches https://github.com/ronypepper/RobotGPT/blob/master/source/RobotGPT/RobotGPT/utils/robots/g1_inspire.py)
    InspireJointIndex.LEFT_INDEX,
    InspireJointIndex.LEFT_MIDDLE,
    InspireJointIndex.LEFT_PINKY,
    InspireJointIndex.LEFT_RING,
    InspireJointIndex.LEFT_THUMB_BEND,
    InspireJointIndex.LEFT_THUMB_ROTATION,
]

INSPIRE_RIGHT_INDICES = [
    # The order matters here! (matches https://github.com/ronypepper/RobotGPT/blob/master/source/RobotGPT/RobotGPT/utils/robots/g1_inspire.py)
    InspireJointIndex.RIGHT_INDEX,
    InspireJointIndex.RIGHT_MIDDLE,
    InspireJointIndex.RIGHT_PINKY,
    InspireJointIndex.RIGHT_RING,
    InspireJointIndex.RIGHT_THUMB_BEND,
    InspireJointIndex.RIGHT_THUMB_ROTATION,
]

INSPIRE_JOINT_RANGES = [
    (0.0, 1.7),
    (0.0, 1.7),
    (0.0, 1.7),
    (0.0, 1.7),
    (0.0, 0.5),
    (-0.1, 1.3),
]

CONTROLLER_TO_INSPIRE_PITCH_SCALES = [
    0.85,
    0.8,
    0.8,
    0.8,
    0.26,
]

CONTROLLER_TO_INSPIRE_MAP_THUMB_ROTATION = 1.3

INSPIRE_CURRENT_ANGLE_REG_ADDRESS = 1546
INSPIRE_TARGET_ANGLE_REG_ADDRESS = 1486
INSPIRE_FORCE_THRESHOLD_REG_ADDRESS = 1498

INSPIRE_FORCE_THRESHOLD = 200  # gramm
