#!/usr/bin/env python3

"""
A ROS1 teleoperation controller for a Franka arm using Virtual Reality through Isaac ROS Teleop.
Allows collecting datasets in hdf5 format for training pi0.5 policies (openpi).

Controls:
 - Right primary (lower) button: Start teleoperation / Stop teleoperation and save dataset recording
 - Left primary (lower) button: Stop teleoperation and discard dataset recording
 - Right secondary (upper) button: Return Franka to home pose (only if teleoperation is not active)

Copyright (c) 2026 ronypepper.

License: Apache 2.0
"""

import rospy

from geometry_msgs.msg import PoseArray, Pose, PoseStamped
from std_msgs.msg import ByteMultiArray
from sensor_msgs.msg import Image, JointState
from franka_msgs.msg import FrankaState # type: ignore

import actionlib
from franka_gripper.msg import ( # type: ignore
    GraspAction, GraspGoal,
    MoveAction, MoveGoal
)

from std_srvs.srv import SetBool

import tf.transformations as tft

from enum import Enum
import os
import threading

import msgpack
import msgpack_numpy
import numpy as np
import h5py

from robotgpt.common.openpi_image_tools import convert_image_for_openpi


EXAMPLE_CARTESIAN_CONTROLLER_IN_USE = False


def decode_controller_data(msg: ByteMultiArray):
    payload = bytes(
            x if x >= 0 else x + 256
            for x in msg.data
        )
    return msgpack.unpackb(
        payload,
        object_hook=msgpack_numpy.decode,
    )


def pose_to_matrix(pose: Pose):
    T = np.eye(4)

    q = [
        pose.orientation.x,
        pose.orientation.y,
        pose.orientation.z,
        pose.orientation.w
    ]
    T[:3, :3] = tft.quaternion_matrix(q)[:3, :3]

    T[:3, 3] = [
        pose.position.x,
        pose.position.y,
        pose.position.z
    ]

    return T


def matrix_to_pose(T: np.ndarray):
    pose = Pose()

    pose.position.x = T[0, 3]
    pose.position.y = T[1, 3]
    pose.position.z = T[2, 3]

    q = tft.quaternion_from_matrix(T)

    pose.orientation.x = q[0]
    pose.orientation.y = q[1]
    pose.orientation.z = q[2]
    pose.orientation.w = q[3]

    return pose


class TeleopControlMode(Enum):
    IDLE = 1,
    TELEOPERATING = 2,


class RobotGptIsaacTeleopRosController:
    def __init__(self):
        rospy.init_node("robotgpt_isaac_teleop_controller")

        # Configuration parameters
        publish_ee_pose_to_marker = rospy.get_param("~publish_ee_pose_to_marker", False)
        self.gripper_max_effort = rospy.get_param("~gripper_max_effort", 1.0)
        dataset_path = rospy.get_param("~dataset_path", "/home/asl_team/RobotGPT/datasets/dataset.hdf5")
        prompt = rospy.get_param("~prompt", "Pick up the blue cube and drop it in the bin")
        self.record_fps = rospy.get_param("~record_fps", 20)

        # Teleoperation state
        self.teleop_control_mode = TeleopControlMode.IDLE
        self.T_ee_anchor_user_inv = None
        self.T_ee_anchor_robot = None

        # Controller input state
        self.last_right_primary_click, self.last_right_secondary_click = 0.0, 0.0
        self.last_T_ee_user, self.last_T_ee_robot = None, None

        # Initialize dataset
        if os.path.exists(dataset_path):
            print(
                "\n------------------------- WARNING -------------------------\n"
                f"     {dataset_path} already exists\n"
                "                   Nothing will be recorded"
                "\n------------------------- WARNING -------------------------\n")
            self.dataset_file = None
        else:
            self.dataset_file = h5py.File(dataset_path, "w")
            self.dataset_file.create_group("data")
            self.dataset_file["data"].attrs["prompt"] = prompt
            self.dataset_file["data"].attrs["fps"] = self.record_fps
            self.dataset_file["data"].attrs["from_real_robot"] = True
        self.dataset = None
        self.dataset_id = 0
        self.demo_joint_obs, self.demo_table_imgs, self.demo_wrist_imgs, self.demo_joint_acts = [], [], [], []
        self.last_joint_obs, self.last_joint_acts = None, None
        self.last_gripper_obs, self.last_gripper_act = None, None
        self.last_table_img, self.last_wrist_img = None, None
        self.dataset_lock = threading.Lock()

        # ROS subscribers
        self.teleop_ee_sub = rospy.Subscriber("xr_teleop/ee_pose", PoseArray, self.teleop_ee_pose_callback)
        self.controller_data_sub = rospy.Subscriber("xr_teleop/controller_data", ByteMultiArray, self.controller_data_callback)
        self.franka_state_sub = rospy.Subscriber("franka_state_controller/franka_states", FrankaState, self.franka_state_callback)
        self.franka_gripper_sub = rospy.Subscriber("franka_gripper/joint_states", JointState, self.franka_gripper_callback)
        self.table_camera_sub = rospy.Subscriber("table_camera/color/image_raw", Image, self.table_camera_callback)
        self.wrist_camera_sub = rospy.Subscriber("wrist_camera/color/image_raw", Image, self.wrist_camera_callback)
        self.joint_actions_sub = rospy.Subscriber("impedance_controller/q_d", JointState, self.joint_actions_callback)

        # ROS publishers
        ee_target_topic = "cartesian_impedance_example_controller/equilibrium_pose" if EXAMPLE_CARTESIAN_CONTROLLER_IN_USE else "impedance_controller/cartesian_command"
        ee_target_topic = ee_target_topic if not publish_ee_pose_to_marker else "robotgpt_isaac_teleop_controller/ee_robot_target"
        self.franka_ee_target_pub = rospy.Publisher(ee_target_topic, PoseStamped, queue_size=10)
        joint_target_topic = "impedance_controller/joint_command" if not publish_ee_pose_to_marker else "robotgpt_isaac_teleop_controller/joint_target"
        self.franka_joint_target_pub = rospy.Publisher(joint_target_topic, JointState, queue_size=10)

        # ROS action servers
        if not publish_ee_pose_to_marker:
            # Grasp action (for closing the gripper)
            self.gripper_grasp_client = actionlib.SimpleActionClient(
                "franka_gripper/grasp",
                GraspAction,
            )
            rospy.loginfo("Waiting for Franka gripper grasp server...")
            if not self.gripper_grasp_client.wait_for_server(
                rospy.Duration(10.0)
            ):
                rospy.logerr("Franka gripper grasp server not available!")
                self.gripper_grasp_client = None
            else:
                rospy.loginfo("Franka gripper grasp server connected.")

            # Move action (for opening the gripper)
            self.gripper_move_client = actionlib.SimpleActionClient(
                "franka_gripper/move",
                MoveAction,
            )
            rospy.loginfo("Waiting for Franka gripper move server...")
            if not self.gripper_move_client.wait_for_server(
                rospy.Duration(10.0)
            ):
                rospy.logerr("Franka gripper move server not available!")
                self.gripper_move_client = None
            else:
                rospy.loginfo("Franka gripper move server connected.")
        else:
            self.gripper_grasp_client, self.gripper_move_client = None, None

        # ROS service clients
        rospy.wait_for_service("impedance_controller/set_cartesian_mode")
        self.franka_control_mode_srv = rospy.ServiceProxy("impedance_controller/set_cartesian_mode", SetBool)

        rospy.spin()

        if self.dataset_file:
            self.dataset_file.close()

    def teleop_ee_pose_callback(self, msg: PoseArray):
        self.last_T_ee_user = pose_to_matrix(msg.poses[1])  # right hand
        if self.teleop_control_mode is TeleopControlMode.TELEOPERATING:
            self.apply_user_ee_pose_to_robot(self.last_T_ee_user)

    def controller_data_callback(self, msg: ByteMultiArray):
        # Check if initial poses have been acquired
        if self.last_T_ee_user is None or self.last_T_ee_robot is None:
            return

        data = decode_controller_data(msg)

        if self.teleop_control_mode is TeleopControlMode.IDLE:
            if data[b'right_primary_click'] == 1.0 and self.last_right_primary_click == 0.0:  # Start teleoperation and dataset recording
                if not EXAMPLE_CARTESIAN_CONTROLLER_IN_USE:
                    try:
                        response = self.franka_control_mode_srv(True)  # Switch to cartesian control mode
                        if response.success:
                            self.initialize_teleoperation()
                            self.teleop_control_mode = TeleopControlMode.TELEOPERATING
                    except rospy.ServiceException:
                        pass
                else:
                    self.initialize_teleoperation()
                    self.teleop_control_mode = TeleopControlMode.TELEOPERATING
            elif data[b'right_secondary_click'] == 1.0 and self.last_right_secondary_click == 0.0:  # Move Franka to home pose
                if not EXAMPLE_CARTESIAN_CONTROLLER_IN_USE:
                    try:
                        response = self.franka_control_mode_srv(False)  # Switch to joint control mode
                        if response.success:
                            print("INFO: Moving Franka to home pose.")
                            self.move_robot_to_home_pose()
                    except rospy.ServiceException:
                        pass
        elif self.teleop_control_mode is TeleopControlMode.TELEOPERATING:
            # Command gripper
            trigger_pos = data[b'right_trigger_value']
            self.command_gripper(trigger_pos)

            if data[b'right_primary_click'] == 1.0 and self.last_right_primary_click == 0.0:  # Stop teleoperation and save dataset recording
                print("INFO: Stopping teleoperation.")
                self.save_dataset()
                self.teleop_control_mode = TeleopControlMode.IDLE
            elif data[b'right_secondary_click'] == 1.0 and self.last_right_secondary_click == 0.0:  # Stop teleoperation and discard dataset recording
                print("INFO: Stopping teleoperation.")
                self.discard_dataset()
                self.teleop_control_mode = TeleopControlMode.IDLE

        # Store last inputs
        self.last_right_primary_click = data[b'right_primary_click']
        self.last_right_secondary_click = data[b'right_secondary_click']

    def franka_state_callback(self, msg: FrankaState):
        self.last_T_ee_robot = np.array(msg.O_T_EE).reshape((4, 4), order='F')
        self.last_joint_obs = msg.q

    def franka_gripper_callback(self, msg):
        self.last_gripper_obs = msg.position

    def table_camera_callback(self, msg: Image):
        self.last_table_img = msg

    def wrist_camera_callback(self, msg: Image):
        self.last_wrist_img = msg

    def joint_actions_callback(self, msg: JointState):
        self.last_joint_acts = msg.position

    def initialize_teleoperation(self):
        print("INFO: Starting teleoperation.")
        
        # Re-anchor to the current VR and robot poses.
        self.T_ee_anchor_user_inv = np.linalg.inv(self.last_T_ee_user)
        self.T_ee_anchor_robot = self.last_T_ee_robot.copy()

        self.start_recording()

    def apply_user_ee_pose_to_robot(self, T_ee_user: np.ndarray):
        # Compute delta
        T_ee_delta = self.T_ee_anchor_user_inv @ T_ee_user

        # Invert translation along x and z
        T_ee_delta[2, 3] *= -1
        T_ee_delta[0, 3] *= -1

        # Invert rotation around x and z
        R = T_ee_delta[:3, :3]
        roll, pitch, yaw = tft.euler_from_matrix(R)
        roll *= -1
        yaw *= -1
        T_ee_delta[:3, :3] = tft.euler_matrix(
            roll,
            pitch,
            yaw
        )[:3, :3]

        # Apply delta
        T_ee_robot_target = self.T_ee_anchor_robot @ T_ee_delta

        msg = PoseStamped()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = "fr3_link0"
        msg.pose = matrix_to_pose(T_ee_robot_target)
        self.franka_ee_target_pub.publish(msg)

    def move_robot_to_home_pose(self):
        msg = JointState()
        msg.header.stamp = rospy.Time.now()
        msg.position = [0.0444, -0.1894, -0.1107, -2.5148, 0.0044, 2.3775, 0.6952]  # matches init_state in https://github.com/ronypepper/RobotGPT/blob/master/source/RobotGPT/RobotGPT/utils/robots/franka_single_arm.py
        self.franka_joint_target_pub.publish(msg)

        # Open gripper
        if self.gripper_grasp_client is None or self.gripper_move_client is None:
            return
        if self.gripper_grasp_client.get_state() == actionlib.GoalStatus.ACTIVE:
            self.gripper_grasp_client.cancel_goal()
        goal = MoveGoal()
        goal.width = 0.08
        goal.speed = 0.1
        self.gripper_move_client.send_goal(goal)
    
    def command_gripper(self, trigger_pos: float):
        if trigger_pos > 0.7:
            # Is gripper already closed?
            if self.last_gripper_act == 0.0:
                return
            self.last_gripper_act = 0.0

            # Close gripper
            if self.gripper_grasp_client is None or self.gripper_move_client is None:
                return
            if self.gripper_move_client.get_state() == actionlib.GoalStatus.ACTIVE:
                self.gripper_move_client.cancel_goal()
            goal = GraspGoal()
            goal.width = 0.0
            goal.speed = 0.1
            goal.force = self.gripper_max_effort
            goal.epsilon.inner = 1.0
            goal.epsilon.outer = 1.0
            self.gripper_grasp_client.send_goal(goal)
        elif trigger_pos < 0.3:
            # Is gripper already open?
            if self.last_gripper_act == 0.04:
                return
            self.last_gripper_act = 0.04

            # Open gripper
            if self.gripper_grasp_client is None or self.gripper_move_client is None:
                return
            if self.gripper_grasp_client.get_state() == actionlib.GoalStatus.ACTIVE:
                self.gripper_grasp_client.cancel_goal()
            goal = MoveGoal()
            goal.width = 0.08
            goal.speed = 0.1
            self.gripper_move_client.send_goal(goal)

    def start_recording(self):
        if not self.dataset_file:
            print(
                "\n------------------------- WARNING -------------------------\n"
                "                   Nothing will be recorded"
                "\n------------------------- WARNING -------------------------\n")
            return
        print(
            "\n--------------------------------------------------\n"
            "                DEMO RECORDING STARTED"
            "\n--------------------------------------------------\n")
        self.clear_recordings()
        self.recording_timer = rospy.Timer(rospy.Duration(1.0 / self.record_fps), self.record_data)

    def record_data(self, event):
        with self.dataset_lock:
            if any(x is None for x in [self.last_joint_obs, self.last_joint_acts, self.last_gripper_obs, 
                                    self.last_gripper_act, self.last_table_img, self.last_wrist_img]):
                return

            # We want to have 7 joints + 2 grippers to match the dataset format captured in Isaac Lab, where positions 
            # for indiviudal gripper fingers are recorded. last_gripper_obs already contains two entries.
            self.demo_joint_obs.append(np.array(self.last_joint_obs + self.last_gripper_obs, dtype=np.float32))
            self.demo_joint_acts.append(np.array(self.last_joint_acts + (self.last_gripper_act, self.last_gripper_act), dtype=np.float32))
            self.demo_table_imgs.append(convert_image_for_openpi(self.last_table_img))
            self.demo_wrist_imgs.append(convert_image_for_openpi(self.last_wrist_img))

    def clear_recordings(self):
        with self.dataset_lock:
            self.demo_joint_obs, self.demo_table_imgs, self.demo_wrist_imgs, self.demo_joint_acts = [], [], [], []
            self.last_joint_obs, self.last_joint_acts = None, None
            self.last_gripper_obs, self.last_gripper_act = None, None
            self.last_table_img, self.last_wrist_img = None, None

    def save_dataset(self):
        if not self.dataset_file:
            return
        print(
            "\n--------------------------------------------------\n"
            f"             {self.dataset_id + 1} DEMO RECORDING(S) SAVED"
            "\n--------------------------------------------------\n")
        self.recording_timer.shutdown()

        dataset_name = f"data/demo_{self.dataset_id}"
        with self.dataset_lock:
            self.dataset_file[f"{dataset_name}/obs/joint_pos"] = list(self.demo_joint_obs)
            self.dataset_file[f"{dataset_name}/obs/table_img"] = list(self.demo_table_imgs)
            self.dataset_file[f"{dataset_name}/obs/wrist_img"] = list(self.demo_wrist_imgs)
            self.dataset_file[f"{dataset_name}/processed_actions"] = list(self.demo_joint_acts)
            self.dataset_file[dataset_name].attrs["num_samples"] = len(self.demo_joint_obs)
        
        self.dataset_id += 1

    def discard_dataset(self):
        if not self.dataset_file:
            return
        print(
            "\n--------------------------------------------------\n"
            "               DEMO RECORDING DISCARDED"
            "\n--------------------------------------------------\n")
        self.recording_timer.shutdown()


if __name__ == "__main__":
    RobotGptIsaacTeleopRosController()
