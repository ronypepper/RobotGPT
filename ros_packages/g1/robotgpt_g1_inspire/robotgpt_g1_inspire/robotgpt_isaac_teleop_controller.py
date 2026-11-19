#!/usr/bin/env python3

"""
A ROS2 teleoperation controller for a Unitree G1 robot with Inspire 6DOF hands using Virtual Reality through Isaac ROS Teleop.
Allows collecting datasets in hdf5 format for training pi0.5 policies (openpi).

Controls:
 - Right primary (lower) button: Start teleoperation / Stop teleoperation and save dataset recording
 - Left primary (lower) button: Stop teleoperation and discard dataset recording
 - Right secondary (upper) button: Return robot and hands to home pose (only if teleoperation is not active)

Copyright (c) 2026 ronypepper.

License: Apache 2.0
"""

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import PoseArray, Pose
from std_msgs.msg import ByteMultiArray
from sensor_msgs.msg import Image, JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from unitree_hg.msg import LowState, LowCmd
from unitree_go.msg import MotorStates, MotorCmd, MotorCmds

from enum import Enum
import os
import sys
import time
import threading

import msgpack
import msgpack_numpy
import numpy as np
import h5py
from pymodbus.client import ModbusTcpClient

from transforms3d import quaternions

from openpi_image_tools import convert_image_for_openpi

from g1_inspire_data import *
from unitree_lowcmd_crc import get_unitree_lowcmd_crc


def decode_controller_data(msg: ByteMultiArray):
    return msgpack.unpackb(
        bytes([ab for a in msg.data for ab in a]),
        object_hook=msgpack_numpy.decode,
    )


def pose_to_matrix(pose: Pose):
    T = np.eye(4)

    q = [
        pose.orientation.w,
        pose.orientation.x,
        pose.orientation.y,
        pose.orientation.z
    ]
    T[:3, :3] = quaternions.quat2mat(q)

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

    w, x, y, z = quaternions.mat2quat(T[:3, :3])

    pose.orientation.x = x
    pose.orientation.y = y
    pose.orientation.z = z
    pose.orientation.w = w

    return pose


class TeleopControlMode(Enum):
    INITIALIZING = 0,
    IDLE = 1,
    TELEOPERATING = 2,
    MOVING_HOME = 3,
    SHUTTING_DOWN = 4


class RobotGptIsaacTeleopRosController(Node):
    def __init__(self):
        super().__init__("robotgpt_isaac_teleop_controller")

        # Configuration parameters
        dataset_path = self.declare_parameter("dataset_path", "/home/asl_team/RobotGPT/datasets/dataset.hdf5").value
        self.prompt = self.declare_parameter("prompt", "MISSING").value
        if self.prompt == "MISSING":
            raise ValueError("No prompt has been specified!")
        self.record_fps = self.declare_parameter("record_fps", 20).value
        self.mode = self.declare_parameter("mode", "g1_isaaclab_simulation").value
        self.use_real_hands = self.declare_parameter("use_real_hands", False).value
        self.disable_left_hand = self.declare_parameter("disable_left_hand", False).value
        self.disable_right_hand = self.declare_parameter("disable_right_hand", False).value 

        # Initialize inspire hands
        if self.use_real_hands:
            if not self.intialize_real_hands():
                self.destroy_node()
                rclpy.shutdown()
                sys.exit(1)

        # Teleoperation state
        self.teleop_control_mode = TeleopControlMode.INITIALIZING
        self.left_T_ee_anchor_user_inv, self.left_T_ee_anchor_robot = None, None
        self.right_T_ee_anchor_user_inv, self.right_T_ee_anchor_robot = None, None

        # Controller input state
        self.last_right_primary_click, self.last_right_secondary_click = 0.0, 0.0
        self.last_left_T_ee_user, self.last_left_T_ee_robot = None, None
        self.last_right_T_ee_user, self.last_right_T_ee_robot = None, None

        # Initialize dataset
        if os.path.exists(dataset_path):
            self.get_logger().warning(
                "\n------------------------- WARNING -------------------------\n"
                f"     {dataset_path} already exists\n"
                "                   Nothing will be recorded"
                "\n------------------------- WARNING -------------------------\n")
            self.dataset_file = None
        else:
            self.dataset_file = h5py.File(dataset_path, "w")
            self.dataset_file.create_group("data")
            self.dataset_file["data"].attrs["prompt"] = self.prompt
            self.dataset_file["data"].attrs["fps"] = self.record_fps
            self.dataset_file["data"].attrs["from_real_robot"] = True
        self.dataset = None
        self.dataset_id = 0
        self.demo_left_joint_obs, self.demo_right_joint_obs, self.demo_joint_acts = [], [], []
        self.demo_table_imgs, self.demo_left_wrist_imgs, self.demo_right_wrist_imgs = [], [], []
        
        # This lock is actually redundant, as the default SingleThreadExecutor for ROS callbacks is used, which does not interleave any callbacks.
        # If actual ROS2 multithreading would be used, other objects concerning robot state and actions would require synchronization as well.
        self.dataset_lock = threading.Lock()

        # Observation buffers
        self.last_left_arm_obs, self.last_left_hand_obs, self.last_left_arm_act, self.last_left_hand_act = None, None, None, None
        self.last_right_arm_obs, self.last_right_hand_obs, self.last_right_arm_act, self.last_right_hand_act = None, None, None, None
        self.last_table_img, self.last_left_wrist_img, self.last_right_wrist_img = None, None, None

        # ROS subscribers
        self.teleop_ee_sub = self.create_subscription(PoseArray, "xr_teleop/ee_poses", self.teleop_ee_pose_callback, 1)
        self.controller_data_sub = self.create_subscription(ByteMultiArray, "xr_teleop/controller_data", self.controller_data_callback, 1)
        self.table_camera_sub = self.create_subscription(Image, "table_camera/color/image_raw", self.table_camera_callback, 1)
        if not self.disable_left_hand:
            self.left_wrist_camera_sub = self.create_subscription(Image, "left_wrist_camera/color/image_raw", self.left_wrist_camera_callback, 1)
        if not self.disable_right_hand:
            self.right_wrist_camera_sub = self.create_subscription(Image, "right_wrist_camera/color/image_raw", self.right_wrist_camera_callback, 1)
        self.ik_joints_sub = self.create_subscription(JointState, "computed_joints", self.ik_joints_callback, 10)
        self.ee_poses_sub = self.create_subscription(PoseArray, "ee_poses", self.ee_poses_callback, 1)
        
        if self.use_real_hands:
            self.inspire_state_timer = self.create_timer(1.0 / (2.0 * self.record_fps), self.inspire_state_callback)
        else:
            self.inspire_state_sub = self.create_subscription(MotorStates, "/inspire/state", self.inspire_state_sim_callback, 1)
        
        if self.mode == "g1_internal_controller":
            self.g1_state_sub = self.create_subscription(LowState, "/rt/lowstate", self.g1_state_callback, 1)
        elif self.mode == "g1_isaaclab_simulation":
            self.g1_state_sub = self.create_subscription(LowState, "/lowstate", self.g1_state_callback, 1)
        elif self.mode == "g1_proprietary_controller":
            self.g1_state_sub = self.create_subscription(JointState, "/joint_states", self.g1_state_proprietary_callback, 1)

        # ROS publishers
        self.g1_ee_target_pub = self.create_publisher(PoseArray, "ee_targets", 10)
        self.g1_joint_state_pub = self.create_publisher(JointState, "current_joints", 10)
        
        if not self.use_real_hands:
            self.inspire_cmd_pub = self.create_publisher(MotorCmds, "/inspire/cmd", 10)
        
        if self.mode == "g1_internal_controller":
            self.g1_cmd_pub = self.create_publisher(LowCmd, "/rt/armsdk", 10)
        elif self.mode == "g1_isaaclab_simulation":
            self.g1_cmd_pub = self.create_publisher(LowCmd, "/lowcmd", 10)
        elif self.mode == "g1_proprietary_controller":
            self.g1_cmd_pub = self.create_publisher(JointTrajectory, "/g1_ub_joint_trajectory_controller/joint_trajectory", 10)
        
        # Prepare command messages
        if self.mode == "g1_internal_controller" or self.mode == "g1_isaaclab_simulation":
            self.arm_cmd_msg = LowCmd()
            self.arm_cmd_msg.mode_pr = 0  # Parallel mechanism (ankle and waist)
            self.arm_cmd_msg.mode_machine = 5  # G1 29-Dof
            for i, (kp, kd) in G1_DYNAMICS.items():
                self.arm_cmd_msg.motor_cmd[i].kp = kp
                self.arm_cmd_msg.motor_cmd[i].kd = kd
                self.arm_cmd_msg.motor_cmd[i].q = 0.0  # 0.0 on all joints is home pose
            self.arm_cmd_msg.motor_cmd[G1JointIndex.CONTROL_WEIGHT].q = 0.0
        else:
            self.arm_cmd_msg = JointTrajectory()
            self.arm_cmd_msg.header.stamp.sec = 0
            self.arm_cmd_msg.header.stamp.nanosec = 0
            self.arm_cmd_msg.header.frame_id = ''
            self.arm_cmd_msg.joint_names = [
                'left_shoulder_pitch_joint',
                'left_shoulder_roll_joint',
                'left_shoulder_yaw_joint',
                'left_elbow_joint',
                'left_wrist_roll_joint',
                'left_wrist_pitch_joint',
                'left_wrist_yaw_joint',
                'right_shoulder_pitch_joint',
                'right_shoulder_roll_joint',
                'right_shoulder_yaw_joint',
                'right_elbow_joint',
                'right_wrist_roll_joint',
                'right_wrist_pitch_joint',
                'right_wrist_yaw_joint',
            ]
            self.arm_cmd_msg

        if not self.use_real_hands:
            self.hand_cmd_msg = MotorCmds()
            for i in range(12):
                self.hand_cmd_msg.cmds.append(MotorCmd())

        # Initialize robot
        self.get_logger().info("INFO: Initializing G1...")
        self.g1_control_weight = 0.0
        self.g1_startup_last_time = time.monotonic()
        self.g1_startup_timer = self.create_timer(1.0 / 50.0, self.g1_startup_callback)

    def shutdown(self):
        self.get_logger().info("INFO: Shutting down teleop node...")
        self.teleop_control_mode = TeleopControlMode.SHUTTING_DOWN

        if self.mode == "g1_internal_controller":
            # Slowly decrease weight on arm commands for G1
            while self.g1_control_weight > 0.0:
                time.sleep(G1_CONTROL_DT)

                self.g1_control_weight -= (G1_MAX_CONTROL_WEIGHT / G1_CONTROL_WEIGHT_ADJUST_DURATION) * G1_CONTROL_DT
                self.g1_control_weight = max(self.g1_control_weight, 0.0)
                self.arm_cmd_msg.motor_cmd[G1JointIndex.CONTROL_WEIGHT].q = self.g1_control_weight
                self.arm_cmd_msg.crc = get_unitree_lowcmd_crc(self.arm_cmd_msg)
                self.g1_cmd_pub.publish(self.arm_cmd_msg)

        if self.use_real_hands:
            if not self.disable_left_hand:
                self.inspire_left_client.close()
            if not self.disable_right_hand:
                self.inspire_right_client.close()

        if self.dataset_file:
            self.dataset_file.close()
        
        self.get_logger().info("INFO: G1 is released")

    def intialize_real_hands(self) -> bool:
        if not self.disable_left_hand:
            self.inspire_left_client = ModbusTcpClient(host="192.168.123.210", port=6000)
            if not self.inspire_left_client.connect():
                self.get_logger().error("ERROR: cannot connect to left inspire hand")
                return False
            response = self.inspire_left_client.write_registers(INSPIRE_FORCE_THRESHOLD_REG_ADDRESS, 
                                                                [INSPIRE_FORCE_THRESHOLD for _ in range(6)])
            if response.isError():
                self.get_logger().error("Error: failed to set force thresholds on left Inspire hand")
                return False

        if not self.disable_right_hand:
            self.inspire_right_client = ModbusTcpClient(host="192.168.123.211", port=6000)
            if not self.inspire_right_client.connect():
                self.get_logger().error("ERROR: cannot connect to right inspire hand")
                return False
            response = self.inspire_right_client.write_registers(INSPIRE_FORCE_THRESHOLD_REG_ADDRESS, 
                                                                [INSPIRE_FORCE_THRESHOLD for _ in range(6)])
            if response.isError():
                self.get_logger().error("Error: failed to set force thresholds on right Inspire hand")
                return False

        return True

    def g1_startup_callback(self):
        if self.teleop_control_mode is not TeleopControlMode.INITIALIZING:
            self.g1_startup_timer.cancel()
            self.get_logger().warning("WARNING: G1 initialization was cancelled")
            return
        
        now = time.monotonic()
        dt = now - self.g1_startup_last_time
        self.g1_startup_last_time = now

        # Wait for first G1 state message to ensure robot connection is active
        if self.last_left_arm_obs is None or self.last_right_arm_obs is None:
            return

        if self.mode == "g1_internal_controller":
            # Slowly increase weight on arm commands for G1
            self.g1_control_weight += (G1_MAX_CONTROL_WEIGHT / G1_CONTROL_WEIGHT_ADJUST_DURATION) * dt
            self.g1_control_weight = min(self.g1_control_weight, G1_MAX_CONTROL_WEIGHT)
            self.arm_cmd_msg.motor_cmd[G1JointIndex.CONTROL_WEIGHT].q = self.g1_control_weight
            self.arm_cmd_msg.crc = get_unitree_lowcmd_crc(self.arm_cmd_msg)
            self.g1_cmd_pub.publish(self.arm_cmd_msg)
            if self.g1_control_weight != G1_MAX_CONTROL_WEIGHT:
                return
        
        # Check if initial end-effector poses of user and robot have been acquired
        if self.last_left_T_ee_robot is None or self.last_right_T_ee_robot is None:
            self.get_logger().info("INFO: Waiting for robot end-effector poses...", throttle_duration_sec=2.0)
            return
        self.get_logger().info("INFO: G1 is ready", once=True)
        if self.last_left_T_ee_user is None or self.last_right_T_ee_user is None:
            self.get_logger().info("INFO: Waiting for user end-effector poses...", throttle_duration_sec=2.0)
            return
        
        self.g1_startup_timer.cancel()
        self.move_robot_to_home_pose()

    def teleop_ee_pose_callback(self, msg: PoseArray):
        self.last_left_T_ee_user = pose_to_matrix(msg.poses[0])
        self.last_right_T_ee_user = pose_to_matrix(msg.poses[1])
        if self.teleop_control_mode is TeleopControlMode.TELEOPERATING:
            self.apply_user_ee_pose_to_robot()

    def controller_data_callback(self, msg: ByteMultiArray):
        data = decode_controller_data(msg)

        if self.teleop_control_mode is TeleopControlMode.IDLE:
            if data['right_primary_click'] == 1.0 and self.last_right_primary_click == 0.0:  # Start teleoperation and dataset recording
                self.start_teleoperation()
            elif data['right_secondary_click'] == 1.0 and self.last_right_secondary_click == 0.0:  # Move robot to home pose
                self.move_robot_to_home_pose()
        elif self.teleop_control_mode is TeleopControlMode.TELEOPERATING:
            # Command gripper
            left_trigger_pos, right_trigger_pos = data['left_trigger_value'], data['right_trigger_value']
            self.command_inspire_hand_grasp(left_trigger_pos, right_trigger_pos)

            if data['right_primary_click'] == 1.0 and self.last_right_primary_click == 0.0:  # Stop teleoperation and save dataset recording
                self.stop_teleoperation(discard_dataset=False)
            elif data['right_secondary_click'] == 1.0 and self.last_right_secondary_click == 0.0:  # Stop teleoperation and discard dataset recording
                self.stop_teleoperation(discard_dataset=True)

        # Store last inputs
        self.last_right_primary_click = data['right_primary_click']
        self.last_right_secondary_click = data['right_secondary_click']

    def g1_state_callback(self, msg: LowState):
        self.last_left_arm_obs = [msg.motor_state[i].q for i in G1_LEFT_ARM_INDICES]
        self.last_right_arm_obs = [msg.motor_state[i].q for i in G1_RIGHT_ARM_INDICES]
        self.publish_joint_state_to_ik_node()

    def g1_state_proprietary_callback(self, msg: JointState):
        self.last_left_arm_obs = [msg.position[G1_PROPRIETARY_JOINT_MAPPING[i]] for i in G1_LEFT_ARM_INDICES]
        self.last_right_arm_obs = [msg.position[G1_PROPRIETARY_JOINT_MAPPING[i]] for i in G1_RIGHT_ARM_INDICES]
        self.publish_joint_state_to_ik_node()

    def publish_joint_state_to_ik_node(self):
        joint_state_msg = JointState()
        joint_state_msg.header.stamp = self.get_clock().now().to_msg()
        joint_state_msg.position = self.last_left_arm_obs + self.last_right_arm_obs
        self.g1_joint_state_pub.publish(joint_state_msg)

    def inspire_state_callback(self):
        if not self.disable_left_hand:
            response = self.inspire_left_client.read_holding_registers(address=INSPIRE_CURRENT_ANGLE_REG_ADDRESS, count=6)
            if response.isError():
                self.get_logger().error("Error: failed reading left Inspire hand state", throttle_duration_sec=0.5)
            else:
                self.last_left_hand_obs = [float(response.registers[i]) / 1000.0 for i in INSPIRE_RIGHT_INDICES]  # INSPIRE_RIGHT_INDICES is correct (because they range from 0 to 5)!
        
        if not self.disable_right_hand:
            response = self.inspire_right_client.read_holding_registers(address=INSPIRE_CURRENT_ANGLE_REG_ADDRESS, count=6)
            if response.isError():
                self.get_logger().error("Error: failed reading right Inspire hand state", throttle_duration_sec=0.5)
            else:
                self.last_right_hand_obs = [float(response.registers[i]) / 1000.0 for i in INSPIRE_RIGHT_INDICES]

    def inspire_state_sim_callback(self, msg: MotorStates):
        self.last_left_hand_obs = [msg.states[i].q for i in INSPIRE_LEFT_INDICES]
        self.last_right_hand_obs = [msg.states[i].q for i in INSPIRE_RIGHT_INDICES]

    def table_camera_callback(self, msg: Image):
        self.last_table_img = msg

    def left_wrist_camera_callback(self, msg: Image):
        self.last_left_wrist_img = msg

    def right_wrist_camera_callback(self, msg: Image):
        self.last_right_wrist_img = msg

    def ik_joints_callback(self, msg: JointState):
        self.last_left_arm_act = list(msg.position[:7])
        self.last_right_arm_act = list(msg.position[7:14])

    def ee_poses_callback(self, msg: PoseArray):
        self.last_left_T_ee_robot = pose_to_matrix(msg.poses[0])
        self.last_right_T_ee_robot = pose_to_matrix(msg.poses[1])

    def start_teleoperation(self):
        self.get_logger().info("INFO: Starting teleoperation.")
        
        # Re-anchor to the current VR and robot poses.
        self.left_T_ee_anchor_user_inv = np.linalg.inv(self.last_left_T_ee_user)
        self.right_T_ee_anchor_user_inv = np.linalg.inv(self.last_right_T_ee_user)
        self.left_T_ee_anchor_robot = self.last_left_T_ee_robot.copy()
        self.right_T_ee_anchor_robot = self.last_right_T_ee_robot.copy()

        # Enable teleoperation
        self.teleop_control_mode = TeleopControlMode.TELEOPERATING
        self.command_g1_ik_arm_actions_timer = self.create_timer(G1_CONTROL_DT, self.command_g1_ik_arm_actions_callback)
        
        self.start_recording()

    def stop_teleoperation(self, discard_dataset: bool):
        self.get_logger().info("INFO: Stopping teleoperation.")
        self.command_g1_ik_arm_actions_timer.cancel()
        self.teleop_control_mode = TeleopControlMode.IDLE
        if discard_dataset:
            self.discard_dataset()
        else:
            self.save_dataset()

    def apply_user_ee_pose_to_robot(self):
        # Compute delta
        left_T_ee_delta = self.left_T_ee_anchor_user_inv @ self.last_left_T_ee_user
        right_T_ee_delta = self.right_T_ee_anchor_user_inv @ self.last_right_T_ee_user

        # Apply delta
        left_T_ee_robot_target = self.left_T_ee_anchor_robot @ left_T_ee_delta
        right_T_ee_robot_target = self.right_T_ee_anchor_robot @ right_T_ee_delta

        msg = PoseArray()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.poses.append(matrix_to_pose(left_T_ee_robot_target))
        msg.poses.append(matrix_to_pose(right_T_ee_robot_target))
        self.g1_ee_target_pub.publish(msg)

    def command_g1_ik_arm_actions_callback(self):
        if self.teleop_control_mode is not TeleopControlMode.TELEOPERATING:
            self.command_g1_ik_arm_actions_timer.cancel()
            self.get_logger().warning("WARNING: G1 teleoperation was cancelled")
            return

        if self.last_left_arm_act is None or self.last_right_arm_act is None:
            return  # No IK solution computed yet
        
        # Limit joint speeds
        if self.mode == "g1_isaaclab_simulation":
            self.send_arm_joint_commands(self.last_left_arm_act + self.last_right_arm_act)
        else:
            # Five lines below taken from G1_29_ArmController.clip_arm_q_target() in https://github.com/unitreerobotics/xr_teleoperate/blob/main/teleop/robot_control/robot_arm.py
            cur_q = np.array(self.last_left_arm_obs + self.last_right_arm_obs)
            target_q = np.array(self.last_left_arm_act + self.last_right_arm_act)
            delta = target_q - cur_q
            motion_scale = np.max(np.abs(delta)) / (G1_TELEOPERATION_MAX_JOINT_SPEED * G1_CONTROL_DT)
            clipped_target_q = cur_q + delta / max(motion_scale, 1.0)            

            self.send_arm_joint_commands(clipped_target_q.tolist())

    def send_arm_joint_commands(self, joint_commands):
        if self.mode == "g1_internal_controller" or self.mode == "g1_isaaclab_simulation":
            for idx, i in enumerate(G1_LEFT_ARM_INDICES + G1_RIGHT_ARM_INDICES):
                self.arm_cmd_msg.motor_cmd[i].q = float(joint_commands[idx])
            self.arm_cmd_msg.crc = get_unitree_lowcmd_crc(self.arm_cmd_msg)
            self.g1_cmd_pub.publish(self.arm_cmd_msg)
        else:
            point = JointTrajectoryPoint()
            point.positions = joint_commands
            point.time_from_start.sec = 0
            point.time_from_start.nanosec = 200_000_000  # Reach the target positions after 0.2 seconds
            self.arm_cmd_msg.points = [point]
            self.g1_cmd_pub.publish(self.arm_cmd_msg)

    def command_inspire_hand_grasp(self, left_grasp_position: float, right_grasp_position: float):
        self.last_left_hand_act = left_grasp_position * -2 + 1
        self.last_right_hand_act = right_grasp_position * -2 + 1

        left_hand_action = [left_grasp_position * CONTROLLER_TO_INSPIRE_PITCH_SCALES[i] for i in range(5)]
        left_hand_action.append(CONTROLLER_TO_INSPIRE_MAP_THUMB_ROTATION)
        for i, (jmin, jmax) in enumerate(INSPIRE_JOINT_RANGES):
            left_hand_action[i] = min(max((jmax - left_hand_action[i]) / (jmax - jmin), 0.0), 1.0)

        right_hand_action = [right_grasp_position * CONTROLLER_TO_INSPIRE_PITCH_SCALES[i] for i in range(5)]
        right_hand_action.append(CONTROLLER_TO_INSPIRE_MAP_THUMB_ROTATION)
        for i, (jmin, jmax) in enumerate(INSPIRE_JOINT_RANGES):
            right_hand_action[i] = min(max((jmax - right_hand_action[i]) / (jmax - jmin), 0.0), 1.0)

        # Command hand
        if self.use_real_hands:
            if not self.disable_left_hand:
                left_hand_action = [int(x * 1000) & 0xFFFF for x in left_hand_action]
                resorted_left_hand_action = [1000 for _ in range(6)]
                for i, idx in enumerate(INSPIRE_RIGHT_INDICES):
                    resorted_left_hand_action[idx] = left_hand_action[i]
                response = self.inspire_left_client.write_registers(INSPIRE_TARGET_ANGLE_REG_ADDRESS, resorted_left_hand_action)
                if response.isError():
                    self.get_logger().error("Error: failed to write commands to left Inspire hand", throttle_duration_sec=0.5)
            
            if not self.disable_right_hand:
                right_hand_action = [int(x * 1000) & 0xFFFF for x in right_hand_action]
                resorted_right_hand_action = [1000 for _ in range(6)]
                for i, idx in enumerate(INSPIRE_RIGHT_INDICES):
                    resorted_right_hand_action[idx] = right_hand_action[i]
                response = self.inspire_right_client.write_registers(INSPIRE_TARGET_ANGLE_REG_ADDRESS, resorted_right_hand_action)
                if response.isError():
                    self.get_logger().error("Error: failed to write commands to right Inspire hand", throttle_duration_sec=0.5)
        else:
            for idx, i in enumerate(INSPIRE_LEFT_INDICES):
                self.hand_cmd_msg.cmds[i].q = left_hand_action[idx]
            for idx, i in enumerate(INSPIRE_RIGHT_INDICES):
                self.hand_cmd_msg.cmds[i].q = right_hand_action[idx]
            self.inspire_cmd_pub.publish(self.hand_cmd_msg)

    def move_robot_to_home_pose(self):
        self.get_logger().info("INFO: Moving robot arms and hands to home pose.")
        self.teleop_control_mode = TeleopControlMode.MOVING_HOME
        self.command_inspire_hand_grasp(0.0, 0.0)
        self.moving_home_last_time = time.monotonic()
        self.moving_home_timer = self.create_timer(1.0 / 50.0, self.move_arms_to_home_pose_callback)

    def move_arms_to_home_pose_callback(self):
        if self.teleop_control_mode is not TeleopControlMode.MOVING_HOME:
            self.moving_home_timer.cancel()
            self.get_logger().warning("WARNING: G1 home move was cancelled")
            return
        
        now = time.monotonic()
        dt = now - self.moving_home_last_time
        self.moving_home_last_time = now

        # Execute arm action on robot
        arm_observations = self.last_left_arm_obs + self.last_right_arm_obs
        joint_commands = []
        for idx, _ in enumerate(G1_LEFT_ARM_INDICES + G1_RIGHT_ARM_INDICES):
            if self.mode == "g1_isaaclab_simulation":
                joint_commands.append(0.0)
            else:
                cur_q = arm_observations[idx]
                max_delta = G1_MOVING_HOME_JOINT_SPEED * dt
                joint_commands.append(0.0 if abs(cur_q) <= max_delta else cur_q - max_delta * float(np.sign(cur_q)))
                
        self.send_arm_joint_commands(joint_commands)

        # Check if home pose was commanded and arm is reasonably close
        robot_at_home = np.all(np.abs(arm_observations) <= G1_HOME_POSE_TOLERANCE)
        if robot_at_home or self.mode == "g1_isaaclab_simulation":
            self.teleop_control_mode = TeleopControlMode.IDLE
            self.moving_home_timer.cancel()
            self.get_logger().info("INFO: G1 has reached home")

    def start_recording(self):
        if not self.dataset_file:
            self.get_logger().warning(
                "\n------------------------- WARNING -------------------------\n"
                "                   Nothing will be recorded"
                "\n------------------------- WARNING -------------------------\n")
            return
        self.get_logger().info(
            "\n--------------------------------------------------\n"
            "                DEMO RECORDING STARTED"
            "\n--------------------------------------------------\n")
        self.clear_recordings()
        self.recording_timer = self.create_timer(1.0 / self.record_fps, self.record_data)

    def record_data(self):
        with self.dataset_lock:
            if any(x is None for x in [self.last_left_arm_obs, self.last_left_arm_act, self.last_left_hand_act,
                                       self.last_right_arm_obs, self.last_right_arm_act, self.last_right_hand_act,
                                       self.last_table_img]):
                return
            if not self.disable_left_hand:
                if self.last_left_hand_obs is None or self.last_left_wrist_img is None:
                    return
            if not self.disable_right_hand:
                if self.last_right_hand_obs is None or self.last_right_wrist_img is None:
                    return
            
            if not self.disable_left_hand:
                self.demo_left_joint_obs.append(np.array(self.last_left_arm_obs + self.last_left_hand_obs, dtype=np.float32))
                self.demo_left_wrist_imgs.append(convert_image_for_openpi(self.last_left_wrist_img)[::-1,::-1,:])  # flip to account for upside-down camera mount
            else:
                self.demo_left_joint_obs.append(np.array(self.last_left_arm_obs + [0.998, 0.998, 0.999, 0.998, 0.998, 0.0], dtype=np.float32))
                self.demo_left_wrist_imgs.append(convert_image_for_openpi(self.last_table_img))  # just for simplicity store table image again
            
            if not self.disable_right_hand:
                self.demo_right_joint_obs.append(np.array(self.last_right_arm_obs + self.last_right_hand_obs, dtype=np.float32))
                self.demo_right_wrist_imgs.append(convert_image_for_openpi(self.last_right_wrist_img)[::-1,::-1,:])  # flip to account for upside-down camera mount
            else:
                self.demo_right_joint_obs.append(np.array(self.last_right_arm_obs + [0.998, 0.998, 0.999, 0.998, 0.998, 0.0], dtype=np.float32))
                self.demo_right_wrist_imgs.append(convert_image_for_openpi(self.last_table_img))  # just for simplicity store table image again
                
            self.demo_joint_acts.append(np.array(self.last_left_arm_act + self.last_right_arm_act + [self.last_left_hand_act, self.last_right_hand_act], dtype=np.float32))
            self.demo_table_imgs.append(convert_image_for_openpi(self.last_table_img))

    def clear_recordings(self):
        with self.dataset_lock:
            self.demo_left_joint_obs, self.demo_right_joint_obs, self.demo_joint_acts = [], [], []
            self.demo_table_imgs, self.demo_left_wrist_imgs, self.demo_right_wrist_imgs = [], [], []
            self.last_left_arm_obs, self.last_left_hand_obs, self.last_left_arm_act, self.last_left_hand_act = None, None, None, None
            self.last_right_arm_obs, self.last_right_hand_obs, self.last_right_arm_act, self.last_right_hand_act = None, None, None, None
            self.last_table_img, self.last_left_wrist_img, self.last_right_wrist_img = None, None, None

    def save_dataset(self):
        if not self.dataset_file:
            return
        self.get_logger().info(
            "\n--------------------------------------------------\n"
            f"             {self.dataset_id + 1} DEMO RECORDING(S) SAVED"
            "\n--------------------------------------------------\n")
        self.recording_timer.cancel()

        dataset_name = f"data/demo_{self.dataset_id}"
        with self.dataset_lock:
            self.dataset_file[f"{dataset_name}/obs/left_joint_pos"] = list(self.demo_left_joint_obs)
            self.dataset_file[f"{dataset_name}/obs/right_joint_pos"] = list(self.demo_right_joint_obs)
            self.dataset_file[f"{dataset_name}/obs/table_img"] = list(self.demo_table_imgs)
            self.dataset_file[f"{dataset_name}/obs/left_wrist_img"] = list(self.demo_left_wrist_imgs)
            self.dataset_file[f"{dataset_name}/obs/right_wrist_img"] = list(self.demo_right_wrist_imgs)
            self.dataset_file[f"{dataset_name}/processed_actions"] = list(self.demo_joint_acts)
            self.dataset_file[f"{dataset_name}/actions"] = list(self.demo_joint_acts)
            self.dataset_file[dataset_name].attrs["num_samples"] = len(self.demo_left_joint_obs)
        
        self.dataset_id += 1

    def discard_dataset(self):
        if not self.dataset_file:
            return
        self.get_logger().info(
            "\n--------------------------------------------------\n"
            "               DEMO RECORDING DISCARDED"
            "\n--------------------------------------------------\n")
        self.recording_timer.cancel()


def main(args=None):
    rclpy.init(args=args)

    try:
        teleop_controller = RobotGptIsaacTeleopRosController()
        rclpy.spin(teleop_controller)
    except Exception as e:
        if type(e) is not KeyboardInterrupt:
            teleop_controller.get_logger().error(f'Node failed with exception: {e}')
    finally:
        teleop_controller.shutdown()
        teleop_controller.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
