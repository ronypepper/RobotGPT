#!/usr/bin/env python3

"""
This script executes the pi0.5 policy on a Unitree G1 robot with Inspire 6DOF hands using ROS2 and a remote policy server.

Portions of this file are derived from code in the openpi project:
https://github.com/Physical-Intelligence/openpi
Licensed under the Apache License, Version 2.0.

All other orignal parts of this file:
Copyright (c) 2026 ronypepper.
Licensed under the Apache License, Version 2.0.
"""

import rclpy
from rclpy.node import Node

from std_msgs.msg import String
from sensor_msgs.msg import Image, JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from unitree_hg.msg import LowState, LowCmd

from enum import Enum
import sys
import time
import contextlib
import signal

import numpy as np
from pymodbus.client import ModbusTcpClient

from openpi_image_tools import convert_image_for_openpi
from openpi_client import websocket_client_policy

from g1_inspire_data import *
from unitree_lowcmd_crc import get_unitree_lowcmd_crc

# We are using Ctrl+C to optionally terminate rollouts early -- however, if we press Ctrl+C while the policy server is
# waiting for a new action chunk, it will raise an exception and the server connection dies.
# This context manager temporarily prevents Ctrl+C and delays it after the server call is complete.
@contextlib.contextmanager
def prevent_keyboard_interrupt():
    """Temporarily prevent keyboard interrupts by delaying them until after the protected code."""
    interrupted = False
    original_handler = signal.getsignal(signal.SIGINT)

    def handler(signum, frame):
        nonlocal interrupted
        interrupted = True

    signal.signal(signal.SIGINT, handler)
    try:
        yield
    finally:
        signal.signal(signal.SIGINT, original_handler)
        if interrupted:
            raise KeyboardInterrupt


class OpenpiClientMode(Enum):
    INITIALIZING = 0,
    IDLE = 1,
    EXECUTING_POLICY = 2,
    MOVING_HOME = 3,
    SHUTTING_DOWN = 4


class RobotGptOpenPiROSClient(Node):
    def __init__(self):
        super().__init__("robotgpt_openpi_agent")

        # Configuration parameters
        self.prompt = self.declare_parameter("prompt", "MISSING").value
        if self.prompt == "MISSING":
            raise ValueError("No prompt has been specified!")
        policy_server_ip = self.declare_parameter("policy_server_ip", "localhost").value
        policy_server_port = self.declare_parameter("policy_server_port", 8000).value
        self.policy_frequency = self.declare_parameter("policy_frequency", 20).value
        self.open_loop_horizon = self.declare_parameter("open_loop_horizon", 16).value
        self.mode = self.declare_parameter("mode", "g1_internal_controller").value
        self.disable_left_hand = self.declare_parameter("disable_left_hand", False).value
        self.disable_right_hand = self.declare_parameter("disable_right_hand", False).value 

        # Client mode
        self.openpi_client_mode = OpenpiClientMode.INITIALIZING
        
        # Initialize inspire hands
        if not self.intialize_real_hands():
            self.destroy_node()
            rclpy.shutdown()
            sys.exit(1)

        # Observation buffers
        self.last_left_arm_obs, self.last_left_hand_obs = None, None
        self.last_right_arm_obs, self.last_right_hand_obs = None, None
        self.last_table_img, self.last_left_wrist_img, self.last_right_wrist_img = None, None, None

        # Action buffers
        self.left_arm_act, self.right_arm_act = None, None

        # ROS subscribers
        self.table_camera_sub = self.create_subscription(Image, "table_camera/color/image_raw", self.table_camera_callback, 1)
        if not self.disable_left_hand:
            self.left_wrist_camera_sub = self.create_subscription(Image, "left_wrist_camera/color/image_raw", self.left_wrist_camera_callback, 1)
        if not self.disable_right_hand:
            self.right_wrist_camera_sub = self.create_subscription(Image, "right_wrist_camera/color/image_raw", self.right_wrist_camera_callback, 1)
        self.inspire_state_timer = self.create_timer(1.0 / (2.0 * self.policy_frequency), self.inspire_state_callback)
        
        if self.mode == "g1_internal_controller":
            self.g1_state_sub = self.create_subscription(LowState, "/rt/lowstate", self.g1_state_callback, 1)
        elif self.mode == "g1_proprietary_controller":
            self.g1_state_sub = self.create_subscription(JointState, "/joint_states", self.g1_state_proprietary_callback, 1)

        self.client_cmd_sub = self.create_subscription(String, "/client_command", self.client_command_callback, 1)

        # ROS publishers
        if self.mode == "g1_internal_controller":
            self.g1_cmd_pub = self.create_publisher(LowCmd, "/rt/armsdk", 10)
        elif self.mode == "g1_proprietary_controller":
            self.g1_cmd_pub = self.create_publisher(JointTrajectory, "/g1_ub_joint_trajectory_controller/joint_trajectory", 10)


        # Prepare command messages
        if self.mode == "g1_internal_controller":
            self.arm_cmd_msg = LowCmd()
            self.arm_cmd_msg.mode_pr = 0  # Parallel mechanism (ankle and waist)
            self.arm_cmd_msg.mode_machine = 5  # G1 29-Dof
            for i, (kp, kd) in G1_DYNAMICS.items():
                self.arm_cmd_msg.motor_cmd[i].kp = kp
                self.arm_cmd_msg.motor_cmd[i].kd = kd
                self.arm_cmd_msg.motor_cmd[i].q = 0.0  # 0.0 on all joints is home pose
            self.arm_cmd_msg.motor_cmd[G1JointIndex.CONTROL_WEIGHT] = 0.0
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

        # Policy client
        self.get_logger().info(f"[INFO]: Attempting to connect to policy server at {policy_server_ip}:{policy_server_port}...")
        self.policy_client = websocket_client_policy.WebsocketClientPolicy(host=policy_server_ip, port=policy_server_port)
        self.pred_action_chunk, self.actions_from_chunk_completed = [], 0
        self.get_logger().info("[INFO]: Connected to policy server")

        # Initialize robot
        self.get_logger().info("INFO: Initializing G1...")
        self.g1_control_weight = 0.0
        self.g1_startup_last_time = time.monotonic()
        self.g1_startup_timer = self.create_timer(1.0 / 50.0, self.g1_startup_callback)

    def shutdown(self):
        self.get_logger().info("INFO: Shutting down openpi client node...")
        self.openpi_client_mode = OpenpiClientMode.SHUTTING_DOWN

        if self.mode == "g1_internal_controller":
            # Slowly decrease weight on arm commands for G1
            while self.g1_control_weight > 0.0:
                time.sleep(G1_CONTROL_DT)

                self.g1_control_weight -= (G1_MAX_CONTROL_WEIGHT / G1_CONTROL_WEIGHT_ADJUST_DURATION) * G1_CONTROL_DT
                self.g1_control_weight = max(self.g1_control_weight, 0.0)
                self.arm_cmd_msg.motor_cmd[G1JointIndex.CONTROL_WEIGHT] = self.g1_control_weight
                self.arm_cmd_msg.crc = get_unitree_lowcmd_crc(self.arm_cmd_msg)
                self.g1_cmd_pub.publish(self.arm_cmd_msg)

        if not self.disable_left_hand:
            self.inspire_left_client.close()
        if not self.disable_right_hand:
            self.inspire_right_client.close()
        
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
        if self.openpi_client_mode is not OpenpiClientMode.INITIALIZING:
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
        
        self.get_logger().info("INFO: G1 is ready")
        self.g1_startup_timer.cancel()
        self.move_robot_to_home_pose()

    def g1_state_callback(self, msg: LowState):
        self.last_left_arm_obs = [msg.motor_state[i].q for i in G1_LEFT_ARM_INDICES]
        self.last_right_arm_obs = [msg.motor_state[i].q for i in G1_RIGHT_ARM_INDICES]
    
    def g1_state_proprietary_callback(self, msg: JointState):
        self.last_left_arm_obs = [msg.position[G1_PROPRIETARY_JOINT_MAPPING[i]] for i in G1_LEFT_ARM_INDICES]
        self.last_right_arm_obs = [msg.position[G1_PROPRIETARY_JOINT_MAPPING[i]] for i in G1_RIGHT_ARM_INDICES]

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

    def table_camera_callback(self, msg: Image):
        self.last_table_img = msg

    def left_wrist_camera_callback(self, msg: Image):
        self.last_left_wrist_img = msg

    def right_wrist_camera_callback(self, msg: Image):
        self.last_right_wrist_img = msg

    def client_command_callback(self, msg: String):
        if msg.data == "START":
            if self.openpi_client_mode is OpenpiClientMode.IDLE:
                self.start_policy_execution()
            else:
                self.get_logger().warning(f"WARNING: Received START command but not in IDLE mode (current mode: {self.openpi_client_mode})")
        elif msg.data == "STOP":
            if self.openpi_client_mode is OpenpiClientMode.EXECUTING_POLICY:
                self.stop_policy_execution()
            else:
                self.get_logger().warning(f"WARNING: Received STOP command but not in EXECUTING_POLICY mode (current mode: {self.openpi_client_mode})")
        elif msg.data == "HOME":
            if self.openpi_client_mode is OpenpiClientMode.IDLE:
                self.move_robot_to_home_pose()
            else:
                self.get_logger().warning(f"WARNING: Received HOME command but not in IDLE mode (current mode: {self.openpi_client_mode})")
        else:
            self.get_logger().warning(f"WARNING: Received invalid command: {msg.data}")

    def start_policy_execution(self):
        self.get_logger().info("INFO: Starting policy execution")
        self.left_arm_act, self.right_arm_act = None, None
        self.openpi_client_mode = OpenpiClientMode.EXECUTING_POLICY
        self.command_g1_arm_actions_timer = self.create_timer(G1_CONTROL_DT, self.command_g1_arm_actions_callback)
        self.policy_loop_timer = self.create_timer(1.0 / self.policy_frequency, self.handle_policy)

    def stop_policy_execution(self):
        self.get_logger().info("INFO: Stopping policy execution")
        self.command_g1_arm_actions_timer.cancel()
        self.policy_loop_timer.cancel()
        self.openpi_client_mode = OpenpiClientMode.IDLE

    def command_g1_arm_actions_callback(self):
        if self.openpi_client_mode is not OpenpiClientMode.EXECUTING_POLICY:
            self.command_g1_arm_actions_timer.cancel()
            self.get_logger().warning("WARNING: G1 policy execution was cancelled")
            return

        if self.left_arm_act is None or self.right_arm_act is None:
            return  # No joint actions received from policy yet
        
        # Limit joint speeds - taken from G1_29_ArmController.clip_arm_q_target() in https://github.com/unitreerobotics/xr_teleoperate/blob/main/teleop/robot_control/robot_arm.py
        cur_q = np.array(self.last_left_arm_obs + self.last_right_arm_obs)
        target_q = np.array(self.left_arm_act + self.right_arm_act)
        delta = target_q - cur_q
        motion_scale = np.max(np.abs(delta)) / (G1_TELEOPERATION_MAX_JOINT_SPEED * G1_CONTROL_DT)
        clipped_target_q = cur_q + delta / max(motion_scale, 1.0)

        self.send_arm_joint_commands(clipped_target_q.tolist())

    def send_arm_joint_commands(self, joint_commands):
        if self.mode == "g1_internal_controller":
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
        left_hand_action = [left_grasp_position * CONTROLLER_TO_INSPIRE_PITCH_SCALES[i] for i in range(5)]
        left_hand_action.append(CONTROLLER_TO_INSPIRE_MAP_THUMB_ROTATION)
        for i, (jmin, jmax) in enumerate(INSPIRE_JOINT_RANGES):
            left_hand_action[i] = min(max((jmax - left_hand_action[i]) / (jmax - jmin), 0.0), 1.0)

        right_hand_action = [right_grasp_position * CONTROLLER_TO_INSPIRE_PITCH_SCALES[i] for i in range(5)]
        right_hand_action.append(CONTROLLER_TO_INSPIRE_MAP_THUMB_ROTATION)
        for i, (jmin, jmax) in enumerate(INSPIRE_JOINT_RANGES):
            right_hand_action[i] = min(max((jmax - right_hand_action[i]) / (jmax - jmin), 0.0), 1.0)

        # Command hand
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

    def move_robot_to_home_pose(self):
        self.get_logger().info("INFO: Moving robot arms and hands to home pose.")
        self.openpi_client_mode = OpenpiClientMode.MOVING_HOME
        self.command_inspire_hand_grasp(0.0, 0.0)
        self.moving_home_last_time = time.monotonic()
        self.moving_home_timer = self.create_timer(1.0 / 50.0, self.move_arms_to_home_pose_callback)

    def move_arms_to_home_pose_callback(self):
        if self.openpi_client_mode is not OpenpiClientMode.MOVING_HOME:
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
            cur_q = arm_observations[idx]
            max_delta = G1_MOVING_HOME_JOINT_SPEED * dt
            joint_commands.append(0.0 if abs(cur_q) <= max_delta else cur_q - max_delta * float(np.sign(cur_q)))
                
        self.send_arm_joint_commands(joint_commands)

        # Check if home pose was commanded and arm is reasonably close
        robot_at_home = np.all(np.abs(arm_observations) <= G1_HOME_POSE_TOLERANCE)
        if robot_at_home:
            self.openpi_client_mode = OpenpiClientMode.IDLE
            self.moving_home_timer.cancel()
            self.get_logger().info("INFO: G1 has reached home")

    def process_observation_for_openpi(self):
        if any(x is None for x in [self.last_left_arm_obs, self.last_right_arm_obs, self.last_table_img]):
            return
        if not self.disable_left_hand:
            if self.last_left_hand_obs is None or self.last_left_wrist_img is None:
                return
        if not self.disable_right_hand:
            if self.last_right_hand_obs is None or self.last_right_wrist_img is None:
                return

        # Pi0 models are trained for gripper positions in [0.0, 1.0], with 0.0 corresponding to fully open and 1.0 corresponding to fully closed.
        # The robot provides observations for the finger joints, which are used to compute a single gripper position in the Pi0 models' format.
        # Proprioceptive state normalization is handled on the server side.
        if not self.disable_left_hand:
            left_fingers_pitch = np.clip(self.last_left_hand_obs[:4], 0.0, 0.7) / 0.7
            left_thumb_pitch = np.clip(self.last_left_hand_obs[4], 0.0, 0.26) / 0.26
            left_gripper_pos = np.clip((np.sum(left_fingers_pitch) + left_thumb_pitch) / 5, 0.0, 1.0, dtype=np.float32)
            left_wrist_img = convert_image_for_openpi(self.last_left_wrist_img)[::-1,::-1,:]  # flip to account for upside-down camera mount
        else:
            left_gripper_pos = 0.0
            left_wrist_img = convert_image_for_openpi(self.last_table_img)
        
        if not self.disable_right_hand:
            right_fingers_pitch = np.clip(self.last_right_hand_obs[:4], 0.0, 0.7) / 0.7
            right_thumb_pitch = np.clip(self.last_right_hand_obs[4], 0.0, 0.26) / 0.26
            right_gripper_pos = np.clip((np.sum(right_fingers_pitch) + right_thumb_pitch) / 5, 0.0, 1.0, dtype=np.float32)
            right_wrist_img = convert_image_for_openpi(self.last_right_wrist_img)[::-1,::-1,:]  # flip to account for upside-down camera mount
        else:
            right_gripper_pos = 0.0
            right_wrist_img = convert_image_for_openpi(self.last_table_img)
        
        joint_pos = np.array(self.last_left_arm_obs + [left_gripper_pos] + self.last_right_arm_obs + [right_gripper_pos])

        policy_server_obs = {
            "observation/table_img": convert_image_for_openpi(self.last_table_img),
            "observation/left_wrist_img": left_wrist_img,
            "observation/right_wrist_img": right_wrist_img,
            "observation/joint_pos": joint_pos,
            "prompt": self.prompt,
        }
        return policy_server_obs

    def process_openpi_action(self, action: np.array):
        # Pi0 models are trained for gripper positions in [0.0, 1.0], with 0.0 corresponding to fully open and 1.0 corresponding to fully closed.
        # The conversion of this single value to finger commands resembling a simple grasp maneuver is done later in command_inspire_hand_grasp().
        return [float(x) for x in action[:7]], float(action[7]), [float(x) for x in action[8:15]], float(action[15])

    def handle_policy(self):
        # Send websocket request to policy server if it's time to predict a new chunk
        if self.actions_from_chunk_completed == 0 or self.actions_from_chunk_completed >= self.open_loop_horizon:
            self.actions_from_chunk_completed = 0

            # Transform observation data to format expected by policy server
            policy_server_obs = self.process_observation_for_openpi()

            # Wrap the server call in a context manager to prevent Ctrl+C from interrupting it
            # Ctrl+C will be handled after the server call is complete
            with prevent_keyboard_interrupt():
                self.pred_action_chunk = self.policy_client.infer(policy_server_obs)["actions"]

        # Select current action to execute from chunk
        action = self.pred_action_chunk[self.actions_from_chunk_completed]
        self.actions_from_chunk_completed += 1

        # Transform action data to format expected by robot
        left_arm_action, left_hand_action, right_arm_action, right_hand_action = self.process_openpi_action(action)

        # Execute arm action on robot
        self.left_arm_act, self.right_arm_act = left_arm_action, right_arm_action

        # Execute hand action on robot
        self.command_inspire_hand_grasp(left_hand_action, right_hand_action)


def main(args=None):
    rclpy.init(args=args)

    try:
        openpi_agent = RobotGptOpenPiROSClient()
        rclpy.spin(openpi_agent)
    except Exception as e:
        if type(e) is not KeyboardInterrupt:
            openpi_agent.get_logger().error(f'Node failed with exception: {e}')
    finally:
        openpi_agent.shutdown()
        openpi_agent.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
