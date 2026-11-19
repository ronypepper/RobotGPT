#!/usr/bin/env python3

"""
This script executes the pi0.5 policy on a Franka arm using ROS1 and (optionally) a remote policy server.

Portions of this file are derived from code in the openpi project:
https://github.com/Physical-Intelligence/openpi
Licensed under the Apache License, Version 2.0.

All other orignal parts of this file:
Copyright (c) 2026 ronypepper.
Licensed under the Apache License, Version 2.0.
"""

import rospy

from control_msgs.msg import GripperCommandActionGoal
from sensor_msgs.msg import Image, JointState
from franka_msgs.msg import FrankaState # type: ignore

import actionlib
from franka_gripper.msg import ( # type: ignore
    GraspAction, GraspGoal,
    MoveAction, MoveGoal
)

import numpy as np

import contextlib
import signal

from robotgpt.common.openpi_image_tools import convert_image_for_openpi

from openpi_client import websocket_client_policy


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


class RobotGptOpenPiROSClient:
    def __init__(self):
        rospy.init_node("robotgpt_openpi_agent")

        # Configuration parameters
        self.prompt = rospy.get_param("~prompt", "Pick up the blue cube and drop it in the bin")
        policy_server_ip = rospy.get_param("~policy_server_ip", "localhost")
        policy_server_port = rospy.get_param("~policy_server_port", 8000)
        control_frequency = rospy.get_param("~control_frequency", 20)
        self.open_loop_horizon = rospy.get_param("~open_loop_horizon", 16)
        self.gripper_max_effort = rospy.get_param("~gripper_max_effort", 1.0)

        # Observation buffers
        self.last_joint_obs, self.last_gripper_obs = None, None
        self.last_table_img, self.last_wrist_img = None, None
        self.last_gripper_act = None

        # ROS subscribers
        self.franka_state_sub = rospy.Subscriber("franka_state_controller/franka_states", FrankaState, self.franka_state_callback)
        self.franka_gripper_sub = rospy.Subscriber("franka_gripper/joint_states", JointState, self.franka_gripper_callback)
        self.table_camera_sub = rospy.Subscriber("table_camera/color/image_raw", Image, self.table_camera_callback)
        self.wrist_camera_sub = rospy.Subscriber("wrist_camera/color/image_raw", Image, self.wrist_camera_callback)

        # ROS publishers
        self.franka_joint_target_pub = rospy.Publisher("impedance_controller/joint_command", JointState, queue_size=10)

        # ROS action servers

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
        
        # Policy client
        rospy.loginfo(f"[INFO]: Attempting to connect to policy server at {policy_server_ip}:{policy_server_port}...")
        self.policy_client = websocket_client_policy.WebsocketClientPolicy(host=policy_server_ip, port=policy_server_port)
        self.pred_action_chunk, self.actions_from_chunk_completed = [], 0
        rospy.loginfo("[INFO]: Connected to policy server.")

        # Intialize robot to home pose
        self.move_robot_to_home_pose()
        input("\n\nPress Enter once the robot is at the home pose...\n\n\n")

        # Control loop
        control_rate = rospy.Rate(control_frequency)
        while not rospy.is_shutdown():
            self.handle_policy()
            control_rate.sleep()

    def franka_state_callback(self, msg: FrankaState):
        self.last_joint_obs = msg.q

    def franka_gripper_callback(self, msg):
        self.last_gripper_obs = msg.position

    def table_camera_callback(self, msg: Image):
        self.last_table_img = msg

    def wrist_camera_callback(self, msg: Image):
        self.last_wrist_img = msg

    def move_robot_to_home_pose(self):  # Function same as in scripts/isaac_teleop/robotgpt_isaac_teleop_controller.py
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

    def command_gripper(self, trigger_pos: float):  # Function same as in scripts/isaac_teleop/robotgpt_isaac_teleop_controller.py
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

    def process_observation_for_openpi(self):
        if any(x is None for x in [self.last_joint_obs, self.last_gripper_obs,
                                   self.last_table_img, self.last_wrist_img]):
            return
        
        # Pi0 models are trained for gripper positions in [0.0, 1.0], with 0.0 corresponding to fully open and 1.0 corresponding to fully closed.
        # Observations from the robot are in [0.0, 0.04], with 0.0 corresponding to fully closed and 0.04 corresponding to fully open.
        # Therefore we adjust the gripper observation to fit the Pi0 models' format.
        # Proprioceptive state normalization is handled on the server side.
        gripper_obs = (self.last_gripper_obs[0] + self.last_gripper_obs[1]) / 2
        gripper_obs = (gripper_obs - 0.04) / -0.04
        joint_pos = np.array(self.last_joint_obs + (gripper_obs, )) # 7 joints + 1 gripper

        policy_server_obs = {
            "observation/table_img": convert_image_for_openpi(self.last_table_img),
            "observation/wrist_img": convert_image_for_openpi(self.last_wrist_img),
            "observation/joint_pos": joint_pos,
            "prompt": self.prompt,
        }
        return policy_server_obs

    def process_openpi_action(self, action: np.array):
        # Pi0 models are trained for gripper positions in [0.0, 1.0], with 0.0 corresponding to fully open and 1.0 corresponding to fully closed.
        # We use these values directly to command discrete "open gripper" and "close gripper" actions to Franka.
        return action[:7].tolist(), action[7]

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
        joint_action, gripper_action = self.process_openpi_action(action)

        # Execute joint action on robot
        msg = JointState()
        msg.header.stamp = rospy.Time.now()
        msg.position = joint_action
        self.franka_joint_target_pub.publish(msg)

        # Execute gripper action on robot
        self.command_gripper(gripper_action)


if __name__ == "__main__":
    RobotGptOpenPiROSClient()
