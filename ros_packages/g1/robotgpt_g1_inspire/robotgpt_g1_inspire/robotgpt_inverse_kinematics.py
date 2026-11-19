"""
This node provides inverse kinematics for the G1 robot's arms.

The IK solution is adapted from:
https://github.com/unitreerobotics/xr_teleoperate/blob/main/teleop/robot_control/robot_arm_ik.py
Licensed under the Apache License, Version 2.0.

The WeightedMovingFilter class is from:
https://github.com/unitreerobotics/xr_teleoperate/blob/main/teleop/utils/weighted_moving_filter.py
Licensed under the Apache License, Version 2.0.

The urdf used by this script is at:
https://github.com/unitreerobotics/xr_teleoperate/tree/main/assets/g1
Licensed under the Apache License, Version 2.0.

All other orignal parts of this file:
Copyright (c) 2026 ronypepper.
Licensed under the Apache License, Version 2.0.
"""

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Pose, PoseArray
from sensor_msgs.msg import JointState

from transforms3d import quaternions

import casadi
import meshcat.geometry as mg
import numpy as np
import pinocchio as pin                             
from pinocchio import casadi as cpin    
from pinocchio.visualize import MeshcatVisualizer   
import os
import pickle


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


def se3_to_pose(T):
    pose = Pose()

    pose.position.x = T.translation[0]
    pose.position.y = T.translation[1]
    pose.position.z = T.translation[2]

    quat = pin.Quaternion(T.rotation)

    pose.orientation.x = quat.x
    pose.orientation.y = quat.y
    pose.orientation.z = quat.z
    pose.orientation.w = quat.w

    return pose


class WeightedMovingFilter:
    def __init__(self, weights, data_size = 14):
        self._window_size = len(weights)
        self._weights = np.array(weights)
        assert np.isclose(np.sum(self._weights), 1.0), "[WeightedMovingFilter] the sum of weights list must be 1.0!"
        self._data_size = data_size
        self._filtered_data = np.zeros(self._data_size)
        self._data_queue = []

    def _apply_filter(self):
        if len(self._data_queue) < self._window_size:
            return self._data_queue[-1]

        data_array = np.array(self._data_queue)
        return np.sum(data_array * self._weights[::-1, None], axis=0)

    def add_data(self, new_data):
        assert len(new_data) == self._data_size

        if len(self._data_queue) > 0 and np.array_equal(new_data, self._data_queue[-1]):
            return  # skip duplicate data
        
        if len(self._data_queue) >= self._window_size:
            self._data_queue.pop(0)

        self._data_queue.append(new_data)
        self._filtered_data = self._apply_filter()

    @property
    def filtered_data(self):
        return self._filtered_data


class RobotGPTG1InverseKinematics(Node):
    def __init__(self):
        super().__init__("robotgpt_g1_inverse_kinematics")

        # Configuration parameters
        self.Visualization = self.declare_parameter("visualization", True).value
        self.assets_directory = self.declare_parameter("assets_directory", "/RobotGPT/robotgpt_ws/src/robotgpt_g1_inspire/assets").value

        # ROS subscribers
        self.ee_targets_sub = self.create_subscription(PoseArray, "ee_targets", self.ee_targets_callback, 1)
        self.current_joints_sub = self.create_subscription(JointState, "current_joints", self.current_joints_callback, 1)

        # ROS publishers
        self.computed_joints_pub = self.create_publisher(JointState, "computed_joints", 10)
        self.ee_poses_pub = self.create_publisher(PoseArray, "ee_poses", 10)

        # Inverse kinematics setup
        self.current_joints = None

        np.set_printoptions(precision=5, suppress=True, linewidth=200)
        
        # fixed cache file path
        self.cache_path = os.path.join(self.assets_directory, "g1_29_model_cache.pkl")

        self.urdf_path = os.path.join(self.assets_directory, 'g1_body29_hand14.urdf')
        self.model_dir = self.assets_directory

        # Try loading cache first
        if os.path.exists(self.cache_path) and (not self.Visualization):
            self.get_logger().info(f"[G1_29_ArmIK] >>> Loading cached robot model: {self.cache_path}")
            self.robot, self.reduced_robot = self.load_cache()
        else:
            self.get_logger().info("[G1_29_ArmIK] >>> Loading URDF (slow)...")
            self.robot = pin.RobotWrapper.BuildFromURDF(self.urdf_path, self.model_dir)

            self.mixed_jointsToLockIDs = [
                "left_hip_pitch_joint" ,
                "left_hip_roll_joint" ,
                "left_hip_yaw_joint" ,
                "left_knee_joint" ,
                "left_ankle_pitch_joint" ,
                "left_ankle_roll_joint" ,
                "right_hip_pitch_joint" ,
                "right_hip_roll_joint" ,
                "right_hip_yaw_joint" ,
                "right_knee_joint" ,
                "right_ankle_pitch_joint" ,
                "right_ankle_roll_joint" ,
                "waist_yaw_joint" ,
                "waist_roll_joint" ,
                "waist_pitch_joint" ,
                
                "left_hand_thumb_0_joint" ,
                "left_hand_thumb_1_joint" ,
                "left_hand_thumb_2_joint" ,
                "left_hand_middle_0_joint" ,
                "left_hand_middle_1_joint" ,
                "left_hand_index_0_joint" ,
                "left_hand_index_1_joint" ,
                
                "right_hand_thumb_0_joint" ,
                "right_hand_thumb_1_joint" ,
                "right_hand_thumb_2_joint" ,
                "right_hand_index_0_joint" ,
                "right_hand_index_1_joint" ,
                "right_hand_middle_0_joint",
                "right_hand_middle_1_joint"
            ]

            self.reduced_robot = self.robot.buildReducedRobot(
                list_of_joints_to_lock=self.mixed_jointsToLockIDs,
                reference_configuration=np.array([0.0] * self.robot.model.nq),
            )

            self.reduced_robot.model.addFrame(
                pin.Frame('L_ee',
                            self.reduced_robot.model.getJointId('left_wrist_yaw_joint'),
                            pin.SE3(np.eye(3),
                                    np.array([0.05,0,0]).T),
                            pin.FrameType.OP_FRAME)
            )
            self.reduced_robot.model.addFrame(
                pin.Frame('R_ee',
                            self.reduced_robot.model.getJointId('right_wrist_yaw_joint'),
                            pin.SE3(np.eye(3),
                                    np.array([0.05,0,0]).T),
                            pin.FrameType.OP_FRAME)
            )

            # Rebuilt data to include added frames
            self.reduced_robot.data = self.reduced_robot.model.createData()

            # Save cache (only after everything is built)
            if not os.path.exists(self.cache_path):
                self.save_cache()
                self.get_logger().info(f">>> Cache saved to {self.cache_path}")

        # for i in range(self.reduced_robot.model.nframes):
        #     frame = self.reduced_robot.model.frames[i]
        #     frame_id = self.reduced_robot.model.getFrameId(frame.name)
        #     self.get_logger().debug(f"Frame ID: {frame_id}, Name: {frame.name}")

        # Creating Casadi models and data for symbolic computing
        self.cmodel = cpin.Model(self.reduced_robot.model)
        self.cdata = self.cmodel.createData()

        # Creating symbolic variables
        self.cq = casadi.SX.sym("q", self.reduced_robot.model.nq, 1) 
        self.cTf_l = casadi.SX.sym("tf_l", 4, 4)
        self.cTf_r = casadi.SX.sym("tf_r", 4, 4)
        cpin.framesForwardKinematics(self.cmodel, self.cdata, self.cq)

        # Get the hand joint ID and define the error function
        self.L_hand_id = self.reduced_robot.model.getFrameId("L_ee")
        self.R_hand_id = self.reduced_robot.model.getFrameId("R_ee")

        self.translational_error = casadi.Function(
            "translational_error",
            [self.cq, self.cTf_l, self.cTf_r],
            [
                casadi.vertcat(
                    self.cdata.oMf[self.L_hand_id].translation - self.cTf_l[:3,3],
                    self.cdata.oMf[self.R_hand_id].translation - self.cTf_r[:3,3]
                )
            ],
        )
        self.rotational_error = casadi.Function(
            "rotational_error",
            [self.cq, self.cTf_l, self.cTf_r],
            [
                casadi.vertcat(
                    cpin.log3(self.cdata.oMf[self.L_hand_id].rotation @ self.cTf_l[:3,:3].T),
                    cpin.log3(self.cdata.oMf[self.R_hand_id].rotation @ self.cTf_r[:3,:3].T)
                )
            ],
        )

        # Defining the optimization problem
        self.opti = casadi.Opti()
        self.var_q = self.opti.variable(self.reduced_robot.model.nq)
        self.var_q_last = self.opti.parameter(self.reduced_robot.model.nq)   # for smooth
        self.param_tf_l = self.opti.parameter(4, 4)
        self.param_tf_r = self.opti.parameter(4, 4)
        self.translational_cost = casadi.sumsqr(self.translational_error(self.var_q, self.param_tf_l, self.param_tf_r))
        self.rotation_cost = casadi.sumsqr(self.rotational_error(self.var_q, self.param_tf_l, self.param_tf_r))
        self.regularization_cost = casadi.sumsqr(self.var_q)
        self.smooth_cost = casadi.sumsqr(self.var_q - self.var_q_last)

        # Setting optimization constraints and goals
        self.opti.subject_to(self.opti.bounded(
            self.reduced_robot.model.lowerPositionLimit,
            self.var_q,
            self.reduced_robot.model.upperPositionLimit)
        )
        self.opti.minimize(50 * self.translational_cost + self.rotation_cost + 0.02 * self.regularization_cost + 0.1 * self.smooth_cost)

        opts = {
            # CasADi-level options
            'expand': True, 
            'detect_simple_bounds': True,
            'calc_lam_p': False,  # https://github.com/casadi/casadi/wiki/FAQ:-Why-am-I-getting-%22NaN-detected%22in-my-optimization%3F
            'print_time':False,   # print or not
            # IPOPT solver options
            'ipopt.sb': 'yes',    # disable Ipopt's license message
            'ipopt.print_level': 0,
            'ipopt.max_iter': 30, 
            'ipopt.tol': 1e-4,
            'ipopt.acceptable_tol': 5e-4,
            'ipopt.acceptable_iter': 5,
            'ipopt.warm_start_init_point': 'yes',
            'ipopt.derivative_test': 'none',
            'ipopt.jacobian_approximation': 'exact',
            # 'ipopt.hessian_approximation': 'limited-memory',
        }
        self.opti.solver("ipopt", opts)

        self.init_data = np.zeros(self.reduced_robot.model.nq)
        self.smooth_filter = WeightedMovingFilter(np.array([0.4, 0.3, 0.2, 0.1]), 14)
        self.vis = None

        if self.Visualization:
            # Initialize the Meshcat visualizer for visualization
            self.vis = MeshcatVisualizer(self.reduced_robot.model, self.reduced_robot.collision_model, self.reduced_robot.visual_model)
            self.vis.initViewer(open=True) 
            self.vis.loadViewerModel("pinocchio") 
            self.vis.displayFrames(True, frame_ids=[107, 108], axis_length = 0.15, axis_width = 5)
            self.vis.display(pin.neutral(self.reduced_robot.model))

            # Enable the display of end effector target frames with short axis lengths and greater width.
            frame_viz_names = ['L_ee_target', 'R_ee_target']
            FRAME_AXIS_POSITIONS = (
                np.array([[0, 0, 0], [1, 0, 0],
                            [0, 0, 0], [0, 1, 0],
                            [0, 0, 0], [0, 0, 1]]).astype(np.float32).T
            )
            FRAME_AXIS_COLORS = (
                np.array([[1, 0, 0], [1, 0.6, 0],
                            [0, 1, 0], [0.6, 1, 0],
                            [0, 0, 1], [0, 0.6, 1]]).astype(np.float32).T
            )
            axis_length = 0.1
            axis_width = 20
            for frame_viz_name in frame_viz_names:
                self.vis.viewer[frame_viz_name].set_object(
                    mg.LineSegments(
                        mg.PointsGeometry(
                            position=axis_length * FRAME_AXIS_POSITIONS,
                            color=FRAME_AXIS_COLORS,
                        ),
                        mg.LineBasicMaterial(
                            linewidth=axis_width,
                            vertexColors=True,
                        ),
                    )
                )

    def ee_targets_callback(self, msg: PoseArray):
        left_target = pose_to_matrix(msg.poses[0])
        right_target = pose_to_matrix(msg.poses[1])
        target_joints, _ = self.solve_ik(left_target, right_target, self.current_joints)

        ik_msg = JointState()
        ik_msg.header.stamp = self.get_clock().now().to_msg()
        ik_msg.position = target_joints
        self.computed_joints_pub.publish(ik_msg)

    def current_joints_callback(self, msg: JointState):
        self.current_joints = msg.position

        # Forward Kinematics
        q = np.asarray(msg.position)
        pin.framesForwardKinematics(self.reduced_robot.model, self.reduced_robot.data, q)
        left_pose = self.reduced_robot.data.oMf[self.L_hand_id]
        right_pose = self.reduced_robot.data.oMf[self.R_hand_id]

        msg_out = PoseArray()
        msg_out.header.stamp = self.get_clock().now().to_msg()
        msg_out.poses.append(se3_to_pose(left_pose))
        msg_out.poses.append(se3_to_pose(right_pose))
        self.ee_poses_pub.publish(msg_out)

    # Save both robot.model and reduced_robot.model
    def save_cache(self):
        data = {
            "robot_model": self.robot.model,
            "reduced_model": self.reduced_robot.model,
        }

        with open(self.cache_path, "wb") as f:
            pickle.dump(data, f)

    # Load both robot.model and reduced_robot.model
    def load_cache(self):
        with open(self.cache_path, "rb") as f:
            data = pickle.load(f)

        robot = pin.RobotWrapper()
        robot.model = data["robot_model"]
        robot.data = robot.model.createData()

        reduced_robot = pin.RobotWrapper()
        reduced_robot.model = data["reduced_model"]
        reduced_robot.data = reduced_robot.model.createData()

        return robot, reduced_robot

    def scale_arms(self, human_left_pose, human_right_pose, human_arm_length=0.60, robot_arm_length=0.75):
        scale_factor = robot_arm_length / human_arm_length
        robot_left_pose = human_left_pose.copy()
        robot_right_pose = human_right_pose.copy()
        robot_left_pose[:3, 3] *= scale_factor
        robot_right_pose[:3, 3] *= scale_factor
        return robot_left_pose, robot_right_pose

    def solve_ik(self, left_wrist, right_wrist, current_lr_arm_motor_q = None, current_lr_arm_motor_dq = None):
        if current_lr_arm_motor_q is not None:
            self.init_data = current_lr_arm_motor_q
        self.opti.set_initial(self.var_q, self.init_data)

        # left_wrist, right_wrist = self.scale_arms(left_wrist, right_wrist)
        if self.Visualization:
            self.vis.viewer['L_ee_target'].set_transform(left_wrist)   # for visualization
            self.vis.viewer['R_ee_target'].set_transform(right_wrist)  # for visualization

        self.opti.set_value(self.param_tf_l, left_wrist)
        self.opti.set_value(self.param_tf_r, right_wrist)
        self.opti.set_value(self.var_q_last, self.init_data) # for smooth

        try:
            sol = self.opti.solve()
            # sol = self.opti.solve_limited()

            sol_q = self.opti.value(self.var_q)
            self.smooth_filter.add_data(sol_q)
            sol_q = self.smooth_filter.filtered_data

            if current_lr_arm_motor_dq is not None:
                v = current_lr_arm_motor_dq * 0.0
            else:
                v = (sol_q - self.init_data) * 0.0

            self.init_data = sol_q

            sol_tauff = pin.rnea(self.reduced_robot.model, self.reduced_robot.data, sol_q, v, np.zeros(self.reduced_robot.model.nv))

            if self.Visualization:
                self.vis.display(sol_q)  # for visualization

            return sol_q, sol_tauff
        
        except Exception as e:
            self.get_logger().error(f"ERROR in convergence, plotting debug info.{e}")

            sol_q = self.opti.debug.value(self.var_q)
            self.smooth_filter.add_data(sol_q)
            sol_q = self.smooth_filter.filtered_data

            if current_lr_arm_motor_dq is not None:
                v = current_lr_arm_motor_dq * 0.0
            else:
                v = (sol_q - self.init_data) * 0.0

            self.init_data = sol_q

            sol_tauff = pin.rnea(self.reduced_robot.model, self.reduced_robot.data, sol_q, v, np.zeros(self.reduced_robot.model.nv))

            self.get_logger().error(f"sol_q:{sol_q} \nmotorstate: \n{current_lr_arm_motor_q} \nleft_pose: \n{left_wrist} \nright_pose: \n{right_wrist}")
            if self.Visualization:
                self.vis.display(sol_q)  # for visualization

            # return sol_q, sol_tauff
            return current_lr_arm_motor_q, np.zeros(self.reduced_robot.model.nv)


def main(args=None):
    rclpy.init(args=args)

    g1_ik = RobotGPTG1InverseKinematics()

    rclpy.spin(g1_ik)

    g1_ik.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
