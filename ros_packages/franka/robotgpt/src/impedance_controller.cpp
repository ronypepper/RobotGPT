// This file is based on: https://github.com/frankarobotics/franka_ros/blob/develop/franka_example_controllers/src/joint_impedance_example_controller.cpp
// The original source code is licensed under the Apache-2.0 license and Copyright (c) 2023 Franka Robotics GmbH
// This file is licensed under the Apache-2.0 license and modifications are Copyright (c) 2026 ronypepper

#include <robotgpt/impedance_controller.h>

#include <cmath>
#include <memory>

#include <controller_interface/controller_base.h>
#include <pluginlib/class_list_macros.h>
#include <ros/ros.h>

#include <franka/robot_state.h>

#include <franka_example_controllers/pseudo_inversion.h>

namespace robotgpt {

bool ImpedanceController::init(hardware_interface::RobotHW* robot_hw,
                                           ros::NodeHandle& node_handle) {
  std::string arm_id;
  if (!node_handle.getParam("arm_id", arm_id)) {
    ROS_ERROR("ImpedanceController: Could not read parameter arm_id");
    return false;
  }

  std::vector<std::string> joint_names;
  if (!node_handle.getParam("joint_names", joint_names) || joint_names.size() != 7) {
    ROS_ERROR(
        "ImpedanceController: Invalid or no joint_names parameters provided, aborting "
        "controller init!");
    return false;
  }

  if (!node_handle.getParam("k_gains", k_gains_) || k_gains_.size() != 7) {
    ROS_ERROR(
        "ImpedanceController:  Invalid or no k_gain parameters provided, aborting "
        "controller init!");
    return false;
  }

  if (!node_handle.getParam("d_gains", d_gains_) || d_gains_.size() != 7) {
    ROS_ERROR(
        "ImpedanceController:  Invalid or no d_gain parameters provided, aborting "
        "controller init!");
    return false;
  }

  if (!node_handle.getParam("coriolis_factor", coriolis_factor_)) {
    ROS_INFO_STREAM("ImpedanceController: coriolis_factor not found. Defaulting to "
                    << coriolis_factor_);
  }

  auto* model_interface = robot_hw->get<franka_hw::FrankaModelInterface>();
  if (model_interface == nullptr) {
    ROS_ERROR_STREAM(
        "ImpedanceController: Error getting model interface from hardware");
    return false;
  }
  try {
    model_handle_ = std::make_unique<franka_hw::FrankaModelHandle>(
        model_interface->getHandle(arm_id + "_model"));
  } catch (hardware_interface::HardwareInterfaceException& ex) {
    ROS_ERROR_STREAM(
        "ImpedanceController: Exception getting model handle from interface: "
        << ex.what());
    return false;
  }

  auto* state_interface = robot_hw->get<franka_hw::FrankaStateInterface>();
  if (state_interface == nullptr) {
    ROS_ERROR_STREAM(
        "ImpedanceController: Error getting state interface from hardware");
    return false;
  }
  try {
    state_handle_ = std::make_unique<franka_hw::FrankaStateHandle>(
        state_interface->getHandle(arm_id + "_robot"));
  } catch (hardware_interface::HardwareInterfaceException& ex) {
    ROS_ERROR_STREAM(
        "ImpedanceController: Exception getting state handle from interface: "
        << ex.what());
    return false;
  }

  auto* effort_joint_interface = robot_hw->get<hardware_interface::EffortJointInterface>();
  if (effort_joint_interface == nullptr) {
    ROS_ERROR_STREAM(
        "ImpedanceController: Error getting effort joint interface from hardware");
    return false;
  }
  for (size_t i = 0; i < 7; ++i) {
    try {
      joint_handles_.push_back(effort_joint_interface->getHandle(joint_names[i]));
    } catch (const hardware_interface::HardwareInterfaceException& ex) {
      ROS_ERROR_STREAM(
          "ImpedanceController: Exception getting joint handles: " << ex.what());
      return false;
    }
  }

  std::fill(dq_filtered_.begin(), dq_filtered_.end(), 0);

  control_mode_buffer_.writeFromNonRT(ControlMode::NONE);

  control_mode_service_ = node_handle.advertiseService(
    "set_cartesian_mode",
    &ImpedanceController::setCartesianMode,
    this);
  
  cartesian_command_sub_ = node_handle.subscribe(
    "cartesian_command",
    1,
    &ImpedanceController::cartesianCommandCallback,
    this);
  
  joint_command_sub_ = node_handle.subscribe(
    "joint_command",
    1,
    &ImpedanceController::jointCommandCallback,
    this);
  
  q_d_pub_ = std::make_unique<realtime_tools::RealtimePublisher<sensor_msgs::JointState>>(
    node_handle,
    "q_d",
    1);
  q_d_pub_->msg_.position.resize(7);

  return true;
}

void ImpedanceController::starting(const ros::Time& time) {
  franka::RobotState robot_state = state_handle_->getRobotState();
  control_mode_buffer_.writeFromNonRT(ControlMode::JOINT);
}

void ImpedanceController::update(const ros::Time& time, const ros::Duration& period) {
  // Get robot state
  franka::RobotState robot_state = state_handle_->getRobotState();
  std::array<double, 7> coriolis = model_handle_->getCoriolis();
  std::array<double, 7> gravity = model_handle_->getGravity();

  // Change control mode if requested
  ControlMode requested_control_mode = *control_mode_buffer_.readFromRT();
  if (requested_control_mode != active_control_mode_) {
    // Reset torques
    for (size_t i = 0; i < 7; ++i) {
      tau_d_filtered_[i] = 0.0;
    }
    if (requested_control_mode == ControlMode::JOINT) {
      switchToJointControlMode(robot_state);
    }
    else if (requested_control_mode == ControlMode::CARTESIAN) {
      switchToCartesianControlMode(robot_state);
    }
  }

  // Command zero joint torques for ControlMode::NONE
  if (active_control_mode_ == ControlMode::NONE) {
    for (size_t i = 0; i < 7; ++i) {
      joint_handles_[i].setCommand(0.0);
    }
    return;
  }

  // Compute or get target joint positions
  std::array<double, 7> q_d;
  if (active_control_mode_ == ControlMode::CARTESIAN) {
    // Get and filter desired pose
    ImpedanceController::EEPose ee_pose_d_ = *ee_pose_d_buffer_.readFromRT();
    position_d_filtered_ = position_filter_ * ee_pose_d_.position + (1.0 - position_filter_) * position_d_filtered_;
    orientation_d_filtered_ = orientation_d_filtered_.slerp(orientation_filter_, ee_pose_d_.orientation);

    // Get current pose
    Eigen::Affine3d current_transform(Eigen::Matrix4d::Map(robot_state.O_T_EE.data()));
    Eigen::Vector3d position_c_ = current_transform.translation();
    Eigen::Quaterniond orientation_c_ = Eigen::Quaterniond(current_transform.rotation());

    // Compute pose error
    Eigen::Matrix<double, 6, 1> pose_error;
    pose_error.head(3) << position_d_filtered_ - position_c_;
    Eigen::Quaterniond orientation_error = orientation_d_filtered_ * orientation_c_.inverse();
    if (orientation_error.w() < 0.0) {
      orientation_error.coeffs() *= -1.0;
    }
    Eigen::AngleAxisd axis_angle_error(orientation_error);
    pose_error.tail(3) << axis_angle_error.angle() * axis_angle_error.axis();
    
    // Get jacobian pseudo-inverse
    std::array<double, 42> jacobian_array = model_handle_->getZeroJacobian(franka::Frame::kEndEffector);
    Eigen::Map<Eigen::Matrix<double, 6, 7>> jacobian(jacobian_array.data());
    Eigen::MatrixXd jacobian_pinv;
    franka_example_controllers::pseudoInverse(jacobian, jacobian_pinv);

    // Compute and limit delta joint positions
    Eigen::VectorXd delta_joint_pos(7);
    delta_joint_pos << jacobian_pinv * pose_error;


    // Eigen::Matrix<double,7,7> identity = Eigen::Matrix<double,7,7>::Identity();
    // Eigen::Matrix<double,7,7> nullspace = identity - jacobian_pinv * jacobian;
    // Eigen::VectorXd q_null(7);
    // q_null.setZero();
    // std::array<double,7> q_home = {0.0, -0.785398163397, 0.0, -2.35619449019, 0.0, 1.57079632679, 0.785398163397};
    // for (size_t i = 0; i < 7; i++) {
    //   q_null[i] = 20.0 * (q_home[i] - robot_state.q[i]);
    // }
    // delta_joint_pos += nullspace * q_null;


    // double max_delta = maxDeltaQ * period.toSec();
    // for (size_t i = 0; i < 7; ++i) {
    //   delta_joint_pos[i] = std::max(-max_delta, std::min(delta_joint_pos[i], max_delta));
    // }

    // Add to current joint positions
    for (size_t i = 0; i < 7; ++i) {
      q_d[i] = robot_state.q[i] + delta_joint_pos[i];
    }
  }
  else if (active_control_mode_ == ControlMode::JOINT) {
    q_d = *q_d_buffer_.readFromRT();
  }

  // Publish desired joint positions
  publish_time_ += period;
  if (publish_time_.toSec() >= 0.01) {  // 100 Hz
    if (q_d_pub_->trylock()) {
      q_d_pub_->msg_.header.stamp = time;
      for (size_t i = 0; i < 7; ++i) {
        q_d_pub_->msg_.position[i] = q_d[i];
      }
      q_d_pub_->unlockAndPublish();
      publish_time_ -= ros::Duration(0.01);
    }
  }

  // Filter desired joint positions
  double q_d_filter_ = active_control_mode_ == ControlMode::CARTESIAN ? q_d_filter_cartesian_mode_ : q_d_filter_joint_mode_;
  for (size_t i = 0; i < 7; ++i) {
    q_d_filtered_[i] = q_d_filter_ * q_d[i] + (1.0 - q_d_filter_) * q_d_filtered_[i];
  }

  // Limit delta joint positions and clamp to joint ranges
  std::array<double, 7> q_target;
  for (size_t i = 0; i < 7; ++i) {
    double max_delta = maxDeltaQs[i] * period.toSec();
    q_target[i] = robot_state.q[i] + std::max(-max_delta, std::min(q_d_filtered_[i] - robot_state.q[i], max_delta));
    q_target[i] = std::max(minJointPositions[i] + joint_margin, std::min(q_target[i], maxJointPositions[i] - joint_margin));
  }

  // Filter joint velocities
  double alpha = 0.99;
  for (size_t i = 0; i < 7; i++) {
    dq_filtered_[i] = (1 - alpha) * dq_filtered_[i] + alpha * robot_state.dq[i];
  }

  // Compute joint torques
  std::array<double, 7> tau_d_calculated;
  for (size_t i = 0; i < 7; ++i) {
    tau_d_calculated[i] = coriolis_factor_ * coriolis[i] +
                          k_gains_[i] * (q_target[i] - robot_state.q[i]) +
                          d_gains_[i] * (-dq_filtered_[i]);
  }

  // Filter desired joint torques
  for (size_t i = 0; i < 7; ++i) {
    tau_d_filtered_[i] = tau_d_filter_ * tau_d_calculated[i] + (1.0 - tau_d_filter_) * tau_d_filtered_[i];
  }

  // Maximum torque difference with a sampling rate of 1 kHz. The maximum torque rate is
  // 1000 * (1 / sampling_time).
  std::array<double, 7> tau_d_saturated = saturateTorqueRate(tau_d_calculated, robot_state.tau_J_d);

  // Command joint torques
  for (size_t i = 0; i < 7; ++i) {
    joint_handles_[i].setCommand(tau_d_saturated[i]);
  }
}

std::array<double, 7> ImpedanceController::saturateTorqueRate(
    const std::array<double, 7>& tau_d_calculated,
    const std::array<double, 7>& tau_J_d) {  // NOLINT (readability-identifier-naming)
  std::array<double, 7> tau_d_saturated{};
  for (size_t i = 0; i < 7; i++) {
    double difference = tau_d_calculated[i] - tau_J_d[i];
    tau_d_saturated[i] = tau_J_d[i] + std::max(std::min(difference, kDeltaTauMax), -kDeltaTauMax);
  }
  return tau_d_saturated;
}

void ImpedanceController::switchToJointControlMode(const franka::RobotState& robot_state) {
  // Set current joint positions as desired joint targets
  for (size_t i = 0; i < 7; ++i) {
    q_d_filtered_[i] = robot_state.q[i];
  }
  q_d_buffer_.writeFromNonRT(q_d_filtered_);
  active_control_mode_ = ControlMode::JOINT;
}

void ImpedanceController::switchToCartesianControlMode(const franka::RobotState& robot_state) {
  // Set current end-effector pose as desired end-effector pose
  Eigen::Affine3d initial_transform(Eigen::Matrix4d::Map(robot_state.O_T_EE.data()));
  position_d_filtered_ = initial_transform.translation();
  orientation_d_filtered_ = Eigen::Quaterniond(initial_transform.rotation());
  ImpedanceController::EEPose ee_pose_initial_ = {position_d_filtered_, orientation_d_filtered_};
  ee_pose_d_buffer_.writeFromNonRT(ee_pose_initial_);
  active_control_mode_ = ControlMode::CARTESIAN;
}

bool ImpedanceController::setCartesianMode(std_srvs::SetBool::Request& req, std_srvs::SetBool::Response& res) {
  control_mode_buffer_.writeFromNonRT(req.data ? ControlMode::CARTESIAN : ControlMode::JOINT);
  res.success = true;
  res.message = req.data ? "Requested cartesian mode" : "Requested joint mode";
  return true;
}

void ImpedanceController::jointCommandCallback(const sensor_msgs::JointStateConstPtr& msg) {
  if (msg->position.size() != 7) {
    ROS_ERROR("Expected 7 joint positions, got %zu", msg->position.size());
    return;
  }

  std::array<double, 7> q_d;
  for (size_t i = 0; i < 7; ++i) {
    q_d[i] = msg->position[i];
  }

  q_d_buffer_.writeFromNonRT(q_d);
}

void ImpedanceController::cartesianCommandCallback(const geometry_msgs::PoseStampedConstPtr& msg) {
  if (msg->header.frame_id.empty() || msg->header.frame_id != "fr3_link0") {
    ROS_WARN_THROTTLE(1.0, "Ignoring Cartesian command not in frame fr3_link0");
    return;
  }

  Eigen::Quaterniond orientation(msg->pose.orientation.w, msg->pose.orientation.x, msg->pose.orientation.y, msg->pose.orientation.z);
  if (orientation.norm() < 1e-6) {
      ROS_WARN("Received invalid Cartesian orientation");
      return;
  }
  orientation.normalize();

  ImpedanceController::EEPose ee_pose_d_ = {
    Eigen::Vector3d(msg->pose.position.x, msg->pose.position.y, msg->pose.position.z),
    orientation
  };
  ee_pose_d_buffer_.writeFromNonRT(ee_pose_d_);
}

}  // namespace robotgpt

PLUGINLIB_EXPORT_CLASS(robotgpt::ImpedanceController,
                       controller_interface::ControllerBase)
