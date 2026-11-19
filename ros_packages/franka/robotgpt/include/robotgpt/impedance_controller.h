// This file has been adapted from: https://github.com/frankarobotics/franka_ros/blob/develop/franka_example_controllers/include/franka_example_controllers/joint_impedance_example_controller.h
// The original source code is licensed under the Apache-2.0 license and Copyright (c) 2023 Franka Robotics GmbH
// This file is licensed under the Apache-2.0 license and modifications are Copyright (c) 2026 ronypepper

#pragma once

#include <memory>
#include <string>
#include <vector>

#include <controller_interface/multi_interface_controller.h>
#include <hardware_interface/joint_command_interface.h>
#include <hardware_interface/robot_hw.h>
#include <ros/node_handle.h>
#include <ros/time.h>

#include <franka_example_controllers/JointTorqueComparison.h>
#include <franka_hw/franka_model_interface.h>
#include <franka_hw/franka_state_interface.h>

#include <realtime_tools/realtime_buffer.h>
#include <realtime_tools/realtime_publisher.h>

#include <std_srvs/SetBool.h>
#include <sensor_msgs/JointState.h>
#include <geometry_msgs/PoseStamped.h>

#include <eigen3/Eigen/Geometry>

namespace robotgpt {

class ImpedanceController : public controller_interface::MultiInterfaceController<
                                            franka_hw::FrankaModelInterface,
                                            hardware_interface::EffortJointInterface,
                                            franka_hw::FrankaStateInterface> {
 public:
  bool init(hardware_interface::RobotHW* robot_hw, ros::NodeHandle& node_handle) override;
  void starting(const ros::Time&) override;
  void update(const ros::Time&, const ros::Duration& period) override;

 private:
  // Saturation
  std::array<double, 7> saturateTorqueRate(
      const std::array<double, 7>& tau_d_calculated,
      const std::array<double, 7>& tau_J_d);  // NOLINT (readability-identifier-naming)

  std::unique_ptr<franka_hw::FrankaStateHandle> state_handle_;
  std::unique_ptr<franka_hw::FrankaModelHandle> model_handle_;
  std::vector<hardware_interface::JointHandle> joint_handles_;

  static constexpr double kDeltaTauMax{1.0};

  static constexpr std::array<double, 7> maxDeltaQs{40.0, 40.0, 40.0, 40.0, 60.0, 60.0, 80.0};

  static constexpr double joint_margin = 3.0 * M_PI / 180.0;
  static constexpr std::array<double, 7> minJointPositions = {-2.3093, -1.5133, -2.4937, -2.7478, -2.4800, 0.8521, -2.6895};
  static constexpr std::array<double, 7> maxJointPositions = {2.3093, 1.5133, 2.4937, -0.4461, 2.4800, 4.2094, 2.6895};

  std::vector<double> k_gains_;
  std::vector<double> d_gains_;
  double coriolis_factor_{1.0};
  std::array<double, 7> dq_filtered_;
  
  double position_filter_{1.0};
  double orientation_filter_{1.0};
  double q_d_filter_cartesian_mode_{0.01};
  double q_d_filter_joint_mode_{0.0008};
  double tau_d_filter_{0.01};
  std::array<double, 7> q_d_filtered_;
  std::array<double, 7> tau_d_filtered_;
  Eigen::Vector3d position_d_filtered_;
  Eigen::Quaterniond orientation_d_filtered_;
  
  enum class ControlMode {
    JOINT,
    CARTESIAN,
    NONE
  };

  struct EEPose
  {
    Eigen::Vector3d position;
    Eigen::Quaterniond orientation;
  };

  void switchToJointControlMode(const franka::RobotState& robot_state);
  void switchToCartesianControlMode(const franka::RobotState& robot_state);

  ros::ServiceServer control_mode_service_;
  bool setCartesianMode(std_srvs::SetBool::Request& req, std_srvs::SetBool::Response& res);
  realtime_tools::RealtimeBuffer<ControlMode> control_mode_buffer_;
  ControlMode active_control_mode_{ControlMode::NONE};

  ros::Subscriber joint_command_sub_;
  void jointCommandCallback(const sensor_msgs::JointStateConstPtr& msg);
  realtime_tools::RealtimeBuffer<std::array<double, 7>> q_d_buffer_;

  ros::Subscriber cartesian_command_sub_;
  void cartesianCommandCallback(const geometry_msgs::PoseStampedConstPtr& msg);
  realtime_tools::RealtimeBuffer<EEPose> ee_pose_d_buffer_;

  std::unique_ptr<realtime_tools::RealtimePublisher<sensor_msgs::JointState>> q_d_pub_;
  ros::Duration publish_time_{0.0};
};

}  // namespace robotgpt
