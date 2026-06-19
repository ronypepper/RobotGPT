# Based on code from the Isaac Lab project:
# https://github.com/isaac-sim/IsaacLab
#
# Original work:
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# Modifications:
# Copyright (c) 2026 ronypepper.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from isaaclab.devices.device_base import DeviceBase
from isaaclab.devices.retargeter_base import RetargeterBase, RetargeterCfg
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG


class ControllerSe3RelRetargeter(RetargeterBase):
    """Retargets OpenXR controller tracking data to end-effector commands using relative positioning.

    This retargeter calculates delta poses between consecutive controller poses to generate incremental robot movements.

    Features:
    - Optional constraint to zero out X/Y rotations (keeping only Z-axis rotation)
    - Motion smoothing with adjustable parameters
    - Optional visualization of the target end-effector pose
    """

    def __init__(
        self,
        cfg: ControllerSe3RelRetargeterCfg,
    ):
        """Initialize the relative motion retargeter.

        Args:
            bound_controller: The controller to track (DeviceBase.TrackingTarget.CONTROLLER_LEFT or DeviceBase.TrackingTarget.CONTROLLER_RIGHT)
            zero_out_xy_rotation: If True, ignore rotations around x and y axes, allowing only z-axis rotation
            delta_pos_scale_factor: Amplification factor for position changes (higher = larger robot movements)
            delta_rot_scale_factor: Amplification factor for rotation changes (higher = larger robot rotations)
            alpha_pos: Position smoothing parameter (0-1); higher values track more closely to input,
                lower values smooth more
            alpha_rot: Rotation smoothing parameter (0-1); higher values track more closely to input,
                lower values smooth more
            enable_visualization: If True, show a visual marker representing the target end-effector pose
            device: The device to place the returned tensor on ('cpu' or 'cuda')
        """
        if cfg.bound_controller not in [DeviceBase.TrackingTarget.CONTROLLER_LEFT, DeviceBase.TrackingTarget.CONTROLLER_RIGHT]:
            raise ValueError(
                "bound_controller must be either DeviceBase.TrackingTarget.CONTROLLER_LEFT or DeviceBase.TrackingTarget.CONTROLLER_RIGHT"
            )
        super().__init__(cfg)
        self.bound_controller = cfg.bound_controller

        self._zero_out_xy_rotation = cfg.zero_out_xy_rotation
        self._delta_pos_scale_factor = cfg.delta_pos_scale_factor
        self._delta_rot_scale_factor = cfg.delta_rot_scale_factor
        self._alpha_pos = cfg.alpha_pos
        self._alpha_rot = cfg.alpha_rot

        # Initialize smoothing state
        self._smoothed_delta_pos = np.zeros(3)
        self._smoothed_delta_rot = np.zeros(3)

        # Define thresholds for small movements
        self._position_threshold = 0.001
        self._rotation_threshold = 0.01

        # Initialize visualization if enabled
        self._enable_visualization = cfg.enable_visualization
        if cfg.enable_visualization:
            frame_marker_cfg = FRAME_MARKER_CFG.copy()
            frame_marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
            self._goal_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_goal"))
            self._goal_marker.set_visibility(True)
            self._visualization_pos = np.zeros(3)
            self._visualization_rot = np.array([0.0, 0.0, 0.0, 1.0])

        self._previous_pose = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)

    def retarget(self, data: dict) -> torch.Tensor:
        """Convert controller pose to robot end-effector command.

        Args:
            data: Dictionary mapping tracking targets to joint data dictionaries.

        Returns:
            torch.Tensor: 6D tensor containing position (xyz) and rotation vector (rx,ry,rz)
                for the robot end-effector
        """
        # Default pose
        pose = self._previous_pose.copy()

        # Get controller data
        if self.bound_controller in data and data[self.bound_controller] is not None:
            controller_data = data[self.bound_controller]
            if len(controller_data) > DeviceBase.MotionControllerDataRowIndex.POSE.value:
                pose = controller_data[DeviceBase.MotionControllerDataRowIndex.POSE.value]

        delta_pose = self._calculate_delta_pose(pose, self._previous_pose)
        ee_command_np = self._retarget_rel(delta_pose)

        self._previous_pose = pose.copy()

        # Convert to torch tensor
        ee_command = torch.tensor(ee_command_np, dtype=torch.float32, device=self._sim_device)

        return ee_command

    def get_requirements(self) -> list[RetargeterBase.Requirement]:
        return [RetargeterBase.Requirement.MOTION_CONTROLLER]

    def _calculate_delta_pose(self, joint_pose: np.ndarray, previous_joint_pose: np.ndarray) -> np.ndarray:
        """Calculate delta pose from previous joint pose.

        Args:
            joint_pose: Current joint pose (position and orientation)
            previous_joint_pose: Previous joint pose for the same joint

        Returns:
            np.ndarray: 6D array with position delta (xyz) and rotation delta as axis-angle (rx,ry,rz)
        """
        delta_pos = joint_pose[:3] - previous_joint_pose[:3]
        abs_rotation = Rotation.from_quat(joint_pose[3:7])
        previous_rot = Rotation.from_quat(previous_joint_pose[3:7])
        relative_rotation = abs_rotation * previous_rot.inv()
        return np.concatenate([delta_pos, relative_rotation.as_rotvec()])

    def _retarget_rel(self, delta_pose: np.ndarray) -> np.ndarray:
        """Handle relative (delta) pose retargeting.

        Args:
            delta_pose: Delta pose of controller

        Returns:
            np.ndarray: 6D array with position delta (xyz) and rotation delta (rx,ry,rz)
        """
        # Get position and rotation
        position = delta_pose[:3]
        rotation = delta_pose[3:6]  # rx, ry, rz

        # Apply zero_out_xy_rotation
        if self._zero_out_xy_rotation:
            rotation[0] = 0  # x-axis
            rotation[1] = 0  # y-axis

        # Smooth and scale position
        self._smoothed_delta_pos = self._alpha_pos * position + (1 - self._alpha_pos) * self._smoothed_delta_pos
        if np.linalg.norm(self._smoothed_delta_pos) < self._position_threshold:
            self._smoothed_delta_pos = np.zeros(3)
        position = self._smoothed_delta_pos * self._delta_pos_scale_factor

        # Smooth and scale rotation
        self._smoothed_delta_rot = self._alpha_rot * rotation + (1 - self._alpha_rot) * self._smoothed_delta_rot
        if np.linalg.norm(self._smoothed_delta_rot) < self._rotation_threshold:
            self._smoothed_delta_rot = np.zeros(3)
        rotation = self._smoothed_delta_rot * self._delta_rot_scale_factor

        # Update visualization if enabled
        if self._enable_visualization:
            # Convert rotation vector to quaternion and combine with current rotation
            delta_quat = Rotation.from_rotvec(rotation).as_quat()
            current_rot = Rotation.from_quat([self._visualization_rot[1:], self._visualization_rot[0]])
            new_rot = Rotation.from_quat(delta_quat) * current_rot
            self._visualization_pos = self._visualization_pos + position
            self._visualization_rot = new_rot.as_quat()
            self._update_visualization()

        return np.concatenate([position, rotation])

    def _update_visualization(self):
        """Update visualization markers with current pose."""
        if self._enable_visualization:
            trans = np.array([self._visualization_pos])
            quat = Rotation.from_matrix(self._visualization_rot).as_quat()
            rot = np.array([quat])
            self._goal_marker.visualize(translations=trans, orientations=rot)


@dataclass
class ControllerSe3RelRetargeterCfg(RetargeterCfg):
    """Configuration for relative position controller retargeter."""

    zero_out_xy_rotation: bool = True
    delta_pos_scale_factor: float = 10.0
    delta_rot_scale_factor: float = 10.0
    alpha_pos: float = 0.5
    alpha_rot: float = 0.5
    enable_visualization: bool = False
    bound_controller: DeviceBase.TrackingTarget = DeviceBase.TrackingTarget.CONTROLLER_RIGHT
    retargeter_type: type[RetargeterBase] = ControllerSe3RelRetargeter
