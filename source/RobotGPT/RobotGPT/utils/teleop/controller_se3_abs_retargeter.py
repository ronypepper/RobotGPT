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


class ControllerSe3AbsRetargeter(RetargeterBase):
    """Retargets OpenXR controller tracking data to end-effector commands using absolute positioning.

    This retargeter maps hand joint poses directly to robot end-effector positions and orientations,
    rather than using relative movements.

    Features:
    - Optional constraint to zero out X/Y rotations (keeping only Z-axis rotation)
    - Optional visualization of the target end-effector pose
    """

    def __init__(
        self,
        cfg: ControllerSe3AbsRetargeterCfg,
    ):
        """Initialize the retargeter.

        Args:
            pos_offset: An offset applied to the tracked controller position in world frame
            rot_offset: An offset applied to the tracked controller rotation in degrees (z, y, x)
            bound_controller: The controller to track (DeviceBase.TrackingTarget.CONTROLLER_LEFT or DeviceBase.TrackingTarget.CONTROLLER_RIGHT)
            zero_out_xy_rotation: If True, zero out rotation around x and y axes
            enable_visualization: If True, visualize the target pose in the scene
            device: The device to place the returned tensor on ('cpu' or 'cuda')
        """
        super().__init__(cfg)
        if cfg.bound_controller not in [DeviceBase.TrackingTarget.CONTROLLER_LEFT, DeviceBase.TrackingTarget.CONTROLLER_RIGHT]:
            raise ValueError(
                "bound_controller must be either DeviceBase.TrackingTarget.CONTROLLER_LEFT or DeviceBase.TrackingTarget.CONTROLLER_RIGHT"
            )
        self.bound_controller = cfg.bound_controller
        self.pos_offset = np.array(cfg.pos_offset)
        self.rot_offset = Rotation.from_euler("ZYX", cfg.rot_offset, degrees=True)
        self._zero_out_xy_rotation = cfg.zero_out_xy_rotation

        # Initialize visualization if enabled
        self._enable_visualization = cfg.enable_visualization
        if cfg.enable_visualization:
            frame_marker_cfg = FRAME_MARKER_CFG.copy()
            frame_marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
            self._goal_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_goal"))
            self._goal_marker.set_visibility(True)

    def retarget(self, data: dict) -> torch.Tensor:
        """Convert controller pose to robot end-effector command.

        Args:
            data: Dictionary mapping tracking targets to joint data dictionaries.

        Returns:
            torch.Tensor: 7D tensor containing position (xyz) and orientation (quaternion)
                for the robot end-effector
        """
        # Default pose
        ee_position = np.array([0.0, 0.0, 0.0])
        ee_rotation = np.array([1.0, 0.0, 0.0, 0.0])

        # Get controller data
        if self.bound_controller in data and data[self.bound_controller] is not None:
            controller_data = data[self.bound_controller]
            if len(controller_data) > DeviceBase.MotionControllerDataRowIndex.POSE.value:
                ee_pose = controller_data[DeviceBase.MotionControllerDataRowIndex.POSE.value]
                ee_position = ee_pose[0:3]
                ee_rotation = ee_pose[3:]

        # Apply frame offsets
        ee_position += self.pos_offset

        # ee_rotation is w,x,y,z but scipy expects x,y,z,w
        ee_rotation = Rotation.from_quat([*ee_rotation[1:], ee_rotation[0]])
        ee_rotation = ee_rotation * self.rot_offset

        # # Helper code for figuring out rotation offsets easily via controller inputs
        # # comment out "ee_rotation = ee_rotation * self.rot_offset" above when using
        # if DeviceBase.TrackingTarget.CONTROLLER_RIGHT in data and DeviceBase.TrackingTarget.CONTROLLER_LEFT in data:
        #     if data[DeviceBase.TrackingTarget.CONTROLLER_RIGHT] is not None and data[DeviceBase.TrackingTarget.CONTROLLER_LEFT] is not None:
        #         rx = data[DeviceBase.TrackingTarget.CONTROLLER_LEFT][1][0] * 90.0
        #         ry = data[DeviceBase.TrackingTarget.CONTROLLER_LEFT][1][2] * 90.0
        #         ry -= data[DeviceBase.TrackingTarget.CONTROLLER_LEFT][1][3] * 90.0
        #         rz = data[DeviceBase.TrackingTarget.CONTROLLER_RIGHT][1][0] * 90.0
        #         #rx: 88.86013627052307, ry: -90.0, rz: 0.0
        #         print(f"rx: {rx}, ry: {ry}, rz: {rz}")
        #         rr = Rotation.from_euler("ZYX", [rz, ry, rx], degrees=True)
        #         ee_rotation = ee_rotation * rr

        if self._zero_out_xy_rotation:
            z, y, x = ee_rotation.as_euler("ZYX")
            y = 0.0  # Zero out rotation around y-axis
            x = 0.0  # Zero out rotation around x-axis
            ee_rotation = Rotation.from_euler("ZYX", [z, y, x]) * Rotation.from_euler("X", np.pi, degrees=False)

        # Convert back to w,x,y,z format
        ee_rotation = ee_rotation.as_quat()
        ee_rotation = np.array([ee_rotation[3], ee_rotation[0], ee_rotation[1], ee_rotation[2]])  # Output remains w,x,y,z

        # Update visualization if enabled
        if self._enable_visualization:
            self._goal_marker.visualize(translations=np.array([ee_position]), orientations=np.array([ee_rotation]))

        # Convert to torch tensor
        ee_command = torch.tensor(np.concatenate([ee_position, ee_rotation]), dtype=torch.float32, device=self._sim_device)

        return ee_command

    def get_requirements(self) -> list[RetargeterBase.Requirement]:
        return [RetargeterBase.Requirement.MOTION_CONTROLLER]

    # def _retarget_abs(self, thumb_tip: np.ndarray, index_tip: np.ndarray, wrist: np.ndarray) -> np.ndarray:
    #     """Handle absolute pose retargeting.

    #     Args:
    #         thumb_tip: 7D array containing position (xyz) and orientation (quaternion)
    #             for the thumb tip
    #         index_tip: 7D array containing position (xyz) and orientation (quaternion)
    #             for the index tip
    #         wrist: 7D array containing position (xyz) and orientation (quaternion)
    #             for the wrist

    #     Returns:
    #         np.ndarray: 7D array containing position (xyz) and orientation (quaternion)
    #             for the robot end-effector
    #     """

    #     # Get position
    #     if self._use_wrist_position:
    #         position = wrist[:3]
    #     else:
    #         position = (thumb_tip[:3] + index_tip[:3]) / 2

    #     # Get rotation
    #     if self._use_wrist_rotation:
    #         # wrist is w,x,y,z but scipy expects x,y,z,w
    #         base_rot = Rotation.from_quat([*wrist[4:], wrist[3]])
    #     else:
    #         # Average the orientations of thumb and index using SLERP
    #         # thumb_tip is w,x,y,z but scipy expects x,y,z,w
    #         r0 = Rotation.from_quat([*thumb_tip[4:], thumb_tip[3]])
    #         # index_tip is w,x,y,z but scipy expects x,y,z,w
    #         r1 = Rotation.from_quat([*index_tip[4:], index_tip[3]])
    #         key_times = [0, 1]
    #         slerp = Slerp(key_times, Rotation.concatenate([r0, r1]))
    #         base_rot = slerp([0.5])[0]

    #     # Apply additional x-axis rotation to align with pinch gesture
    #     final_rot = base_rot * Rotation.from_euler("x", 90, degrees=True)

    #     if self._zero_out_xy_rotation:
    #         z, y, x = final_rot.as_euler("ZYX")
    #         y = 0.0  # Zero out rotation around y-axis
    #         x = 0.0  # Zero out rotation around x-axis
    #         final_rot = Rotation.from_euler("ZYX", [z, y, x]) * Rotation.from_euler("X", np.pi, degrees=False)

    #     # Convert back to w,x,y,z format
    #     quat = final_rot.as_quat()
    #     rotation = np.array([quat[3], quat[0], quat[1], quat[2]])  # Output remains w,x,y,z

    #     # Update visualization if enabled
    #     if self._enable_visualization:
    #         self._visualization_pos = position
    #         self._visualization_rot = rotation
    #         self._update_visualization()

    #     return np.concatenate([position, rotation])

    # def _update_visualization(self):
    #     """Update visualization markers with current pose.

    #     If visualization is enabled, the target end-effector pose is visualized in the scene.
    #     """
    #     if self._enable_visualization:
    #         trans = np.array([self._visualization_pos])
    #         quat = Rotation.from_matrix(self._visualization_rot).as_quat()
    #         rot = np.array([np.array([quat[3], quat[0], quat[1], quat[2]])])
    #         self._goal_marker.visualize(translations=trans, orientations=rot)


@dataclass
class ControllerSe3AbsRetargeterCfg(RetargeterCfg):
    """Configuration for absolute position controller retargeter."""

    pos_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
    rot_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
    zero_out_xy_rotation: bool = True
    enable_visualization: bool = False
    bound_controller: DeviceBase.TrackingTarget = DeviceBase.TrackingTarget.CONTROLLER_RIGHT
    retargeter_type: type[RetargeterBase] = ControllerSe3AbsRetargeter
