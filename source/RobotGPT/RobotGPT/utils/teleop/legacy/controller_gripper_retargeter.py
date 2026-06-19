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

import torch

from isaaclab.devices.device_base import DeviceBase
from isaaclab.devices.retargeter_base import RetargeterBase, RetargeterCfg


class ControllerGripperRetargeter(RetargeterBase):
    """
    Retargeter specifically for gripper control based on controller tracking data.
    Analog trigger on controller directly commands gripper.
    """

    def __init__(
        self,
        cfg: ControllerGripperRetargeterCfg,
    ):
        super().__init__(cfg)
        """Initialize the gripper retargeter."""
        # Store configuration
        if cfg.bound_controller not in [DeviceBase.TrackingTarget.CONTROLLER_LEFT, DeviceBase.TrackingTarget.CONTROLLER_RIGHT]:
            raise ValueError(
                "bound_controller must be either DeviceBase.TrackingTarget.CONTROLLER_LEFT or DeviceBase.TrackingTarget.CONTROLLER_RIGHT"
            )
        self.bound_controller = cfg.bound_controller

        if cfg.num_joints < 1:
            raise ValueError("num_joints must be greater or equal to 1")
        self.num_joints = cfg.num_joints

    def retarget(self, data: dict) -> torch.Tensor:
        """Convert controller data to gripper command.

        Args:
            data: Dictionary mapping tracking targets to joint data dictionaries.

        Returns:
            torch.Tensor: Tensor containing a single scalar representing gripper command where 1.0 = gripper closed, and 0.0 = gripper opened
        """
        # Default gripper command
        gripper_command = 0.0

        # Get controller data
        if self.bound_controller in data and data[self.bound_controller] is not None:
            controller_data = data[self.bound_controller]
            if len(controller_data) > DeviceBase.MotionControllerDataRowIndex.INPUTS.value:
                inputs = controller_data[DeviceBase.MotionControllerDataRowIndex.INPUTS.value]
                if len(inputs) > DeviceBase.MotionControllerInputIndex.TRIGGER.value:
                    gripper_command = inputs[DeviceBase.MotionControllerInputIndex.TRIGGER.value]

        return torch.tensor([gripper_command for _ in range(self.num_joints)], dtype=torch.float32, device=self._sim_device)

    def get_requirements(self) -> list[RetargeterBase.Requirement]:
        return [RetargeterBase.Requirement.MOTION_CONTROLLER]


@dataclass
class ControllerGripperRetargeterCfg(RetargeterCfg):
    """Configuration for gripper retargeter."""

    bound_controller: DeviceBase.TrackingTarget = DeviceBase.TrackingTarget.CONTROLLER_RIGHT
    num_joints: int = 1
    retargeter_type: type[RetargeterBase] = ControllerGripperRetargeter
