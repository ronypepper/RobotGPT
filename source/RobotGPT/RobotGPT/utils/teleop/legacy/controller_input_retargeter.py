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


class ControllerInputRetargeter(RetargeterBase):
    """
    Retargeter that returns input from the controller.
    """

    def __init__(
        self,
        cfg: ControllerInputRetargeterCfg,
    ):
        super().__init__(cfg)
        """Initialize the input retargeter."""
        # Store configuration
        if cfg.bound_controller not in [DeviceBase.TrackingTarget.CONTROLLER_LEFT, DeviceBase.TrackingTarget.CONTROLLER_RIGHT]:
            raise ValueError(
                "bound_controller must be either DeviceBase.TrackingTarget.CONTROLLER_LEFT or DeviceBase.TrackingTarget.CONTROLLER_RIGHT"
            )
        self.bound_controller = cfg.bound_controller

        if cfg.input_selection is not DeviceBase.MotionControllerInputIndex:
            raise ValueError(
                "input_selection must be a member of DeviceBase.MotionControllerInputIndex"
            )
        self.input_selection = cfg.input_selection

    def retarget(self, data: dict) -> torch.Tensor:
        """Extract input from controller data.

        Args:
            data: Dictionary mapping tracking targets to joint data dictionaries.

        Returns:
            torch.Tensor: Tensor containing a single scalar equal to the controller's selected input.
        """
        # Default controller input
        controller_input = 0.0

        # Get controller data
        if self.bound_controller in data and data[self.bound_controller] is not None:
            controller_data = data[self.bound_controller]
            if len(controller_data) > DeviceBase.MotionControllerDataRowIndex.INPUTS.value:
                inputs = controller_data[DeviceBase.MotionControllerDataRowIndex.INPUTS.value]
                if len(inputs) > self.input_selection.value:
                    controller_input = inputs[self.input_selection.value]

        return torch.tensor(controller_input, dtype=torch.float32, device=self._sim_device)

    def get_requirements(self) -> list[RetargeterBase.Requirement]:
        return [RetargeterBase.Requirement.MOTION_CONTROLLER]


@dataclass
class ControllerInputRetargeterCfg(RetargeterCfg):
    """Configuration for input retargeter."""

    bound_controller: DeviceBase.TrackingTarget = DeviceBase.TrackingTarget.CONTROLLER_RIGHT
    input_selection: DeviceBase.MotionControllerInputIndex = DeviceBase.MotionControllerInputIndex.BUTTON_0
    retargeter_type: type[RetargeterBase] = ControllerInputRetargeter
