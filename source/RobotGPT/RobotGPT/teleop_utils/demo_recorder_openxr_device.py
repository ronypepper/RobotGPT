# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""OpenXR-powered device for teleoperation and interaction with additional capabilities to signal reset."""

from __future__ import annotations

import contextlib
from dataclasses import dataclass

from isaaclab.devices.device_base import DeviceBase, RetargeterBase
from isaaclab.devices.openxr.openxr_device import OpenXRDevice, OpenXRDeviceCfg


class DemoRecorderOpenXRDevice(OpenXRDevice):
    def __init__(
        self,
        cfg: DemoRecorderOpenXRDeviceCfg,
        retargeters: list[RetargeterBase] | None = None,
    ):
        super().__init__(cfg, retargeters)
        with contextlib.suppress(Exception):
            self._bind_button_press(
                "/user/hand/left",
                "x",
                "signal_success",
                lambda ev: self._signal_success(),
            )
            self._bind_button_press(
                "/user/hand/right",
                "b",
                "signal_reset",
                lambda ev: self._signal_reset(),
            )
            self._bind_button_press(
                "/user/hand/right",
                "a",
                "signal_start_stop",
                lambda ev: self._signal_start_stop(),
            )

    def _signal_reset(self):
        if "RESET" in self._additional_callbacks:
            self._additional_callbacks["RESET"]()
        self.reset()

    def _signal_success(self):
        if "SUCCESS" in self._additional_callbacks:
            self._additional_callbacks["SUCCESS"]()

    def _signal_start_stop(self):
        if "START_STOP" in self._additional_callbacks:
            self._additional_callbacks["START_STOP"]()

@dataclass
class DemoRecorderOpenXRDeviceCfg(OpenXRDeviceCfg):
    class_type: type[DeviceBase] = DemoRecorderOpenXRDevice
