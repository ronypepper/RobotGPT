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

from collections.abc import Sequence

import torch

from isaaclab.assets.articulation.articulation import Articulation
from isaaclab.envs.manager_based_env import ManagerBasedEnv
from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg
from isaaclab.utils.configclass import configclass

INSPIRE_JOINT_NAMES = [
    # Hand joints
    ".*_index_proximal_joint",  # 0...1.7 / 0...0.7
    ".*_middle_proximal_joint",
    ".*_pinky_proximal_joint",
    ".*_ring_proximal_joint",
    ".*_thumb_proximal_pitch_joint",  # -0.1...0.6 / 0...0.26
    ".*_index_intermediate_joint",
    ".*_middle_intermediate_joint",
    ".*_pinky_intermediate_joint",
    ".*_ring_intermediate_joint",
    ".*_thumb_intermediate_joint",
    ".*_thumb_distal_joint",
    ".*_thumb_proximal_yaw_joint",  # -0.1...1.3 / 1.3
]

INSPIRE_JOINT_SCALES = [
    0.7,  # ".*_index_proximal_joint",
    0.7,  # ".*_middle_proximal_joint",
    0.7,  # ".*_pinky_proximal_joint",
    0.7,  # ".*_ring_proximal_joint",
    0.26,  # ".*_thumb_proximal_pitch_joint",
    0.7,  # ".*_index_intermediate_joint",
    0.7,  # ".*_middle_intermediate_joint",
    0.7,  # ".*_pinky_intermediate_joint",
    0.7,  # ".*_ring_intermediate_joint",
    0.26 * 1.6,  # ".*_thumb_intermediate_joint",
    0.26 * 2.4,  # ".*_thumb_distal_joint",
]

INSPIRE_THUMB_YAW_OFFSET = 1.3


class InspireHandAction(ActionTerm):
    """Action term that maps a single action value in range [-1, 1] to joint position commands for the Inspire 6-DOF
    dexterous hand, with 1.0 corresponding to the hand fully open and -1.0 corresponding to fully closed."""

    cfg: InspireHandActionCfg
    """The configuration of the action term."""
    _asset: Articulation
    """The articulation asset on which the action term is applied."""

    def __init__(self, cfg: InspireHandActionCfg, env: ManagerBasedEnv):
        # initialize the action term
        super().__init__(cfg, env)

        # resolve the joints over which the action term is applied
        self._inspire_joint_names = [s.replace(".*", "L" if cfg.left_hand else "R") for s in INSPIRE_JOINT_NAMES]
        self._joint_ids, self._joint_names = self._asset.find_joints(
            self._inspire_joint_names, preserve_order=True
        )
        self._num_joints = len(self._joint_ids)
        assert self._num_joints == 12

        # create tensors for raw and processed actions
        self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self._processed_actions = torch.zeros(self.num_envs, self._num_joints, device=self.device)

        # create tensor for joint scales
        self._inspire_joint_scales = torch.tensor(INSPIRE_JOINT_SCALES, device=self.device)

    """
    Properties.
    """

    @property
    def action_dim(self) -> int:
        return 1

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    """
    Operations.
    """

    def process_actions(self, actions: torch.Tensor):
        # store the raw actions
        self._raw_actions[:] = (actions - 1.0) * -0.5

        # Map raw action to inspire hand joint position actions
        self._processed_actions[:, :-1] = self._raw_actions * self._inspire_joint_scales
        self._processed_actions[:, -1] = INSPIRE_THUMB_YAW_OFFSET

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        self._raw_actions[env_ids] = 0.0

    def apply_actions(self):
        # set position targets
        self._asset.set_joint_position_target_index(target=self.processed_actions, joint_ids=self._joint_ids)


@configclass
class InspireHandActionCfg(ActionTermCfg):
    """Configuration for the Inspire hand action term.

    See :class:`InspireHandAction` for more details.
    """

    class_type: type[ActionTerm] = InspireHandAction

    left_hand: bool = False
