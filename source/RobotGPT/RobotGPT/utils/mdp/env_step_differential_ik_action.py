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

import logging
from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils
import isaaclab.utils.string as string_utils
from isaaclab.assets.articulation import Articulation
from isaaclab.controllers import DifferentialIKControllerCfg
from isaaclab.controllers.differential_ik import DifferentialIKController
from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv
    from isaaclab.envs.utils.io_descriptors import GenericActionIODescriptor

# import logger
logger = logging.getLogger(__name__)


class EnvStepDifferentialInverseKinematicsAction(ActionTerm):
    r"""Inverse Kinematics action term.

    This action term performs pre-processing of the raw actions using scaling transformation.

    .. math::
        \text{action} = \text{scaling} \times \text{input action}
        \text{joint position} = J^{-} \times \text{action}

    where :math:`\text{scaling}` is the scaling applied to the input action, and :math:`\text{input action}`
    is the input action from the user, :math:`J` is the Jacobian over the articulation's actuated joints,
    and \text{joint position} is the desired joint position command for the articulation's joints.
    """

    cfg: EnvStepDifferentialInverseKinematicsActionCfg
    """The configuration of the action term."""
    _asset: Articulation
    """The articulation asset on which the action term is applied."""
    _scale: torch.Tensor
    """The scaling factor applied to the input action. Shape is (1, action_dim)."""
    _clip: torch.Tensor
    """The clip applied to the input action."""

    def __init__(self, cfg: EnvStepDifferentialInverseKinematicsActionCfg, env: ManagerBasedEnv):
        # initialize the action term
        super().__init__(cfg, env)

        # resolve the joints over which the action term is applied
        self._joint_ids, self._joint_names = self._asset.find_joints(self.cfg.joint_names)
        self._num_joints = len(self._joint_ids)
        # parse the body index
        body_ids, body_names = self._asset.find_bodies(self.cfg.body_name)
        if len(body_ids) != 1:
            raise ValueError(
                f"Expected one match for the body name: {self.cfg.body_name}. Found {len(body_ids)}: {body_names}."
            )
        # save only the first body index
        self._body_idx = body_ids[0]
        self._body_name = body_names[0]
        self._jacobi_body_idx = self._body_idx - 1 if self._asset.is_fixed_base else self._body_idx
        self._jacobi_joint_ids = [j + self._asset.num_base_dofs for j in self._joint_ids]

        # log info for debugging
        logger.info(
            f"Resolved joint names for the action term {self.__class__.__name__}:"
            f" {self._joint_names} [{self._joint_ids}]"
        )
        logger.info(
            f"Resolved body name for the action term {self.__class__.__name__}: {self._body_name} [{self._body_idx}]"
        )
        # Avoid indexing across all joints for efficiency
        if self._num_joints == self._asset.num_joints:
            self._joint_ids = slice(None)

        # create the differential IK controller
        self._ik_controller = DifferentialIKController(
            cfg=self.cfg.controller, num_envs=self.num_envs, device=self.device
        )

        # create tensors for raw, scaled_clipped and processed actions
        self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self._scaled_clipped_actions = torch.zeros_like(self._raw_actions)
        self._processed_actions = torch.zeros(self.num_envs, self._num_joints, device=self.device)

        # owned buffer; _compute_frame_jacobian mutates this, not the data-layer view.
        self._jacobian_b = torch.zeros(self.num_envs, 6, len(self._jacobi_joint_ids), device=self.device)

        # save the scale as tensors
        self._scale = torch.zeros((self.num_envs, self.action_dim), device=self.device)
        self._scale[:] = torch.tensor(self.cfg.scale, device=self.device)

        # convert the fixed offsets to torch tensors of batched shape
        if self.cfg.body_offset is not None:
            self._offset_pos = torch.tensor(self.cfg.body_offset.pos, device=self.device).repeat(self.num_envs, 1)
            self._offset_rot = torch.tensor(self.cfg.body_offset.rot, device=self.device).repeat(self.num_envs, 1)
        else:
            self._offset_pos, self._offset_rot = None, None
        if self.cfg.world_offset is not None:
            if self.cfg.auto_world_offset:
                raise ValueError("world_offset must not be set if auto_world_offset is True")
            self._world_offset_pos = torch.tensor(self.cfg.world_offset.pos, device=self.device).repeat(self.num_envs, 1)
            self._world_offset_rot = torch.tensor(self.cfg.world_offset.rot, device=self.device).repeat(self.num_envs, 1)
        else:
            self._world_offset_pos, self._world_offset_rot = None, None

        # parse clip
        if self.cfg.clip is not None:
            if isinstance(cfg.clip, dict):
                self._clip = torch.tensor([[-float("inf"), float("inf")]], device=self.device).repeat(
                    self.num_envs, self.action_dim, 1
                )
                index_list, _, value_list = string_utils.resolve_matching_names_values(self.cfg.clip, self._joint_names)
                self._clip[:, index_list] = torch.tensor(value_list, device=self.device)
            else:
                raise ValueError(f"Unsupported clip type: {type(cfg.clip)}. Supported types are dict.")

    """
    Properties.
    """

    @property
    def action_dim(self) -> int:
        return self._ik_controller.action_dim

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    @property
    def jacobian_w(self) -> torch.Tensor:
        return self._asset.data.body_link_jacobian_w.torch[:, self._jacobi_body_idx, :, self._jacobi_joint_ids]

    @property
    def jacobian_b(self) -> torch.Tensor:
        jacobian = self.jacobian_w
        base_rot = self._asset.data.root_quat_w.torch
        base_rot_matrix = math_utils.matrix_from_quat(math_utils.quat_inv(base_rot))
        jacobian[:, :3, :] = torch.bmm(base_rot_matrix, jacobian[:, :3, :])
        jacobian[:, 3:, :] = torch.bmm(base_rot_matrix, jacobian[:, 3:, :])
        return jacobian

    @property
    def IO_descriptor(self) -> GenericActionIODescriptor:
        """The IO descriptor of the action term.

        This descriptor is used to describe the action term of the pink inverse kinematics action.
        It adds the following information to the base descriptor:
        - body_name: The name of the body.
        - joint_names: The names of the joints.
        - scale: The scale of the action term.
        - clip: The clip of the action term.
        - controller_cfg: The configuration of the controller.
        - body_offset: The offset of the body.

        Returns:
            The IO descriptor of the action term.
        """
        super().IO_descriptor
        self._IO_descriptor.shape = (self.action_dim,)
        self._IO_descriptor.dtype = str(self.raw_actions.dtype)
        self._IO_descriptor.action_type = "TaskSpaceAction"
        self._IO_descriptor.body_name = self._body_name
        self._IO_descriptor.joint_names = self._joint_names
        self._IO_descriptor.scale = self._scale
        if self.cfg.clip is not None:
            self._IO_descriptor.clip = self.cfg.clip
        else:
            self._IO_descriptor.clip = None
        self._IO_descriptor.extras["controller_cfg"] = self.cfg.controller.__dict__
        self._IO_descriptor.extras["body_offset"] = self.cfg.body_offset.__dict__
        return self._IO_descriptor

    """
    Operations.
    """

    def process_actions(self, actions: torch.Tensor):
        # Auto set world offset on first call
        if self.cfg.auto_world_offset and self._world_offset_pos is None:
            # obtain quantities from simulation
            ee_pos_w = self._asset.data.body_pos_w.torch[:, self._body_idx]
            ee_quat_w = self._asset.data.body_quat_w.torch[:, self._body_idx]

            self._world_offset_pos, self._world_offset_rot = math_utils.subtract_frame_transforms(
                actions[:, :3], actions[:, 3:], ee_pos_w, ee_quat_w
            )

        # Apply world offset
        if self.cfg.world_offset is not None:
            actions[:, :3], actions[:, 3:] = math_utils.combine_frame_transforms(
                self._world_offset_pos, self._world_offset_rot, actions[:, :3], actions[:, 3:]
            )

        # store the raw actions
        self._raw_actions[:] = actions
        self._scaled_clipped_actions[:] = self.raw_actions * self._scale
        if self.cfg.clip is not None:
            self._scaled_clipped_actions = torch.clamp(
                self._scaled_clipped_actions, min=self._clip[:, :, 0], max=self._clip[:, :, 1]
            )
        # obtain quantities from simulation
        ee_pos_curr, ee_quat_curr = self._compute_frame_pose()
        # set command into controller
        self._ik_controller.set_command(self._scaled_clipped_actions, ee_pos_curr, ee_quat_curr)

        # ------  Moved here from apply_actions() ------
        joint_pos = self._asset.data.joint_pos.torch[:, self._joint_ids]
        # compute the delta in joint-space
        if ee_quat_curr.norm() != 0:
            jacobian = self._compute_frame_jacobian()
            self._processed_actions = self._ik_controller.compute(ee_pos_curr, ee_quat_curr, jacobian, joint_pos)
        else:
            self._processed_actions = joint_pos.clone()

    def apply_actions(self):
        # set the joint position command
        self._asset.set_joint_position_target_index(target=self._processed_actions, joint_ids=self._joint_ids)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        self._raw_actions[env_ids] = 0.0

    """
    Helper functions.
    """

    def _compute_frame_pose(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Computes the pose of the target frame in the root frame.

        Returns:
            A tuple of the body's position and orientation in the root frame.
        """
        # obtain quantities from simulation
        ee_pos_w = self._asset.data.body_pos_w.torch[:, self._body_idx]
        ee_quat_w = self._asset.data.body_quat_w.torch[:, self._body_idx]
        root_pos_w = self._asset.data.root_pos_w.torch
        root_quat_w = self._asset.data.root_quat_w.torch
        # compute the pose of the body in the root frame
        ee_pose_b, ee_quat_b = math_utils.subtract_frame_transforms(root_pos_w, root_quat_w, ee_pos_w, ee_quat_w)
        # account for the offset
        if self.cfg.body_offset is not None:
            ee_pose_b, ee_quat_b = math_utils.combine_frame_transforms(
                ee_pose_b, ee_quat_b, self._offset_pos, self._offset_rot
            )

        return ee_pose_b, ee_quat_b

    def _compute_frame_jacobian(self):
        """Computes the geometric Jacobian of the target frame in the root frame.

        This function accounts for the target frame offset and applies the necessary transformations to obtain
        the right Jacobian from the parent body Jacobian.
        """
        self._jacobian_b[:] = self.jacobian_b
        # account for the offset
        if self.cfg.body_offset is not None:
            # Modify the jacobian to account for the offset
            # -- translational part
            # v_link = v_ee + w_ee x r_link_ee = v_J_ee * q + w_J_ee * q x r_link_ee
            #        = (v_J_ee + w_J_ee x r_link_ee ) * q
            #        = (v_J_ee - r_link_ee_[x] @ w_J_ee) * q
            self._jacobian_b[:, 0:3, :] += torch.bmm(
                -math_utils.skew_symmetric_matrix(self._offset_pos), self._jacobian_b[:, 3:, :]
            )
            # -- rotational part
            # w_link = R_link_ee @ w_ee
            self._jacobian_b[:, 3:, :] = torch.bmm(
                math_utils.matrix_from_quat(self._offset_rot), self._jacobian_b[:, 3:, :]
            )

        return self._jacobian_b


@configclass
class EnvStepDifferentialInverseKinematicsActionCfg(ActionTermCfg):
    """Configuration for inverse differential kinematics action term.

    See :class:`DifferentialInverseKinematicsAction` for more details.
    """

    @configclass
    class OffsetCfg:
        """The offset pose from parent frame to child frame.

        On many robots, end-effector frames are fictitious frames that do not have a corresponding
        rigid body. In such cases, it is easier to define this transform w.r.t. their parent rigid body.
        For instance, for the Franka Emika arm, the end-effector is defined at an offset to the the
        "panda_hand" frame.
        """

        pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
        """Translation w.r.t. the parent frame. Defaults to (0.0, 0.0, 0.0)."""
        rot: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
        """Quaternion rotation ``(x, y, z, w)`` w.r.t. the parent frame. Defaults to (0.0, 0.0, 0.0, 1.0)."""

    class_type: type[ActionTerm] = EnvStepDifferentialInverseKinematicsAction

    joint_names: list[str] = MISSING
    """List of joint names or regex expressions that the action will be mapped to."""
    body_name: str = MISSING
    """Name of the body or frame for which IK is performed."""
    body_offset: OffsetCfg | None = None
    """Offset of target frame w.r.t. to the body frame. Defaults to None, in which case no offset is applied."""
    scale: float | tuple[float, ...] = 1.0
    """Scale factor for the action. Defaults to 1.0."""
    controller: DifferentialIKControllerCfg = MISSING
    """The configuration for the differential IK controller."""

    world_offset: OffsetCfg | None = None
    """Additional world offset directly applied to the raw actions"""
    auto_world_offset: bool = False
    """Sets the world offset to match the first pose received with the gripper, ie all input poses are transformed to be
      relative to the first input pose."""
