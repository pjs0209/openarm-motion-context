# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Explicit native3 boundaries for saved OpenArm WXYZ observation semantics."""

import torch
import warp as wp
from isaaclab.utils.warp import ProxyArray

from .legacy_math import from_native, to_native

_QUATERNIONS = {"root_quat_w", "body_quat_w", "quat_w_ros", "quat_w_world", "quat_w_opengl"}
_STATES = {"root_state_w", "body_state_w", "default_root_state"}


def tensor(value):
    if isinstance(value, ProxyArray):
        return value.torch
    if isinstance(value, wp.array):
        return wp.to_torch(value)
    return value


def read_data(data, field):
    value = tensor(getattr(data, field))
    if field in _QUATERNIONS:
        return from_native(value)
    if field in _STATES:
        converted = value.clone()
        converted[..., 3:7] = from_native(value[..., 3:7])
        return converted
    return value


def write_root_pose(asset, pose, env_ids=None):
    converted = pose.clone()
    converted[..., 3:7] = to_native(pose[..., 3:7])
    asset.write_root_pose_to_sim_index(root_pose=converted, env_ids=env_ids)


def set_joint_position(asset, value, joint_ids=None, env_ids=None):
    asset.actuators.target_command.set_position_index(value=value, joint_ids=joint_ids, env_ids=env_ids)
