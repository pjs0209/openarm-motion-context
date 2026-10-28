# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""WXYZ policy/task interface implemented through native3 XYZW math.

Old actor observations and saved task targets remain WXYZ. Simulator data and
write/config boundaries explicitly convert; do not pass these helpers raw3.x
quaternions. USD quaternion attributes remain scalar-first and are unchanged.
"""

import torch
from isaaclab.utils import math as native


@torch.jit.script
def to_native(q: torch.Tensor) -> torch.Tensor:
    return torch.cat((q[..., 1:], q[..., :1]), dim=-1)


@torch.jit.script
def from_native(q: torch.Tensor) -> torch.Tensor:
    return torch.cat((q[..., -1:], q[..., :3]), dim=-1)


@torch.jit.script
def quat_apply(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    return native.quat_apply(to_native(q), v)


@torch.jit.script
def quat_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    return from_native(native.quat_mul(to_native(q1), to_native(q2)))


@torch.jit.script
def quat_inv(q: torch.Tensor) -> torch.Tensor:
    return from_native(native.quat_inv(to_native(q)))


@torch.jit.script
def matrix_from_quat(q: torch.Tensor) -> torch.Tensor:
    return native.matrix_from_quat(to_native(q))


@torch.jit.script
def quat_error_magnitude(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    return native.quat_error_magnitude(to_native(q1), to_native(q2))


@torch.jit.script
def quat_from_euler_xyz(roll: torch.Tensor, pitch: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
    return from_native(native.quat_from_euler_xyz(roll, pitch, yaw))


def combine_frame_transforms(t01, q01, t12=None, q12=None):
    pos, quat = native.combine_frame_transforms(t01, to_native(q01), t12, to_native(q12) if q12 is not None else None)
    return pos, from_native(quat)


def subtract_frame_transforms(t01, q01, t02=None, q02=None):
    pos, quat = native.subtract_frame_transforms(t01, to_native(q01), t02, to_native(q02) if q02 is not None else None)
    return pos, from_native(quat)
