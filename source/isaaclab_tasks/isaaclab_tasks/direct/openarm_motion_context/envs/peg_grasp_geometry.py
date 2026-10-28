# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Geometry-only retention metrics for the free-object Peg task."""

import torch

from ..legacy_api import read_data
from ..legacy_math import quat_apply


def grasp_point_errors(env, ctx):
    """Measure actual part grasp frames relative to the two current TCPs."""
    body_positions = read_data(env.robot.data, "body_pos_w")
    errors = []
    for side, role in (("left", "peg"), ("right", "hole")):
        part = getattr(env, role)
        point = read_data(part.data, "root_pos_w") + quat_apply(
            read_data(part.data, "root_quat_w"), ctx[f"{role}_grip_local_pos"]
        )
        errors.append(torch.linalg.vector_norm(point - body_positions[:, ctx[f"{side}_ee_body_id"]], dim=-1))
    return tuple(errors)


def physical_grasp_validity(env, ctx):
    """Require both commanded grasps to retain their actual parts near the TCP.

    This kinematic evaluation gate does not apply forces or create contacts.
    Native release probes independently validate that retention comes from
    physical jaw contact rather than a hidden attachment.
    """
    left, right = grasp_point_errors(env, ctx)
    return (
        (left <= env.cfg.grasp_success_distance)
        & (right <= env.cfg.grasp_success_distance)
        & env._left_gripper_closed[:, 0]
        & env._right_gripper_closed[:, 0]
    )
