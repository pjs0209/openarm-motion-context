# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Optional physical joint bounds and anti-windup for Lift arm references."""

import math
from types import MethodType

import torch

from ..legacy_api import read_data


def _pre_physics_step(self, actions):
    self._arm_target_bounds_original_pre_physics_step(actions)
    positions = read_data(self.robot.data, "joint_pos")
    limits = read_data(self.robot.data, "joint_pos_limits")
    debug = self._ctx.setdefault("debug_stats", {})
    for side in ("left", "right"):
        ids = getattr(self, f"{side}_arm_joint_ids")
        target = getattr(self, f"_{side}_arm_incremental_target")
        default = getattr(self, f"{side}_arm_default_joint_pos")
        lower = torch.maximum(limits[:, ids, 0], default - self.cfg.action_scale)
        upper = torch.minimum(limits[:, ids, 1], default + self.cfg.action_scale)
        requested = target.clone()
        target.clamp_(min=lower, max=upper)
        current = positions[:, ids]
        bound = self._arm_target_bounds_error_rad
        if bound > 0.0:
            # A contact impulse may move actual q outside the legal interval.
            # In that case prioritize a legal reference; never write q back.
            near_lower = (current - bound).clamp(min=lower, max=upper)
            near_upper = (current + bound).clamp(min=lower, max=upper)
            target.clamp_(min=near_lower, max=near_upper)
        error = target - current
        debug.update(
            {
                f"{side}_joint_target_error_norm": error.norm(dim=-1).mean().item(),
                f"{side}_joint_target_error_abs_max": error.abs().max().item(),
                f"arm_target_bounds/{side}_clipped_ratio": (target != requested).float().mean().item(),
                f"arm_target_bounds/{side}_outside_physical_limits_ratio": (
                    (target < limits[:, ids, 0]) | (target > limits[:, ids, 1])
                )
                .float()
                .mean()
                .item(),
                f"arm_target_bounds/{side}_actual_outside_reference_interval_ratio": (
                    (current < lower) | (current > upper)
                )
                .float()
                .mean()
                .item(),
            }
        )
    debug["arm_target_bounds/error_rad"] = self._arm_target_bounds_error_rad


def install_arm_target_bounds(env, max_error_rad=0.2):
    """Constrain references without clamping physical state or changing the policy.

    The task owns target/reset buffers and gripper trajectories. A zero error
    bound disables only anti-windup; authored joint and action bounds still
    apply. This is a separate experiment from the reference-rate controller.
    """
    env = getattr(env, "unwrapped", env)
    if hasattr(env, "_arm_target_bounds_error_rad") or hasattr(env, "_arm_reference_rate_scale"):
        raise ValueError("Do not install competing OpenArm arm reference controllers")
    if env.cfg.intent_task_label not in ("openarm_lift", "openarm_lift_with_ball"):
        raise ValueError("Arm target bounds support Lift and WithBall only")
    if not math.isfinite(max_error_rad) or max_error_rad < 0.0:
        raise ValueError("Maximum arm target error must be finite and nonnegative")
    limits = read_data(env.robot.data, "joint_pos_limits")
    for side in ("left", "right"):
        ids = getattr(env, f"{side}_arm_joint_ids")
        default = getattr(env, f"{side}_arm_default_joint_pos")
        lower = torch.maximum(limits[:, ids, 0], default - env.cfg.action_scale)
        upper = torch.minimum(limits[:, ids, 1], default + env.cfg.action_scale)
        if not torch.isfinite(lower).all() or not torch.isfinite(upper).all() or (lower > upper).any():
            raise ValueError("Arm action bounds must intersect finite physical joint limits")
    env._arm_target_bounds_error_rad = float(max_error_rad)
    env._arm_target_bounds_original_pre_physics_step = env._pre_physics_step
    env._pre_physics_step = MethodType(_pre_physics_step, env)
