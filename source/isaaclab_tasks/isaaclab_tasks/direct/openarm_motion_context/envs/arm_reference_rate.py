# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Optional OpenArm position-reference rates for the Newton Lift experiments.

This limits command trajectories, not solver velocities or contact impulses.
It never writes joint states, velocities, object poses or actuator properties.
The task retains ownership of reference buffers and resets them normally.
"""

import math
from types import MethodType

import torch

from ..legacy_api import read_data


def _pre_physics_step(self, actions):
    previous = {side: getattr(self, f"_{side}_arm_incremental_target").clone() for side in ("left", "right")}
    self._arm_reference_rate_original_pre_physics_step(actions)
    limits = read_data(self.robot.data, "joint_vel_limits")
    positions = read_data(self.robot.data, "joint_pos")
    velocities = read_data(self.robot.data, "joint_vel")
    debug = self._ctx.setdefault("debug_stats", {})
    for side in ("left", "right"):
        ids = getattr(self, f"{side}_arm_joint_ids")
        target = getattr(self, f"_{side}_arm_incremental_target")
        max_delta = limits[:, ids] * self.step_dt * self._arm_reference_rate_scale
        delta = (target - previous[side]).clamp(min=-max_delta, max=max_delta)
        target.copy_(previous[side] + delta)
        error = target - positions[:, ids]
        debug.update(
            {
                f"{side}_joint_target_error_norm": error.norm(dim=-1).mean().item(),
                f"{side}_joint_target_error_abs_max": error.abs().max().item(),
                f"arm_reference_rate/{side}_max_rad_s": (delta / self.step_dt).abs().max().item(),
                f"arm_reference_rate/{side}_actual_max_rad_s": velocities[:, ids].abs().max().item(),
                f"arm_reference_rate/{side}_actual_limit_ratio_max": (velocities[:, ids].abs() / limits[:, ids])
                .max()
                .item(),
            }
        )
    debug["arm_reference_rate/scale"] = self._arm_reference_rate_scale


def install_arm_reference_rate_limit(env, scale=1.0):
    """Cap each arm reference using declared rad/s limits once per policy tick.

    Scope this experiment to Lift and WithBall. The contact Peg task has a
    separate target-error guard, which must not be silently relaxed by a
    competing reference constraint. Frozen task defaults remain unchanged.
    """
    # skrl delegates task attributes but env.step calls the underlying task.
    # Binding the hook on the delegating wrapper would never limit commands.
    env = getattr(env, "unwrapped", env)
    if hasattr(env, "_arm_reference_rate_scale"):
        raise ValueError("An arm reference-rate limiter is already installed")
    if not math.isfinite(scale) or not 0.0 < scale <= 1.0:
        raise ValueError("Arm reference-rate scale must lie in (0, 1]")
    if env.cfg.intent_task_label not in ("openarm_lift", "openarm_lift_with_ball"):
        raise ValueError("This reference-rate experiment supports Lift and WithBall only")
    limits = read_data(env.robot.data, "joint_vel_limits")
    for side in ("left", "right"):
        selected = limits[:, getattr(env, f"{side}_arm_joint_ids")]
        if not torch.isfinite(selected).all() or not (selected > 0).all():
            raise ValueError("Every arm joint requires a finite positive declared velocity limit")
    env._arm_reference_rate_scale = float(scale)
    env._arm_reference_rate_original_pre_physics_step = env._pre_physics_step
    env._pre_physics_step = MethodType(_pre_physics_step, env)
