# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Optional own-controller state; no partner, task phase, force or reward inputs."""

import torch

CONTROL_STATE_DIM = 10
PROFILE = "own_arm_target_error_gripper_reference_latch_reset_grace/v1"


def control_state_observation(env, side: str, joint_pos: torch.Tensor) -> torch.Tensor:
    """Return 7 target errors [rad], jaw reference [m], latch and grace time [s].

    The reference is the current rate-limited symmetric jaw command, not the
    measured opening or the final open/close setpoint. These features expose
    local controller memory that can otherwise differ at the same measured pose.
    """
    target = getattr(env, f"_{side}_arm_incremental_target")
    reference = getattr(env, f"_{side}_gripper_position_command").mean(dim=-1, keepdim=True)
    closed = getattr(env, f"_{side}_gripper_closed").to(dtype=joint_pos.dtype)
    grace = env._reset_grace_left.to(dtype=joint_pos.dtype).unsqueeze(-1) * float(env.cfg.sim.dt * env.cfg.decimation)
    return torch.cat((target - joint_pos, reference, closed, grace), dim=-1)
