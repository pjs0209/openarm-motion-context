# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Control-tick gripper trajectories for backends without a hard speed clamp."""

import torch


def update_gripper_command(
    command: torch.Tensor,
    closed: torch.Tensor,
    open_command: torch.Tensor,
    close_command: torch.Tensor,
    max_delta: float,
) -> None:
    """Advance an existing position command once; zero retains the legacy step target.

    This bounds reference speed, not velocities caused by contact impulses.
    The task owns and resets the command; actuator writes hold it unchanged
    through physics substeps on either decimation path.
    """
    desired = torch.where(closed, close_command, open_command)
    if max_delta > 0.0:
        command.add_((desired - command).clamp(-max_delta, max_delta))
    else:
        command.copy_(desired)
