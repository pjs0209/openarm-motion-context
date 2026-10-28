# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Record pre-action physical starts without changing simulation or random state."""

from collections.abc import Callable

from ..legacy_api import read_data


def capture_physical_start(resolve: Callable, index: int) -> dict[str, list[float]]:
    """Copy one world's robot and task bodies in the saved WXYZ convention.

    This is an observation of the actual start, not a reset mechanism. Equal
    records do not certify equality of hidden solver or random-generator state.
    """
    robot = resolve("robot")
    fields = {
        "robot_root_state_w": read_data(robot.data, "root_state_w"),
        "robot_joint_pos": read_data(robot.data, "joint_pos"),
        "robot_joint_vel": read_data(robot.data, "joint_vel"),
    }
    for name in ("object", "ball", "peg", "hole"):
        asset = resolve(name, None)
        if asset is not None:
            fields[f"{name}_root_state_w"] = read_data(asset.data, "root_state_w")
    return {name: value[index].detach().float().cpu().tolist() for name, value in fields.items()}
