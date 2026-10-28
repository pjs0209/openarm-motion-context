# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Optional gripper initialization for the open-jaw Lift reset curriculum.

A zero-mean Gaussian combined with a slow bidirectional gripper reference can
take many control ticks to discover a bilateral close. This experiment starts
the gripper mean slightly toward closing, without changing the controller,
Gaussian sampling, likelihood, rewards, observations or partner message. The
initial mean is learned normally thereafter. Use the same initialization for
every communication method compared within this training protocol.
"""

import math

import torch

from .grasp_reset_curriculum import build_grasp_reset_curriculum_mappo_agent


def build_grasp_exploration_mappo_agent(env, experiment_cfg):
    """Initialize only the gripper output layer; checkpoint loading restores learned weights."""
    mean = float(experiment_cfg.get("runtime", {}).get("gripper_initial_mean", -0.2))
    if not math.isfinite(mean) or not -1.0 <= mean <= 1.0:
        raise ValueError("gripper_initial_mean must be finite and within [-1, 1]")
    agent = build_grasp_reset_curriculum_mappo_agent(env, experiment_cfg)
    for modules in agent.models.values():
        policy = modules["policy"]
        if policy.gripper_action_head is None:
            raise ValueError("Grasp exploration initialization requires a gripper policy output")
        output = policy.gripper_action_head[-1]
        with torch.no_grad():
            output.weight.zero_()
            output.bias.fill_(mean)
    print(f"[INFO] Lift gripper initial Gaussian mean={mean}; weights remain trainable; execution unchanged")
    return agent
