# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Checkpointed Lift control experiment with physically bounded joint targets."""

import torch

from ..envs.arm_target_bounds import install_arm_target_bounds

_VERSION = "_arm_target_bounds_control_version"
_ERROR = "_arm_target_bounds_error_rad"


def _validate_checkpoint(module, state, prefix, *args):
    for name in (_VERSION, _ERROR):
        value = state.get(prefix + name)
        if value is None or not torch.equal(value.detach().cpu(), getattr(module, name).detach().cpu()):
            raise ValueError("Arm target-bounds checkpoint protocol mismatch")


def build_target_bounded_mappo_agent(env, experiment_cfg):
    """Retain Gaussian actions, likelihoods and gripper policy; constrain arm references."""
    from .actor import build_motion_context_mappo_agent

    bound = float(experiment_cfg.get("runtime", {}).get("arm_target_error_rad", 0.2))
    install_arm_target_bounds(env, bound)
    agent = build_motion_context_mappo_agent(env, experiment_cfg)
    for modules in agent.models.values():
        policy = modules["policy"]
        template = policy.arm_action_head.weight
        policy.register_buffer(_VERSION, template.new_tensor(1.0))
        policy.register_buffer(_ERROR, template.new_tensor(bound))
        policy.register_load_state_dict_pre_hook(_validate_checkpoint)
    print(f"[INFO] arm target-bounds v1: physical joint limits, target-error={bound}rad; no state clamping")
    return agent
