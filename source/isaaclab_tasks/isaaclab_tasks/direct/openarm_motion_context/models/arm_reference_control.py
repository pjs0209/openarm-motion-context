# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Checkpointed control experiment: plain MAPPO with nominal arm reference rates."""

import torch

from ..envs.arm_reference_rate import install_arm_reference_rate_limit

_VERSION = "_arm_reference_rate_control_version"
_SCALE = "_arm_reference_rate_control_scale"


def _validate_checkpoint(module, state, prefix, *args):
    for name in (_VERSION, _SCALE):
        value = state.get(prefix + name)
        expected = getattr(module, name)
        if value is None or not torch.equal(value.detach().cpu(), expected.detach().cpu()):
            raise ValueError("Arm reference-rate checkpoint protocol mismatch")


def build_rate_limited_mappo_agent(env, experiment_cfg):
    """Keep the plain policy/distribution; change only arm command slew rates."""
    from .actor import _unwrap_env, build_motion_context_mappo_agent

    task = _unwrap_env(env)
    scale = float(experiment_cfg.get("runtime", {}).get("arm_reference_rate_scale", 1.0))
    install_arm_reference_rate_limit(task, scale)
    agent = build_motion_context_mappo_agent(env, experiment_cfg)
    for modules in agent.models.values():
        policy = modules["policy"]
        template = policy.arm_action_head.weight
        policy.register_buffer(_VERSION, template.new_tensor(2.0))
        policy.register_buffer(_SCALE, template.new_tensor(scale))
        policy.register_load_state_dict_pre_hook(_validate_checkpoint)
    print(f"[INFO] arm reference-rate v2: declared-rate scale={scale}; plain policy; no state/velocity clamping")
    return agent
