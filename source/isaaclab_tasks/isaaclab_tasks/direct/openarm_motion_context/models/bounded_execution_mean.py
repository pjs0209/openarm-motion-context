# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Opt-in arm-mean ablation, preserving raw Gaussian likelihoods.

The same parameter-free mapping applies to every communication method. Gripper
means stay unchanged: their magnitude controls the probability of close/open
switches, rather than a continuous joint increment. Messages, observations,
rewards, variances and execution clipping remain unchanged. Samples are not
squashed: PPO scores the raw command under its actual bounded arm mean.
"""

from types import MethodType

import torch

_VERSION = "_bounded_execution_mean_version"


def _compute(self, inputs, role=""):
    mean, log_std, outputs = self._unbounded_execution_mean_compute(inputs, role)
    width = self.arm_action_head.out_features
    arm = mean[..., :width]
    arm = arm / (1.0 + arm.abs())
    bounded = torch.cat((arm, mean[..., width:]), dim=-1)
    return bounded, log_std, {**outputs, "mean_actions": bounded}


def _validate_checkpoint(module, state, prefix, *args):
    for name in (_VERSION,):
        actual = state.get(prefix + name)
        expected = getattr(module, name)
        if actual is None or not torch.equal(actual.detach().cpu(), expected.detach().cpu()):
            raise ValueError("Bounded arm-mean checkpoint version mismatch")


def install_mean_bound(policy):
    """Add no parameters; keep Gaussian sampling, variance and gripper intact."""
    if hasattr(policy, _VERSION) or policy._g_clip_actions:
        raise ValueError("Requires an unmodified Gaussian policy that scores raw commands")
    width = policy.arm_action_head.out_features
    if policy.num_actions not in (width, width + 1):
        raise ValueError("Requires arm actions and at most one gripper command")
    template = policy.arm_action_head.weight
    # Version one was an integration smoke that also bounded gripper means.
    # Reject it explicitly rather than replaying a different switch density.
    policy.register_buffer(_VERSION, template.new_tensor(2.0))
    policy.register_load_state_dict_pre_hook(_validate_checkpoint)
    policy._unbounded_execution_mean_compute = policy.compute
    policy.compute = MethodType(_compute, policy)


def build_bounded_execution_mean_mappo_agent(env, experiment_cfg):
    """Use the plain MAPPO actor with a common, explicitly checkpointed mean map."""
    from .actor import build_motion_context_mappo_agent

    if experiment_cfg["models"]["policy"]["clip_actions"]:
        raise ValueError("Bounded-mean experiments require policy.clip_actions=False")
    agent = build_motion_context_mappo_agent(env, experiment_cfg)
    for modules in agent.models.values():
        install_mean_bound(modules["policy"])
    print("[INFO] bounded arm means v2: softsign; gripper unchanged; raw Gaussian likelihood")
    return agent
