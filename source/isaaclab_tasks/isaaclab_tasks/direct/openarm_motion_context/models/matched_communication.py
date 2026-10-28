"""Same neutral initialization and action-weight cap for all message baselines.

Historical Full 6D builders deliberately reject other modes. This separate
benchmark builder applies their identical initialization/projection to each
fixed-width message mode; no-share retains the same encoder with zero inputs.
"""

import math
from types import MethodType

import torch

from .actor import build_motion_context_mappo_agent
from .full6d_balanced import _balanced_update


def build_matched_communication_mappo_agent(env, experiment_cfg):
    mode = env.communication_mode
    if mode not in {
        "motion_context",
        "motion_context_la",
        "motion_context_s",
        "motion_only",
        "context_only",
        "previous_action",
        "full_partner_observation",
    }:
        raise ValueError("Matched benchmark requires a fixed-width active encoder; no-share is a sharing mode")
    ratio = float(experiment_cfg["models"]["policy"].get("partner_action_max_ratio", 0.5))
    if not math.isfinite(ratio) or ratio <= 0:
        raise ValueError("Partner action ratio must be finite and positive")
    agent = build_motion_context_mappo_agent(env, experiment_cfg)
    for modules in agent.models.values():
        policy = modules["policy"]
        if not policy.uses_context:
            raise ValueError("Matched no-share control must retain the same encoder")
        with torch.no_grad():
            policy.arm_action_head.weight[:, policy.policy_feature_dim :].zero_()
            if policy.gripper_action_head is not None:
                policy.gripper_action_head[0].weight[
                    :, policy.policy_feature_dim : policy.actor_action_input_dim
                ].zero_()
    agent._full6d_original_update = agent._update
    agent._full6d_partner_max_ratio = ratio
    agent._update = MethodType(_balanced_update, agent)
    print(f"[INFO] matched communication decoder mode={mode}; neutral partner init; weight ratio={ratio}")
    return agent
