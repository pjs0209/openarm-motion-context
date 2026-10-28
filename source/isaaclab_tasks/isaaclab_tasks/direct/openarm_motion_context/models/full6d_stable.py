"""Full 6D Trend MAPPO builder with a neutral initial partner-action pathway."""

from __future__ import annotations

import torch

from .actor import MotionContextPolicyModel, build_motion_context_mappo_agent


def initialize_partner_action_columns(policy: MotionContextPolicyModel) -> None:
    """Let the policy start local-only while keeping message connections trainable."""

    if policy.communication_mode != "motion_context" or not policy.uses_context:
        raise ValueError("Stable partner initialization is only for the full 6D motion_context policy")
    partner_start = policy.policy_feature_dim
    with torch.no_grad():
        policy.arm_action_head.weight[:, partner_start:].zero_()
        if policy.gripper_action_head is not None:
            policy.gripper_action_head[0].weight[:, partner_start : policy.actor_action_input_dim].zero_()


def build_full6d_stable_mappo_agent(env, experiment_cfg: dict):
    """Build the unchanged MAPPO agent, then neutralize only initial partner-action weights."""

    agent = build_motion_context_mappo_agent(env, experiment_cfg)
    for agent_id, modules in agent.models.items():
        policy = modules["policy"]
        initialize_partner_action_columns(policy)
        print(f"[INFO] {agent_id} full6d_partner_action_init=zero_trainable")
    return agent
