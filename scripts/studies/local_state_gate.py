"""Opt-in, checkpointed local-state gate for unchanged hand-crafted 6D messages.

Load builders with PYTHONPATH containing the repository root. This file is
outside the frozen task package, so current ungated experiments are unchanged.
"""

import torch
from torch import nn
from torch.nn import functional as F


class LocalStateGatedArmHead(nn.Module):
    """Own action plus sigmoid(local hidden) times the partner action branch."""

    def __init__(self, original, own_width):
        super().__init__()
        if not isinstance(original, nn.Linear) or not 0 < own_width < original.in_features:
            raise ValueError("Requires a linear own-plus-partner arm head")
        self.own_width = int(own_width)
        self.in_features, self.out_features = original.in_features, original.out_features
        # Preserve parameter identities already registered with Adam.
        self.weight, self.bias = original.weight, original.bias
        # Initial gate is exactly 0.5; neutral partner initialization still gives
        # the exact original own-only action. Do not advance experiment RNG.
        with torch.random.fork_rng(devices=[]):
            self.local_gate = nn.Linear(own_width, original.out_features)
        self.local_gate.to(device=original.weight.device, dtype=original.weight.dtype)
        nn.init.zeros_(self.local_gate.weight)
        nn.init.zeros_(self.local_gate.bias)
        self.register_buffer("_local_state_gate_marker", original.weight.new_tensor(1.0))

    def forward(self, features):
        own, partner = features[..., : self.own_width], features[..., self.own_width :]
        local_action = F.linear(own, self.weight[:, : self.own_width], self.bias)
        partner_action = F.linear(partner, self.weight[:, self.own_width :])
        return local_action + torch.sigmoid(self.local_gate(own)) * partner_action


def install_gate(policy, optimizer):
    if isinstance(policy.arm_action_head, LocalStateGatedArmHead):
        raise ValueError("Gate already installed")
    head = LocalStateGatedArmHead(policy.arm_action_head, policy.policy_feature_dim)
    gate_parameters = list(head.local_gate.parameters())
    registered = {id(p) for group in optimizer.param_groups for p in group["params"]}
    if id(head.weight) not in registered or (head.bias is not None and id(head.bias) not in registered):
        raise ValueError("Original head parameters missing from optimizer")
    if any(id(p) in registered for p in gate_parameters):
        raise ValueError("Gate parameters already registered")
    # Same parameter group preserves existing LR scheduler/group semantics.
    optimizer.param_groups[0]["params"].extend(gate_parameters)
    policy.arm_action_head = head


def augment_warmstart(payload):
    """Add only neutral gate state to the identical common Lift actor source."""
    result = {}
    for uid, checkpoint in payload.items():
        state = checkpoint["policy"]
        if any("local_gate" in key or "_local_state_gate_marker" in key for key in state):
            raise ValueError("Warm-start source is already gated")
        arm = state["arm_action_head.weight"]
        partner_width = state["partner_intent_encoder.2.weight"].shape[0]
        own_width = arm.shape[1] - partner_width
        if own_width <= 0 or not torch.isfinite(arm).all():
            raise ValueError("Invalid warm-start head")
        extra = {
            "arm_action_head.local_gate.weight": arm.new_zeros((arm.shape[0], own_width)),
            "arm_action_head.local_gate.bias": arm.new_zeros(arm.shape[0]),
            "arm_action_head._local_state_gate_marker": arm.new_tensor(1.0),
        }
        result[uid] = {**checkpoint, "policy": {**state, **extra}}
    return result


def build_local_state_gate_mappo_agent(env, experiment_cfg):
    from isaaclab_tasks.direct.openarm_motion_context.models.matched_communication import (
        build_matched_communication_mappo_agent,
    )

    if env.communication_mode != "motion_context" or env.sharing_mode not in ("intent_share", "no_share"):
        raise ValueError("Gate pilot requires Full6D or matched NoShare")
    agent = build_matched_communication_mappo_agent(env, experiment_cfg)
    for uid, modules in agent.models.items():
        install_gate(modules["policy"], agent.optimizers[uid])
    print("[INFO] local-state gate v1: unchanged 6D message; bounded per-joint partner influence; initial gate=0.5")
    return agent


def build_center_goal_local_state_gate_mappo_agent(env, experiment_cfg):
    from isaaclab_tasks.direct.openarm_motion_context.envs.ball_center_goal import (
        install_centered_hold_goal,
        unwrap_ball_task,
    )

    install_centered_hold_goal(unwrap_ball_task(env))
    return build_local_state_gate_mappo_agent(env, experiment_cfg)
