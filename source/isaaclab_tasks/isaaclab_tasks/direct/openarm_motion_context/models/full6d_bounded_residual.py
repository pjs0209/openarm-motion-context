"""Bound actual partner-induced action corrections, not only weight norms.

Both heads implement local_action + bound*tanh(partner_correction/bound).
Existing trainable parameters and optimizer references are retained. Explicit
checkpoint markers prevent silently evaluating with an unbounded head or a
different bound. No task phase, object state, or contact signal is introduced.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from .full6d_balanced import build_full6d_balanced_mappo_agent
from .full6d_factorized import build_full6d_factorized_mappo_agent


def validate_bound(bound: float) -> float:
    bound = float(bound)
    if not math.isfinite(bound) or bound <= 0:
        raise ValueError("partner_action_residual_bound must be finite and positive")
    return bound


class ResidualBoundCheckpointGuard:
    """Fail loading if the checkpoint's action semantics do not match config."""

    def _initialize_bound(self, bound: float, device):
        self.bound = validate_bound(bound)
        self.register_buffer("_residual_bound_marker", torch.tensor(self.bound, device=device))

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        marker = state_dict.get(prefix + "_residual_bound_marker")
        if marker is not None and (
            marker.numel() != 1 or not math.isclose(float(marker.item()), self.bound, rel_tol=0, abs_tol=1e-7)
        ):
            error_msgs.append(f"{prefix} residual bound differs from configured bound {self.bound}")
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )


class BoundedPartnerArmHead(ResidualBoundCheckpointGuard, nn.Linear):
    """Reuse a linear head's parameters without consuming initialization RNG."""

    def __init__(self, source: nn.Linear, own_width: int, bound: float):
        nn.Module.__init__(self)
        self.in_features = source.in_features
        self.out_features = source.out_features
        self.weight = source.weight
        self.bias = source.bias
        self.own_width = int(own_width)
        self._initialize_bound(bound, source.weight.device)
        if not 0 < self.own_width < self.in_features:
            raise ValueError("Invalid own/partner feature split")

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        local = F.linear(features[..., : self.own_width], self.weight[:, : self.own_width], self.bias)
        partner = F.linear(features[..., self.own_width :], self.weight[:, self.own_width :])
        return local + self.bound * torch.tanh(partner / self.bound)


class BoundedPartnerGripperHead(ResidualBoundCheckpointGuard, nn.Sequential):
    """Bound shared-versus-local gripper means using the same underlying MLP."""

    def __init__(self, source: nn.Sequential, partner_start: int, partner_end: int, bound: float):
        super().__init__(*list(source.children()))
        self.partner_start = int(partner_start)
        self.partner_end = int(partner_end)
        self._initialize_bound(bound, source[0].weight.device)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        shared = super().forward(features)
        local_features = features.clone()
        local_features[..., self.partner_start : self.partner_end] = 0
        local = super().forward(local_features)
        return local + self.bound * torch.tanh((shared - local) / self.bound)


def build_full6d_bounded_residual_mappo_agent(env, experiment_cfg: dict):
    """Full 6D with bounded corrections; default to the best verified joint encoder."""

    bound = validate_bound(experiment_cfg["models"]["policy"].get("partner_action_residual_bound", 0.25))
    factorized = experiment_cfg["models"]["policy"].get("partner_encoder_factorized", False)
    if not isinstance(factorized, bool):
        raise ValueError("partner_encoder_factorized must be a boolean")
    builder = build_full6d_factorized_mappo_agent if factorized else build_full6d_balanced_mappo_agent
    agent = builder(env, experiment_cfg)
    for agent_id, modules in agent.models.items():
        policy = modules["policy"]
        policy.arm_action_head = BoundedPartnerArmHead(policy.arm_action_head, policy.policy_feature_dim, bound)
        if policy.gripper_action_head is not None:
            policy.gripper_action_head = BoundedPartnerGripperHead(
                policy.gripper_action_head, policy.policy_feature_dim, policy.actor_action_input_dim, bound
            )
        print(f"[INFO] {agent_id} actual_partner_action_residual_bound={bound}")
    return agent
