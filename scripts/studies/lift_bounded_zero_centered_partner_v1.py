"""Bounded Lift actor whose zero partner message is exactly neutral.

The frozen bounded actor only initialized partner-encoder biases to zero.
Training on nonzero Full6D messages can later move those biases, yielding a
large constant arm correction even at an all-zero message. This version
centers the encoder output at its *current* zero-message output before the
unchanged ±0.25 action head. It keeps architecture widths and optimizer
parameters, adds no task signal, and is a new developmental actor variant.
"""

from __future__ import annotations

from types import MethodType

from torch import nn
import torch

from .lift_orientation_bounded_partner_v1 import build_lift_orientation_bounded_partner_v1_mappo_agent


class ZeroCenteredPartnerEncoder(nn.Sequential):
    """Reuse the sequential MLP parameters, but remove its zero-input offset."""

    def __init__(self, source: nn.Sequential):
        if not isinstance(source, nn.Sequential) or isinstance(source, ZeroCenteredPartnerEncoder) or len(source) != 4:
            raise ValueError("Expected the frozen four-layer partner encoder")
        super().__init__(*list(source.children()))
        device = next(source.parameters()).device
        self.register_buffer("_zero_centered_message_marker", torch.tensor(1.0, device=device))

    def forward(self, message: torch.Tensor) -> torch.Tensor:
        if message.ndim != 2 or message.shape[1] != self[0].in_features:
            raise ValueError("Expected [batch, communication_feature_dim] partner message")
        encoded = super().forward(message)
        zero_encoded = super().forward(torch.zeros_like(message[:1]))
        centered = encoded - zero_encoded
        # Batched and singleton GEMMs can differ by a few ulps, so enforce
        # literal zero on zero-message rows, including NoShare and reset.
        return torch.where(message.eq(0).all(dim=-1, keepdim=True), torch.zeros_like(centered), centered)

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        marker = state_dict.get(prefix + "_zero_centered_message_marker")
        if marker is not None and (marker.numel() != 1 or float(marker) != 1.0):
            error_msgs.append(f"{prefix} zero-centered message marker differs")
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )


def _zero_neutral_policy_preprocess(self, uid, states, train=False):
    """Keep a raw absent message neutral after the actor state normalizer."""
    processed = self._zero_center_original_preprocess(uid, states, train=train)
    policy = self.models[uid]["policy"]
    own = policy.own_observation_dim
    width = policy.communication_feature_dim
    if states.shape[-1] < own + width or processed.shape != states.shape or width <= 0:
        raise RuntimeError("Zero-centered actor observation layout changed")
    raw_slot = states[..., own : own + width]
    normalized_slot = processed[..., own : own + width]
    absent = raw_slot.eq(0).all(dim=-1, keepdim=True)
    result = processed.clone()
    result[..., own : own + width] = torch.where(absent, torch.zeros_like(normalized_slot), normalized_slot)
    return result


def install_zero_centered_partner(agent) -> None:
    """Make raw-absent messages exactly neutral in each arm's actor path."""
    if not callable(getattr(agent, "_preprocess_policy_states", None)):
        raise RuntimeError("Zero-centered variant requires the matched MAPPO actor normalizer")
    if hasattr(agent, "_zero_center_original_preprocess"):
        raise RuntimeError("Zero-centered actor path already installed")
    for uid, modules in agent.models.items():
        policy = modules["policy"]
        source = policy.partner_intent_encoder
        if source is None:
            raise RuntimeError("Zero-centered variant requires the fixed partner slot")
        before = {id(param) for param in source.parameters()}
        policy.partner_intent_encoder = ZeroCenteredPartnerEncoder(source)
        if {id(param) for param in policy.partner_intent_encoder.parameters()} != before:
            raise RuntimeError("Centering unexpectedly replaced trainable encoder parameters")

        def validate_center_marker(module, incompatible):
            name = "partner_intent_encoder._zero_centered_message_marker"
            if name in incompatible.missing_keys:
                raise ValueError("Zero-centered partner checkpoint identity missing")
            if float(module.partner_intent_encoder._zero_centered_message_marker) != 1.0:
                raise ValueError("Zero-centered partner checkpoint identity changed")

        policy.register_load_state_dict_post_hook(validate_center_marker)
        print(f"[INFO] {uid} zero_centered_partner_message=1")
    agent._zero_center_original_preprocess = agent._preprocess_policy_states
    agent._preprocess_policy_states = MethodType(_zero_neutral_policy_preprocess, agent)


def build_lift_bounded_zero_centered_partner_v1_mappo_agent(env, experiment_cfg):
    """Apply the centering operation identically in Full6D and NoShare."""
    agent = build_lift_orientation_bounded_partner_v1_mappo_agent(env, experiment_cfg)
    install_zero_centered_partner(agent)
    return agent
