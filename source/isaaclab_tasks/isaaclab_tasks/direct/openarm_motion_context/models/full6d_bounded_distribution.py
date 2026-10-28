"""Exploratory bounded control-distribution decoder for the unchanged 6D message.

Local means use tanh; partner corrections retain the existing 0.25 residual
bound. Each learned std uses a differentiable bounded sigmoid rather than a
flat hard clip. PPO samples/scores raw Gaussian commands; the environment still
clamps execution. This changes the control distribution, not the descriptor.
"""

from __future__ import annotations

import math
from types import MethodType

import torch
from torch.nn import functional as F

from .full6d_bounded_residual import (
    BoundedPartnerArmHead,
    BoundedPartnerGripperHead,
    build_full6d_bounded_residual_mappo_agent,
)


def validate_std_spec(minimum, maximum, initial):
    minimum, maximum, initial = float(minimum), float(maximum), float(initial)
    if not all(math.isfinite(value) for value in (minimum, maximum, initial)) or not 0 < minimum < initial < maximum:
        raise ValueError("Std bounds must be finite with 0 < min < initial < max")
    return minimum, maximum, initial


def bounded_log_std(raw, minimum, maximum):
    return (minimum + (maximum - minimum) * torch.sigmoid(raw)).log()


class LocalMeanCheckpointGuard:
    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        marker = state_dict.get(prefix + "_bounded_local_mean_marker")
        if marker is not None and (
            marker.numel() != 1 or not torch.isfinite(marker).all() or float(marker.item()) != 1
        ):
            error_msgs.append(f"{prefix} local mean must use the unit tanh bound")
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )


class BoundedMeanPartnerArmHead(LocalMeanCheckpointGuard, BoundedPartnerArmHead):
    def forward(self, features):
        local = F.linear(features[..., : self.own_width], self.weight[:, : self.own_width], self.bias)
        partner = F.linear(features[..., self.own_width :], self.weight[:, self.own_width :])
        return local.tanh() + self.bound * torch.tanh(partner / self.bound)


class BoundedMeanPartnerGripperHead(LocalMeanCheckpointGuard, BoundedPartnerGripperHead):
    def forward(self, features):
        # Call the underlying Sequential, not the inherited residual wrapper.
        shared = torch.nn.Sequential.forward(self, features)
        local_features = features.clone()
        local_features[..., self.partner_start : self.partner_end] = 0
        local = torch.nn.Sequential.forward(self, local_features)
        return local.tanh() + self.bound * torch.tanh((shared - local) / self.bound)


def _distribution_compute(self, inputs, role=""):
    means, raw_std, outputs = self._distribution_original_compute(inputs, role)
    return means, bounded_log_std(raw_std, *self._bounded_std_limits), outputs


def _distribution_load(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
    marker = state_dict.get(prefix + "_bounded_std_marker")
    expected = self._bounded_std_marker
    if marker is not None and (
        marker.shape != expected.shape
        or not torch.isfinite(marker).all()
        or not torch.allclose(marker.to(expected), expected, rtol=0, atol=1e-7)
    ):
        error_msgs.append(f"{prefix} bounded std limits differ from configured limits")
    self._distribution_original_loader(
        state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    )


def build_full6d_bounded_distribution_mappo_agent(env, experiment_cfg):
    policy_config = experiment_cfg["models"]["policy"]
    if policy_config["clip_actions"] is not False:
        raise ValueError("Bounded distribution requires raw Gaussian sampling (clip_actions=False)")
    if policy_config.get("partner_encoder_factorized", False) is not False:
        raise ValueError("Bounded distribution pilot uses the better verified joint encoder")
    minimum, maximum, initial = validate_std_spec(
        policy_config.get("bounded_distribution_std_min", 0.05),
        policy_config.get("bounded_distribution_std_max", 0.75),
        policy_config.get("bounded_distribution_std_initial", 0.35),
    )
    # Returned log std must not be subsequently altered by GaussianMixin.
    if policy_config["clip_log_std"] and (
        math.log(minimum) < policy_config["min_log_std"] or math.log(maximum) > policy_config["max_log_std"]
    ):
        raise ValueError("Bounded std limits exceed GaussianMixin log-std limits")
    agent = build_full6d_bounded_residual_mappo_agent(env, experiment_cfg)
    raw_initial = math.log((initial - minimum) / (maximum - initial))
    for uid, modules in agent.models.items():
        policy = modules["policy"]
        policy.arm_action_head = BoundedMeanPartnerArmHead(
            policy.arm_action_head, policy.policy_feature_dim, policy.arm_action_head.bound
        )
        heads = [policy.arm_action_head]
        if policy.gripper_action_head is not None:
            policy.gripper_action_head = BoundedMeanPartnerGripperHead(
                policy.gripper_action_head,
                policy.policy_feature_dim,
                policy.actor_action_input_dim,
                policy.gripper_action_head.bound,
            )
            heads.append(policy.gripper_action_head)
        for head in heads:
            head.register_buffer("_bounded_local_mean_marker", torch.tensor(1.0, device=policy.device))
        policy._bounded_std_limits = (minimum, maximum)
        policy.register_buffer("_bounded_std_marker", torch.tensor([minimum, maximum], device=policy.device))
        with torch.no_grad():
            policy.log_std_parameter.fill_(raw_initial)
        policy._distribution_original_compute = policy.compute
        policy.compute = MethodType(_distribution_compute, policy)
        policy._distribution_original_loader = policy._load_from_state_dict
        policy._load_from_state_dict = MethodType(_distribution_load, policy)
        print(
            f"[INFO] {uid} local_mean_bound=1 partner_bound={policy.arm_action_head.bound} "
            f"bounded_std=[{minimum},{maximum}] initial_std={initial}; latent Gaussian commands"
        )
    return agent
