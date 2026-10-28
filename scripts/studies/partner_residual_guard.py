"""Uninstalled next-candidate prototype: bounded partner arm residual.

head = local_linear + c*tanh(partner_linear/c), c=.25, before the existing
softsign mean and bounded sigma. Absolute correction relative to a ZERO
embedding is <=c. Two arbitrary raw-message interventions differ by <=2c,
because zero raw features need not yield a zero learned embedding.
This is a structural bound, NOT successful control or communication evidence.
"""

from types import MethodType

import torch
from torch.nn import functional as F

from scripts.studies import arm_noise_guard as noise
from scripts.studies import likelihood_mean_guard as mean_guard

MARKER = "_partner_residual_guard_version"
CAP = "_partner_residual_guard_cap"
MAX_CORRECTION = 0.25


def _forward(self, features):
    own = self._partner_residual_own_width
    local = F.linear(features[..., :own], self.weight[:, :own], self.bias)
    partner = F.linear(features[..., own:], self.weight[:, own:], None)
    cap = getattr(self, CAP)
    correction = cap * torch.tanh(partner / cap)
    # Optional composite consumes these same-graph terms after local softsign.
    # Standalone prototype output remains unchanged.
    self._partner_residual_latest_local = local
    self._partner_residual_latest_correction = correction
    return local + correction


def _validate(module, state, prefix, *args):
    for key, expected in ((MARKER, 1.0), (CAP, MAX_CORRECTION)):
        value = state.get(prefix + key)
        if value is None or value.numel() != 1 or not torch.isfinite(value).all() or float(value) != expected:
            raise ValueError("Partner-residual checkpoint identity mismatch")


def install_guard(policy):
    head = policy.arm_action_head
    if (
        type(head) is not torch.nn.Linear
        or hasattr(head, MARKER)
        or not getattr(policy, "uses_context", False)
        or not hasattr(policy, mean_guard.MARKER)
        or not hasattr(policy, noise.MARKER)
        or policy._g_clip_actions
        or head.in_features != policy.policy_feature_dim + policy.partner_intent_embed_dim
    ):
        raise ValueError("Requires unmodified factorized linear head with mean/noise guards")
    head._partner_residual_own_width = policy.policy_feature_dim
    head.register_buffer(MARKER, head.weight.new_tensor(1.0))
    head.register_buffer(CAP, head.weight.new_tensor(MAX_CORRECTION))
    head.register_load_state_dict_pre_hook(_validate)
    head.forward = MethodType(_forward, head)


def augment_source(payload):
    if set(payload) != {"left_arm", "right_arm"}:
        raise ValueError("Both source actors required")
    result = {}
    for uid, state in payload.items():
        policy = state["policy"]
        if noise.MARKER not in policy or mean_guard.MARKER not in policy or "arm_action_head." + MARKER in policy:
            raise ValueError("Requires unmodified bounded-noise source")
        template = policy["arm_action_head.weight"]
        peer_width = policy["partner_intent_encoder.0.weight"].shape[0]
        if torch.count_nonzero(template[:, -peer_width:]) != 0:
            raise ValueError("Only a neutral peer source preserves its initial forward map")
        result[uid] = {
            **state,
            "policy": {
                **policy,
                "arm_action_head." + MARKER: template.new_tensor(1.0),
                "arm_action_head." + CAP: template.new_tensor(MAX_CORRECTION),
            },
        }
    return result


def build_partner_residual_guard_mappo_agent(env, experiment_cfg):
    agent = noise.build_bounded_noise_mappo_agent(env, experiment_cfg)
    for modules in agent.models.values():
        install_guard(modules["policy"])
    return agent


def build_fresh_camera_partner_residual_guard_mappo_agent(env, experiment_cfg):
    agent = noise.build_fresh_camera_center_goal_bounded_noise_mappo_agent(env, experiment_cfg)
    for modules in agent.models.values():
        install_guard(modules["policy"])
    return agent
