"""Uninstalled extension: bound gripper sigma as well as existing arm sigma.

Arm-only Circle remains unchanged by this noise extension. No means, sampled
actions, weights or RNG draws change at installation; gripper distribution
changes when present. Gaussian likelihood uses that actual bounded sigma.
"""

from types import MethodType

import torch
from torch.nn import functional as F

from scripts.studies import arm_noise_guard as arm

MARKER = "_all_action_noise_guard_version"
CAP = "_all_action_noise_guard_cap"
MAX_SIGMA = 0.5


def _compute(self, inputs, role=""):
    mean, log_std, outputs = self._all_action_noise_original_compute(inputs, role)
    width = self.arm_action_head.out_features
    tail = getattr(self, CAP).log() + F.logsigmoid(log_std[..., width:])
    return mean, torch.cat((log_std[..., :width], tail), dim=-1), outputs


def _validate(module, state, prefix, *args):
    for key, expected in ((MARKER, 1.0), (CAP, MAX_SIGMA)):
        value = state.get(prefix + key)
        if value is None or value.numel() != 1 or not torch.isfinite(value).all() or float(value) != expected:
            raise ValueError("All-action-noise checkpoint identity mismatch")


def install_guard(policy):
    if (
        not hasattr(policy, arm.MARKER)
        or hasattr(policy, MARKER)
        or policy._g_clip_actions
        or float(getattr(policy, arm.CAP)) != MAX_SIGMA
        or policy.num_actions not in (policy.arm_action_head.out_features, policy.arm_action_head.out_features + 1)
    ):
        raise ValueError("Requires existing arm-noise guard and at most one gripper")
    template = policy.arm_action_head.weight
    policy.register_buffer(MARKER, template.new_tensor(1.0))
    policy.register_buffer(CAP, template.new_tensor(MAX_SIGMA))
    policy.register_load_state_dict_pre_hook(_validate)
    policy._all_action_noise_original_compute = policy.compute
    policy.compute = MethodType(_compute, policy)


def augment_source(payload):
    if set(payload) != {"left_arm", "right_arm"}:
        raise ValueError("Both source actors required")
    result = {}
    for uid, state in payload.items():
        policy = state["policy"]
        if arm.MARKER not in policy or MARKER in policy:
            raise ValueError("Requires unmodified arm-noise source")
        template = policy["arm_action_head.weight"]
        result[uid] = {
            **state,
            "policy": {**policy, MARKER: template.new_tensor(1.0), CAP: template.new_tensor(MAX_SIGMA)},
        }
    return result
