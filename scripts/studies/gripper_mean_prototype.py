"""Unqueued CPU prototype: switch-preserving bounded Gaussian gripper mean.

Identity on[-epsilon,epsilon], smooth C1 outside, bounded by1. This preserves
the deterministic close/retain/open categories mathematically, not raw mean
values or stochastic sampling. No performance or integration claim.
"""

from types import MethodType
import torch

from scripts.studies import bounded_peer_control as control

MARKER = "_gripper_mean_guard_version"
EPSILON = "_gripper_mean_guard_switch_epsilon"
CAP = "_gripper_mean_guard_cap"


def bounded_switch_mean(raw, epsilon, cap):
    # Retain-band identity; outer derivative is1 at the join and decreases.
    # Rearranged ratio keeps large finite raw values numerically stable.
    span = cap - epsilon
    delta = (raw.abs() - epsilon).clamp_min(0)
    outer = raw.sign() * (epsilon + span * (delta / (span + delta)))
    return torch.where(raw.abs() <= epsilon, raw, outer)


def _compute(self, inputs, role=""):
    mean, log_std, outputs = self._gripper_mean_original_compute(inputs, role)
    width = self.arm_action_head.out_features
    tail = bounded_switch_mean(mean[..., width:], getattr(self, EPSILON), getattr(self, CAP))
    mean = torch.cat((mean[..., :width], tail), dim=-1)
    return mean, log_std, {**outputs, "mean_actions": mean}


def _validate(module, state, prefix, *args):
    for key, value in ((MARKER, 1.0), (EPSILON, 0.02), (CAP, 1.0)):
        actual = state.get(prefix + key)
        if (
            actual is None
            or actual.numel() != 1
            or not torch.isfinite(actual).all()
            or not torch.equal(actual, actual.new_tensor(value))
        ):
            raise ValueError("Gripper-mean identity mismatch")


def install(policy):
    if (
        not hasattr(policy, control.MARKER)
        or hasattr(policy, MARKER)
        or policy._g_clip_actions
        or policy.num_actions not in (7, 8)
    ):
        raise ValueError("Requires unmodified combined Gaussian candidate")
    template = policy.arm_action_head.weight
    for key, value in ((MARKER, 1.0), (EPSILON, 0.02), (CAP, 1.0)):
        policy.register_buffer(key, template.new_tensor(value))
    policy.register_load_state_dict_pre_hook(_validate)
    policy._gripper_mean_original_compute = policy.compute
    policy.compute = MethodType(_compute, policy)


def augment_source(payload):
    if set(payload) != {"left_arm", "right_arm"}:
        raise ValueError("Both source actors required")
    result = {}
    for uid, state in payload.items():
        policy = state["policy"]
        if control.MARKER not in policy or MARKER in policy:
            raise ValueError("Requires unmodified combined source")
        template = policy["arm_action_head.weight"]
        buffers = {key: template.new_tensor(value) for key, value in ((MARKER, 1.0), (EPSILON, 0.02), (CAP, 1.0))}
        result[uid] = {**state, "policy": {**policy, **buffers}}
    return result
