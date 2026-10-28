"""Checkpointed smooth arm sigma bound on top of mean-guard raw Gaussian.

sigma_arm = .5*sigmoid(raw_log_sigma). Gripper sigma is unchanged. This bounds
exploration, not sampled commands: the environment still clips physical copies.
"""

from types import MethodType

import torch
from torch.nn import functional as F

from scripts.studies import likelihood_mean_guard as mean_guard

MARKER = "_arm_noise_guard_version"
CAP = "_arm_noise_guard_max_sigma"
MAX_SIGMA = 0.5


def bounded_arm_log_std(raw, cap):
    return cap.log() + F.logsigmoid(raw)


def _compute(self, inputs, role=""):
    mean, log_std, outputs = self._arm_noise_guard_original_compute(inputs, role)
    width = self.arm_action_head.out_features
    bounded = torch.cat((bounded_arm_log_std(log_std[..., :width], getattr(self, CAP)), log_std[..., width:]), dim=-1)
    return mean, bounded, outputs


def _validate(module, state, prefix, *args):
    for key, expected in ((MARKER, 1.0), (CAP, MAX_SIGMA)):
        value = state.get(prefix + key)
        if value is None or value.numel() != 1 or not torch.isfinite(value).all() or float(value) != expected:
            raise ValueError("Arm-noise checkpoint identity mismatch")


def install_noise_guard(policy):
    if hasattr(policy, MARKER):
        raise ValueError("Arm-noise guard already installed")
    if not hasattr(policy, mean_guard.MARKER) or policy._g_clip_actions or not policy._g_clip_log_std:
        raise ValueError("Requires mean-guard raw Gaussian with finite log-std limits")
    template = policy.arm_action_head.weight
    policy.register_buffer(MARKER, template.new_tensor(1.0))
    policy.register_buffer(CAP, template.new_tensor(MAX_SIGMA))
    policy.register_load_state_dict_pre_hook(_validate)
    policy._arm_noise_guard_original_compute = policy.compute
    policy.compute = MethodType(_compute, policy)


def augment_source(payload):
    result = {}
    if set(payload) != {"left_arm", "right_arm"}:
        raise ValueError("Both actors required")
    for uid, state in payload.items():
        policy = state["policy"]
        if MARKER in policy or mean_guard.MARKER not in policy:
            raise ValueError("Requires unaugmented mean-guard actors")
        template = policy["arm_action_head.weight"]
        result[uid] = {
            "policy": {**policy, MARKER: template.new_tensor(1.0), CAP: template.new_tensor(MAX_SIGMA)},
            "state_preprocessor": state["state_preprocessor"],
        }
    return result


def build_bounded_noise_mappo_agent(env, experiment_cfg):
    agent = mean_guard.build_likelihood_mean_guard_mappo_agent(env, experiment_cfg)
    for modules in agent.models.values():
        install_noise_guard(modules["policy"])
    return agent


def build_fresh_camera_center_goal_bounded_noise_mappo_agent(env, experiment_cfg):
    agent = mean_guard.build_fresh_camera_center_goal_mean_guard_mappo_agent(env, experiment_cfg)
    for modules in agent.models.values():
        install_noise_guard(modules["policy"])
    return agent
