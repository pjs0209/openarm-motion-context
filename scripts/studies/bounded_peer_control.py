"""Uninstalled combined candidate: bounded mean-space correction and all sigma.

Same6D Trend H5; no new learned parameters, sensor channel, goal or reward.
This is a combined optimization candidate, not an isolated causal ablation.
"""

from scripts.studies import arm_noise_guard as arm
from scripts.studies import all_action_noise_guard as all_noise
from scripts.studies import partner_residual_guard as peer
from scripts.studies import likelihood_mean_guard as mean_guard
from types import MethodType
import torch

MARKER = "_bounded_peer_control_version"


def _compute(self, inputs, role=""):
    mean, log_std, outputs = self._bounded_peer_control_original_compute(inputs, role)
    head = self.arm_action_head
    local = mean_guard.softsign_mean(head._partner_residual_latest_local, getattr(self, mean_guard.BOUND))
    corrected = (local + head._partner_residual_latest_correction).clamp(-1.0, 1.0)
    mean = torch.cat((corrected, mean[..., head.out_features :]), dim=-1)
    return mean, log_std, {**outputs, "mean_actions": mean}


def _validate(module, state, prefix, *args):
    value = state.get(prefix + MARKER)
    if value is None or value.numel() != 1 or not torch.isfinite(value).all() or float(value) != 1.0:
        raise ValueError("Bounded-peer-control checkpoint identity mismatch")


def install(policy):
    if hasattr(policy, MARKER):
        raise ValueError("Bounded peer control already installed")
    all_noise.install_guard(policy)
    peer.install_guard(policy)
    policy.register_buffer(MARKER, policy.arm_action_head.weight.new_tensor(1.0))
    policy.register_load_state_dict_pre_hook(_validate)
    policy._bounded_peer_control_original_compute = policy.compute
    policy.compute = MethodType(_compute, policy)


def augment_source(payload):
    result = peer.augment_source(all_noise.augment_source(payload))
    for state in result.values():
        state["policy"][MARKER] = state["policy"]["arm_action_head.weight"].new_tensor(1.0)
    return result


def build_bounded_peer_control_mappo_agent(env, experiment_cfg):
    agent = arm.build_bounded_noise_mappo_agent(env, experiment_cfg)
    for modules in agent.models.values():
        install(modules["policy"])
    return agent


def build_fresh_camera_bounded_peer_control_mappo_agent(env, experiment_cfg):
    agent = arm.build_fresh_camera_center_goal_bounded_noise_mappo_agent(env, experiment_cfg)
    for modules in agent.models.values():
        install(modules["policy"])
    return agent
