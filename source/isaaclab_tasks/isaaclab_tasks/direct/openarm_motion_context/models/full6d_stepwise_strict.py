"""Stepwise Full 6D curriculum with an exact closed-path gradient mask.

During local-only warmup, gradient hooks remove partner-action gradients before
global gradient clipping. The underlying stepwise builder also clears these
gradients before Adam and projects weights after every optimizer step.
"""

from __future__ import annotations

from .full6d_stepwise import build_full6d_stepwise_mappo_agent


def make_closed_path_gradient_hook(agent, policy):
    """Return a hook that masks partner columns only while their gate is closed."""

    partner_start = policy.policy_feature_dim
    partner_end = policy.actor_action_input_dim

    def hook(gradient):
        if agent._full6d_partner_current_ratio != 0.0:
            return gradient
        masked = gradient.clone()
        masked[:, partner_start:partner_end] = 0
        return masked

    return hook


def build_full6d_stepwise_strict_mappo_agent(env, experiment_cfg: dict):
    """Keep warmup truly local-only, including MAPPO minibatch gradients."""

    agent = build_full6d_stepwise_mappo_agent(env, experiment_cfg)
    handles = []
    for modules in agent.models.values():
        policy = modules["policy"]
        hook = make_closed_path_gradient_hook(agent, policy)
        handles.append(policy.arm_action_head.weight.register_hook(hook))
        if policy.gripper_action_head is not None:
            handles.append(policy.gripper_action_head[0].weight.register_hook(hook))
    agent._full6d_partner_gradient_hooks = handles
    print("[INFO] full6d_partner_warmup_gradient_mask=before_clip")
    return agent
