"""Opt-in arm-mean guard with likelihood-correct latent Gaussian actions.

Prototype only: no running experiment uses this builder. The environment
still clips physical commands, but PPO records and scores the raw samples.
Softsign bounds only the arm mean, not samples or gripper outputs. This is
an optimization ablation, not a new message or an inferred partner intention.
"""

from types import MethodType

import torch

VERSION = 1.0
MARKER = "_likelihood_mean_guard_version"
BOUND = "_likelihood_mean_guard_bound"


def softsign_mean(value, bound):
    # Rearranged form avoids overflow in bound * value for large finite inputs.
    return bound * (value / (bound + value.abs()))


def _guarded_compute(self, inputs, role=""):
    mean, log_std, outputs = self._mean_guard_original_compute(inputs, role)
    width = self.arm_action_head.out_features
    guarded = torch.cat((softsign_mean(mean[..., :width], getattr(self, BOUND)), mean[..., width:]), dim=-1)
    return guarded, log_std, {**outputs, "mean_actions": guarded}


def _validate_checkpoint(module, state, prefix, *args):
    for name, expected in ((MARKER, VERSION), (BOUND, 1.0)):
        value = state.get(prefix + name)
        if value is None or value.numel() != 1 or not torch.isfinite(value).all() or float(value) != expected:
            raise ValueError("Mean-guard checkpoint version/bound mismatch")


def install_guard(policy):
    if hasattr(policy, MARKER):
        raise ValueError("Mean guard already installed")
    if not hasattr(policy, "_g_clip_actions") or not hasattr(policy, "arm_action_head"):
        raise ValueError("Requires Gaussian arm policy")
    if hasattr(policy.arm_action_head, "_local_state_gate_marker"):
        raise ValueError("Do not mix the gate and mean-guard ablations")
    template = policy.arm_action_head.weight
    policy.register_buffer(MARKER, template.new_tensor(VERSION))
    policy.register_buffer(BOUND, template.new_tensor(1.0))
    policy.register_load_state_dict_pre_hook(_validate_checkpoint)
    policy._mean_guard_original_compute = policy.compute
    policy.compute = MethodType(_guarded_compute, policy)
    policy._g_clip_actions = False


def augment_warmstart(payload):
    """Preserve source weights/scalers, adding explicit prototype identity only."""
    result = {}
    if set(payload) != {"left_arm", "right_arm"}:
        raise ValueError("Requires both common-source actors")
    for uid, checkpoint in payload.items():
        state = checkpoint["policy"]
        if MARKER in state or "arm_action_head._local_state_gate_marker" in state:
            raise ValueError("Requires ungated, unguarded source")
        template = state["arm_action_head.weight"]
        if not torch.isfinite(template).all():
            raise ValueError("Invalid source head")
        result[uid] = {
            **checkpoint,
            "policy": {**state, MARKER: template.new_tensor(VERSION), BOUND: template.new_tensor(1.0)},
        }
    return result


def build_likelihood_mean_guard_mappo_agent(env, experiment_cfg):
    from isaaclab_tasks.direct.openarm_motion_context.models.matched_communication import (
        build_matched_communication_mappo_agent,
    )

    if env.communication_mode != "motion_context" or env.sharing_mode not in ("intent_share", "no_share"):
        raise ValueError("Requires Full6D or matched NoShare")
    if experiment_cfg["models"]["policy"]["clip_actions"]:
        raise ValueError("Saved configuration must explicitly set policy.clip_actions=False")
    agent = build_matched_communication_mappo_agent(env, experiment_cfg)
    for modules in agent.models.values():
        install_guard(modules["policy"])
    print("[INFO] mean guard v1: softsign arm mean, raw Gaussian likelihood, environment-only physical clipping")
    return agent


def build_fresh_camera_center_goal_mean_guard_mappo_agent(env, experiment_cfg):
    from isaaclab_tasks.direct.openarm_motion_context.envs.ball_center_goal import (
        install_centered_hold_goal,
        unwrap_ball_task,
    )
    from scripts.studies.fresh_camera_scaler import install_vector_counts

    install_centered_hold_goal(unwrap_ball_task(env))
    agent = build_likelihood_mean_guard_mappo_agent(env, experiment_cfg)
    install_vector_counts(agent)
    return agent
