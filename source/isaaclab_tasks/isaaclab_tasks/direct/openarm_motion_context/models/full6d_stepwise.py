"""Full 6D Trend fusion constrained at every optimizer step.

The early local-only stage is exact even within a MAPPO update's minibatches.
After warmup, partner-to-action weights open gradually and are projected after
each optimizer step, not only after an entire multi-epoch MAPPO update.
"""

from __future__ import annotations

from types import MethodType

from .full6d_balanced import project_partner_action_columns
from .full6d_curriculum import partner_ratio_at_step
from .full6d_stable import build_full6d_stable_mappo_agent, initialize_partner_action_columns


def zero_partner_action_gradients(policy) -> None:
    """Prevent optimizer moments from accumulating on the closed partner path."""

    partner_start = policy.policy_feature_dim
    partner_end = policy.actor_action_input_dim
    arm_grad = policy.arm_action_head.weight.grad
    if arm_grad is not None:
        arm_grad[:, partner_start:partner_end] = 0
    if policy.gripper_action_head is not None:
        gripper_grad = policy.gripper_action_head[0].weight.grad
        if gripper_grad is not None:
            gripper_grad[:, partner_start:partner_end] = 0


def _stepwise_update(self, timestep: int, timesteps: int) -> None:
    self._full6d_partner_current_ratio = partner_ratio_at_step(
        timestep,
        self._full6d_partner_warmup_timesteps,
        self._full6d_partner_ramp_timesteps,
        self._full6d_partner_max_ratio,
    )
    self._full6d_original_update(timestep, timesteps)


def _install_stepwise_projection(agent, uid: str) -> None:
    optimizer = agent.optimizers[uid]
    policy = agent.models[uid]["policy"]
    original_step = optimizer.step

    def projected_step(self, *args, **kwargs):
        ratio = agent._full6d_partner_current_ratio
        if ratio == 0.0:
            zero_partner_action_gradients(policy)
        result = original_step(*args, **kwargs)
        if ratio == 0.0:
            initialize_partner_action_columns(policy)
        else:
            project_partner_action_columns(policy, ratio)
        return result

    optimizer.step = MethodType(projected_step, optimizer)


def build_full6d_stepwise_mappo_agent(env, experiment_cfg: dict):
    """Build standard MAPPO, then constrain each inner optimizer step."""

    policy_cfg = experiment_cfg["models"]["policy"]
    warmup = int(policy_cfg.get("partner_warmup_timesteps", 20000))
    ramp = int(policy_cfg.get("partner_ramp_timesteps", 20000))
    max_ratio = float(policy_cfg.get("partner_action_max_ratio", 0.75))
    partner_ratio_at_step(0, warmup, ramp, max_ratio)
    agent = build_full6d_stable_mappo_agent(env, experiment_cfg)
    agent._full6d_original_update = agent._update
    agent._full6d_partner_warmup_timesteps = warmup
    agent._full6d_partner_ramp_timesteps = ramp
    agent._full6d_partner_max_ratio = max_ratio
    agent._full6d_partner_current_ratio = 0.0
    for uid in agent.models:
        _install_stepwise_projection(agent, uid)
    agent._update = MethodType(_stepwise_update, agent)
    print(f"[INFO] full6d_partner_stepwise_curriculum=warmup:{warmup},ramp:{ramp},max_ratio:{max_ratio}")
    return agent
