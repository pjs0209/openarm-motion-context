"""Full 6D Trend policy with a gradual partner-to-action learning pathway.

The actor and message are unchanged. The local pathway learns first; after a
fixed warmup, the trainable partner action columns are allowed to grow under
a relative weight-norm cap. The schedule is task-agnostic and independent of
success labels, object state, and reward signals.
"""

from __future__ import annotations

from types import MethodType

from .full6d_balanced import project_partner_action_columns
from .full6d_stable import build_full6d_stable_mappo_agent, initialize_partner_action_columns


def partner_ratio_at_step(
    timestep: int,
    warmup_timesteps: int,
    ramp_timesteps: int,
    max_ratio: float,
) -> float:
    """Linearly open the partner pathway after a local-policy warmup."""

    if warmup_timesteps < 0 or ramp_timesteps <= 0 or max_ratio <= 0:
        raise ValueError("warmup must be nonnegative, ramp and max_ratio must be positive")
    progress = (int(timestep) - warmup_timesteps) / ramp_timesteps
    return max_ratio * min(max(progress, 0.0), 1.0)


def _curriculum_update(self, timestep: int, timesteps: int) -> None:
    self._full6d_original_update(timestep, timesteps)
    ratio = partner_ratio_at_step(
        timestep,
        self._full6d_partner_warmup_timesteps,
        self._full6d_partner_ramp_timesteps,
        self._full6d_partner_max_ratio,
    )
    for modules in self.models.values():
        policy = modules["policy"]
        if ratio == 0.0:
            initialize_partner_action_columns(policy)
        else:
            project_partner_action_columns(policy, ratio)


def build_full6d_curriculum_mappo_agent(env, experiment_cfg: dict):
    """Build standard MAPPO with a staged, bounded Full 6D message pathway."""

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
    agent._update = MethodType(_curriculum_update, agent)
    print(f"[INFO] full6d_partner_curriculum=warmup:{warmup},ramp:{ramp},max_ratio:{max_ratio}")
    return agent
