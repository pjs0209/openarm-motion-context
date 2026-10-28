"""Full 6D Trend actor with bounded standardized partner-message inputs.

The MAPPO state preprocessor gives low-variance trend features unit variance.
This keeps their sign and usual magnitude, but caps rare standardized spikes
before they enter the partner encoder. Checkpoint parameter names are unchanged.
"""

from __future__ import annotations

from types import MethodType

import torch

from .actor import MotionContextPolicyModel
from .full6d_stable import build_full6d_stable_mappo_agent


def _bounded_split_states(self: MotionContextPolicyModel, states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    own, partner = MotionContextPolicyModel._split_states(self, states)
    return own, partner.clamp(-self.partner_message_clip, self.partner_message_clip)


def build_full6d_bounded_mappo_agent(env, experiment_cfg: dict):
    """Keep Full 6D learnable while limiting normalized message outliers."""

    clip = float(experiment_cfg["models"]["policy"].get("partner_message_clip", 1.5))
    if clip <= 0:
        raise ValueError(f"partner_message_clip must be positive, got {clip}")
    agent = build_full6d_stable_mappo_agent(env, experiment_cfg)
    for agent_id, modules in agent.models.items():
        policy = modules["policy"]
        policy.partner_message_clip = clip
        policy._split_states = MethodType(_bounded_split_states, policy)
        print(f"[INFO] {agent_id} standardized_partner_message_clip={clip}")
    return agent
