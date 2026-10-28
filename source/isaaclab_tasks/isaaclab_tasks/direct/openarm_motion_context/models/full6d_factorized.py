"""Factorized, task-agnostic encoder for the Full 6D Motion Trend descriptor.

The 30D fixed communication slot is retained for architecture-matched
comparisons, but only the first six descriptor channels are used. Current
signed motion and temporal trends have separate 3D-to-16D encoders. The
resulting 32D feature enters the unchanged bounded partner-action pathway.
"""

from __future__ import annotations

import torch
from torch import nn

from .full6d_balanced import build_full6d_balanced_mappo_agent


class FactorizedMotionTrendEncoder(nn.Module):
    """Preserve the semantic motion/trend split until the action head."""

    def __init__(self, branch_width: int = 16):
        super().__init__()
        if branch_width < 1:
            raise ValueError("branch_width must be positive")
        self.motion = nn.Sequential(
            nn.Linear(3, branch_width), nn.ELU(), nn.Linear(branch_width, branch_width), nn.ELU()
        )
        self.trend = nn.Sequential(
            nn.Linear(3, branch_width), nn.ELU(), nn.Linear(branch_width, branch_width), nn.ELU()
        )
        with torch.no_grad():
            for module in self.modules():
                if isinstance(module, nn.Linear):
                    module.bias.zero_()

    def forward(self, descriptor: torch.Tensor) -> torch.Tensor:
        if descriptor.shape[-1] < 6:
            raise ValueError(f"Full 6D descriptor expected, got shape {tuple(descriptor.shape)}")
        return torch.cat((self.motion(descriptor[..., :3]), self.trend(descriptor[..., 3:6])), dim=-1)


def refresh_agent_optimizer_parameters(agent, agent_id: str) -> None:
    """Replace pre-training optimizer references after swapping policy modules."""

    optimizer = agent.optimizers[agent_id]
    if len(optimizer.param_groups) != 1 or optimizer.state:
        raise RuntimeError("Factorized encoder must be installed before the first optimizer step")
    policy = agent.models[agent_id]["policy"]
    value = agent.models[agent_id]["value"]
    parameters = list(policy.parameters()) + list(value.parameters())
    if len({id(param) for param in parameters}) != len(parameters):
        raise RuntimeError("Policy/value optimizer parameters overlap")
    optimizer.param_groups[0]["params"] = parameters


def build_full6d_factorized_mappo_agent(env, experiment_cfg: dict):
    """Build bounded Full 6D MAPPO with separate Motion and Trend branches."""

    agent = build_full6d_balanced_mappo_agent(env, experiment_cfg)
    for agent_id, modules in agent.models.items():
        policy = modules["policy"]
        if (
            policy.communication_mode != "motion_context"
            or policy.motion_intent_dim != 3
            or policy.motion_context_dim != 3
        ):
            raise ValueError("Factorized encoder requires 3D Motion + 3D Trend")
        if policy.partner_intent_embed_dim != 32:
            raise ValueError("Factorized encoder requires 32D partner embedding")
        policy.partner_intent_encoder = FactorizedMotionTrendEncoder(branch_width=16).to(env.device)
        refresh_agent_optimizer_parameters(agent, agent_id)
        print(f"[INFO] {agent_id} factorized_motion_trend_encoder=3+3_to_16+16")
    return agent
