"""Full 6D Trend training with a bounded partner-to-action weight ratio.

The partner message remains available and trainable. After each MAPPO update,
its action-head columns are projected relative to the local proprioception
columns, preventing the partner pathway from dominating either arm.
"""

from __future__ import annotations

from types import MethodType

import torch

from .actor import MotionContextPolicyModel
from .full6d_stable import build_full6d_stable_mappo_agent


def project_partner_action_columns(policy: MotionContextPolicyModel, max_ratio: float) -> None:
    """Cap the partner-column Frobenius norm relative to local action columns."""

    if max_ratio <= 0:
        raise ValueError(f"max_ratio must be positive, got {max_ratio}")
    partner_start = policy.policy_feature_dim
    partner_end = policy.actor_action_input_dim
    with torch.no_grad():
        arm = policy.arm_action_head.weight
        arm_local_norm = arm[:, :partner_start].norm()
        arm_partner = arm[:, partner_start:partner_end]
        arm_partner_norm = arm_partner.norm()
        arm_scale = (max_ratio * arm_local_norm / arm_partner_norm.clamp_min(1e-12)).clamp(max=1.0)
        arm_partner.mul_(arm_scale)

        if policy.gripper_action_head is not None:
            gripper = policy.gripper_action_head[0].weight
            gripper_local_norm = torch.cat((gripper[:, :partner_start], gripper[:, partner_end:]), dim=1).norm()
            gripper_partner = gripper[:, partner_start:partner_end]
            gripper_partner_norm = gripper_partner.norm()
            gripper_scale = (max_ratio * gripper_local_norm / gripper_partner_norm.clamp_min(1e-12)).clamp(max=1.0)
            gripper_partner.mul_(gripper_scale)


def _balanced_update(self, timestep: int, timesteps: int) -> None:
    self._full6d_original_update(timestep, timesteps)
    for modules in self.models.values():
        project_partner_action_columns(modules["policy"], self._full6d_partner_max_ratio)


def build_full6d_balanced_mappo_agent(env, experiment_cfg: dict):
    """Use the standard Full 6D model and constrain only message-action weights."""

    max_ratio = float(experiment_cfg["models"]["policy"].get("partner_action_max_ratio", 0.5))
    if max_ratio <= 0:
        raise ValueError(f"partner_action_max_ratio must be positive, got {max_ratio}")
    agent = build_full6d_stable_mappo_agent(env, experiment_cfg)
    agent._full6d_original_update = agent._update
    agent._full6d_partner_max_ratio = max_ratio
    agent._update = MethodType(_balanced_update, agent)
    print(f"[INFO] full6d_partner_action_max_ratio={max_ratio}")
    return agent
