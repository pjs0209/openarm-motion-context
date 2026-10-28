"""Opt-in Circle V11 reward candidate for safe pre-insertion approach.

The reward-only penalty discourages the sidewall-invalid combination of a
near-mouth peg, positive insertion depth, and excessive lateral offset.
Native success and invalid guards, actor observations, and 6D communication
are not changed. This candidate is not part of any frozen queued study.
"""

import math
from types import MethodType

import torch


SCALE = 4.0
SAFE_LATERAL_M = 0.020
LATERAL_RAMP_M = 0.030
TIP_GATE_START_M = 0.10
TIP_GATE_FULL_M = 0.06
DEPTH_GATE_START_M = -0.01
DEPTH_GATE_FULL_M = 0.01
MARKER = "_circle_sidewall_approach_margin_reward_version"


def sidewall_approach_penalty(
    lateral: torch.Tensor, tip_distance: torch.Tensor, insertion_depth: torch.Tensor
) -> torch.Tensor:
    """Return bounded nonpositive credit before the native sidewall guard fires."""
    if (
        lateral.ndim != 1
        or tip_distance.shape != lateral.shape
        or insertion_depth.shape != lateral.shape
        or any(not torch.isfinite(x).all() for x in (lateral, tip_distance, insertion_depth))
        or torch.any(lateral < 0)
        or torch.any(tip_distance < 0)
        or not all(
            math.isfinite(v)
            for v in (
                SCALE,
                SAFE_LATERAL_M,
                LATERAL_RAMP_M,
                TIP_GATE_START_M,
                TIP_GATE_FULL_M,
                DEPTH_GATE_START_M,
                DEPTH_GATE_FULL_M,
            )
        )
        or SCALE < 0
        or SAFE_LATERAL_M < 0
        or LATERAL_RAMP_M <= 0
        or TIP_GATE_START_M <= TIP_GATE_FULL_M
        or DEPTH_GATE_START_M >= DEPTH_GATE_FULL_M
    ):
        raise ValueError("Finite aligned Circle pre-insertion sidewall signals required")
    lateral_risk = ((lateral - SAFE_LATERAL_M) / LATERAL_RAMP_M).clamp(0.0, 1.0)
    tip_gate = ((TIP_GATE_START_M - tip_distance) / (TIP_GATE_START_M - TIP_GATE_FULL_M)).clamp(0.0, 1.0)
    depth_gate = ((insertion_depth - DEPTH_GATE_START_M) / (DEPTH_GATE_FULL_M - DEPTH_GATE_START_M)).clamp(0.0, 1.0)
    return -SCALE * lateral_risk * tip_gate * depth_gate


def install_sidewall_approach_margin(base, logic) -> None:
    if (
        getattr(base, "_circle_sidewall_approach_margin_installed", False)
        or getattr(type(base.cfg), "__name__", "") != "OpenArmFixedCirclePegAndHoleV11EnvCfg"
        or float(getattr(base.cfg, "wall_penetration_lateral_threshold", -1)) != 0.025
        or float(getattr(base.cfg, "wall_penetration_depth_threshold", -1)) != 0.01
        or float(getattr(base.cfg, "wall_penetration_max_tip_distance", -1)) != 0.06
    ):
        raise ValueError("Exact Circle V11 native sidewall guard required")
    original = base._get_rewards

    def rewards(self):
        values = original()
        metrics = logic.compute_peg_in_hole_metrics(self, self._ctx)
        penalty = sidewall_approach_penalty(metrics["lateral_error"], metrics["tip_dist"], metrics["insertion_depth"])
        if not values or any(value.shape != penalty.shape for value in values.values()):
            raise ValueError("Circle team-reward shape changed")
        stats = self._ctx["debug_stats"]
        amount = float(penalty.mean())
        stats["reward_circle_sidewall_approach_margin"] = amount
        for key in ("team_reward", "left_reward", "right_reward", "left_reward_raw", "right_reward_raw"):
            if key in stats:
                stats[key] += amount
        self.extras["log"] = dict(stats)
        return {uid: value + penalty for uid, value in values.items()}

    base._get_rewards = MethodType(rewards, base)
    base._circle_sidewall_approach_margin_installed = True


def build_circle_sidewall_approach_margin_mappo_agent(env, experiment_cfg):
    from isaaclab_tasks.direct.openarm_motion_context.envs import peg_in_hole_task_logic as logic
    from isaaclab_tasks.direct.openarm_motion_context.models.matched_communication import (
        build_matched_communication_mappo_agent,
    )

    base = env.unwrapped
    if (
        experiment_cfg["agent"].get("clip_predicted_values") is not False
        or getattr(getattr(base._get_rewards, "__func__", None), "__qualname__", "")
        != "OpenArmFixedPegAndHoleEnv._get_rewards"
    ):
        raise ValueError("Exact Circle V11 reward and unclipped value loss required")
    agent = build_matched_communication_mappo_agent(env, experiment_cfg)
    install_sidewall_approach_margin(base, logic)
    for modules in agent.models.values():
        critic = modules["value"]
        critic.register_buffer(MARKER, torch.tensor(1.0, device=critic.device))

        def validate(module, incompatible):
            if MARKER in incompatible.missing_keys or float(getattr(module, MARKER)) != 1.0:
                raise ValueError("Circle sidewall approach critic identity missing or changed")

        critic.register_load_state_dict_post_hook(validate)
    return agent
