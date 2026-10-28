"""Opt-in Circle V11 reward candidate for coincident approach, axis and depth.

This is a prospective study module, not part of any registered running job.
The bonus peaks when the peg approaches the mouth with its axis aligned and
its insertion depth near the physical hole depth. The native success guard is
not changed. Ground-truth task geometry is used only by the reward/critic;
the communicating actor observation remains the existing proprioceptive one.
"""

import math
from types import MethodType

import torch


SCALE = 4.0
APPROACH_WIDTH_M = 0.15
DEPTH_WIDTH_M = 0.05
ALIGNMENT_POWER = 4
MARKER = "_circle_target_depth_alignment_reward_version"


def target_depth_alignment_bonus(
    axis: torch.Tensor,
    lateral: torch.Tensor,
    depth: torch.Tensor,
    target_depth: torch.Tensor,
) -> torch.Tensor:
    if (
        axis.ndim != 1
        or lateral.shape != axis.shape
        or depth.shape != axis.shape
        or target_depth.shape != axis.shape
        or not all(torch.isfinite(x).all() for x in (axis, lateral, depth, target_depth))
        or torch.any(axis < -1)
        or torch.any(axis > 1)
        or torch.any(lateral < 0)
        or torch.any(target_depth <= 0)
        or not math.isfinite(SCALE)
        or SCALE <= 0
        or not math.isfinite(APPROACH_WIDTH_M)
        or APPROACH_WIDTH_M <= 0
        or not math.isfinite(DEPTH_WIDTH_M)
        or DEPTH_WIDTH_M <= 0
    ):
        raise ValueError("Finite Circle axis, lateral distance and positive target depth required")
    axis_quality = ((axis + 1.0) * 0.5).pow(ALIGNMENT_POWER)
    approach_quality = torch.exp(-lateral / APPROACH_WIDTH_M)
    depth_quality = torch.exp(-(depth - target_depth).abs() / DEPTH_WIDTH_M)
    return SCALE * axis_quality * approach_quality * depth_quality


def install_target_depth_alignment_reward(base, logic) -> None:
    if getattr(base, "_circle_target_depth_alignment_installed", False):
        raise ValueError("Circle target-depth/alignment reward already installed")
    original = base._get_rewards

    def rewards(self):
        values = original()
        metrics = logic.compute_peg_in_hole_metrics(self, self._ctx)
        bonus = target_depth_alignment_bonus(
            metrics["axis_alignment"],
            metrics["lateral_error"],
            metrics["insertion_depth"],
            metrics["hole_depth"],
        )
        if not values or any(value.shape != bonus.shape for value in values.values()):
            raise ValueError("Circle team-reward shape changed")
        adjusted = {uid: value + bonus for uid, value in values.items()}
        stats = self._ctx["debug_stats"]
        amount = float(bonus.mean())
        stats["reward_circle_target_depth_alignment"] = amount
        for key in ("team_reward", "left_reward", "right_reward", "left_reward_raw", "right_reward_raw"):
            if key in stats:
                stats[key] += amount
        self.extras["log"] = dict(stats)
        return adjusted

    base._get_rewards = MethodType(rewards, base)
    base._circle_target_depth_alignment_installed = True


def build_circle_target_depth_alignment_mappo_agent(env, experiment_cfg):
    from isaaclab_tasks.direct.openarm_motion_context.envs import peg_in_hole_task_logic as logic
    from isaaclab_tasks.direct.openarm_motion_context.models.matched_communication import (
        build_matched_communication_mappo_agent,
    )

    base = env.unwrapped
    if (
        getattr(type(base.cfg), "__name__", "") != "OpenArmFixedCirclePegAndHoleV11EnvCfg"
        or int(getattr(base.cfg, "rotational_symmetry_order", -1)) != 0
        or getattr(getattr(base._get_rewards, "__func__", None), "__qualname__", "")
        != "OpenArmFixedPegAndHoleEnv._get_rewards"
        or experiment_cfg["agent"].get("clip_predicted_values") is not False
    ):
        raise ValueError("Exact Circle V11 reward, matched actor and unclipped value loss required")
    agent = build_matched_communication_mappo_agent(env, experiment_cfg)
    install_target_depth_alignment_reward(base, logic)
    for modules in agent.models.values():
        critic = modules["value"]
        critic.register_buffer(MARKER, torch.tensor(1.0, device=critic.device))

        def validate(module, incompatible):
            if MARKER in incompatible.missing_keys or float(getattr(module, MARKER)) != 1.0:
                raise ValueError("Circle target-depth/alignment critic identity missing or changed")

        critic.register_load_state_dict_post_hook(validate)
    return agent
