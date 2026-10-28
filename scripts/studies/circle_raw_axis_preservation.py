"""Opt-in Circle V11 reward for preserving alignment during approach.

Raw seed47 starts with axis≈1 but loses it by step20, before entering the
mouth band near step51. The added term is zero at perfect alignment and
negative only when tilted, so it offers no positive reward for camping at
the distant initial pose. Actor, observations, physics, and strict goal stay
unchanged. Both communication methods must use the same reward.
"""

import math
from types import MethodType

import torch


SCALE = 8.0
APPROACH_WIDTH_M = 0.15
MARKER = "_circle_raw_axis_preservation_reward_version"


def axis_preservation_penalty(axis, lateral):
    if (
        axis.ndim != 1
        or lateral.shape != axis.shape
        or not torch.isfinite(axis).all()
        or not torch.isfinite(lateral).all()
        or torch.any(axis < -1)
        or torch.any(axis > 1)
        or torch.any(lateral < 0)
        or not math.isfinite(SCALE)
        or SCALE <= 0
        or not math.isfinite(APPROACH_WIDTH_M)
        or APPROACH_WIDTH_M <= 0
    ):
        raise ValueError("Finite Circle axis [-1,1] and nonnegative lateral distance required")
    proximity = torch.exp(-lateral / APPROACH_WIDTH_M)
    return -SCALE * proximity * (1.0 - axis)


def install_axis_preservation_reward(base, logic):
    if getattr(base, "_circle_raw_axis_preservation_installed", False):
        raise ValueError("Circle axis-preservation reward already installed")
    original = base._get_rewards

    def rewards(self):
        values = original()
        metrics = logic.compute_peg_in_hole_metrics(self, self._ctx)
        delta = axis_preservation_penalty(metrics["axis_alignment"], metrics["lateral_error"])
        if not values or any(value.shape != delta.shape for value in values.values()):
            raise ValueError("Circle team-reward shape changed")
        adjusted = {uid: value + delta for uid, value in values.items()}
        stats = self._ctx["debug_stats"]
        stats["reward_circle_axis_preservation_penalty"] = float(delta.mean())
        for key in ("team_reward", "left_reward", "right_reward", "left_reward_raw", "right_reward_raw"):
            if key in stats:
                stats[key] += float(delta.mean())
        self.extras["log"] = dict(stats)
        return adjusted

    base._get_rewards = MethodType(rewards, base)
    base._circle_raw_axis_preservation_installed = True


def build_circle_raw_axis_preservation_mappo_agent(env, experiment_cfg):
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
        or experiment_cfg["agent"].get("clip_predicted_values") is not True
    ):
        raise ValueError("Exact raw Circle V11 matched actor and original reward required")
    agent = build_matched_communication_mappo_agent(env, experiment_cfg)
    install_axis_preservation_reward(base, logic)
    for modules in agent.models.values():
        critic = modules["value"]
        critic.register_buffer(MARKER, torch.tensor(1.0, device=critic.device))

        def validate(module, incompatible):
            if MARKER in incompatible.missing_keys or float(getattr(module, MARKER)) != 1.0:
                raise ValueError("Circle axis-preservation critic identity missing or changed")

        critic.register_load_state_dict_post_hook(validate)
    return agent
