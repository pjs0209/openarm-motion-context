"""Separate Lift reward ablation: halve premature-closure cost, not success.

Base grasp score r*c-(1-r)*c becomes r*c-.5*(1-r)*c. The
correction uses the original per-environment measured reward terms; no sensor,
observation, phase, grasp predicate, contact signal or physical setting changes.
This is not potential-based shaping or a guarantee of policy improvement.
"""

from types import MethodType
import math

import torch

COEFFICIENT = 0.5
MARKER = "_grasp_closure_balance_version"
COEF_MARKER = "_grasp_closure_balance_coefficient"


def correction(left, right, reward_scale, coefficient=COEFFICIENT):
    if (
        not math.isfinite(reward_scale)
        or reward_scale < 0
        or not math.isfinite(coefficient)
        or not 0 < coefficient <= 1
    ):
        raise ValueError("Finite nonnegative scale and coefficient in(0,1] required")
    if (
        left.shape != right.shape
        or left.ndim != 1
        or not torch.isfinite(left).all()
        or not torch.isfinite(right).all()
        or (left < 0).any()
        or (right < 0).any()
        or (left > 1).any()
        or (right > 1).any()
    ):
        raise ValueError("Finite per-environment premature-closure scores in[0,1] required")
    return 0.5 * reward_scale * (1 - coefficient) * (left + right)


def install_environment(base):
    if hasattr(base, "_grasp_closure_balance_original"):
        raise ValueError("Reward adjustment already installed")
    original = base._get_rewards
    base._grasp_closure_balance_original = original

    def rewards(self):
        values = original()
        signals = self._ctx["trace_signals"]
        delta = correction(
            signals["left_premature_closure"], signals["right_premature_closure"], float(self.cfg.reward_grasp_scale)
        )
        if not values or any(value.shape != delta.shape for value in values.values()):
            raise ValueError("Reward shape changed")
        self.extras["log"].update(
            grasp_premature_closure_coefficient=COEFFICIENT, reward_grasp_closure_balance_delta=float(delta.mean())
        )
        return {uid: value + delta for uid, value in values.items()}

    base._get_rewards = MethodType(rewards, base)


def build_closure_balance_project_value_off_mappo_agent(env, experiment_cfg):
    from scripts.studies.project_unclipped_value_regression import build_project_unclipped_value_bounded_all_mappo_agent

    base = env.unwrapped
    # This version is deliberately plain Lift only; no accidental Circle use.
    if (
        int(getattr(base, "own_observation_dim", 0)) != 30
        or getattr(getattr(base._get_rewards, "__func__", None), "__qualname__", None)
        != "OpenArmReBimanualLiftEnv._get_rewards"
    ):
        raise ValueError("Plain30D Lift own observation required")
    install_environment(base)
    agent = build_project_unclipped_value_bounded_all_mappo_agent(env, experiment_cfg)
    for modules in agent.models.values():
        critic = modules["value"]
        critic.register_buffer(MARKER, torch.tensor(1.0, device=critic.device))
        critic.register_buffer(COEF_MARKER, torch.tensor(COEFFICIENT, device=critic.device))

        def validate(module, incompatible):
            if (
                MARKER in incompatible.missing_keys
                or COEF_MARKER in incompatible.missing_keys
                or float(getattr(module, MARKER)) != 1.0
                or float(getattr(module, COEF_MARKER)) != COEFFICIENT
            ):
                raise ValueError("Closure reward critic identity missing or changed")

        critic.register_load_state_dict_post_hook(validate)
    return agent
