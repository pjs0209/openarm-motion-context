"""Opt-in Ball reward candidate anchored to the successful 68D camera actor.

The actor still receives only rendered depth/segmentation-derived ball pose.
Simulator ball pose is read only by the existing reward-only ``_ball_metrics``.
This exploratory initializer is selected from a Full6D-successful Lift model;
it does not by itself establish a fair, independent-seed communication effect.
No active study uses this module yet.
"""

import math
from pathlib import Path
from types import MethodType

import torch

from scripts.studies.ball_policy_transfer import initialize_from_ball


EXTRA_SCALE = 4.0
FULL_ERROR_M = 0.10
MARKER = "_ball_early_center_guard_reward_version"
SOURCE = (
    Path(__file__).resolve().parents[2]
    / "logs/motion_context_analysis"
    / "ball_successful_teacher_transfer_v1/common_actor.pt"
)


def extra_early_center_penalty(error: torch.Tensor, inside: torch.Tensor, threshold_m: float) -> torch.Tensor:
    """Bounded extra penalty concentrated between center threshold and 10 cm."""

    if (
        error.ndim != 1
        or inside.shape != error.shape
        or inside.dtype != torch.bool
        or not math.isfinite(threshold_m)
        or not 0 <= threshold_m < FULL_ERROR_M
    ):
        raise ValueError("Finite Ball center threshold and aligned per-environment signals required")
    safe = torch.nan_to_num(error, nan=FULL_ERROR_M, posinf=FULL_ERROR_M, neginf=FULL_ERROR_M)
    severity = ((safe - threshold_m) / (FULL_ERROR_M - threshold_m)).clamp(0.0, 1.0)
    valid = torch.isfinite(error) & (error >= 0) & inside
    return -EXTRA_SCALE * torch.where(valid, severity, torch.ones_like(severity))


def install_early_center_guard(base):
    if getattr(base, "_ball_early_center_guard_installed", False):
        raise ValueError("Ball early-center guard already installed")
    original = base._get_rewards

    def rewards(self):
        values = original()
        _, metrics = self._ball_metrics()  # Ground truth is reward-only here.
        extra = extra_early_center_penalty(
            metrics["error"], metrics["inside"], float(self.cfg.ball_center_success_threshold)
        )
        if not values or any(value.shape != extra.shape for value in values.values()):
            raise ValueError("Ball team-reward shape changed")
        adjusted = {uid: value + extra for uid, value in values.items()}
        self.extras["log"]["reward_ball_early_center_guard"] = float(extra.mean())
        return adjusted

    base._get_rewards = MethodType(rewards, base)
    base._ball_early_center_guard_installed = True


def build_ball_early_center_guard_mappo_agent(env, experiment_cfg):
    from scripts.studies.raw_camera_ball_reference import build_raw_camera_ball_reference_mappo_agent

    base = env.unwrapped
    if (
        getattr(type(base.cfg), "__name__", "") != "OpenArmLiftWithBallEnvCfg"
        or getattr(base.cfg, "ball_reward_profile", "") != "center_penalty"
        or float(base.cfg.ball_center_penalty_scale) != 2.0
        or float(base.cfg.ball_center_penalty_full_error) != FULL_ERROR_M
        or float(base.cfg.ball_center_success_threshold) != 0.05
        or int(base.cfg.camera_extra_observation_dim) != 4
        or int(base.cfg.actor_input_dim) != 68
        or getattr(getattr(base._get_rewards, "__func__", None), "__qualname__", "")
        != "OpenArmLiftWithBallEnv._get_rewards"
        or experiment_cfg["models"]["policy"].get("clip_actions") is not False
        or experiment_cfg["agent"].get("clip_predicted_values") is not True
    ):
        raise ValueError("Exact rendered-camera Ball center-penalty task and raw actor required")
    # This installs the existing centered-held60 success predicate and a
    # featurewise 68D camera normalizer before the audited actor is copied.
    agent = build_raw_camera_ball_reference_mappo_agent(env, experiment_cfg)
    if getattr(base, "_ball_goal_variant", None) != "centered_hold_v1":
        raise ValueError("Native centered-held Ball goal was not installed")
    provenance = initialize_from_ball(agent, SOURCE)
    print(
        "[BALL_EARLY_CENTER_WARMSTART] "
        + provenance["source_sha256"]
        + " trained_camera68 copied; critic_optimizer_fresh",
        flush=True,
    )
    install_early_center_guard(base)
    agent._ball_early_center_guard_warmstart = provenance
    for modules in agent.models.values():
        critic = modules["value"]
        critic.register_buffer(MARKER, torch.tensor(1.0, device=critic.device))

        def validate(module, incompatible):
            if MARKER in incompatible.missing_keys or float(getattr(module, MARKER)) != 1.0:
                raise ValueError("Ball early-center guard critic identity missing or changed")

        critic.register_load_state_dict_post_hook(validate)
    return agent
