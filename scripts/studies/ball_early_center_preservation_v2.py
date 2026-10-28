"""Opt-in early center-preservation cost for the camera-Ball combo package.

This candidate leaves the original Lift reward, native off-center penalty,
post-lift retention and pre-lift margin unchanged. Simulator ball pose enters
only this reward; the actor still receives the rendered camera estimate.
"""

import math
from types import MethodType

import torch

from scripts.studies.ball_postlift_margin_combo import (
    build_ball_postlift_margin_combo_mappo_agent,
)


SCALE = 1.0
FREE_ERROR_M = 0.01
FLOOR_WEIGHT = 0.5
MARKER = "_ball_early_center_preservation_v2_reward_version"


def early_center_guard(
    error: torch.Tensor,
    inside: torch.Tensor,
    height_progress: torch.Tensor,
    threshold_m: float,
    scale: float = SCALE,
    free_error_m: float = FREE_ERROR_M,
    floor_weight: float = FLOOR_WEIGHT,
) -> torch.Tensor:
    """Penalize drift from 1 to 5 cm, including before the first box hold.

    The term is nonpositive and zero when centered, so it cannot grant a
    stationary-floor *bonus*. A half-strength floor term gives feedback
    before measurable elevation; it grows smoothly with actual lift progress.
    """
    if (
        error.ndim != 1
        or inside.shape != error.shape
        or height_progress.shape != error.shape
        or inside.dtype != torch.bool
        or not all(math.isfinite(float(v)) for v in (threshold_m, scale, free_error_m, floor_weight))
        or threshold_m <= 0
        or scale < 0
        or not 0 <= free_error_m < threshold_m
        or not 0 <= floor_weight <= 1
    ):
        raise ValueError("Finite aligned early-center guard signals required")
    safe_error = torch.nan_to_num(error, nan=threshold_m, posinf=threshold_m, neginf=threshold_m)
    safe_height = torch.nan_to_num(height_progress, nan=0, posinf=0, neginf=0)
    drift = ((safe_error - free_error_m) / (threshold_m - free_error_m)).clamp(0, 1)
    weight = floor_weight + (1 - floor_weight) * safe_height.clamp(0, 1)
    # Dropping this term outside the box would make escape less costly than
    # severe in-box drift, despite the unchanged native off-center penalty.
    drift = torch.where(inside, drift, torch.ones_like(drift))
    valid = torch.isfinite(error) & (error >= 0) & torch.isfinite(height_progress)
    return torch.where(valid, -scale * weight * drift, torch.zeros_like(error))


def install_early_center_preservation(task) -> None:
    if (
        getattr(task, "_ball_early_center_preservation_v2_installed", False)
        or not getattr(task, "_ball_prelift_center_margin_installed", False)
        or not getattr(task, "_ball_postlift_center_retention_installed", False)
        or getattr(task, "_ball_goal_variant", None) != "centered_hold_v1"
        or getattr(task.cfg, "ball_reward_profile", None) != "center_penalty"
        or float(getattr(task.cfg, "ball_center_success_threshold", -1)) != 0.05
        or float(getattr(task.cfg, "ball_center_penalty_full_error", -1)) != 0.25
        or int(getattr(task.cfg, "hold_required_steps", -1)) != 60
    ):
        raise ValueError("Exact camera-Ball centered-held combo package required")
    original = task._get_rewards

    def rewards(self):
        values = original()
        _, metrics = self._ball_metrics()  # Simulator truth remains reward-only.
        penalty = early_center_guard(
            metrics["error"],
            metrics["inside"],
            self._ctx["trace_signals"]["height_progress"],
            float(self.cfg.ball_center_success_threshold),
        )
        if not values or any(value.shape != penalty.shape for value in values.values()):
            raise ValueError("Expected one team reward per environment")
        mean_penalty = float(penalty.mean())
        logs = self.extras.setdefault("log", {})
        logs["reward_ball_early_center_preservation_v2"] = mean_penalty
        for key in (
            "team_reward",
            "left_reward",
            "right_reward",
            "left_reward_raw",
            "right_reward_raw",
            "left_reward_total",
            "right_reward_total",
        ):
            if key in logs:
                logs[key] += mean_penalty
        return {uid: value + penalty for uid, value in values.items()}

    task._get_rewards = MethodType(rewards, task)
    task._ball_early_center_preservation_v2_installed = True


def build_ball_early_center_preservation_v2_mappo_agent(env, experiment_cfg):
    from isaaclab_tasks.direct.openarm_motion_context.envs.ball_center_goal import unwrap_ball_task

    agent = build_ball_postlift_margin_combo_mappo_agent(env, experiment_cfg)
    task = unwrap_ball_task(env)
    install_early_center_preservation(task)
    for modules in agent.models.values():
        critic = modules["value"]
        critic.register_buffer(MARKER, torch.tensor(1.0, device=critic.device))

        def validate(module, incompatible):
            if MARKER in incompatible.missing_keys or float(getattr(module, MARKER)) != 1.0:
                raise ValueError("Ball early-center guard critic identity missing or changed")

        critic.register_load_state_dict_post_hook(validate)
    print("[BALL_EARLY_CENTER_PRESERVATION_V2] combo plus early nonpositive center cost", flush=True)
    return agent
