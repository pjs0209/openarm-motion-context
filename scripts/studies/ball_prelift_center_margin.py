"""Opt-in Ball reward candidate: preserve a centered margin during the lift.

This is a developmental reward study sharing the same selected camera68 actor
as the frozen post-lift candidate. Simulator Ball position enters reward only;
the actor still receives rendered depth and semantic-segmentation features.
Native Lift reward and strict 60-step centered-held termination are unchanged.
"""

import math
from pathlib import Path
from types import MethodType

import torch

from scripts.studies.ball_policy_transfer import initialize_from_ball
from scripts.studies.raw_camera_ball_reference import build_raw_camera_ball_reference_mappo_agent


BONUS_SCALE = 2.0
TARGET_ERROR_M = 0.035
TRANSITION_WIDTH_M = 0.015
MARKER = "_ball_prelift_center_margin_reward_version"
SOURCE = (
    Path(__file__).resolve().parents[2]
    / "logs/motion_context_analysis"
    / "ball_successful_teacher_transfer_v1/common_actor.pt"
)


def prelift_center_margin_bonus(
    error: torch.Tensor,
    inside: torch.Tensor,
    lift_progress: torch.Tensor,
    threshold_m: float,
    scale: float = BONUS_SCALE,
    target_m: float = TARGET_ERROR_M,
    width_m: float = TRANSITION_WIDTH_M,
) -> torch.Tensor:
    """Positive reward inside a 3.5 cm target, gated by actual lift progress.

    Centeredness starts changing before the 5 cm success boundary. Multiplying
    by lift progress prevents a stationary floor-level reward optimum.
    """
    if (
        error.ndim != 1
        or inside.shape != error.shape
        or lift_progress.shape != error.shape
        or inside.dtype != torch.bool
        or not all(math.isfinite(float(v)) for v in (threshold_m, scale, target_m, width_m))
        or threshold_m <= 0
        or scale < 0
        or not 0 < target_m < threshold_m
        or width_m <= 0
    ):
        raise ValueError("Finite aligned pre-lift Ball margin signals required")
    safe_error = torch.nan_to_num(error, nan=1.0e6, posinf=1.0e6, neginf=1.0e6)
    centeredness = 0.5 * (1.0 + torch.tanh((target_m - safe_error) / width_m))
    safe_progress = torch.nan_to_num(lift_progress, nan=0.0, posinf=0.0, neginf=0.0)
    valid = inside & torch.isfinite(error) & (error >= 0) & torch.isfinite(lift_progress)
    return scale * torch.where(valid, centeredness * safe_progress.clamp(0.0, 1.0), torch.zeros_like(centeredness))


def install_prelift_center_margin(task) -> None:
    if (
        getattr(task, "_ball_prelift_center_margin_installed", False)
        or getattr(task, "_ball_goal_variant", None) != "centered_hold_v1"
        or getattr(task.cfg, "ball_reward_profile", None) != "center_penalty"
        or float(getattr(task.cfg, "ball_center_success_threshold", -1)) != 0.05
        or float(getattr(task.cfg, "ball_center_penalty_full_error", -1)) != 0.25
        or int(getattr(task.cfg, "hold_required_steps", -1)) != 60
    ):
        raise ValueError("Exact camera-Ball centered-held goal and penalty required")
    original = task._get_rewards

    def rewards(self):
        values = original()
        _, metrics = self._ball_metrics()  # Simulator pose stays reward-only.
        bonus = prelift_center_margin_bonus(
            metrics["error"],
            metrics["inside"],
            self._ctx["trace_signals"]["height_progress"],
            float(self.cfg.ball_center_success_threshold),
        )
        if not values or any(value.shape != bonus.shape for value in values.values()):
            raise ValueError("Expected one team reward per environment")
        logs = self.extras.setdefault("log", {})
        mean_bonus = float(bonus.mean())
        logs["reward_ball_prelift_center_margin"] = mean_bonus
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
                logs[key] += mean_bonus
        return {uid: value + bonus for uid, value in values.items()}

    task._get_rewards = MethodType(rewards, task)
    task._ball_prelift_center_margin_installed = True


def build_ball_prelift_center_margin_mappo_agent(env, experiment_cfg):
    from isaaclab_tasks.direct.openarm_motion_context.envs.ball_center_goal import unwrap_ball_task

    agent = build_raw_camera_ball_reference_mappo_agent(env, experiment_cfg)
    task = unwrap_ball_task(env)
    provenance = initialize_from_ball(agent, SOURCE)
    install_prelift_center_margin(task)
    print(
        "[BALL_PRELIFT_MARGIN_WARMSTART] "
        + provenance["source_sha256"]
        + " trained_camera68 copied; critic_optimizer_fresh",
        flush=True,
    )
    agent._ball_prelift_center_margin_warmstart = provenance
    for modules in agent.models.values():
        critic = modules["value"]
        critic.register_buffer(MARKER, torch.tensor(1.0, device=critic.device))

        def validate(module, incompatible):
            if MARKER in incompatible.missing_keys or float(getattr(module, MARKER)) != 1.0:
                raise ValueError("Ball pre-lift margin critic identity missing or changed")

        critic.register_load_state_dict_post_hook(validate)
    return agent
