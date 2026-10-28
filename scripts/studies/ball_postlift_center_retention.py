"""Unqueued, opt-in reward candidate for sustained post-lift ball centering.

The source camera actor is the exploratory Full6D-selected 26/30 box-lift
model. Simulator ball pose is read only by the existing reward path. The actor
still receives rendered depth/semantic-segmentation-derived ball position.
The native centered-held success predicate and original Lift rewards remain.
"""

import math
from pathlib import Path
from types import MethodType

import torch

from scripts.studies.ball_policy_transfer import initialize_from_ball
from scripts.studies.raw_camera_ball_reference import build_raw_camera_ball_reference_mappo_agent


BONUS_SCALE = 8.0
TRANSITION_WIDTH_M = 0.05
MARKER = "_ball_postlift_center_retention_reward_version"
SOURCE = (
    Path(__file__).resolve().parents[2]
    / "logs/motion_context_analysis"
    / "ball_successful_teacher_transfer_v1/common_actor.pt"
)


def postlift_center_bonus(
    error: torch.Tensor,
    inside: torch.Tensor,
    box_valid: torch.Tensor,
    threshold_m: float,
    scale: float = BONUS_SCALE,
    width_m: float = TRANSITION_WIDTH_M,
) -> torch.Tensor:
    """Dense, bounded centeredness credit only while original box hold is valid."""
    if (
        error.ndim != 1
        or inside.shape != error.shape
        or box_valid.shape != error.shape
        or inside.dtype != torch.bool
        or box_valid.dtype != torch.bool
        or not all(math.isfinite(float(v)) for v in (threshold_m, scale, width_m))
        or threshold_m <= 0
        or scale < 0
        or width_m <= 0
    ):
        raise ValueError("Finite aligned post-lift ball center signals required")
    safe_error = torch.nan_to_num(error, nan=1.0e6, posinf=1.0e6, neginf=1.0e6)
    centeredness = 0.5 * (1.0 + torch.tanh((threshold_m - safe_error) / width_m))
    valid = box_valid & inside & torch.isfinite(error) & (error >= 0)
    return scale * torch.where(valid, centeredness, torch.zeros_like(centeredness))


def install_postlift_center_retention(task) -> None:
    if (
        getattr(task, "_ball_postlift_center_retention_installed", False)
        or getattr(task, "_ball_goal_variant", None) != "centered_hold_v1"
        or getattr(task.cfg, "ball_reward_profile", None) != "center_penalty"
        or float(getattr(task.cfg, "ball_center_success_threshold", -1)) != 0.05
        or int(getattr(task.cfg, "hold_required_steps", -1)) != 60
        or "ball_box_only_hold_count" not in task._ctx
    ):
        raise ValueError("Exact camera-Ball centered-held goal required")
    original = task._get_rewards

    def rewards(self):
        values = original()
        _, metrics = self._ball_metrics()  # Simulator pose remains reward-only.
        bonus = postlift_center_bonus(
            metrics["error"],
            metrics["inside"],
            self._ctx["ball_box_only_hold_count"] > 0,
            float(self.cfg.ball_center_success_threshold),
        )
        if not values or any(value.shape != bonus.shape for value in values.values()):
            raise ValueError("Expected one team reward per environment")
        logs = self.extras.setdefault("log", {})
        logs["reward_ball_postlift_center_retention"] = float(bonus.mean())
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
                logs[key] += float(bonus.mean())
        return {uid: value + bonus for uid, value in values.items()}

    task._get_rewards = MethodType(rewards, task)
    task._ball_postlift_center_retention_installed = True


def build_ball_postlift_center_retention_mappo_agent(env, experiment_cfg):
    from isaaclab_tasks.direct.openarm_motion_context.envs.ball_center_goal import unwrap_ball_task

    agent = build_raw_camera_ball_reference_mappo_agent(env, experiment_cfg)
    task = unwrap_ball_task(env)
    provenance = initialize_from_ball(agent, SOURCE)
    install_postlift_center_retention(task)
    print(
        "[BALL_POSTLIFT_CENTER_WARMSTART] "
        + provenance["source_sha256"]
        + " trained_camera68 copied; critic_optimizer_fresh",
        flush=True,
    )
    agent._ball_postlift_center_retention_warmstart = provenance
    for modules in agent.models.values():
        critic = modules["value"]
        critic.register_buffer(MARKER, torch.tensor(1.0, device=critic.device))

        def validate(module, incompatible):
            if MARKER in incompatible.missing_keys or float(getattr(module, MARKER)) != 1.0:
                raise ValueError("Ball post-lift center retention critic identity missing or changed")

        critic.register_load_state_dict_post_hook(validate)
    return agent
