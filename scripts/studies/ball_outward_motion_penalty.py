"""Unqueued reward-only penalty for outward ball motion during a box lift.

The camera68 actor and its Full6D/NoShare communication inputs are unchanged.
Simulator ball/box velocities are used only in this reward. Native Lift and
strict centered-held success predicates are untouched. This is a candidate,
not a measured strict-success improvement.
"""

import math
from pathlib import Path
from types import MethodType

import torch

from scripts.studies.ball_policy_transfer import initialize_from_ball
from scripts.studies.raw_camera_ball_reference import build_raw_camera_ball_reference_mappo_agent


SOURCE = (
    Path(__file__).resolve().parents[2]
    / "logs/motion_context_analysis"
    / "ball_successful_teacher_transfer_v1/common_actor.pt"
)
MARKER = "_ball_outward_motion_penalty_reward_version"
SCALE = 2.0
SPEED_WIDTH_MPS = 0.10
START_ERROR_M = 0.02


def outward_motion_penalty(
    relative_xy: torch.Tensor,
    relative_velocity_xy: torch.Tensor,
    center_xy: torch.Tensor,
    lift_progress: torch.Tensor,
    inside: torch.Tensor,
    *,
    scale: float = SCALE,
    speed_width: float = SPEED_WIDTH_MPS,
    start_error: float = START_ERROR_M,
) -> torch.Tensor:
    """Penalize only radial motion away from the box center while lifting."""
    if (
        relative_xy.ndim != 2
        or relative_xy.shape[1] != 2
        or relative_velocity_xy.shape != relative_xy.shape
        or center_xy.shape not in ((2,), relative_xy.shape)
        or lift_progress.shape != relative_xy.shape[:1]
        or inside.shape != lift_progress.shape
        or inside.dtype != torch.bool
        or any(
            not torch.isfinite(value).all() for value in (relative_xy, relative_velocity_xy, center_xy, lift_progress)
        )
        or not all(math.isfinite(float(value)) and float(value) > 0 for value in (scale, speed_width, start_error))
    ):
        raise ValueError("Finite aligned ball position, velocity and lift signals required")
    displacement = relative_xy - center_xy
    distance = torch.linalg.vector_norm(displacement, dim=1)
    radial_speed = torch.sum(displacement * relative_velocity_xy, dim=1) / distance.clamp_min(1e-6)
    outward = radial_speed.clamp_min(0)
    active = inside & (distance > start_error)
    progress = lift_progress.clamp(0, 1)
    return -scale * torch.where(active, torch.tanh(outward / speed_width), torch.zeros_like(outward)) * progress


def install_outward_motion_penalty(task):
    from isaaclab_tasks.direct.openarm_motion_context.legacy_math import quat_apply, quat_inv

    cfg = task.cfg
    if (
        getattr(task, MARKER, False)
        or getattr(task, "_ball_goal_variant", None) != "centered_hold_v1"
        or cfg.ball_reward_profile != "center_penalty"
        or float(cfg.ball_center_penalty_full_error) != 0.25
        or float(cfg.ball_center_success_threshold) != 0.05
        or int(cfg.hold_required_steps) != 60
    ):
        raise ValueError("Exact original camera-Ball centered-held goal required")
    original = task._get_rewards

    def rewards(self):
        values = original()
        local, metrics = self._ball_metrics()
        box = self.object.data
        ball = self.ball.data
        relative_pos_w = ball.root_pos_w - box.root_pos_w
        relative_velocity_w = (
            ball.root_lin_vel_w - box.root_lin_vel_w - torch.cross(box.root_ang_vel_w, relative_pos_w, dim=1)
        )
        velocity_local = quat_apply(quat_inv(box.root_quat_w), relative_velocity_w)
        center_xy = torch.as_tensor(self.cfg.ball_center_xy, device=local.device, dtype=local.dtype)
        bonus = outward_motion_penalty(
            local[:, :2],
            velocity_local[:, :2],
            center_xy,
            self._ctx["trace_signals"]["height_progress"],
            metrics["inside"],
        )
        if not values or any(value.shape != bonus.shape for value in values.values()):
            raise ValueError("Expected one team reward per environment")
        mean = float(bonus.mean())
        logs = self.extras.setdefault("log", {})
        logs["reward_ball_outward_motion_penalty"] = mean
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
                logs[key] += mean
        return {uid: value + bonus for uid, value in values.items()}

    task._get_rewards = MethodType(rewards, task)
    setattr(task, MARKER, True)


def build_ball_outward_motion_penalty_mappo_agent(env, experiment_cfg):
    from isaaclab_tasks.direct.openarm_motion_context.envs.ball_center_goal import unwrap_ball_task

    agent = build_raw_camera_ball_reference_mappo_agent(env, experiment_cfg)
    task = unwrap_ball_task(env)
    provenance = initialize_from_ball(agent, SOURCE)
    install_outward_motion_penalty(task)
    for modules in agent.models.values():
        critic = modules["value"]
        critic.register_buffer(MARKER, torch.tensor(1.0, device=critic.device))

        def validate(module, incompatible):
            if MARKER in incompatible.missing_keys or float(getattr(module, MARKER)) != 1.0:
                raise ValueError("Ball outward-motion reward identity missing or changed")

        critic.register_load_state_dict_post_hook(validate)
    agent._ball_outward_motion_warmstart = provenance
    print(
        "[BALL_OUTWARD_MOTION_WARMSTART] "
        + provenance["source_sha256"]
        + " trained camera68 copied; critic_optimizer_fresh",
        flush=True,
    )
    return agent
