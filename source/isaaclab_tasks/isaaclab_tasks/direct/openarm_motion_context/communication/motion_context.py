"""Task-agnostic proprioceptive motion-context descriptor."""

import torch


DEFAULT_CALIBRATION_ENABLED = True
DEFAULT_CALIBRATION_WINDOW = 256
DEFAULT_CALIBRATION_CV_THRESHOLD = 0.02
DEFAULT_CALIBRATION_CONFIRMATIONS = 64
DEFAULT_CALIBRATION_EPSILON = 1.0e-6
DEFAULT_TREND_HORIZON = 5
DEFAULT_TREND_BETA = 0.8
DEFAULT_TREND_LINEAR_EPSILON = 1.0e-2
DEFAULT_TREND_ANGULAR_EPSILON = 5.0e-2
DEFAULT_TREND_ACTION_EPSILON = 1.0e-2
REPRESENTATION_LAYOUTS = {
    "trend": ("translation_trend", "rotation_trend", "stability_trend"),
    "absolute_fixed": ("linear_activity", "angular_activity", "action_smoothness"),
    "signed_twist": ("angular_velocity_x", "angular_velocity_y", "angular_velocity_z"),
}


def motion_context_layout(representation: str) -> tuple[str, str, str]:
    """Validate a task-agnostic 3D context representation and name its slots."""

    try:
        return REPRESENTATION_LAYOUTS[str(representation)]
    except KeyError as exc:
        raise ValueError(f"Unsupported motion_context_representation={representation!r}") from exc


def select_motion_context(
    representation: str,
    trend: torch.Tensor,
    linear_speed: torch.Tensor,
    angular_speed: torch.Tensor,
    action_variation_ema: torch.Tensor,
    angular_velocity_base: torch.Tensor,
    fixed_scale: torch.Tensor,
    absolute_norm_max: float = 1.5,
    twist_angular_scale: float = 3.0,
) -> torch.Tensor:
    """Select a 3D payload without changing the common signed translation 3D.

    ``absolute_fixed`` deliberately uses fixed, declared physical scales; it
    does not inherit the historical online P90 calibration or a sharing gate.
    ``signed_twist`` encodes base-frame angular velocity, not angular speed.
    """

    motion_context_layout(representation)
    if representation == "trend":
        return trend
    if representation == "absolute_fixed":
        return motion_context_from_raw(
            linear_speed,
            angular_speed,
            action_variation_ema,
            fixed_scale,
            norm_max=absolute_norm_max,
        )
    angular_scale = max(float(twist_angular_scale), 1.0e-6)
    return torch.nan_to_num(angular_velocity_base / angular_scale, nan=0.0, posinf=1.0, neginf=-1.0).clamp(-1.0, 1.0)


def full_action_variation(current_action: torch.Tensor, previous_action: torch.Tensor) -> torch.Tensor:
    """Return the per-agent full-action RMS variation as a column vector."""

    previous_action = previous_action.to(device=current_action.device, dtype=current_action.dtype)
    return torch.sqrt(torch.mean((current_action - previous_action).square(), dim=-1, keepdim=True))


def update_variation_ema(
    variation: torch.Tensor,
    previous_ema: torch.Tensor,
    valid: torch.Tensor,
    beta: float = 0.8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Update short-horizon variation EMA without introducing a reset-step spike."""

    beta = min(max(float(beta), 0.0), 0.9999)
    valid = valid.to(device=variation.device, dtype=torch.bool)
    effective_variation = torch.where(valid, variation, torch.zeros_like(variation))
    updated = beta * previous_ema + (1.0 - beta) * effective_variation
    updated = torch.where(valid, updated, torch.zeros_like(updated))
    return effective_variation, updated


def relative_motion_trend(
    current_ema: torch.Tensor,
    lagged_ema: torch.Tensor,
    epsilon: torch.Tensor | float,
) -> torch.Tensor:
    """Return bounded relative trends [T_v, T_omega, T_S]."""

    epsilon = torch.as_tensor(epsilon, device=current_ema.device, dtype=current_ema.dtype)
    denominator = current_ema.abs() + lagged_ema.abs() + epsilon
    trend = (current_ema - lagged_ema) / denominator
    trend = trend.clone()
    trend[..., 2] = -trend[..., 2]
    return torch.nan_to_num(trend, nan=0.0, posinf=1.0, neginf=-1.0).clamp(-1.0, 1.0)


def update_motion_trend_context(
    env,
    left_raw: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    right_raw: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Update reset-safe EMA/history state and return each arm's trend context."""

    horizon = max(int(getattr(env.cfg, "motion_context_trend_horizon", DEFAULT_TREND_HORIZON)), 1)
    beta = min(max(float(getattr(env.cfg, "motion_context_trend_beta", DEFAULT_TREND_BETA)), 0.0), 0.9999)
    current = torch.stack([torch.cat(left_raw, dim=-1), torch.cat(right_raw, dim=-1)], dim=1).clamp_min(0.0)
    expected_history_shape = (horizon + 1, current.shape[0], 2, 3)
    history = getattr(env, "_motion_context_trend_history", None)
    if not isinstance(history, torch.Tensor) or tuple(history.shape) != expected_history_shape:
        env._motion_context_trend_ema = torch.zeros_like(current)
        env._motion_context_trend_valid = torch.zeros(current.shape[:2], device=current.device, dtype=torch.bool)
        env._motion_context_trend_age = torch.zeros(current.shape[:2], device=current.device, dtype=torch.long)
        env._motion_context_trend_history = torch.zeros(
            expected_history_shape, device=current.device, dtype=current.dtype
        )
        env._motion_context_trend_history_index = 0

    valid = env._motion_context_trend_valid
    updated_ema = torch.where(
        valid.unsqueeze(-1), beta * env._motion_context_trend_ema + (1.0 - beta) * current, current
    )
    index = int(env._motion_context_trend_history_index) % (horizon + 1)
    lag_index = (index - horizon) % (horizon + 1)
    lagged_ema = env._motion_context_trend_history[lag_index].clone()
    enough_history = valid & (env._motion_context_trend_age >= horizon)
    epsilon = torch.tensor(
        [
            max(
                float(getattr(env.cfg, "motion_context_trend_linear_epsilon", DEFAULT_TREND_LINEAR_EPSILON)),
                1.0e-12,
            ),
            max(
                float(getattr(env.cfg, "motion_context_trend_angular_epsilon", DEFAULT_TREND_ANGULAR_EPSILON)),
                1.0e-12,
            ),
            max(
                float(getattr(env.cfg, "motion_context_trend_action_epsilon", DEFAULT_TREND_ACTION_EPSILON)),
                1.0e-12,
            ),
        ],
        device=current.device,
        dtype=current.dtype,
    )
    trend = relative_motion_trend(updated_ema, lagged_ema, epsilon)
    trend = torch.where(enough_history.unsqueeze(-1), trend, torch.zeros_like(trend))

    env._motion_context_trend_history[index].copy_(updated_ema.detach())
    env._motion_context_trend_history_index = (index + 1) % (horizon + 1)
    env._motion_context_trend_ema.copy_(updated_ema.detach())
    env._motion_context_trend_valid.fill_(True)
    env._motion_context_trend_age.add_(1).clamp_(max=horizon)
    env._motion_context_trend_lagged_ema = lagged_ema.detach()
    return {"left_arm": trend[:, 0], "right_arm": trend[:, 1]}


def reset_motion_trend_state(env, env_ids: torch.Tensor) -> None:
    """Clear temporal context state for selected vectorized environments."""

    if hasattr(env, "_motion_context_trend_ema"):
        env._motion_context_trend_ema[env_ids] = 0.0
        env._motion_context_trend_valid[env_ids] = False
        env._motion_context_trend_age[env_ids] = 0
        env._motion_context_trend_history[:, env_ids] = 0.0


def motion_context_from_raw(
    linear_speed: torch.Tensor,
    angular_speed: torch.Tensor,
    action_variation_ema: torch.Tensor,
    running_scale: torch.Tensor,
    norm_max: float = 1.5,
) -> torch.Tensor:
    """Build [linear activity, angular activity, motion smoothness]."""

    scale = running_scale.clamp_min(1.0e-6)
    maximum = max(float(norm_max), 1.0e-6)
    linear_normalized = torch.clamp(linear_speed / scale[0], 0.0, maximum)
    angular_normalized = torch.clamp(angular_speed / scale[1], 0.0, maximum)
    action_normalized = torch.clamp(action_variation_ema / scale[2], 0.0, maximum)
    context = torch.cat(
        [
            linear_normalized,
            angular_normalized,
            torch.exp(-action_normalized),
        ],
        dim=-1,
    )
    context = torch.nan_to_num(context, nan=0.0, posinf=maximum, neginf=0.0)
    context[..., 0:2].clamp_(0.0, maximum)
    context[..., 2:3].clamp_(0.0, 1.0)
    return context


def _calibration_cfg(env) -> tuple[bool, int, float, int, float]:
    """Return the shared task-agnostic calibration configuration."""

    cfg = env.cfg

    def value(name: str, default):
        return getattr(env, name, getattr(cfg, name, default))

    return (
        bool(value("motion_context_calibration_enabled", DEFAULT_CALIBRATION_ENABLED)),
        max(int(value("motion_context_calibration_window", DEFAULT_CALIBRATION_WINDOW)), 2),
        max(
            float(value("motion_context_calibration_cv_threshold", DEFAULT_CALIBRATION_CV_THRESHOLD)),
            0.0,
        ),
        max(
            int(value("motion_context_calibration_confirmations", DEFAULT_CALIBRATION_CONFIRMATIONS)),
            1,
        ),
        max(float(value("motion_context_calibration_epsilon", DEFAULT_CALIBRATION_EPSILON)), 1.0e-12),
    )


def ensure_motion_context_calibration_state(env) -> None:
    """Initialize global running-scale and calibration state without fake history samples."""

    enabled, window, _, _, _ = _calibration_cfg(env)
    if not hasattr(env, "_motion_context_running_scale"):
        env._motion_context_running_scale = torch.tensor(
            [
                float(getattr(env.cfg, "motion_context_lin_scale_init", 0.10)),
                float(getattr(env.cfg, "motion_context_ang_scale_init", 0.50)),
                float(getattr(env.cfg, "motion_context_action_scale_init", 0.50)),
            ],
            device=env.device,
            dtype=torch.float32,
        ).clamp_min(1.0e-6)
    if not hasattr(env, "_motion_context_scale_update_step"):
        env._motion_context_scale_update_step = -1
    if not hasattr(env, "_motion_context_scale_frozen"):
        env._motion_context_scale_frozen = False
    if not hasattr(env, "_motion_context_calibrated"):
        env._motion_context_calibrated = not enabled
    history = getattr(env, "_motion_context_calibration_history", None)
    if not isinstance(history, torch.Tensor) or tuple(history.shape) != (window, 3):
        env._motion_context_calibration_history = torch.zeros((window, 3), device=env.device, dtype=torch.float32)
        env._motion_context_calibration_history_index = 0
        env._motion_context_calibration_history_count = 0
    if not hasattr(env, "_motion_context_calibration_history_index"):
        env._motion_context_calibration_history_index = 0
    if not hasattr(env, "_motion_context_calibration_history_count"):
        env._motion_context_calibration_history_count = 0
    if not hasattr(env, "_motion_context_calibration_stable_count"):
        env._motion_context_calibration_stable_count = 0
    if not hasattr(env, "_motion_context_calibration_cv"):
        env._motion_context_calibration_cv = torch.zeros(3, device=env.device, dtype=torch.float32)
    if not hasattr(env, "_motion_context_calibration_event_emitted"):
        env._motion_context_calibration_event_emitted = bool(env._motion_context_calibrated)


def update_motion_context_scale_and_calibration(
    env,
    left_raw: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    right_raw: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    current_step: int,
) -> torch.Tensor:
    """Update P90-EMA scales throughout training; calibration gates sharing only.

    The signed motion descriptor is independent of this state. Only context
    visibility uses calibration, and the criterion never reads reward, task
    progress, success, or manipulation-phase signals.
    """

    ensure_motion_context_calibration_state(env)
    enabled, window, threshold, confirmations, epsilon = _calibration_cfg(env)
    update_enabled = bool(
        getattr(env, "motion_context_update_scale", getattr(env.cfg, "motion_context_update_scale", True))
    )
    training = bool(getattr(env, "training", True))
    should_update = update_enabled and training
    # A checkpoint may contain the legacy post-calibration frozen flag. Runtime
    # training/evaluation mode, not calibration, now determines scale updates.
    env._motion_context_scale_frozen = not should_update
    if should_update and int(env._motion_context_scale_update_step) != int(current_step):
        beta = min(max(float(getattr(env.cfg, "motion_context_scale_beta", 0.99)), 0.0), 0.9999)
        quantile = min(max(float(getattr(env.cfg, "motion_context_scale_percentile", 0.90)), 0.0), 1.0)
        batch_scale = (
            torch.stack(
                [
                    torch.quantile(torch.cat([left_raw[0], right_raw[0]], dim=0).flatten(), quantile),
                    torch.quantile(torch.cat([left_raw[1], right_raw[1]], dim=0).flatten(), quantile),
                    torch.quantile(torch.cat([left_raw[2], right_raw[2]], dim=0).flatten(), quantile),
                ]
            )
            .detach()
            .clamp_min(1.0e-6)
        )
        env._motion_context_running_scale.mul_(beta).add_(batch_scale, alpha=1.0 - beta)
        env._motion_context_scale_update_step = int(current_step)

        history_index = int(env._motion_context_calibration_history_index) % window
        env._motion_context_calibration_history[history_index].copy_(env._motion_context_running_scale.detach())
        env._motion_context_calibration_history_index = (history_index + 1) % window
        env._motion_context_calibration_history_count = min(
            int(env._motion_context_calibration_history_count) + 1, window
        )

        if enabled and not bool(env._motion_context_calibrated):
            if int(env._motion_context_calibration_history_count) == window:
                history = env._motion_context_calibration_history
                cv = history.std(dim=0, unbiased=False) / (history.mean(dim=0).abs() + epsilon)
                finite_max = torch.finfo(cv.dtype).max
                cv = torch.nan_to_num(cv, nan=finite_max, posinf=finite_max, neginf=finite_max)
                env._motion_context_calibration_cv.copy_(cv)
                if float(cv.max().detach().cpu()) < threshold:
                    env._motion_context_calibration_stable_count += 1
                else:
                    env._motion_context_calibration_stable_count = 0
                if int(env._motion_context_calibration_stable_count) >= confirmations:
                    env._motion_context_calibrated = True
                    if not bool(env._motion_context_calibration_event_emitted):
                        values = cv.detach().float().cpu().tolist()
                        scales = env._motion_context_running_scale.detach().float().cpu().tolist()
                        print(
                            "[Motion Context Calibration Complete]\n"
                            f"  step={int(current_step)}\n"
                            f"  cv_linear={values[0]:.6f} cv_angular={values[1]:.6f} "
                            f"cv_action_variation={values[2]:.6f} cv_max={max(values):.6f}\n"
                            f"  stable_count={int(env._motion_context_calibration_stable_count)}\n"
                            f"  current_scales={scales} (continuing adaptation during training)"
                        )
                        env._motion_context_calibration_event_emitted = True
            else:
                env._motion_context_calibration_stable_count = 0
        elif not enabled:
            env._motion_context_calibrated = True

    return env._motion_context_running_scale.clamp_min(1.0e-6)


def motion_context_sharing_active(env) -> bool:
    """Return whether the normalized Context 3D may enter a partner payload."""

    if str(getattr(env.cfg, "motion_context_representation", "absolute")) in REPRESENTATION_LAYOUTS:
        env._motion_context_calibrated = True
        env._motion_context_scale_frozen = True
        return True
    ensure_motion_context_calibration_state(env)
    enabled, _, _, _, _ = _calibration_cfg(env)
    return (not enabled) or bool(env._motion_context_calibrated)


def motion_context_calibration_diagnostics(env) -> dict[str, float | int | bool]:
    """Return scalar calibration diagnostics for logs, traces, and checkpoints."""

    ensure_motion_context_calibration_state(env)
    enabled, window, threshold, confirmations, epsilon = _calibration_cfg(env)
    cv = env._motion_context_calibration_cv.detach().float().cpu().tolist()
    mode = str(getattr(env, "communication_mode", getattr(env.cfg, "communication_mode", "motion_context")))
    globally_enabled = bool(
        getattr(env, "communication_enabled", mode != "none" and getattr(env, "sharing_mode", "") != "no_share")
    )
    calibrated = motion_context_sharing_active(env)
    return {
        "motion_context_calibration_enabled": enabled,
        "motion_context_calibrated": bool(env._motion_context_calibrated),
        "motion_sharing_active": globally_enabled
        and mode in ("motion_only", "motion_context", "motion_context_la", "motion_context_s"),
        "context_sharing_active": globally_enabled
        and mode in ("context_only", "motion_context", "motion_context_la", "motion_context_s")
        and calibrated,
        "calibration_cv_linear": float(cv[0]),
        "calibration_cv_angular": float(cv[1]),
        "calibration_cv_action_variation": float(cv[2]),
        "calibration_cv_max": float(max(cv)),
        "calibration_stable_count": int(env._motion_context_calibration_stable_count),
        "calibration_history_count": int(env._motion_context_calibration_history_count),
        "calibration_window": window,
        "calibration_window_filled": int(env._motion_context_calibration_history_count) == window,
        "calibration_cv_threshold": threshold,
        "calibration_confirmations": confirmations,
        "calibration_epsilon": epsilon,
        "motion_context_scale_frozen": bool(env._motion_context_scale_frozen),
    }
