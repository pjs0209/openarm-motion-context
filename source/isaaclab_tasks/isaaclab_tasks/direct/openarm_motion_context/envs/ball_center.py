"""Reward-only ball metrics. Never appended to actor or critic observations."""

import math

import torch


def ball_center_metrics(local_position, center_xy, half_extents_xy, height_range, std, center_threshold):
    """Score horizontal centering in the box frame; exclude balls outside it.

    A containment proxy checks ball-center coordinates, not physical contact.
    The caller uses simulator pose only for reward/evaluation, not observation.
    """
    if local_position.ndim != 2 or local_position.shape[1] != 3:
        raise ValueError("Ball positions must be [environments,3]")
    spec = (*center_xy, *half_extents_xy, *height_range, std, center_threshold)
    if not all(math.isfinite(float(v)) for v in spec) or std <= 0 or center_threshold <= 0:
        raise ValueError("Ball metric specification must be finite with positive scales")
    if (
        len(center_xy) != 2
        or len(half_extents_xy) != 2
        or min(half_extents_xy) <= 0
        or height_range[0] >= height_range[1]
    ):
        raise ValueError("Invalid ball containment geometry")
    delta = local_position[:, :2] - local_position.new_tensor(center_xy)
    finite = torch.isfinite(local_position).all(dim=1)
    error = torch.linalg.vector_norm(delta, dim=1)
    inside = (delta.abs() <= local_position.new_tensor(half_extents_xy)).all(dim=1)
    inside &= (local_position[:, 2] >= height_range[0]) & (local_position[:, 2] <= height_range[1]) & finite
    score = torch.where(inside, torch.exp(-(error / std).square()), torch.zeros_like(error))
    return {
        "error": torch.nan_to_num(error, nan=1e6, posinf=1e6, neginf=1e6),
        "inside": inside,
        "center_ok": inside & (error <= center_threshold),
        "score": score,
    }


def ball_offcenter_penalty(error, inside, threshold, full_error, scale):
    """Zero inside the central dead zone; bounded negative credit outside it.

    Error is horizontal distance in the box frame. Leaving the containment
    proxy (including vertical escape) receives the full penalty.
    """
    if not all(math.isfinite(float(v)) for v in (threshold, full_error, scale)):
        raise ValueError("Penalty parameters must be finite")
    if threshold < 0 or full_error <= threshold or scale < 0:
        raise ValueError("Invalid off-center penalty scale or geometry")
    if error.shape != inside.shape:
        raise ValueError("Off-center error/inside shapes must match")
    safe_error = torch.nan_to_num(error, nan=full_error, posinf=full_error, neginf=full_error)
    severity = ((safe_error - threshold) / (full_error - threshold)).clamp(0, 1)
    valid = torch.isfinite(error) & (error >= 0) & inside.bool()
    severity = torch.where(valid, severity, torch.ones_like(severity))
    return -scale * severity
