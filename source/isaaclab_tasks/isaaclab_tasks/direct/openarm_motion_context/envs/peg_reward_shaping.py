"""Pure, versioned reward shaping for physical peg-in-hole alignment."""

from __future__ import annotations

import math

import torch


def axis_alignment_quality(axis_alignment: torch.Tensor, power: float = 1.0) -> torch.Tensor:
    """Map axis dot product to [0, 1] without changing the v7 result at power 1.

    A larger power reduces the reward for a tilted but laterally centered peg.
    This changes only the dense training reward, never the success predicate.
    """

    if not math.isfinite(power) or power <= 0.0:
        raise ValueError("Axis alignment shaping power must be positive and finite")
    return torch.clamp(0.5 * (axis_alignment + 1.0), 0.0, 1.0).pow(power)


def target_depth_quality(
    insertion_depth: torch.Tensor, hole_depth: torch.Tensor, overrun_std: float = 0.0
) -> torch.Tensor:
    """Reward reaching the hole bottom without rewarding unlimited over-insertion.

    A zero width exactly preserves the original saturating depth reward. A
    positive width smoothly discounts motion beyond the physical hole
    depth; the success predicate remains independent of this reward.
    """

    if not math.isfinite(overrun_std) or overrun_std < 0.0:
        raise ValueError("Overrun width must be nonnegative and finite")
    progress = torch.clamp(insertion_depth / hole_depth, 0.0, 1.0)
    if overrun_std == 0.0:
        return progress
    overrun = torch.clamp(insertion_depth - hole_depth, min=0.0)
    return progress / (1.0 + overrun / overrun_std)


def insertion_settle_quality(
    axis_alignment: torch.Tensor,
    depth_quality: torch.Tensor,
    ee_speed: torch.Tensor,
    axis_threshold: float,
    axis_width: float,
    speed_std: float,
) -> torch.Tensor:
    """Dense alignment/depth/low-speed quality for the insertion stage."""

    if not all(math.isfinite(x) and x > 0.0 for x in (axis_width, speed_std)):
        raise ValueError("Settle axis width and speed scale must be positive and finite")
    if not math.isfinite(axis_threshold):
        raise ValueError("Settle axis threshold must be finite")
    axis_score = torch.sigmoid((axis_alignment - axis_threshold) / axis_width)
    speed_score = 1.0 / (1.0 + torch.clamp(ee_speed, min=0.0) / speed_std)
    return axis_score * torch.clamp(depth_quality, 0.0, 1.0) * speed_score
