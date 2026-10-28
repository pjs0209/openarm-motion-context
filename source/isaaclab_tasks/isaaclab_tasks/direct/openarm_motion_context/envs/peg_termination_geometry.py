"""Pure geometric guard for peg/hole wall-penetration termination."""

from __future__ import annotations

import math

import torch


def side_wall_penetration(
    insertion_depth: torch.Tensor,
    lateral_error: torch.Tensor,
    tip_dist: torch.Tensor,
    *,
    depth_threshold: float,
    lateral_threshold: float,
    max_tip_distance: float = math.inf,
) -> torch.Tensor:
    """Flag a tip beyond the entrance plane and near its physical mouth.

    Axial depth alone is a projection: a distant tip can lie past the plane
    while remaining far to the side. ``max_tip_distance`` may be infinite to
    preserve historical termination behavior for non-Circle geometries.
    """

    if insertion_depth.shape != lateral_error.shape or insertion_depth.shape != tip_dist.shape:
        raise ValueError("Peg termination geometry tensors must have equal shapes")
    if not (math.isfinite(depth_threshold) and math.isfinite(lateral_threshold)):
        raise ValueError("Peg penetration thresholds must be finite")
    if max_tip_distance <= 0 or math.isnan(max_tip_distance):
        raise ValueError("Maximum tip distance must be positive")
    return (insertion_depth > depth_threshold) & (lateral_error > lateral_threshold) & (tip_dist < max_tip_distance)
