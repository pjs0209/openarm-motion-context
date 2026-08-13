"""Pure safety checks used by the ROS2 deployment node."""

import torch


def message_is_fresh(now: float, stamp: float, max_age: float) -> bool:
    return float(now) - float(stamp) <= float(max_age)


def visible_tag_count(visible: torch.Tensor) -> int:
    return int((visible.sum(dim=0) > 0.0).sum().item())


def target_delta_within_limit(target_delta: torch.Tensor, maximum: float) -> bool:
    return float(torch.linalg.vector_norm(target_delta)) <= float(maximum)
