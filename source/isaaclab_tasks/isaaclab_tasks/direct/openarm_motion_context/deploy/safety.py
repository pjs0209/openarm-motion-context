"""Pure safety checks used by the ROS2 deployment node."""

import math
import xml.etree.ElementTree as ET

import torch


def joint_position_limits(urdf: str, names: list[str]) -> tuple[list[float], list[float]]:
    """Read finite position limits in the requested controller order from the live URDF."""
    try:
        joints = {joint.get("name"): joint for joint in ET.fromstring(urdf).findall("joint")}
    except ET.ParseError as exc:
        raise ValueError("Malformed robot_description XML") from exc
    lower, upper = [], []
    for name in names:
        joint = joints.get(name)
        limit = joint.find("limit") if joint is not None else None
        if limit is None or "lower" not in limit.attrib or "upper" not in limit.attrib:
            raise ValueError(f"Missing bounded joint limits for {name}")
        lo, hi = float(limit.get("lower")), float(limit.get("upper"))
        if not math.isfinite(lo) or not math.isfinite(hi) or lo >= hi:
            raise ValueError(f"Invalid joint limits for {name}")
        lower.append(lo)
        upper.append(hi)
    return lower, upper


def message_is_fresh(now: float, stamp: float, max_age: float) -> bool:
    return 0.0 <= float(now) - float(stamp) <= float(max_age)


def visible_tag_count(visible: torch.Tensor) -> int:
    return int((visible.sum(dim=0) > 0.0).sum().item())


def target_delta_within_limit(target_delta: torch.Tensor, maximum: float) -> bool:
    return float(torch.linalg.vector_norm(target_delta)) <= float(maximum)
