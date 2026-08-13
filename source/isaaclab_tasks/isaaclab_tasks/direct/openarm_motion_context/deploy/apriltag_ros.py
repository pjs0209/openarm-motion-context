"""Adapters for common ROS2 AprilTag detection messages."""

from typing import Any


def detection_id(detection: Any) -> int | None:
    """Extract one integer tag ID from common apriltag_ros layouts."""

    tag_id = getattr(detection, "id", None)
    if tag_id is None:
        return None
    if isinstance(tag_id, (list, tuple)):
        return int(tag_id[0]) if tag_id else None
    try:
        if hasattr(tag_id, "__len__"):
            return int(tag_id[0]) if len(tag_id) else None
    except TypeError:
        pass
    return int(tag_id)


def detection_pose(detection: Any):
    """Extract a geometry pose from nested AprilTag detection messages."""

    pose = getattr(detection, "pose", None)
    for _ in range(4):
        if pose is None:
            break
        if hasattr(pose, "position") and hasattr(pose, "orientation"):
            return pose
        pose = getattr(pose, "pose", None)
    return None
