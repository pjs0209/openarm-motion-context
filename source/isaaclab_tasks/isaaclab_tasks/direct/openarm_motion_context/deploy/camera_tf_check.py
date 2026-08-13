#!/usr/bin/env python3
"""Compare live ROS camera TF extrinsics against transforms exported from USD."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
import yaml


def _quat_wxyz_from_ros(rotation) -> torch.Tensor:
    quat = torch.tensor([rotation.w, rotation.x, rotation.y, rotation.z], dtype=torch.float64)
    return quat / torch.linalg.vector_norm(quat).clamp_min(1.0e-12)


def _rotation_error_deg(reference: torch.Tensor, measured: torch.Tensor) -> float:
    dot = torch.abs(torch.dot(reference, measured)).clamp(0.0, 1.0)
    return math.degrees(2.0 * math.acos(float(dot)))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Calibrated deploy YAML")
    parser.add_argument("--position-tolerance-mm", type=float, default=5.0)
    parser.add_argument("--rotation-tolerance-deg", type=float, default=2.0)
    parser.add_argument("--timeout", type=float, default=3.0)
    args = parser.parse_args()

    import rclpy
    from rclpy.duration import Duration
    from rclpy.time import Time
    from tf2_ros import Buffer, TransformListener

    with Path(args.config).open("r", encoding="utf-8") as stream:
        cfg = yaml.safe_load(stream) or {}
    references = cfg.get("camera_extrinsics_usd", {})
    cameras = cfg.get("frames", {}).get("cameras", {})
    mounts = cfg.get("frames", {}).get("camera_mounts", {})
    missing = sorted(set(cameras) - set(references))
    if missing:
        raise RuntimeError(f"USD camera references are missing for: {missing}. Run deploy/calibration.py first.")

    rclpy.init()
    node = rclpy.create_node("openarm_camera_tf_check")
    buffer = Buffer()
    listener = TransformListener(buffer, node)
    deadline = node.get_clock().now() + Duration(seconds=args.timeout)
    while node.get_clock().now() < deadline:
        rclpy.spin_once(node, timeout_sec=0.1)

    failed = False
    base_frame = str(cfg.get("base_frame", "openarm_body_link0"))
    required_frames = {
        "left_ee": str(cfg.get("frames", {}).get("left_ee", "")),
        "right_ee": str(cfg.get("frames", {}).get("right_ee", "")),
    }
    print(f"OpenArm base frame: {base_frame}")
    for label, child_frame in required_frames.items():
        if not child_frame:
            failed = True
            print(f"{label}: frame is not configured")
            continue
        try:
            buffer.lookup_transform(base_frame, child_frame, Time(), timeout=Duration(seconds=0.5))
            print(f"{label}: {base_frame} -> {child_frame}: PASS")
        except Exception as exc:
            failed = True
            print(f"{label}: {base_frame} -> {child_frame}: TF ERROR: {exc}")

    print("camera       mount -> optical                                      pos_err(mm)  rot_err(deg)  result")
    for name, optical_frame in cameras.items():
        mount_frame = str(mounts[name])
        try:
            tf = buffer.lookup_transform(mount_frame, str(optical_frame), Time(), timeout=Duration(seconds=0.5))
        except Exception as exc:
            failed = True
            print(f"{name:<12} {mount_frame} -> {optical_frame}: TF ERROR: {exc}")
            continue

        ref = references[name]
        ref_pos = torch.tensor(ref["pos_mount_camera"], dtype=torch.float64)
        ref_quat = torch.tensor(ref["quat_mount_camera"], dtype=torch.float64)
        ref_quat = ref_quat / torch.linalg.vector_norm(ref_quat).clamp_min(1.0e-12)
        translation = tf.transform.translation
        measured_pos = torch.tensor([translation.x, translation.y, translation.z], dtype=torch.float64)
        measured_quat = _quat_wxyz_from_ros(tf.transform.rotation)
        pos_error_mm = float(torch.linalg.vector_norm(measured_pos - ref_pos)) * 1000.0
        rot_error_deg = _rotation_error_deg(ref_quat, measured_quat)
        passed = pos_error_mm <= args.position_tolerance_mm and rot_error_deg <= args.rotation_tolerance_deg
        failed |= not passed
        print(
            f"{name:<12} {mount_frame} -> {str(optical_frame):<34} "
            f"{pos_error_mm:11.3f}  {rot_error_deg:12.3f}  {'PASS' if passed else 'FAIL'}"
        )
        print(f"  USD pos={ref_pos.tolist()} quat(wxyz)={ref_quat.tolist()}")
        print(f"  TF  pos={measured_pos.tolist()} quat(wxyz)={measured_quat.tolist()}")

    node.destroy_node()
    rclpy.shutdown()
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
