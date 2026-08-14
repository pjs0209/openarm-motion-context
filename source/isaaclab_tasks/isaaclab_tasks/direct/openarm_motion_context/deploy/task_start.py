"""Move a real OpenArm from its bringup pose to the task start pose.

OpenArm v1.0's hardware plugin moves every arm joint to zero during activation.
This utility waits for fresh joint states and then streams a smooth position
trajectory to the same start pose used by simulation and deployment.
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path
from typing import Any

import yaml


def cosine_blend(start: list[float], target: list[float], progress: float) -> list[float]:
    """Interpolate joint positions with zero velocity at both ends."""

    if len(start) != len(target):
        raise ValueError(f"Start/target dimensions differ: {len(start)} != {len(target)}")
    progress = min(max(float(progress), 0.0), 1.0)
    weight = 0.5 - 0.5 * math.cos(math.pi * progress)
    return [a + weight * (b - a) for a, b in zip(start, target, strict=True)]


def positions_by_name(names: list[str], positions: list[float], required: list[str]) -> list[float]:
    """Return positions ordered by ``required``, rejecting incomplete states."""

    state = dict(zip(names, positions, strict=False))
    missing = [name for name in required if name not in state]
    if missing:
        raise ValueError(f"JointState is missing joints: {missing}")
    return [float(state[name]) for name in required]


def _load_config(path: str) -> dict[str, Any]:
    with Path(path).expanduser().open("r", encoding="utf-8") as stream:
        cfg = yaml.safe_load(stream)
    if not isinstance(cfg, dict):
        raise ValueError(f"Expected a mapping in {path}")
    return cfg


def _trajectory_message(joint_names, positions, seconds_from_start):
    from builtin_interfaces.msg import Duration
    from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

    message = JointTrajectory()
    message.joint_names = list(joint_names)
    point = JointTrajectoryPoint()
    point.positions = [float(value) for value in positions]
    point.time_from_start = Duration(
        sec=int(seconds_from_start),
        nanosec=int((seconds_from_start % 1.0) * 1.0e9),
    )
    message.points = [point]
    return message


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Real-robot deployment YAML")
    parser.add_argument("--duration", type=float, default=8.0, help="Motion duration in seconds")
    parser.add_argument("--rate", type=float, default=30.0, help="Stream rate in Hz")
    parser.add_argument("--state-timeout", type=float, default=5.0, help="JointState wait timeout")
    parser.add_argument("--command-lead", type=float, default=0.10, help="Controller command lead time")
    parser.add_argument("--goal-tolerance", type=float, default=0.05, help="Final joint tolerance in radians")
    parser.add_argument("--gripper-tolerance", type=float, default=0.005, help="Final gripper tolerance in meters")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually command the robot. Without this flag only validation is performed.",
    )
    args = parser.parse_args()

    if args.duration <= 0.0 or args.rate <= 0.0 or args.command_lead <= 0.0:
        parser.error("duration, rate, and command-lead must be positive")

    import rclpy
    from sensor_msgs.msg import JointState
    from trajectory_msgs.msg import JointTrajectory

    cfg = _load_config(args.config)
    joints = cfg["joints"]
    targets = cfg["default_joint_pos"]
    topics = cfg["topics"]
    command_topics = topics["commands"]

    group_names = {
        "left_arm": list(joints["left_arm"]),
        "right_arm": list(joints["right_arm"]),
        "left_gripper": list(joints["left_gripper"]),
        "right_gripper": list(joints["right_gripper"]),
    }
    group_targets = {
        "left_arm": [float(value) for value in targets["left_arm"]],
        "right_arm": [float(value) for value in targets["right_arm"]],
        "left_gripper": [float(cfg["action"]["gripper_open_target"])],
        "right_gripper": [float(cfg["action"]["gripper_open_target"])],
    }
    for side in ("left_arm", "right_arm"):
        if len(group_names[side]) != 7 or len(group_targets[side]) != 7:
            raise ValueError(f"{side} must contain exactly seven joints and target values")
    for side in ("left_gripper", "right_gripper"):
        if len(group_names[side]) != 1:
            raise ValueError(f"{side} must contain exactly one joint")

    rclpy.init()
    node = rclpy.create_node("openarm_move_to_task_start")
    latest_state = None
    latest_state_time = 0.0

    def on_joint_state(message):
        nonlocal latest_state, latest_state_time
        latest_state = message
        latest_state_time = time.monotonic()

    node.create_subscription(JointState, topics["joint_states"], on_joint_state, 10)
    publishers = {
        side: node.create_publisher(JointTrajectory, command_topics[side], 10) for side in group_names
    }

    deadline = time.monotonic() + args.state_timeout
    while rclpy.ok() and latest_state is None and time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.1)
    if latest_state is None:
        node.destroy_node()
        rclpy.shutdown()
        raise RuntimeError(f"No JointState received from {topics['joint_states']} within {args.state_timeout}s")

    starts = {
        side: positions_by_name(list(latest_state.name), list(latest_state.position), names)
        for side, names in group_names.items()
    }
    max_moves = {
        side: max(abs(target - start) for start, target in zip(starts[side], group_targets[side], strict=True))
        for side in group_names
    }
    node.get_logger().info(
        "Task start target validated. "
        f"max joint move: left={max_moves['left_arm']:.3f} rad, right={max_moves['right_arm']:.3f} rad"
    )

    if not args.execute:
        node.get_logger().warning("Validation only: no command published. Re-run with --execute to move.")
        node.destroy_node()
        rclpy.shutdown()
        return 0

    node.get_logger().warning(
        f"Moving both arms to the task start pose over {args.duration:.1f}s. Keep the emergency stop ready."
    )
    # Send one complete trajectory instead of repeatedly replacing the
    # controller goal with short single-point trajectories.
    #
    # First hold the measured pose briefly, then follow a cosine trajectory
    # whose velocity is zero at both ends.
    hold_seconds = 1.0
    steps = max(2, int(math.ceil(args.duration * args.rate)))

    for side in group_names:
        trajectory = JointTrajectory()
        trajectory.joint_names = list(group_names[side])

        # Point 0: current measured pose.
        trajectory.points.append(
            _trajectory_message(
                group_names[side],
                starts[side],
                args.command_lead,
            ).points[0]
        )

        # Point 1: hold the current pose before motion begins.
        trajectory.points.append(
            _trajectory_message(
                group_names[side],
                starts[side],
                args.command_lead + hold_seconds,
            ).points[0]
        )

        # Complete smooth trajectory.
        for step in range(1, steps + 1):
            progress = step / steps
            command = cosine_blend(
                starts[side],
                group_targets[side],
                progress,
            )
            point_time = (
                args.command_lead
                + hold_seconds
                + args.duration * progress
            )
            trajectory.points.append(
                _trajectory_message(
                    group_names[side],
                    command,
                    point_time,
                ).points[0]
            )

        publishers[side].publish(trajectory)

    node.get_logger().info(
        f"Published complete task-start trajectory: "
        f"hold={hold_seconds:.1f}s, motion={args.duration:.1f}s, "
        f"points={steps + 2}"
    )

    motion_deadline = (
        time.monotonic()
        + args.command_lead
        + hold_seconds
        + args.duration
        + 0.5
    )

    while rclpy.ok() and time.monotonic() < motion_deadline:
        rclpy.spin_once(node, timeout_sec=0.05)

        if time.monotonic() - latest_state_time > 0.5:
            raise RuntimeError(
                "JointState stream became stale during task-start motion"
            )

    settle_deadline = time.monotonic() + 1.0
    while rclpy.ok() and time.monotonic() < settle_deadline:
        rclpy.spin_once(node, timeout_sec=0.05)

    final_errors = {}
    if latest_state is not None:
        for side, names in group_names.items():
            actual = positions_by_name(list(latest_state.name), list(latest_state.position), names)
            final_errors[side] = max(
                abs(target - value) for target, value in zip(group_targets[side], actual, strict=True)
            )

    node.destroy_node()
    rclpy.shutdown()
    arm_error = max((final_errors.get("left_arm", math.inf), final_errors.get("right_arm", math.inf)))
    gripper_error = max(
        (final_errors.get("left_gripper", math.inf), final_errors.get("right_gripper", math.inf))
    )
    if arm_error > args.goal_tolerance or gripper_error > args.gripper_tolerance:
        raise RuntimeError(
            "Task start pose was not reached within tolerance: "
            f"errors={final_errors}, arm_tolerance={args.goal_tolerance:.3f} rad, "
            f"gripper_tolerance={args.gripper_tolerance:.3f} m"
        )
    print(
        "[OK] Task start pose reached: "
        f"left error={final_errors['left_arm']:.4f} rad, "
        f"right error={final_errors['right_arm']:.4f} rad, "
        f"grippers={final_errors['left_gripper']:.4f}/{final_errors['right_gripper']:.4f} m"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
