# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Publish policy observations and candidate targets for deployment monitoring."""

import json


class DeploymentTelemetry:
    """Own observation publishers without adding a control input to the bridge."""

    def __init__(self, node, base_frame: str):
        from geometry_msgs.msg import PoseStamped
        from sensor_msgs.msg import JointState
        from std_msgs.msg import String

        self.node, self.base_frame = node, base_frame
        self.PoseStamped, self.JointState, self.String = PoseStamped, JointState, String
        self.pose_publishers = {
            name: node.create_publisher(PoseStamped, f"/openarm/deploy/{path}", 1)
            for name, path in {
                "box": "objects/box",
                "left_grasp": "grasp_targets/left",
                "right_grasp": "grasp_targets/right",
            }.items()
        }
        self.target_publisher = node.create_publisher(JointState, "/openarm/deploy/policy_target", 1)
        self.state_publisher = node.create_publisher(String, "/openarm/deploy/control_state", 1)
        self._last_state = None
        self._last_state_time = float("-inf")
        self._last_pose_time = float("-inf")

    def needs_poses(self) -> bool:
        """Avoid extra pose fusion when the monitor is absent; cap updates at 15 Hz."""
        elapsed = self.node.get_clock().now().nanoseconds * 1e-9 - self._last_pose_time
        return not 0 <= elapsed < 1 / 15 and any(
            publisher.get_subscription_count() for publisher in self.pose_publishers.values()
        )

    def state(self, state: str, detail: str = "") -> None:
        """Publish state changes immediately and refresh the heartbeat at 5 Hz."""
        now = self.node.get_clock().now().nanoseconds * 1e-9
        current = (state, detail)
        if current == self._last_state and 0 <= now - self._last_state_time < 0.2:
            return
        message = self.String()
        message.data = json.dumps({"state": state, "detail": detail, "stamp": now})
        self.state_publisher.publish(message)
        self._last_state, self._last_state_time = current, now

    def poses(self, stamp: float, poses: dict) -> None:
        """Retain the oldest contributing measurement stamp; quaternions arrive as WXYZ."""
        self._last_pose_time = self.node.get_clock().now().nanoseconds * 1e-9
        for name, (position, quaternion) in poses.items():
            message = self.PoseStamped()
            seconds = int(stamp)
            message.header.stamp.sec, message.header.stamp.nanosec = seconds, int((stamp - seconds) * 1e9)
            message.header.frame_id = self.base_frame
            xyz, wxyz = position.detach().cpu().tolist(), quaternion.detach().cpu().tolist()
            message.pose.position.x, message.pose.position.y, message.pose.position.z = xyz
            message.pose.orientation.w, message.pose.orientation.x = wxyz[:2]
            message.pose.orientation.y, message.pose.orientation.z = wxyz[2:]
            self.pose_publishers[name].publish(message)

    def target(self, commands: dict) -> None:
        """Publish the complete bounded candidate independently of motor command topics."""
        message = self.JointState()
        message.header.stamp = self.node.get_clock().now().to_msg()
        for names, positions in commands.values():
            message.name.extend(names)
            message.position.extend(positions.detach().cpu().tolist())
        self.target_publisher.publish(message)
