# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Dashboard contracts at real ROS message and HTTP boundaries; no motor interface is used."""

import copy
import http.client
import json
import runpy
import threading
import time
import uuid
from pathlib import Path

import pytest
import yaml

rclpy = pytest.importorskip("rclpy")
from geometry_msgs.msg import PoseStamped, TransformStamped
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray, String
from tf2_ros import StaticTransformBroadcaster
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

DEPLOY = Path(__file__).parents[1] / "isaaclab_tasks/direct/openarm_motion_context/deploy"
runtime = runpy.run_path(str(DEPLOY / "visualization.py"))


def test_observed_targets_frame_transforms_and_stale_poses(monkeypatch):
    """A displayed dry-run candidate is distinct from a published command; stale/invalid poses disappear."""
    monkeypatch.setenv("ROS_DOMAIN_ID", "175")
    monkeypatch.setenv("ROS_LOCALHOST_ONLY", "1")
    namespace = f"/dashboard_test_{uuid.uuid4().hex}"
    config = yaml.safe_load((DEPLOY / "configs/lift_real.yaml").read_text())
    config["visualization"] = {
        "max_pose_age_s": 0.3,
        "rate_hz": 30,
        "objects": {name: {"topic": f"{namespace}/{name}"} for name in ("box", "ball", "peg", "hole")},
    }
    config["topics"]["joint_states"] = f"{namespace}/joints"
    config["topics"]["commands"] = {group: f"{namespace}/{group}" for group in config["joints"]}
    config["topics"]["apriltag"] = {}
    rclpy.init(
        args=[
            "--ros-args",
            "-r",
            f"/openarm/deploy/policy_target:={namespace}/preview",
            "-r",
            f"/openarm/deploy/control_state:={namespace}/control",
        ]
    )
    state = runtime["SceneState"]()
    scene = runtime["DeploymentScene"](config, state)
    fixture = Node("dashboard_fixture")
    executor = SingleThreadedExecutor()
    executor.add_node(scene)
    executor.add_node(fixture)
    measured = fixture.create_publisher(JointState, config["topics"]["joint_states"], 1)
    command = fixture.create_publisher(Float64MultiArray, config["topics"]["commands"]["left_arm"], 1)
    preview = fixture.create_publisher(JointState, f"{namespace}/preview", 1)
    control = fixture.create_publisher(String, f"{namespace}/control", 1)
    objects = {name: fixture.create_publisher(PoseStamped, f"{namespace}/{name}", 1) for name in ("box", "ball")}
    broadcaster = StaticTransformBroadcaster(fixture)

    def wait_for(predicate, publish=lambda: None, timeout=4.0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            publish()
            executor.spin_once(timeout_sec=0.01)
            if predicate():
                return
        raise AssertionError(state.get())

    def feedback():
        message = JointState()
        message.header.stamp = fixture.get_clock().now().to_msg()
        message.name = [name for names in config["joints"].values() for name in names]
        message.position = [0.0] * len(message.name)
        measured.publish(message)
        return message

    try:
        wait_for(
            lambda: all(
                pub.get_subscription_count() for pub in [measured, command, preview, control, *objects.values()]
            )
        )
        wait_for(lambda: state.get().get("robot", {}).get("live", False), feedback)
        assert not state.get()["robot"]["target"]
        command.publish(Float64MultiArray(data=[0.3] + [0.0] * 6))
        wait_for(lambda: state.get()["robot"]["target_source"] == "command", feedback)
        left_joint = config["joints"]["left_arm"][0]
        assert state.get()["robot"]["target"][left_joint] == pytest.approx(0.3)

        def candidate():
            message = feedback()
            message.position[0] = 0.45
            preview.publish(message)
            control.publish(
                String(data=json.dumps({"state": "dry_run", "stamp": scene.get_clock().now().nanoseconds * 1e-9}))
            )

        wait_for(lambda: state.get()["robot"]["target_source"] == "preview", candidate)
        assert state.get()["robot"]["target"][left_joint] == pytest.approx(0.45)
        assert state.get()["robot"]["measured"][left_joint] == 0.0
        for topic in config["topics"]["commands"].values():
            assert all(info.node_name != scene.get_name() for info in fixture.get_publishers_info_by_topic(topic))

        transform = TransformStamped()
        transform.header.stamp = fixture.get_clock().now().to_msg()
        transform.header.frame_id, transform.child_frame_id = config["base_frame"], f"test_camera_{uuid.uuid4().hex}"
        transform.transform.translation.x = 1.0
        transform.transform.rotation.z = transform.transform.rotation.w = 2**-0.5
        broadcaster.sendTransform(transform)
        pose = PoseStamped()
        pose.header.frame_id = transform.child_frame_id
        pose.pose.position.x, pose.pose.position.z = 0.2, 0.3
        pose.pose.orientation.w = 1.0

        def observation():
            feedback()
            pose.header.stamp = fixture.get_clock().now().to_msg()
            objects["ball"].publish(pose)

        wait_for(lambda: state.get()["objects"]["ball"]["state"] == "LIVE", observation)
        assert state.get()["objects"]["ball"]["pose"]["position"] == pytest.approx([1.0, 0.2, 0.3])
        wait_for(lambda: state.get()["objects"]["ball"]["state"] == "STALE", feedback)
        assert state.get()["objects"]["ball"]["pose"] is None
        invalid = copy.deepcopy(pose)
        invalid.pose.position.x = float("nan")
        objects["ball"].publish(invalid)
        wait_for(lambda: state.get()["objects"]["ball"]["state"] == "INVALID", feedback)
        assert state.get()["objects"]["ball"]["pose"] is None
        future = copy.deepcopy(pose)
        future.header.frame_id = config["base_frame"]
        future.header.stamp = fixture.get_clock().now().to_msg()
        future.header.stamp.sec += 10
        objects["box"].publish(future)
        wait_for(lambda: state.get()["objects"]["box"]["state"] == "INVALID TIME", feedback)
        assert state.get()["objects"]["box"]["pose"] is None
    finally:
        executor.shutdown()
        scene.destroy_node()
        fixture.destroy_node()
        rclpy.shutdown()


def test_http_demo_identity_and_scoped_asset_routes(tmp_path):
    """Demo is explicit in the public response, and HTTP cannot read outside the asset roots."""
    config = yaml.safe_load((DEPLOY / "configs/lift_real.yaml").read_text())
    state = runtime["SceneState"]()
    state.set(runtime["demo_snapshot"](config, 0.0))
    frontend, assets = tmp_path / "frontend", tmp_path / "assets"
    (frontend / "public").mkdir(parents=True)
    assets.mkdir()
    (frontend / "public/index.html").write_text("<title>Dashboard</title>")
    (assets / "robot.dae").write_text("<COLLADA/>")
    (tmp_path / "outside.txt").write_text("not an asset")
    server = runtime["create_server"]("127.0.0.1", 0, state, frontend, "<robot/>", assets)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=2)
    try:
        connection.request("GET", "/api/state")
        response = connection.getresponse()
        assert response.status == 200
        snapshot = json.loads(response.read())
        assert snapshot["source"] == "demo"
        assert snapshot["robot"]["target_source"] == "preview"
        assert "synthetic" in snapshot["control"]["detail"].lower()
        for path, expected in [
            ("/api/robot.urdf", 200),
            ("/assets/openarm_description/assets/robot.dae", 200),
            ("/assets/openarm_description/assets/%2e%2e/outside.txt", 404),
            ("/.git/config", 404),
        ]:
            connection.request("GET", path)
            response = connection.getresponse()
            assert response.status == expected
            response.read()
        connection.request("POST", "/api/command", "{}")
        response = connection.getresponse()
        assert response.status == 501
        response.read()
    finally:
        connection.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_bridge_preview_matches_bounded_commands_without_dry_run_actuation(monkeypatch):
    """Observe the bridge's real publishers: previews match clamps and dry-run emits no motor command."""
    import torch
    from builtin_interfaces.msg import Duration

    bridge_runtime = runpy.run_path(str(DEPLOY / "deploy_node.py"))
    telemetry_runtime = runpy.run_path(str(DEPLOY / "telemetry.py"))
    monkeypatch.setenv("ROS_DOMAIN_ID", "175")
    monkeypatch.setenv("ROS_LOCALHOST_ONLY", "1")
    namespace = f"/dashboard_command_test_{uuid.uuid4().hex}"
    rclpy.init(args=["--ros-args", "-r", f"/openarm/deploy/policy_target:={namespace}/preview"])
    node, observer = Node("test_bridge_output"), Node("test_bridge_observer")
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    executor.add_node(observer)
    bridge = object.__new__(bridge_runtime["OpenArmPaperReLiftDeployNode"])
    bridge.node, bridge.telemetry = node, telemetry_runtime["DeploymentTelemetry"](node, "base")
    config = yaml.safe_load((DEPLOY / "configs/lift_real.yaml").read_text())
    bridge.left_arm_joints, bridge.right_arm_joints = config["joints"]["left_arm"], config["joints"]["right_arm"]
    bridge.left_gripper_joints, bridge.right_gripper_joints = (
        config["joints"]["left_gripper"],
        config["joints"]["right_gripper"],
    )
    bridge.arm_command_type, bridge.Float64MultiArray = "forward_position", Float64MultiArray
    bridge.JointTrajectory, bridge.JointTrajectoryPoint, bridge.DurationMsg = (
        JointTrajectory,
        JointTrajectoryPoint,
        Duration,
    )
    bridge.dt, bridge.cfg = 1 / 60, {"action": {"command_time_from_start": 1 / 60}}
    bridge.last_command_time = time.time()
    bridge.hardware_limits = {
        group: (
            torch.full((len(names),), -0.1 if group.endswith("_arm") else 0.0),
            torch.full((len(names),), 0.1 if group.endswith("_arm") else 0.04),
        )
        for group, names in config["joints"].items()
    }
    bridge.command_pubs, received = {}, {}
    for group in config["joints"]:
        kind = Float64MultiArray if group.endswith("_arm") else JointTrajectory
        topic = f"{namespace}/{group}"
        bridge.command_pubs[group] = node.create_publisher(kind, topic, 1)
        observer.create_subscription(kind, topic, lambda message, group=group: received.update({group: message}), 1)
    observer.create_subscription(
        JointState, f"{namespace}/preview", lambda message: received.update(preview=message), 1
    )

    def wait_for(predicate):
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            executor.spin_once(timeout_sec=0.01)
            if predicate():
                return
        raise AssertionError(received)

    try:
        wait_for(
            lambda: all(
                pub.get_subscription_count()
                for pub in [*bridge.command_pubs.values(), bridge.telemetry.target_publisher]
            )
        )
        commands = (torch.full((7,), 0.5), torch.zeros(7), torch.full((1,), 0.2), torch.full((1,), 0.2))
        bridge.dry_run = True
        bridge._publish_command(*commands)
        wait_for(lambda: "preview" in received)
        deadline = time.monotonic() + 0.15
        while time.monotonic() < deadline:
            executor.spin_once(timeout_sec=0.01)
        assert set(received) == {"preview"}
        preview = dict(zip(received["preview"].name, received["preview"].position))
        assert [preview[name] for name in bridge.left_arm_joints] == pytest.approx([0.1] * 7)
        assert [preview[name] for name in bridge.left_gripper_joints] == pytest.approx([0.04])
        bridge.dry_run = False
        bridge._publish_command(*commands)
        wait_for(lambda: set(config["joints"]).issubset(received))
        assert list(received["left_arm"].data) == pytest.approx([preview[name] for name in bridge.left_arm_joints])
        assert list(received["left_gripper"].points[0].positions) == pytest.approx(
            [preview[name] for name in bridge.left_gripper_joints]
        )
    finally:
        executor.shutdown()
        node.destroy_node()
        observer.destroy_node()
        rclpy.shutdown()
