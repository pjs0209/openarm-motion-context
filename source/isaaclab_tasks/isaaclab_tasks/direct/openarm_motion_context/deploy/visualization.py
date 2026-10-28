# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Serve a local 3D deployment dashboard from observed ROS messages, without motor inputs."""

import argparse
import contextlib
import copy
import json
import math
import mimetypes
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlsplit

import rclpy
import tf2_geometry_msgs  # noqa: F401  # Register PoseStamped transformations with tf2.
import xacro
import yaml
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray, String
from tf2_ros import Buffer, TransformException, TransformListener
from trajectory_msgs.msg import JointTrajectory

# Schematic display dimensions, in metres; configure these to match measured hardware.
OBJECTS = {
    "box": {"shape": "box", "size": [0.412, 0.3, 0.25], "color": "#e7ae59"},
    "ball": {"shape": "sphere", "size": [0.03, 0.03, 0.03], "color": "#f2737d"},
    "peg": {"shape": "cylinder", "size": [0.056, 0.056, 0.07], "color": "#6cafff"},
    "hole": {"shape": "ring", "size": [0.07, 0.07, 0.025], "color": "#6de0bb"},
}


def _seconds(stamp) -> float:
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


def _object_styles(config):
    styles = copy.deepcopy(OBJECTS)
    for name, style in styles.items():
        style.update(config.get("visualization", {}).get("objects", {}).get(name, {}))
        if len(style["size"]) != 3 or not all(math.isfinite(value) and value > 0 for value in style["size"]):
            raise ValueError(f"Display size for {name} must contain three positive finite metre values")
    return styles


class SceneState:
    """Exchange immutable JSON snapshots between the ROS executor and HTTP threads."""

    def __init__(self):
        self._lock = threading.Lock()
        self._snapshot = {"source": "ros", "ready": False}
        self._updated = time.monotonic()

    def set(self, snapshot):
        with self._lock:
            self._snapshot = snapshot
            self._updated = time.monotonic()

    def get(self):
        with self._lock:
            return {**self._snapshot, "snapshot_age_s": time.monotonic() - self._updated}


class DeploymentScene(Node):
    """Observe robot, target and object topics; publish no motor commands or joint states."""

    def __init__(self, config: dict, state: SceneState):
        super().__init__("openarm_deployment_dashboard")
        self.config, self.state = config, state
        view = config.get("visualization", {})
        self.base_frame = config.get("base_frame", "openarm_body_link0")
        self.max_age = float(view.get("max_pose_age_s", 0.5))
        self.rate_hz = float(view.get("rate_hz", 15.0))
        if not math.isfinite(self.max_age) or self.max_age <= 0 or not 1 <= self.rate_hz <= 60:
            raise ValueError("Dashboard needs positive max_pose_age_s and rate_hz between 1 and 60")
        self.styles = _object_styles(config)
        self.joints = config["joints"]
        self.required_joints = {name for names in self.joints.values() for name in names}
        self.measured, self.preview, self.control_state = None, None, None
        self.commands, self.command_times, self.poses, self.cameras = {}, {}, {}, {}
        self.invalid_poses = set()
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.create_subscription(JointState, config["topics"]["joint_states"], self._measured, qos_profile_sensor_data)
        self.create_subscription(JointState, "/openarm/deploy/policy_target", self._preview, 1)
        self.create_subscription(String, "/openarm/deploy/control_state", self._control, 1)
        for group, topic in config["topics"]["commands"].items():
            direct = group.endswith("_arm") and config["action"].get("arm_command_type") == "forward_position"
            self.create_subscription(
                Float64MultiArray if direct else JointTrajectory,
                topic,
                lambda message, group=group, direct=direct: self._command(group, message, direct),
                1,
            )
        for name, style in self.styles.items():
            topic = style.get("topic", f"/openarm/deploy/objects/{name}")
            self.create_subscription(PoseStamped, topic, lambda message, name=name: self._pose(name, message), 1)
        for side in ("left", "right"):
            self.create_subscription(
                PoseStamped,
                f"/openarm/deploy/grasp_targets/{side}",
                lambda message, name=f"{side}_grasp": self._pose(name, message),
                1,
            )
        try:
            from apriltag_msgs.msg import AprilTagDetectionArray
        except ImportError:
            self.get_logger().info("AprilTag messages are unavailable; robot-only monitoring is enabled")
        else:
            for camera, topic in config["topics"].get("apriltag", {}).items():
                self.create_subscription(
                    AprilTagDetectionArray,
                    topic,
                    lambda message, camera=camera: self.cameras.update(
                        {camera: (_seconds(message.header.stamp), len(message.detections))}
                    ),
                    qos_profile_sensor_data,
                )
        self.create_timer(1.0 / self.rate_hz, self._update)

    def _fresh(self, stamp, now):
        return 0 <= now - stamp <= self.max_age

    def _valid_joints(self, message):
        return (
            len(message.name) == len(message.position)
            and len(set(message.name)) == len(message.name)
            and self.required_joints.issubset(message.name)
            and all(math.isfinite(value) for value in message.position)
        )

    def _measured(self, message):
        if self._valid_joints(message):
            self.measured = message

    def _preview(self, message):
        if self._valid_joints(message) and set(message.name) == self.required_joints:
            self.preview = message

    def _control(self, message):
        try:
            state = json.loads(message.data)
            if (
                isinstance(state.get("state"), str)
                and isinstance(state.get("stamp"), (float, int))
                and math.isfinite(state["stamp"])
            ):
                self.control_state = state
        except (ValueError, AttributeError):
            pass

    def _command(self, group, message, direct):
        if direct:
            names, positions = self.joints[group], message.data
        elif message.points:
            names, positions = message.joint_names, message.points[-1].positions
        else:
            return
        if (
            len(names) != len(positions)
            or len(set(names)) != len(names)
            or set(names) != set(self.joints[group])
            or not all(math.isfinite(value) for value in positions)
        ):
            return
        self.commands[group] = dict(zip(names, positions))
        self.command_times[group] = self.get_clock().now().nanoseconds * 1e-9

    def _pose(self, name, message):
        position, rotation = message.pose.position, message.pose.orientation
        values = [position.x, position.y, position.z, rotation.x, rotation.y, rotation.z, rotation.w]
        norm = math.sqrt(sum(value * value for value in values[3:]))
        if (
            not message.header.frame_id
            or not all(math.isfinite(value) for value in values)
            or not math.isfinite(norm)
            or norm < 1e-9
        ):
            self.invalid_poses.add(name)
            self.poses.pop(name, None)
            return
        message = copy.deepcopy(message)
        quaternion = message.pose.orientation
        quaternion.x, quaternion.y, quaternion.z, quaternion.w = [value / norm for value in values[3:]]
        self.poses[name] = message
        self.invalid_poses.discard(name)

    def _object(self, name, now):
        pose = self.poses.get(name)
        result = {"state": "INVALID" if name in self.invalid_poses else "NO DATA", "pose": None, "age_s": None}
        if pose is None:
            return result
        stamp = _seconds(pose.header.stamp)
        result["age_s"] = now - stamp
        if not self._fresh(stamp, now):
            result["state"] = "STALE" if now >= stamp else "INVALID TIME"
            return result
        if pose.header.frame_id != self.base_frame:
            try:
                pose = self.tf_buffer.transform(pose, self.base_frame)
            except TransformException:
                result["state"] = "TF MISSING"
                return result
        position, rotation = pose.pose.position, pose.pose.orientation
        result.update(
            state="LIVE",
            pose={
                "position": [position.x, position.y, position.z],
                "quaternion": [rotation.x, rotation.y, rotation.z, rotation.w],
            },
        )
        return result

    def _update(self):
        now = self.get_clock().now().nanoseconds * 1e-9
        measured_live = self.measured is not None and self._fresh(_seconds(self.measured.header.stamp), now)
        state_live = self.control_state is not None and self._fresh(self.control_state["stamp"], now)
        mode = self.control_state["state"] if state_live else "monitor"
        measured = dict(zip(self.measured.name, self.measured.position)) if self.measured else {}
        target, source, target_age = {}, None, None
        if measured_live:
            if mode == "dry_run" and self.preview and self._fresh(_seconds(self.preview.header.stamp), now):
                target = {**measured, **dict(zip(self.preview.name, self.preview.position))}
                source, target_age = "preview", now - _seconds(self.preview.header.stamp)
            elif self.commands:
                target = measured.copy()
                for values in self.commands.values():
                    target.update(values)
                source, target_age = "command", now - max(self.command_times.values())
        errors = {}
        if target:
            for side in ("left", "right"):
                names = self.joints[f"{side}_arm"]
                errors[side] = max(abs(measured[name] - target[name]) for name in names)
        objects = {name: {**style, **self._object(name, now)} for name, style in self.styles.items()}
        cameras = {}
        for name in self.config["topics"].get("apriltag", {}):
            stamp, count = self.cameras.get(name, (None, 0))
            cameras[name] = {
                "live": stamp is not None and self._fresh(stamp, now),
                "tags": count,
                "age_s": now - stamp if stamp is not None else None,
            }
        self.state.set(
            {
                "source": "ros",
                "ready": True,
                "stamp": now,
                "base_frame": self.base_frame,
                "control_rate_hz": self.config.get("control_rate_hz", 60),
                "joint_order": self.joints,
                "robot": {
                    "live": measured_live,
                    "measured": measured,
                    "target": target,
                    "target_source": source,
                    "age_s": now - _seconds(self.measured.header.stamp) if self.measured else None,
                    "target_age_s": target_age,
                    "max_error_rad": errors,
                    "command_age_s": {name: now - stamp for name, stamp in self.command_times.items()},
                },
                "control": {
                    "state": mode,
                    "detail": self.control_state.get("detail", "") if state_live else "Policy heartbeat unavailable",
                },
                "objects": objects,
                "grasps": {side: self._object(f"{side}_grasp", now) for side in ("left", "right")},
                "cameras": cameras,
            }
        )


def demo_snapshot(config, elapsed):
    """Generate explicitly labelled browser-only data; never initialize ROS or publish topics."""
    measured, target = {}, {}
    for group, names in config["joints"].items():
        defaults = config.get("default_joint_pos", {}).get(group, [0.02] * len(names))
        for index, name in enumerate(names):
            baseline = float(defaults[index]) if index < len(defaults) else 0.0
            value = baseline + (0.045 * math.sin(elapsed * 0.45 + index) if group.endswith("_arm") else 0)
            measured[name] = value
            target[name] = value + (0.065 * math.sin(index + elapsed * 0.25) if group.endswith("_arm") else 0)
    positions = {
        "box": [0.42, 0.0, 0.28],
        "ball": [0.43, 0.0, 0.42],
        "peg": [0.32, -0.3, 0.35],
        "hole": [0.4, -0.3, 0.33],
    }
    objects = {
        name: {
            **style,
            "state": "LIVE",
            "age_s": 0.025,
            "pose": {
                "position": positions[name],
                "quaternion": [0.0, 0.0, 0.0, 1.0],
            },
        }
        for name, style in _object_styles(config).items()
    }
    errors = {
        side: max(abs(measured[name] - target[name]) for name in config["joints"][f"{side}_arm"])
        for side in ("left", "right")
    }
    return {
        "source": "demo",
        "ready": True,
        "stamp": time.time(),
        "base_frame": config.get("base_frame", "openarm_body_link0"),
        "control_rate_hz": 60,
        "joint_order": config["joints"],
        "robot": {
            "live": True,
            "measured": measured,
            "target": target,
            "target_source": "preview",
            "age_s": 0.018,
            "target_age_s": 0.025,
            "max_error_rad": errors,
            "command_age_s": {},
        },
        "control": {"state": "dry_run", "detail": "Synthetic preview. No ROS connection or motor commands."},
        "objects": objects,
        "grasps": {},
        "cameras": {name: {"live": True, "tags": 3, "age_s": 0.025} for name in config["topics"].get("apriltag", {})},
    }


def create_server(host, port, state, frontend, model_xml, asset_root, usd_scene=None):
    """Expose only dashboard files, pinned JS modules, and the robot's asset directory."""
    frontend, asset_root = Path(frontend).resolve(), Path(asset_root).resolve()

    scene_bytes = json.dumps(usd_scene, allow_nan=False).encode()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            path = unquote(urlsplit(self.path).path)
            if path == "/api/state":
                self._send(json.dumps(state.get(), allow_nan=False).encode(), "application/json", False)
                return
            if path == "/api/usd-scene":
                self._send(scene_bytes, "application/json", False)
                return
            if path == "/api/robot.urdf":
                self._send(model_xml.encode(), "application/xml", True)
                return
            roots = {
                "/assets/openarm_description/assets/": asset_root,
                "/vendor/three/": frontend / "node_modules/three",
                "/vendor/urdf-loader/": frontend / "node_modules/urdf-loader",
            }
            for prefix, root in roots.items():
                if path.startswith(prefix):
                    candidate = (root / path[len(prefix) :]).resolve()
                    if candidate.is_relative_to(root.resolve()) and candidate.is_file():
                        self._file(candidate)
                    else:
                        self.send_error(404)
                    return
            if path in ("/", "/index.html", "/app.js", "/style.css"):
                self._file(frontend / "public" / ("index.html" if path == "/" else path[1:]))
            else:
                self.send_error(404)

        def _file(self, path):
            if not path.is_file():
                self.send_error(404, "Dashboard asset missing; run setup.sh")
                return
            content_type = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
            self._send(path.read_bytes(), content_type, True)

        def _send(self, content, content_type, cache):
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Cache-Control", "max-age=3600" if cache else "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            with contextlib.suppress(BrokenPipeError, ConnectionResetError):
                self.wfile.write(content)

        def log_message(self, format, *args):
            if len(args) > 1 and str(args[1]).startswith(("4", "5")):
                super().log_message(format, *args)

    return ThreadingHTTPServer((host, port), Handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--demo", action="store_true", help="Use labelled synthetic data without initializing ROS")
    parser.add_argument("--usd-scene", type=Path, help="Object geometry/reference layout from export_usd_scene.py")
    args, ros_arguments = parser.parse_known_args()
    usd_scene = json.loads(args.usd_scene.read_text()) if args.usd_scene else None
    if usd_scene is not None and (
        usd_scene.get("schema") != 1 or usd_scene.get("units") != "metres" or usd_scene.get("up_axis") != "Z"
    ):
        parser.error("USD scene must be a schema 1, Z-up, metre export")
    config = yaml.safe_load(args.config.read_text())
    root = Path(__file__).resolve().parents[6]
    frontend = root / "scripts/deploy/dashboard"
    if not (frontend / "node_modules/three/build/three.module.js").is_file():
        raise RuntimeError("Run npm ci --prefix scripts/deploy/dashboard before starting the dashboard")
    share = Path(get_package_share_directory("openarm_description"))
    model = share / "assets/robot/openarm_v1.0/urdf/openarm_v10.urdf.xacro"
    xml = xacro.process_file(str(model), mappings={"ros2_control": "false", "bimanual": "true"}).toxml()
    state = SceneState()
    # colcon symlink installs each asset file individually. Resolve the source
    # model first so mesh files stay inside their actual asset directory.
    asset_root = model.resolve().parents[3]
    server = create_server(args.host, args.port, state, frontend, xml, asset_root, usd_scene)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    mode = "DEMO, synthetic data" if args.demo else "ROS monitor"
    print(f"OpenArm dashboard: http://{args.host}:{server.server_port} ({mode})", flush=True)
    node = None
    try:
        if args.demo:
            start = time.monotonic()
            while True:
                state.set(demo_snapshot(config, time.monotonic() - start))
                time.sleep(1 / 15)
        else:
            rclpy.init(args=ros_arguments)
            node = DeploymentScene(config, state)
            rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()
        server.server_close()
        if node is not None:
            node.destroy_node()
            rclpy.shutdown()


if __name__ == "__main__":
    main()
