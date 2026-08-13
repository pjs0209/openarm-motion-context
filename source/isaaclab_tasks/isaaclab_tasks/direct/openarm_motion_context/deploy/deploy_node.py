#!/usr/bin/env python3
# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""ROS2 deploy bridge for the Paper OpenArm Re-Lift policy.

Runtime data flow:

1. D435i AprilTag detector publishes ``T_camera_tag`` per camera.
2. TF provides ``T_base_camera`` and ``T_base_ee``.
3. The deploy pose provider estimates ``T_base_box`` and left/right grip poses.
4. The same 30D own observation and 30D partner-message convention used by the
   Paper Lift task is built on the real robot.
5. The policy output is applied as incremental joint-position commands.

The script is intentionally independent from Isaac Sim. It reuses only the
small geometry and policy modules in ``openarm_motion_context``.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any

import torch


REPO_ROOT = Path(__file__).resolve().parents[6]
SOURCE_ROOT = REPO_ROOT / "source"
for path in (SOURCE_ROOT / "isaaclab_tasks", SOURCE_ROOT / "isaaclab"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from isaaclab_tasks.direct.openarm_motion_context.perception.apriltag_geometry import (  # noqa: E402
    AprilTagDeployPoseProvider,
    compute_actor_target_from_grip,
    normalize_quat_wxyz,
    quat_angle_error_wxyz,
)
from isaaclab_tasks.direct.openarm_motion_context.models.actor import MotionContextPolicyModel  # noqa: E402
from isaaclab_tasks.direct.openarm_motion_context.communication.message_builder import pad_message  # noqa: E402
from isaaclab_tasks.direct.openarm_motion_context.communication.motion import signed_motion_intent  # noqa: E402
from isaaclab_tasks.direct.openarm_motion_context.communication.motion_context import (  # noqa: E402
    motion_context_from_raw,
)
from isaaclab_tasks.direct.openarm_motion_context.deploy.apriltag_ros import (  # noqa: E402
    detection_id as _extract_detection_id,
    detection_pose as _extract_detection_pose,
)
from isaaclab_tasks.direct.openarm_motion_context.deploy.controller import (  # noqa: E402
    incremental_joint_target,
    update_gripper_closed,
)
from isaaclab_tasks.direct.openarm_motion_context.deploy.observation import ObservationNormalizer  # noqa: E402
from isaaclab_tasks.direct.openarm_motion_context.deploy.robot_state import ArmState, JointSnapshot  # noqa: E402
from isaaclab_tasks.direct.openarm_motion_context.deploy.safety import (  # noqa: E402
    message_is_fresh,
    target_delta_within_limit,
    visible_tag_count,
)


def _load_yaml(path: str | os.PathLike[str]) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required for the deploy config: pip install pyyaml") from exc
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


def _tensor(data: Any, *, device: torch.device, shape: tuple[int, ...] | None = None) -> torch.Tensor:
    out = torch.as_tensor(data, dtype=torch.float32, device=device)
    if shape is not None and tuple(out.shape) != shape:
        raise ValueError(f"Expected tensor shape {shape}, got {tuple(out.shape)}")
    return out


def _xyzw_to_wxyz(values: Any, *, device: torch.device) -> torch.Tensor:
    quat_xyzw = _tensor(values, device=device, shape=(4,))
    return normalize_quat_wxyz(torch.stack([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]]))


def _ros_transform_to_pose(transform, *, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    t = transform.transform.translation
    q = transform.transform.rotation
    pos = torch.tensor([t.x, t.y, t.z], dtype=torch.float32, device=device)
    quat = _xyzw_to_wxyz([q.x, q.y, q.z, q.w], device=device)
    return pos, quat


def _ros_pose_to_pose(pose, *, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    p = pose.position
    q = pose.orientation
    pos = torch.tensor([p.x, p.y, p.z], dtype=torch.float32, device=device)
    quat = _xyzw_to_wxyz([q.x, q.y, q.z, q.w], device=device)
    return pos, quat


def _now_sec(node) -> float:
    return node.get_clock().now().nanoseconds * 1.0e-9


def _normalize_frame_id(frame_id: str) -> str:
    """Normalize ROS frame IDs for exact detector/TF contract checks."""

    return str(frame_id).strip().lstrip("/")


def _module_state_from_checkpoint(checkpoint: Any, agent: str) -> dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict):
        for key in ("paper_intent_modules", "modules", "model", "models"):
            modules = checkpoint.get(key)
            if isinstance(modules, dict):
                if agent in modules and isinstance(modules[agent], dict):
                    return modules[agent]
                if f"{agent}/policy" in modules and isinstance(modules[f"{agent}/policy"], dict):
                    return modules[f"{agent}/policy"]
                nested = modules.get(agent)
                if isinstance(nested, dict) and "policy" in nested:
                    return nested["policy"]
        if agent in checkpoint and isinstance(checkpoint[agent], dict):
            agent_modules = checkpoint[agent]
            policy = agent_modules.get("policy")
            if isinstance(policy, dict):
                return policy
            if all(isinstance(v, torch.Tensor) for v in agent_modules.values()):
                return agent_modules
        if all(isinstance(v, torch.Tensor) for v in checkpoint.values()):
            return checkpoint
    raise RuntimeError(
        "Could not find policy weights in checkpoint. Expected a skrl checkpoint "
        "with per-agent policy modules or a direct state_dict."
    )


def _maybe_load_motion_context_scale(cfg: dict[str, Any], *, device: torch.device) -> torch.Tensor:
    """Load the frozen motion-context scale saved next to the policy checkpoint."""

    comm_cfg = cfg.setdefault("communication", {})
    configured = comm_cfg.get("context_scale", [0.10, 0.50, 0.50])
    scale = _tensor(configured, device=device, shape=(3,))
    sidecar_path = comm_cfg.get("motion_context_sidecar", None)
    if sidecar_path is None:
        checkpoint = cfg.get("checkpoint", "")
        if checkpoint:
            root, ext = os.path.splitext(str(checkpoint))
            sidecar_path = f"{root}_motion_context{ext or '.pt'}"
    if not sidecar_path or not os.path.exists(str(sidecar_path)):
        print(
            "[WARN] Motion-context sidecar not found; using configured fallback scale "
            f"{scale.detach().cpu().tolist()}: {sidecar_path or '<unset>'}"
        )
        return scale.clamp_min(1.0e-6)
    try:
        state = torch.load(str(sidecar_path), map_location=device)
    except Exception as exc:
        print(f"[WARN] Failed to load motion-context sidecar {sidecar_path}: {exc}")
        return scale.clamp_min(1.0e-6)
    if isinstance(state, dict):
        nested = state.get("motion_context")
        if isinstance(nested, dict):
            state = nested
        raw = state.get("running_scale", state.get("motion_context_running_scale", None))
        if raw is not None:
            loaded = torch.as_tensor(raw, dtype=torch.float32, device=device).flatten()
            if loaded.numel() >= 3:
                return loaded[:3].clamp_min(1.0e-6)
    print(f"[WARN] Motion-context sidecar has no valid running_scale: {sidecar_path}")
    return scale.clamp_min(1.0e-6)


def _make_policy(agent: str, cfg: dict[str, Any], checkpoint: Any, *, device: torch.device):
    from gymnasium import spaces

    policy_cfg = cfg["policy"]
    obs_dim = int(policy_cfg.get("own_observation_dim", 30)) + int(policy_cfg.get("communication_feature_dim", 30))
    action_dim = 8
    model = MotionContextPolicyModel(
        observation_space=spaces.Box(-float("inf"), float("inf"), shape=(obs_dim,)),
        action_space=spaces.Box(-1.0, 1.0, shape=(action_dim,)),
        device=device,
        hidden_sizes=[int(v) for v in policy_cfg.get("hidden_sizes", [128, 128])],
        own_observation_dim=int(policy_cfg.get("own_observation_dim", 30)),
        motion_intent_dim=int(policy_cfg.get("motion_intent_dim", 3)),
        motion_context_dim=int(policy_cfg.get("motion_context_dim", 3)),
        intent_variant="share_intent",
        intent_arch="shared_intent_encoder",
        agent_id=agent,
        clip_actions=True,
        initial_log_std=0.0,
        partner_intent_embed_dim=int(policy_cfg.get("partner_intent_embed_dim", 32)),
        communication_feature_dim=int(policy_cfg.get("communication_feature_dim", 30)),
    )
    model.load_state_dict(_module_state_from_checkpoint(checkpoint, agent), strict=True)
    model.eval()
    return model


class OpenArmPaperReLiftDeployNode:
    """ROS2 node that turns AprilTag detections and TF into policy commands."""

    def __init__(self, cfg: dict[str, Any]):
        import rclpy
        from apriltag_msgs.msg import AprilTagDetectionArray
        from sensor_msgs.msg import JointState
        from geometry_msgs.msg import TransformStamped
        from tf2_ros import Buffer, StaticTransformBroadcaster, TransformListener
        from trajectory_msgs.msg import JointTrajectory

        self.rclpy = rclpy
        self.JointTrajectory = JointTrajectory
        self.JointTrajectoryPoint = __import__("trajectory_msgs.msg", fromlist=["JointTrajectoryPoint"]).JointTrajectoryPoint
        self.DurationMsg = __import__("builtin_interfaces.msg", fromlist=["Duration"]).Duration
        self.cfg = cfg
        self.device = torch.device(str(cfg.get("policy", {}).get("device", "cpu")))

        self.node = rclpy.create_node("openarm_paper_re_lift_deploy")
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self.node)
        self.base_frame = str(cfg.get("base_frame", "openarm_body_link0"))
        self.rate_hz = float(cfg.get("control_rate_hz", 30.0))
        self.dt = 1.0 / max(self.rate_hz, 1.0e-6)
        self.dry_run = bool(cfg.get("dry_run", True))
        self.context_scale = _maybe_load_motion_context_scale(cfg, device=self.device)
        self._validate_calibration_or_dry_run()

        topics = cfg["topics"]
        self.node.create_subscription(JointState, topics["joint_states"], self._on_joint_state, 10)
        command_topics = topics["commands"]
        self.command_pubs = {
            name: self.node.create_publisher(JointTrajectory, str(topic), 10)
            for name, topic in command_topics.items()
        }

        # Preserve the exact TCP used by the Isaac policy without changing the
        # upstream OpenArm v1.0 URDF. Camera optical transforms remain owned by
        # the RealSense/hand-eye TF chain and are intentionally not broadcast here.
        self.static_tf_broadcaster = StaticTransformBroadcaster(self.node)
        static_transforms = []
        for transform_cfg in cfg.get("static_policy_transforms", []):
            transform = TransformStamped()
            transform.header.stamp = self.node.get_clock().now().to_msg()
            transform.header.frame_id = str(transform_cfg["parent"])
            transform.child_frame_id = str(transform_cfg["child"])
            xyz = transform_cfg["xyz"]
            quat = transform_cfg["quat_wxyz"]
            transform.transform.translation.x = float(xyz[0])
            transform.transform.translation.y = float(xyz[1])
            transform.transform.translation.z = float(xyz[2])
            transform.transform.rotation.w = float(quat[0])
            transform.transform.rotation.x = float(quat[1])
            transform.transform.rotation.y = float(quat[2])
            transform.transform.rotation.z = float(quat[3])
            static_transforms.append(transform)
        if static_transforms:
            self.static_tf_broadcaster.sendTransform(static_transforms)

        self.camera_names = list(cfg["frames"]["cameras"].keys())
        self.camera_frames = [str(cfg["frames"]["cameras"][name]) for name in self.camera_names]
        mount_cfg = cfg["frames"].get("camera_mounts", {})
        self.camera_mount_frames = [str(mount_cfg.get(name, self.base_frame)) for name in self.camera_names]
        self.require_matching_detection_frame = bool(
            cfg.get("safety", {}).get("require_matching_detection_frame", True)
        )
        self._validated_camera_tf_chains: set[int] = set()
        self.tag_ids = [int(v) for v in cfg["apriltag"]["ids"]]
        self.tag_id_to_index = {tag_id: idx for idx, tag_id in enumerate(self.tag_ids)}
        self.latest_detections: list[dict[int, tuple[float, torch.Tensor, torch.Tensor]]] = [
            {} for _ in self.camera_names
        ]
        for camera_index, camera_name in enumerate(self.camera_names):
            topic = topics["apriltag"][camera_name]
            self.node.create_subscription(
                AprilTagDetectionArray,
                topic,
                lambda msg, idx=camera_index: self._on_apriltag(msg, idx),
                10,
            )

        checkpoint_path = str(cfg.get("checkpoint", ""))
        if not checkpoint_path:
            raise ValueError("config.checkpoint is required")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.policies = {
            "left_arm": _make_policy("left_arm", cfg, checkpoint, device=self.device),
            "right_arm": _make_policy("right_arm", cfg, checkpoint, device=self.device),
        }
        actor_dim = int(cfg["policy"].get("own_observation_dim", 30)) + int(cfg["policy"].get("communication_feature_dim", 30))
        normalize_obs = bool(cfg["policy"].get("use_checkpoint_state_normalizer", True))
        self.normalizers = {
            "left_arm": ObservationNormalizer(checkpoint, "left_arm", actor_dim, device=self.device, enabled=normalize_obs),
            "right_arm": ObservationNormalizer(checkpoint, "right_arm", actor_dim, device=self.device, enabled=normalize_obs),
        }
        missing_normalizers = [agent for agent, normalizer in self.normalizers.items() if not normalizer.loaded]
        if missing_normalizers:
            raise RuntimeError(
                "Checkpoint state-preprocessor statistics are missing or malformed for "
                f"{missing_normalizers}. Set policy.use_checkpoint_state_normalizer=false only "
                "for a policy trained without RunningStandardScaler."
            )

        tag_pos_box = _tensor(cfg["apriltag"]["tag_pos_box"], device=self.device, shape=(len(self.tag_ids), 3))
        tag_quat_box = _tensor(cfg["apriltag"]["tag_quat_box"], device=self.device, shape=(len(self.tag_ids), 4))
        grip_cfg = cfg["grip_targets"]
        self.pose_provider = AprilTagDeployPoseProvider(
            tag_pos_box=tag_pos_box,
            tag_quat_box=tag_quat_box,
            left_grip_pos_box=_tensor(grip_cfg["left_pos_box"], device=self.device, shape=(3,)),
            left_grip_quat_box=_tensor(grip_cfg["left_quat_box"], device=self.device, shape=(4,)),
            right_grip_pos_box=_tensor(grip_cfg["right_pos_box"], device=self.device, shape=(3,)),
            right_grip_quat_box=_tensor(grip_cfg["right_quat_box"], device=self.device, shape=(4,)),
        )

        joints = cfg["joints"]
        self.left_arm_joints = [str(v) for v in joints["left_arm"]]
        self.right_arm_joints = [str(v) for v in joints["right_arm"]]
        self.left_gripper_joints = [str(v) for v in joints["left_gripper"]]
        self.right_gripper_joints = [str(v) for v in joints["right_gripper"]]
        expected_command_topics = {"left_arm", "right_arm", "left_gripper", "right_gripper"}
        if set(self.command_pubs) != expected_command_topics:
            raise ValueError(
                "topics.commands must define exactly "
                f"{sorted(expected_command_topics)}, got {sorted(self.command_pubs)}"
            )
        default_cfg = cfg["default_joint_pos"]
        self.left_default = _tensor(default_cfg["left_arm"], device=self.device, shape=(7,))
        self.right_default = _tensor(default_cfg["right_arm"], device=self.device, shape=(7,))
        self.arm_state = {
            "left_arm": ArmState(q_target=self.left_default.clone(), prev_action=torch.zeros(8, device=self.device)),
            "right_arm": ArmState(q_target=self.right_default.clone(), prev_action=torch.zeros(8, device=self.device)),
        }
        width = int(self.cfg["policy"].get("communication_feature_dim", 30))
        self.communication_buffer = {
            "left_arm": torch.zeros(width, dtype=torch.float32, device=self.device),
            "right_arm": torch.zeros(width, dtype=torch.float32, device=self.device),
        }
        self.last_joint_state: JointSnapshot | None = None
        self.last_command_time = 0.0

        self.timer = self.node.create_timer(self.dt, self._control_tick)
        self.node.get_logger().info(
            "OpenArm Paper Re-Lift deploy node ready "
            f"(dry_run={self.dry_run}, cameras={self.camera_names}, tags={self.tag_ids})"
        )

    def _validate_calibration_or_dry_run(self) -> None:
        tag_pos = torch.as_tensor(self.cfg.get("apriltag", {}).get("tag_pos_box", []), dtype=torch.float32)
        left_grip = torch.as_tensor(self.cfg.get("grip_targets", {}).get("left_pos_box", []), dtype=torch.float32)
        right_grip = torch.as_tensor(self.cfg.get("grip_targets", {}).get("right_pos_box", []), dtype=torch.float32)
        looks_empty = tag_pos.numel() == 0 or float(tag_pos.abs().sum() + left_grip.abs().sum() + right_grip.abs().sum()) <= 1.0e-9
        if looks_empty and not self.dry_run:
            raise RuntimeError(
                "Deploy calibration still looks like placeholder zeros. Fill T_box_tag and T_box_grip in the YAML, "
                "or run with --dry-run while testing subscriptions/TF."
            )

    def _on_joint_state(self, msg) -> None:
        self.last_joint_state = JointSnapshot(stamp=_now_sec(self.node), msg=msg)

    def _on_apriltag(self, msg, camera_index: int) -> None:
        expected_frame = _normalize_frame_id(self.camera_frames[camera_index])
        header = getattr(msg, "header", None)
        measured_frame = _normalize_frame_id(getattr(header, "frame_id", ""))
        if measured_frame != expected_frame:
            message = (
                f"Ignoring {self.camera_names[camera_index]} AprilTag detections: "
                f"header.frame_id={measured_frame!r}, expected={expected_frame!r}"
            )
            if self.require_matching_detection_frame:
                self.node.get_logger().error(message, throttle_duration_sec=2.0)
                return
            self.node.get_logger().warn(message, throttle_duration_sec=2.0)

        stamp = self._message_stamp_sec(msg)
        detections = {}
        for detection in msg.detections:
            tag_id = _extract_detection_id(detection)
            if tag_id not in self.tag_id_to_index:
                continue
            pose = _extract_detection_pose(detection)
            if pose is None:
                continue
            tag_index = self.tag_id_to_index[tag_id]
            detections[tag_index] = (stamp, *_ros_pose_to_pose(pose, device=self.device))
        # All detections in one array share the same image timestamp. Replace
        # the camera snapshot so a tag left over from an older image is never
        # transformed with the current wrist-camera pose.
        self.latest_detections[camera_index] = detections

    def _lookup_transform_pose(
        self,
        parent_frame: str,
        child_frame: str,
        stamp_sec: float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        try:
            stamp = (
                self.rclpy.time.Time()
                if stamp_sec is None
                else self.rclpy.time.Time(nanoseconds=max(0, int(stamp_sec * 1.0e9)))
            )
            transform = self.tf_buffer.lookup_transform(parent_frame, child_frame, stamp)
        except Exception as exc:
            when = "latest" if stamp_sec is None else f"t={stamp_sec:.6f}"
            self.node.get_logger().warn(
                f"TF unavailable {parent_frame}->{child_frame} ({when}): {exc}",
                throttle_duration_sec=2.0,
            )
            return None
        return _ros_transform_to_pose(transform, device=self.device)

    def _lookup_pose(
        self,
        child_frame: str,
        stamp_sec: float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        return self._lookup_transform_pose(self.base_frame, child_frame, stamp_sec)

    def _message_stamp_sec(self, msg) -> float:
        header = getattr(msg, "header", None)
        stamp = getattr(header, "stamp", None)
        if stamp is None:
            return _now_sec(self.node)
        return float(stamp.sec) + float(stamp.nanosec) * 1.0e-9

    def _safety_cfg(self, name: str, default: float) -> float:
        return float(self.cfg.get("safety", {}).get(name, default))

    def _is_joint_state_fresh(self) -> bool:
        if self.last_joint_state is None:
            return False
        now = _now_sec(self.node)
        age = now - self.last_joint_state.stamp
        max_age = self._safety_cfg("max_joint_state_age", 0.25)
        if not message_is_fresh(now, self.last_joint_state.stamp, max_age):
            self.node.get_logger().warn(
                f"JointState stale ({age:.3f}s > {max_age:.3f}s); holding command",
                throttle_duration_sec=1.0,
            )
            return False
        return True

    def _joint_values(self, names: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        if self.last_joint_state is None:
            raise RuntimeError("No JointState received yet")
        msg = self.last_joint_state.msg
        index = {name: i for i, name in enumerate(msg.name)}
        pos = []
        vel = []
        for name in names:
            if name not in index:
                raise RuntimeError(f"JointState is missing joint {name!r}")
            i = index[name]
            pos.append(float(msg.position[i]))
            if i < len(msg.velocity):
                vel.append(float(msg.velocity[i]))
            else:
                vel.append(0.0)
        return torch.tensor(pos, dtype=torch.float32, device=self.device), torch.tensor(vel, dtype=torch.float32, device=self.device)

    def _collect_apriltag_tensors(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[float | None]]:
        num_cameras = len(self.camera_names)
        num_tags = len(self.tag_ids)
        tag_pos_camera = torch.zeros((num_cameras, num_tags, 3), dtype=torch.float32, device=self.device)
        tag_quat_camera = torch.tensor(
            [[[1.0, 0.0, 0.0, 0.0]] * num_tags] * num_cameras,
            dtype=torch.float32,
            device=self.device,
        )
        visible = torch.zeros((num_cameras, num_tags), dtype=torch.float32, device=self.device)
        camera_stamps: list[float | None] = [None] * num_cameras
        now = _now_sec(self.node)
        max_detection_age = self._safety_cfg("max_detection_age", 0.35)
        for camera_idx, detection_map in enumerate(self.latest_detections):
            for tag_idx, (stamp, pos, quat) in detection_map.items():
                if not message_is_fresh(now, stamp, max_detection_age):
                    continue
                tag_pos_camera[camera_idx, tag_idx] = pos
                tag_quat_camera[camera_idx, tag_idx] = quat
                visible[camera_idx, tag_idx] = 1.0
                previous_stamp = camera_stamps[camera_idx]
                camera_stamps[camera_idx] = stamp if previous_stamp is None else max(previous_stamp, stamp)
        return tag_pos_camera, tag_quat_camera, visible, camera_stamps

    def _camera_poses(
        self,
        camera_stamps: list[float | None],
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        poses: list[tuple[torch.Tensor, torch.Tensor]] = []
        for camera_index, (camera_name, camera_frame, mount_frame, stamp) in enumerate(
            zip(self.camera_names, self.camera_frames, self.camera_mount_frames, camera_stamps)
        ):
            if stamp is None:
                poses.append(
                    (
                        torch.zeros(3, dtype=torch.float32, device=self.device),
                        torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32, device=self.device),
                    )
                )
                continue

            # Wrist cameras require FK plus their fixed hand-eye transform. Looking
            # up the optical frame at the image timestamp preserves both pieces.
            pose = self._lookup_pose(camera_frame, stamp)
            if pose is None:
                return None
            poses.append(pose)

            if camera_index not in self._validated_camera_tf_chains:
                mount_to_camera = self._lookup_transform_pose(mount_frame, camera_frame)
                if mount_to_camera is None:
                    self.node.get_logger().error(
                        f"Camera TF chain is incomplete for {camera_name}: "
                        f"missing {mount_frame}->{camera_frame}",
                        throttle_duration_sec=2.0,
                    )
                    return None
                mount_pos, mount_quat = mount_to_camera
                self.node.get_logger().info(
                    f"Validated camera TF {camera_name}: {mount_frame}->{camera_frame} "
                    f"pos={mount_pos.detach().cpu().tolist()} "
                    f"quat_wxyz={mount_quat.detach().cpu().tolist()}"
                )
                self._validated_camera_tf_chains.add(camera_index)

        pos = torch.stack([pose[0] for pose in poses], dim=0)
        quat = torch.stack([pose[1] for pose in poses], dim=0)
        return pos, quat

    def _update_ee_motion(self, agent: str, ee_pos: torch.Tensor, ee_quat: torch.Tensor) -> tuple[torch.Tensor, float]:
        state = self.arm_state[agent]
        if state.prev_ee_pos is None:
            lin_vel = torch.zeros(3, dtype=torch.float32, device=self.device)
            ang_speed = 0.0
        else:
            lin_vel = (ee_pos - state.prev_ee_pos) / self.dt
            ang_speed = float(quat_angle_error_wxyz(state.prev_ee_quat.unsqueeze(0), ee_quat.unsqueeze(0)).item() / self.dt)
        state.prev_ee_pos = ee_pos.detach().clone()
        state.prev_ee_quat = ee_quat.detach().clone()
        state.ee_lin_vel = lin_vel
        state.ee_ang_speed = ang_speed
        return lin_vel, ang_speed

    def _motion_message(self, agent: str, action: torch.Tensor | None = None) -> torch.Tensor:
        comm_cfg = self.cfg["communication"]
        state = self.arm_state[agent]
        lin_vel = state.ee_lin_vel if state.ee_lin_vel is not None else torch.zeros(3, device=self.device)
        motion = signed_motion_intent(
            lin_vel,
            self.dt,
            int(comm_cfg.get("motion_intent_horizon", 15)),
            float(comm_cfg.get("interaction_motion_scale", 0.05)),
        )

        prev_action = state.prev_action if state.prev_action is not None else torch.zeros(8, device=self.device)
        current_action = action if action is not None else prev_action
        action_change = torch.linalg.vector_norm(current_action[:7] - prev_action[:7]).reshape(())
        raw = torch.stack(
            [
                torch.linalg.vector_norm(lin_vel),
                torch.tensor(float(state.ee_ang_speed), dtype=torch.float32, device=self.device),
                action_change,
            ]
        )
        context = motion_context_from_raw(
            raw[0:1],
            raw[1:2],
            raw[2:3],
            self.context_scale,
            float(comm_cfg.get("context_norm_max", 1.5)),
        )

        width = int(self.cfg["policy"].get("communication_feature_dim", 30))
        mode = str(comm_cfg.get("mode", "motion_context"))
        if mode == "none":
            payload = torch.zeros(0, device=self.device)
        elif mode == "motion_only":
            payload = motion
        elif mode == "context_only":
            payload = context
        elif mode == "motion_context":
            payload = torch.cat([motion, context], dim=-1)
        elif mode == "previous_action":
            payload = prev_action
        else:
            raise ValueError(f"Unsupported deploy communication mode: {mode}")
        return pad_message(payload, width, mode)

    def _own_obs(
        self,
        agent: str,
        ee_pos: torch.Tensor,
        ee_quat: torch.Tensor,
        grip_pos: torch.Tensor,
        grip_quat: torch.Tensor,
    ) -> torch.Tensor:
        if agent == "left_arm":
            arm_joints = self.left_arm_joints
            gripper_joints = self.left_gripper_joints
        else:
            arm_joints = self.right_arm_joints
            gripper_joints = self.right_gripper_joints
        joint_pos, joint_vel = self._joint_values(arm_joints)
        gripper_pos, _ = self._joint_values(gripper_joints)
        gripper_opening = gripper_pos.mean().reshape(1)
        open_target = max(float(self.cfg["action"].get("gripper_open_target", 0.04)), 1.0e-6)
        closure = torch.clamp(1.0 - gripper_opening / open_target, 0.0, 1.0)
        target_delta, target_quat_error = compute_actor_target_from_grip(ee_pos, ee_quat, grip_pos, grip_quat)
        ee_lin_vel = self.arm_state[agent].ee_lin_vel
        if ee_lin_vel is None:
            ee_lin_vel = torch.zeros(3, dtype=torch.float32, device=self.device)
        return torch.cat(
            [
                ee_pos,
                ee_quat,
                gripper_opening,
                joint_pos,
                joint_vel,
                target_delta,
                target_quat_error,
                closure,
                ee_lin_vel,
            ],
            dim=0,
        )

    def _policy_action(self, agent: str, own_obs: torch.Tensor, partner_message: torch.Tensor) -> torch.Tensor:
        states = torch.cat([own_obs, partner_message], dim=0).unsqueeze(0)
        states = self.normalizers[agent](states)
        with torch.no_grad():
            mean, _, _ = self.policies[agent].compute({"states": states}, role="policy")
        return torch.clamp(mean.squeeze(0), -1.0, 1.0)

    def _apply_incremental_action(self, agent: str, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        state = self.arm_state[agent]
        default = self.left_default if agent == "left_arm" else self.right_default
        if not state.target_initialized:
            arm_joints = self.left_arm_joints if agent == "left_arm" else self.right_arm_joints
            state.q_target = self._joint_values(arm_joints)[0]
            state.target_initialized = True
        action_cfg = self.cfg["action"]
        target_limit = float(action_cfg.get("action_scale", 0.7))
        step_scale = min(
            float(action_cfg.get("action_step_scale", 0.04)),
            self._safety_cfg("max_joint_step", 0.04),
        )
        target_limit = min(target_limit, self._safety_cfg("max_command_abs_from_default", target_limit))
        state.q_target = incremental_joint_target(
            state.q_target,
            action[:7],
            default,
            step_scale,
            target_limit,
        )
        eps = float(action_cfg.get("gripper_switch_epsilon", 0.02))
        state.gripper_closed = update_gripper_closed(state.gripper_closed, action[7], eps)
        grip_target = (
            float(action_cfg.get("gripper_close_target", 0.0))
            if state.gripper_closed
            else float(action_cfg.get("gripper_open_target", 0.04))
        )
        state.prev_action = action.detach().clone()
        gripper_joints = self.left_gripper_joints if agent == "left_arm" else self.right_gripper_joints
        return state.q_target, torch.full((len(gripper_joints),), grip_target, dtype=torch.float32, device=self.device)

    def _trajectory_message(self, joint_names: list[str], positions: torch.Tensor):
        msg = self.JointTrajectory()
        msg.joint_names = joint_names
        point = self.JointTrajectoryPoint()
        point.positions = positions.detach().cpu().tolist()
        duration = float(self.cfg["action"].get("command_time_from_start", 0.08))
        point.time_from_start = self.DurationMsg(sec=int(duration), nanosec=int((duration % 1.0) * 1.0e9))
        msg.points = [point]
        return msg

    def _publish_command(self, left_q: torch.Tensor, right_q: torch.Tensor, left_grip: torch.Tensor, right_grip: torch.Tensor) -> None:
        if self.dry_run:
            now = time.time()
            if now - self.last_command_time > 1.0:
                self.node.get_logger().info(
                    "dry_run command "
                    f"Lq={left_q.detach().cpu().numpy().round(3).tolist()} "
                    f"Rq={right_q.detach().cpu().numpy().round(3).tolist()} "
                    f"Lgrip={float(left_grip[0]):.3f} Rgrip={float(right_grip[0]):.3f}"
                )
                self.last_command_time = now
            return
        commands = {
            "left_arm": (self.left_arm_joints, left_q),
            "right_arm": (self.right_arm_joints, right_q),
            "left_gripper": (self.left_gripper_joints, left_grip),
            "right_gripper": (self.right_gripper_joints, right_grip),
        }
        for name, (joint_names, positions) in commands.items():
            self.command_pubs[name].publish(self._trajectory_message(joint_names, positions))

    def _control_tick(self) -> None:
        if not self._is_joint_state_fresh():
            return
        left_ee = self._lookup_pose(str(self.cfg["frames"]["left_ee"]))
        right_ee = self._lookup_pose(str(self.cfg["frames"]["right_ee"]))
        if left_ee is None or right_ee is None:
            return
        tag_pos_camera, tag_quat_camera, visible, camera_stamps = self._collect_apriltag_tensors()
        min_visible_tags = int(self.cfg.get("safety", {}).get("min_visible_tags", 1))
        visible_tags = visible_tag_count(visible)
        if visible_tags < min_visible_tags:
            self.node.get_logger().warn(
                f"Visible AprilTags below safety threshold ({visible_tags} < {min_visible_tags}); holding previous command",
                throttle_duration_sec=1.0,
            )
            return
        camera_poses = self._camera_poses(camera_stamps)
        if camera_poses is None:
            return

        left_ee_pos, left_ee_quat = left_ee
        right_ee_pos, right_ee_quat = right_ee
        self._update_ee_motion("left_arm", left_ee_pos, left_ee_quat)
        self._update_ee_motion("right_arm", right_ee_pos, right_ee_quat)

        left_grip_pos, left_grip_quat, right_grip_pos, right_grip_quat = self.pose_provider.compute_grip_targets(
            camera_poses[0],
            camera_poses[1],
            tag_pos_camera,
            tag_quat_camera,
            visible,
        )
        left_own = self._own_obs("left_arm", left_ee_pos, left_ee_quat, left_grip_pos, left_grip_quat)
        right_own = self._own_obs("right_arm", right_ee_pos, right_ee_quat, right_grip_pos, right_grip_quat)
        max_target_delta_norm = self._safety_cfg("max_target_delta_norm", 1.0)
        if not target_delta_within_limit(left_own[22:25], max_target_delta_norm) or not target_delta_within_limit(
            right_own[22:25], max_target_delta_norm
        ):
            self.node.get_logger().warn(
                "Grasp target delta exceeded safety limit; holding previous command",
                throttle_duration_sec=1.0,
            )
            return

        # Match training semantics: each arm observes the partner message from
        # the previous control tick, then current messages are stored below.
        left_partner = self.communication_buffer["right_arm"].clone()
        right_partner = self.communication_buffer["left_arm"].clone()
        left_action = self._policy_action("left_arm", left_own, left_partner)
        right_action = self._policy_action("right_arm", right_own, right_partner)
        self.communication_buffer["left_arm"] = self._motion_message("left_arm", left_action)
        self.communication_buffer["right_arm"] = self._motion_message("right_arm", right_action)
        left_q, left_grip = self._apply_incremental_action("left_arm", left_action)
        right_q, right_grip = self._apply_incremental_action("right_arm", right_action)
        self._publish_command(left_q, right_q, left_grip, right_grip)

    def spin(self) -> None:
        try:
            self.rclpy.spin(self.node)
        finally:
            self.node.destroy_node()
            self.rclpy.shutdown()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Deploy YAML path")
    parser.add_argument("--checkpoint", default=None, help="Override checkpoint path from YAML")
    parser.add_argument("--dry-run", action="store_true", help="Log commands instead of publishing them")
    parser.add_argument("--publish", action="store_true", help="Publish JointTrajectory commands")
    args = parser.parse_args()

    cfg = _load_yaml(args.config)
    if args.checkpoint is not None:
        cfg["checkpoint"] = args.checkpoint
    if args.dry_run:
        cfg["dry_run"] = True
    if args.publish:
        cfg["dry_run"] = False

    node = OpenArmPaperReLiftDeployNode(cfg)
    node.spin()


if __name__ == "__main__":
    main()
