# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Contact-free cooperative OpenArm bimanual KLT lift task logic.

Reward:
    pose approach
    + bilateral kinematic grasp quality
    + object-centric tilt-aware lift
    + terminal success
    - action-rate penalty

The task reward intentionally avoids stage logic, attachment gates, contact
sensors, force thresholds, learned modes, and auxiliary-intent rewards. The
same reward is used for every paper communication baseline; only the actor
communication input changes.

Grasp shaping and object-level lift shaping are additive rather than
multiplicatively coupled. The bilateral grasp term encourages a plausible
two-arm grasp configuration, while the lift term directly rewards object
elevation and stability. A stricter bilateral grasp condition is retained for
terminal task success. No explicit manipulation phase or reward state machine
is used.
"""

from __future__ import annotations

import torch

from .control_state_observation import control_state_observation

from isaaclab_tasks.direct.openarm_motion_context.legacy_math import (
    combine_frame_transforms,
    matrix_from_quat,
    quat_apply,
    quat_error_magnitude,
    quat_inv,
    subtract_frame_transforms,
)

from ..perception.apriltag_geometry import (
    fuse_camera_tag_detections_to_base as deploy_camera_tag_measurements_to_world,
    estimate_box_pose_from_tags as deploy_estimate_box_pose_from_tags,
    estimate_grip_targets_from_tags as deploy_estimate_grip_targets_from_tags,
)
from ..communication.message_builder import MOTION_CONTEXT_MODES, motion_context_payload, pad_message
from ..communication.motion import signed_motion_intent
from ..communication.motion_context import (
    full_action_variation,
    motion_context_calibration_diagnostics,
    motion_context_layout,
    motion_context_sharing_active,
    reset_motion_trend_state,
    select_motion_context,
    update_motion_trend_context,
)
from isaaclab_tasks.direct.openarm_motion_context.legacy_api import (
    read_data,
    write_root_pose,
    set_joint_position,
    tensor,
)


def _resolve_env_path(path_template: str, env_id: int) -> str:
    """Resolve an Isaac Lab env-regex path for one cloned environment."""

    env_ns = f"/World/envs/env_{env_id}"
    return (
        str(path_template)
        .replace("{ENV_REGEX_NS}", env_ns)
        .replace("/World/envs/env_.*/", f"{env_ns}/")
        .replace("/World/envs/env_[^/]+/", f"{env_ns}/")
    )


def _resolve_robot_child_path(env, child_path: str, env_id: int) -> str:
    """Resolve a robot-local child path for one cloned environment."""

    robot_root = _resolve_env_path(env.scene["robot"].cfg.prim_path, env_id)
    child_path = str(child_path).strip("/")
    return f"{robot_root}/{child_path}"


def _usd_authored_local_pose(env, prim_path: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Read a USD grasp prim's authored local pose in wxyz convention."""

    from isaaclab.sim.utils import get_current_stage
    from pxr import Gf, Usd, UsdGeom  # type: ignore

    stage = get_current_stage()
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise RuntimeError(f"OpenArmRE grasp target prim not found: {prim_path}")
    xformable = UsdGeom.Xformable(prim)
    local_tf = xformable.GetLocalTransformation(Usd.TimeCode.Default())
    local_tf = local_tf[0] if isinstance(local_tf, tuple) else local_tf
    decomposition = Gf.Transform(local_tf)
    translation = decomposition.GetTranslation()
    rotation = decomposition.GetRotation().GetQuat()
    imag = rotation.GetImaginary()

    pos = torch.tensor(
        [translation[0], translation[1], translation[2]],
        device=env.device,
        dtype=torch.float32,
    )
    quat = torch.tensor(
        [rotation.GetReal(), imag[0], imag[1], imag[2]],
        device=env.device,
        dtype=torch.float32,
    )
    quat = quat / torch.linalg.vector_norm(quat).clamp_min(1.0e-8)
    return pos, quat


def _usd_relative_pose(env, root_path: str, prim_path: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Read ``prim_path`` pose expressed in ``root_path`` coordinates."""

    from isaaclab.sim.utils import get_current_stage
    from pxr import Gf, Usd, UsdGeom  # type: ignore

    stage = get_current_stage()
    root = stage.GetPrimAtPath(root_path)
    prim = stage.GetPrimAtPath(prim_path)
    if not root.IsValid():
        raise RuntimeError(f"OpenArmRE root prim not found: {root_path}")
    if not prim.IsValid():
        raise RuntimeError(f"OpenArmRE child prim not found: {prim_path}")

    root_tf = UsdGeom.Xformable(root).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    prim_tf = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    rel_tf = prim_tf * root_tf.GetInverse()
    # AprilTag meshes carry non-uniform visual scale. Gf.Transform removes it
    # before extracting rotation; Matrix4d.ExtractRotationQuat does not.
    decomposition = Gf.Transform(rel_tf)
    translation = decomposition.GetTranslation()
    rotation = decomposition.GetRotation().GetQuat()
    imag = rotation.GetImaginary()

    pos = torch.tensor(
        [translation[0], translation[1], translation[2]],
        device=env.device,
        dtype=torch.float32,
    )
    quat = torch.tensor(
        [rotation.GetReal(), imag[0], imag[1], imag[2]],
        device=env.device,
        dtype=torch.float32,
    )
    quat = quat / torch.linalg.vector_norm(quat).clamp_min(1.0e-8)
    return pos, quat


def _usd_world_pose(env, prim_path: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Read a runtime USD prim's current world pose in wxyz convention."""

    from isaaclab.sim.utils import get_current_stage
    from pxr import Gf, Usd, UsdGeom  # type: ignore

    stage = get_current_stage()
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise RuntimeError(f"OpenArmRE runtime prim not found: {prim_path}")

    world_tf = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    decomposition = Gf.Transform(world_tf)
    translation = decomposition.GetTranslation()
    rotation = decomposition.GetRotation().GetQuat()
    imag = rotation.GetImaginary()

    pos = torch.tensor(
        [translation[0], translation[1], translation[2]],
        device=env.device,
        dtype=torch.float32,
    )
    quat = torch.tensor(
        [rotation.GetReal(), imag[0], imag[1], imag[2]],
        device=env.device,
        dtype=torch.float32,
    )
    quat = quat / torch.linalg.vector_norm(quat).clamp_min(1.0e-8)
    return pos, quat


def _find_child_prim_paths(root_path: str, child_names: tuple[str, ...]) -> dict[str, str]:
    """Find named child prims under a runtime USD root in one stage traversal."""

    from isaaclab.sim.utils import get_current_stage

    stage = get_current_stage()
    root = stage.GetPrimAtPath(root_path)
    if not root.IsValid():
        return {}

    wanted = set(child_names)
    found: dict[str, str] = {}
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        if not path.startswith(f"{root_path}/"):
            continue
        name = path.rsplit("/", 1)[-1]
        if name in wanted:
            found[name] = path
            if len(found) == len(wanted):
                break
    return found


def _quat_mul_wxyz(q0: torch.Tensor, q1: torch.Tensor) -> torch.Tensor:
    """Multiply two quaternions in wxyz convention."""

    w0, x0, y0, z0 = q0.unbind(dim=-1)
    w1, x1, y1, z1 = q1.unbind(dim=-1)
    out = torch.stack(
        [
            w0 * w1 - x0 * x1 - y0 * y1 - z0 * z1,
            w0 * x1 + x0 * w1 + y0 * z1 - z0 * y1,
            w0 * y1 - x0 * z1 + y0 * w1 + z0 * x1,
            w0 * z1 + x0 * y1 - y0 * x1 + z0 * w1,
        ],
        dim=-1,
    )
    return out / torch.linalg.vector_norm(out, dim=-1, keepdim=True).clamp_min(1.0e-8)


def _compose_pose_wxyz(
    parent_pos: torch.Tensor,
    parent_quat: torch.Tensor,
    child_pos: torch.Tensor,
    child_quat: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compose two poses represented with wxyz quaternions."""

    pos = parent_pos + quat_apply(parent_quat, child_pos)
    quat = _quat_mul_wxyz(parent_quat, child_quat)
    return pos, quat


def _invert_pose_wxyz(pos: torch.Tensor, quat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Invert a pose represented with a wxyz quaternion."""

    inv_quat = quat_inv(quat)
    inv_pos = -quat_apply(inv_quat, pos)
    return inv_pos, inv_quat


def _relative_pose_wxyz(
    root_pos: torch.Tensor,
    root_quat: torch.Tensor,
    child_pos: torch.Tensor,
    child_quat: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return child pose expressed in root coordinates."""

    inv_root_pos, inv_root_quat = _invert_pose_wxyz(root_pos, root_quat)
    return _compose_pose_wxyz(inv_root_pos, inv_root_quat, child_pos, child_quat)


def _apply_collision_target_pose_offset(
    env, pos: torch.Tensor, quat: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply small authored-target corrections for collision-box targets."""

    pos = pos.clone()
    pos[2] = pos[2] + float(getattr(env.cfg, "collision_target_z_offset", 0.0))

    x_rot_deg = float(getattr(env.cfg, "collision_target_x_rot_offset_deg", 0.0))
    if abs(x_rot_deg) > 1.0e-6:
        half = torch.deg2rad(torch.tensor(0.5 * x_rot_deg, device=env.device, dtype=torch.float32))
        offset = torch.stack(
            [
                torch.cos(half),
                torch.sin(half),
                torch.zeros((), device=env.device, dtype=torch.float32),
                torch.zeros((), device=env.device, dtype=torch.float32),
            ]
        )
        quat = _quat_mul_wxyz(quat, offset)

    yaw_deg = float(getattr(env.cfg, "collision_target_yaw_offset_deg", 0.0))
    if abs(yaw_deg) > 1.0e-6:
        half = torch.deg2rad(torch.tensor(0.5 * yaw_deg, device=env.device, dtype=torch.float32))
        yaw_offset = torch.stack(
            [
                torch.cos(half),
                torch.zeros((), device=env.device, dtype=torch.float32),
                torch.zeros((), device=env.device, dtype=torch.float32),
                torch.sin(half),
            ]
        )
        # This correction is expressed around the box-local up axis.
        quat = _quat_mul_wxyz(yaw_offset, quat)
    return pos, quat


def ensure_openarm_re_context(env) -> dict:
    """Create and cache ids, target buffers, hold counters, and debug stats."""

    if hasattr(env, "_openarm_re_ctx"):
        return env._openarm_re_ctx

    robot = env.robot

    left_arm_joint_ids, _ = robot.find_joints("openarm_left_joint[1-7]", preserve_order=True)
    right_arm_joint_ids, _ = robot.find_joints("openarm_right_joint[1-7]", preserve_order=True)
    left_gripper_joint_ids, _ = robot.find_joints("openarm_left_finger_joint.*", preserve_order=True)
    right_gripper_joint_ids, _ = robot.find_joints("openarm_right_finger_joint.*", preserve_order=True)
    left_ee_body_ids, _ = robot.find_bodies("openarm_left_ee_tcp", preserve_order=True)
    right_ee_body_ids, _ = robot.find_bodies("openarm_right_ee_tcp", preserve_order=True)
    left_inner_finger_body_ids, _ = robot.find_bodies("openarm_left_left_finger", preserve_order=True)
    left_outer_finger_body_ids, _ = robot.find_bodies("openarm_left_right_finger", preserve_order=True)
    right_inner_finger_body_ids, _ = robot.find_bodies("openarm_right_left_finger", preserve_order=True)
    right_outer_finger_body_ids, _ = robot.find_bodies("openarm_right_right_finger", preserve_order=True)
    left_camera_mount_body_ids, _ = robot.find_bodies("openarm_left_link7", preserve_order=True)
    right_camera_mount_body_ids, _ = robot.find_bodies("openarm_right_link7", preserve_order=True)
    chest_camera_mount_body_ids, _ = robot.find_bodies("openarm_body_link", preserve_order=True)

    if not left_arm_joint_ids or not right_arm_joint_ids:
        raise RuntimeError("OpenArmRE could not resolve left/right arm joints.")
    if not left_gripper_joint_ids or not right_gripper_joint_ids:
        raise RuntimeError("OpenArmRE could not resolve left/right gripper joints.")
    if not left_ee_body_ids or not right_ee_body_ids:
        raise RuntimeError("OpenArmRE could not resolve left/right EE TCP bodies.")
    if not (
        left_inner_finger_body_ids
        and left_outer_finger_body_ids
        and right_inner_finger_body_ids
        and right_outer_finger_body_ids
    ):
        raise RuntimeError("OpenArmRE could not resolve finger bodies for kinematic grasp scoring.")
    if not left_camera_mount_body_ids or not right_camera_mount_body_ids or not chest_camera_mount_body_ids:
        raise RuntimeError("OpenArmRE could not resolve camera mount bodies.")
    ctx = {
        "robot": env.robot,
        "object": env.object,
        "left_arm_joint_ids": left_arm_joint_ids,
        "right_arm_joint_ids": right_arm_joint_ids,
        "left_gripper_joint_ids": left_gripper_joint_ids,
        "right_gripper_joint_ids": right_gripper_joint_ids,
        "left_ee_body_id": left_ee_body_ids[0],
        "right_ee_body_id": right_ee_body_ids[0],
        "left_inner_finger_body_id": left_inner_finger_body_ids[0],
        "left_outer_finger_body_id": left_outer_finger_body_ids[0],
        "right_inner_finger_body_id": right_inner_finger_body_ids[0],
        "right_outer_finger_body_id": right_outer_finger_body_ids[0],
        "left_camera_mount_body_id": left_camera_mount_body_ids[0],
        "right_camera_mount_body_id": right_camera_mount_body_ids[0],
        "chest_camera_mount_body_id": chest_camera_mount_body_ids[0],
        "left_target_local_pos": torch.zeros((env.num_envs, 3), device=env.device),
        "right_target_local_pos": torch.zeros((env.num_envs, 3), device=env.device),
        "left_target_local_quat": torch.zeros((env.num_envs, 4), device=env.device),
        "right_target_local_quat": torch.zeros((env.num_envs, 4), device=env.device),
        "apriltag_local_pos": torch.zeros(
            (env.num_envs, len(tuple(getattr(env.cfg, "apriltag_names", ()))), 3),
            device=env.device,
        ),
        "apriltag_local_quat": torch.zeros(
            (env.num_envs, len(tuple(getattr(env.cfg, "apriltag_names", ()))), 4),
            device=env.device,
        ),
        "apriltag_paths": [],
        "apriltag_paths_by_env": [[] for _ in range(env.num_envs)],
        "apriltag_camera_paths": [],
        "apriltag_camera_paths_by_env": [[] for _ in range(env.num_envs)],
        "apriltag_camera_mount_local_pos": torch.zeros(
            (env.num_envs, len(tuple(getattr(env.cfg, "apriltag_camera_prims", ()))), 3),
            device=env.device,
        ),
        "apriltag_camera_mount_local_quat": torch.zeros(
            (env.num_envs, len(tuple(getattr(env.cfg, "apriltag_camera_prims", ()))), 4),
            device=env.device,
        ),
        "external_apriltag_measurements": None,
        "apriltag_measurement_history": [],
        "actor_target_cache_step": -1,
        "actor_target_cache": None,
        "actor_object_pose_cache_step": -1,
        "actor_object_pose_cache": None,
        "initial_object_pos_w": read_data(env.object.data, "root_pos_w")[:, 0:3].clone(),
        "initial_object_quat_w": read_data(env.object.data, "root_quat_w").clone(),
        "initial_actor_object_pos_w": read_data(env.object.data, "root_pos_w")[:, 0:3].clone(),
        "initial_actor_object_quat_w": read_data(env.object.data, "root_quat_w").clone(),
        "target_object_xy_w": read_data(env.object.data, "root_pos_w")[:, 0:2].clone(),
        "success_hold_count": torch.zeros(env.num_envs, device=env.device, dtype=torch.long),
        "reward_temporal_valid": torch.zeros(env.num_envs, device=env.device, dtype=torch.bool),
        "debug_stats": {},
    }
    env._openarm_re_ctx = ctx
    refresh_collision_targets_from_usd(env, ctx)
    refresh_apriltag_targets_from_usd(env, ctx)
    return ctx


def resolve_collision_target_paths(env, env_id: int) -> tuple[str, str]:
    """Return runtime USD paths for the left/right box grasp frames."""

    return (
        _resolve_env_path(env.cfg.left_collision_target_prim, env_id),
        _resolve_env_path(env.cfg.right_collision_target_prim, env_id),
    )


def refresh_collision_targets_from_usd(env, ctx: dict, env_ids: torch.Tensor | None = None) -> None:
    """Refresh object-local grasp targets from USD-authored grasp frames.

    There is deliberately no hard-coded fallback offset. A missing prim means
    the asset/task contract is broken, so we fail loudly.
    """

    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)

    selected_env_ids = [int(env_id.item()) for env_id in env_ids]
    if not selected_env_ids:
        return
    reference_env_id = selected_env_ids[0]
    left_path, right_path = resolve_collision_target_paths(env, reference_env_id)
    left_local_pos, left_local_quat = _usd_authored_local_pose(env, left_path)
    right_local_pos, right_local_quat = _usd_authored_local_pose(env, right_path)
    left_local_pos, left_local_quat = _apply_collision_target_pose_offset(env, left_local_pos, left_local_quat)
    right_local_pos, right_local_quat = _apply_collision_target_pose_offset(env, right_local_pos, right_local_quat)
    ctx["left_target_local_pos"][selected_env_ids] = left_local_pos
    ctx["right_target_local_pos"][selected_env_ids] = right_local_pos
    ctx["left_target_local_quat"][selected_env_ids] = left_local_quat
    ctx["right_target_local_quat"][selected_env_ids] = right_local_quat

    if not bool(ctx.get("_printed_target_debug", False)):
        left_target_w, right_target_w, _, _ = compute_collision_target_poses_w(env, ctx)
        print("[OpenArmRE Target Debug]")
        print(f"  left_path={resolve_collision_target_paths(env, 0)[0]}")
        print(f"  right_path={resolve_collision_target_paths(env, 0)[1]}")
        print(f"  object_root={env.scene['object'].cfg.prim_path}")
        print(
            "  target_offset="
            f"z{float(getattr(env.cfg, 'collision_target_z_offset', 0.0)):+.4f}m, "
            f"xrot{float(getattr(env.cfg, 'collision_target_x_rot_offset_deg', 0.0)):+.1f}deg, "
            f"yaw{float(getattr(env.cfg, 'collision_target_yaw_offset_deg', 0.0)):+.1f}deg"
        )
        print(f"  left_target_local={ctx['left_target_local_pos'][0].detach().cpu().tolist()}")
        print(f"  right_target_local={ctx['right_target_local_pos'][0].detach().cpu().tolist()}")
        print(f"  object_pos={read_data(env.object.data, 'root_pos_w')[0, 0:3].detach().cpu().tolist()}")
        print(f"  left_target_world={left_target_w[0].detach().cpu().tolist()}")
        print(f"  right_target_world={right_target_w[0].detach().cpu().tolist()}")
        ctx["_printed_target_debug"] = True


def refresh_apriltag_targets_from_usd(env, ctx: dict, env_ids: torch.Tensor | None = None) -> None:
    """Cache each AprilTag's KLT-local pose from the authored USD asset."""

    tag_names = tuple(getattr(env.cfg, "apriltag_names", ()))
    if not tag_names:
        return

    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)

    if ctx["apriltag_local_pos"].shape[1] != len(tag_names):
        ctx["apriltag_local_pos"] = torch.zeros((env.num_envs, len(tag_names), 3), device=env.device)
        ctx["apriltag_local_quat"] = torch.zeros((env.num_envs, len(tag_names), 4), device=env.device)
        ctx["apriltag_paths_by_env"] = [[] for _ in range(env.num_envs)]
    if len(ctx.get("apriltag_paths_by_env", [])) != env.num_envs:
        ctx["apriltag_paths_by_env"] = [[] for _ in range(env.num_envs)]

    camera_prims = tuple(getattr(env.cfg, "apriltag_camera_prims", ()))
    camera_mount_prims = tuple(getattr(env.cfg, "apriltag_camera_mount_prims", ()))
    if len(camera_prims) != len(camera_mount_prims):
        raise ValueError("apriltag_camera_prims and apriltag_camera_mount_prims must have equal length.")
    if len(ctx.get("apriltag_camera_paths_by_env", [])) != env.num_envs:
        ctx["apriltag_camera_paths_by_env"] = [[] for _ in range(env.num_envs)]

    selected_env_ids = [int(env_id.item()) for env_id in env_ids]
    if not selected_env_ids:
        return

    # Cloned environments share identical object/tag and mount/camera local
    # transforms. Read one authored instance, then replicate its extrinsics.
    reference_env_id = selected_env_ids[0]
    reference_object_root = _resolve_env_path(env.scene["object"].cfg.prim_path, reference_env_id)
    tag_suffixes: list[str] = []
    tag_local_pos: list[torch.Tensor] = []
    tag_local_quat: list[torch.Tensor] = []
    tag_paths = _find_child_prim_paths(reference_object_root, tuple(str(name) for name in tag_names))
    for tag_name in tag_names:
        tag_path = tag_paths.get(str(tag_name))
        if tag_path is None:
            raise RuntimeError(f"OpenArmRE AprilTag prim not found under {reference_object_root}: {tag_name}")
        local_pos, local_quat = _usd_relative_pose(env, reference_object_root, tag_path)
        tag_suffixes.append(tag_path[len(reference_object_root) :])
        tag_local_pos.append(local_pos)
        tag_local_quat.append(local_quat)

    camera_local_pos: list[torch.Tensor] = []
    camera_local_quat: list[torch.Tensor] = []
    for camera_prim, mount_prim in zip(camera_prims, camera_mount_prims):
        camera_path = _resolve_robot_child_path(env, str(camera_prim), reference_env_id)
        mount_path = _resolve_robot_child_path(env, str(mount_prim), reference_env_id)
        local_pos, local_quat = _usd_relative_pose(env, mount_path, camera_path)
        camera_local_pos.append(local_pos)
        camera_local_quat.append(local_quat)

    tag_local_pos_tensor = torch.stack(tag_local_pos)
    tag_local_quat_tensor = torch.stack(tag_local_quat)
    camera_local_pos_tensor = torch.stack(camera_local_pos)
    camera_local_quat_tensor = torch.stack(camera_local_quat)
    for env_id in selected_env_ids:
        object_root = _resolve_env_path(env.scene["object"].cfg.prim_path, env_id)
        ctx["apriltag_local_pos"][env_id] = tag_local_pos_tensor
        ctx["apriltag_local_quat"][env_id] = tag_local_quat_tensor
        ctx["apriltag_paths_by_env"][env_id] = [f"{object_root}{suffix}" for suffix in tag_suffixes]
        ctx["apriltag_camera_mount_local_pos"][env_id] = camera_local_pos_tensor
        ctx["apriltag_camera_mount_local_quat"][env_id] = camera_local_quat_tensor
        ctx["apriltag_camera_paths_by_env"][env_id] = [
            _resolve_robot_child_path(env, str(camera_prim), env_id) for camera_prim in camera_prims
        ]

    first_paths = ctx["apriltag_paths_by_env"][reference_env_id]
    first_camera_paths = ctx["apriltag_camera_paths_by_env"][reference_env_id]

    if first_paths:
        ctx["apriltag_paths"] = first_paths
    if first_camera_paths:
        ctx["apriltag_camera_paths"] = first_camera_paths

    if first_paths and not bool(ctx.get("_printed_apriltag_debug", False)):
        print("[OpenArmRE AprilTag Debug]")
        for tag_id, tag_path in enumerate(first_paths):
            print(
                f"  tag[{tag_id}] path={tag_path} "
                f"local_pos={ctx['apriltag_local_pos'][0, tag_id].detach().cpu().tolist()}"
            )
        tag_positions = ctx["apriltag_local_pos"][0]
        for side in ("left", "right"):
            grasp_pos = ctx[f"{side}_target_local_pos"][0]
            nearest_tag_id = int(torch.argmin(torch.linalg.vector_norm(tag_positions - grasp_pos, dim=-1)))
            tag_to_grasp_pos, _ = subtract_frame_transforms(
                tag_positions[nearest_tag_id].unsqueeze(0),
                ctx["apriltag_local_quat"][0, nearest_tag_id].unsqueeze(0),
                grasp_pos.unsqueeze(0),
                ctx[f"{side}_target_local_quat"][0].unsqueeze(0),
            )
            print(
                f"  nearest_tag_to_{side}_grasp=tag[{nearest_tag_id}] "
                f"delta_tag={tag_to_grasp_pos[0].detach().cpu().tolist()}"
            )
        for camera_id, camera_path in enumerate(first_camera_paths):
            print(f"  camera[{camera_id}] path={camera_path}")
        camera_pos_fk, camera_quat_fk = read_apriltag_camera_poses_w(env, ctx)
        for camera_id, camera_path in enumerate(first_camera_paths):
            usd_pos, usd_quat = _usd_world_pose(env, camera_path)
            pos_error = torch.linalg.vector_norm(camera_pos_fk[0, camera_id] - usd_pos)
            rot_error = quat_error_magnitude(
                camera_quat_fk[0, camera_id].unsqueeze(0),
                usd_quat.unsqueeze(0),
            )[0]
            print(
                f"  camera_tf_check[{camera_id}] "
                f"position_error={float(pos_error):.6f}m "
                f"rotation_error={float(torch.rad2deg(rot_error)):.4f}deg"
            )

        gt_left, gt_right, gt_left_q, gt_right_q = compute_collision_target_poses_w(env, ctx)
        actor_left, actor_right, actor_left_q, actor_right_q = estimate_collision_target_poses_from_apriltags_w(
            env,
            ctx,
        )
        left_pos_error = torch.linalg.vector_norm(actor_left - gt_left, dim=-1)
        right_pos_error = torch.linalg.vector_norm(actor_right - gt_right, dim=-1)
        left_rot_error = torch.rad2deg(quat_error_magnitude(actor_left_q, gt_left_q))
        right_rot_error = torch.rad2deg(quat_error_magnitude(actor_right_q, gt_right_q))
        max_position_error = torch.maximum(left_pos_error.max(), right_pos_error.max())
        max_rotation_error = torch.maximum(left_rot_error.max(), right_rot_error.max())
        print(
            "  grasp_target_round_trip_check "
            f"position_error_max={float(max_position_error):.8f}m "
            f"rotation_error_max={float(max_rotation_error):.6f}deg"
        )

        clean_virtual_measurement = (
            str(getattr(env.cfg, "apriltag_measurement_source", "camera_transform")).lower() == "camera_transform"
            and float(getattr(env.cfg, "apriltag_sim_position_noise_std", 0.0)) == 0.0
            and float(getattr(env.cfg, "apriltag_sim_rotation_noise_deg", 0.0)) == 0.0
            and float(getattr(env.cfg, "apriltag_sim_dropout_prob", 0.0)) == 0.0
            and int(getattr(env.cfg, "apriltag_sim_latency_steps", 0)) == 0
        )
        if clean_virtual_measurement and (float(max_position_error) > 1.0e-4 or float(max_rotation_error) > 1.0e-2):
            raise RuntimeError(
                "Clean AprilTag grasp-target round trip does not match the USD/object target: "
                f"position={float(max_position_error):.8f}m, rotation={float(max_rotation_error):.6f}deg."
            )
        ctx["_printed_apriltag_debug"] = True


def compute_collision_target_poses_w(env, ctx: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Attach the cached object-local collision targets to the current object pose."""

    object_pos = read_data(env.object.data, "root_pos_w")[:, 0:3]
    object_quat = read_data(env.object.data, "root_quat_w")
    return compute_collision_target_poses_from_object_pose_w(env, ctx, object_pos, object_quat)


def compute_collision_target_poses_from_object_pose_w(
    env,
    ctx: dict,
    object_pos: torch.Tensor,
    object_quat: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Attach cached object-local grasp targets to a supplied object pose.

    Reward and termination keep using the simulated rigid-object pose. Actor
    observations can route through this function with an estimated AprilTag
    pose, which mirrors the real deployment interface without changing the
    reward definition.
    """

    left_pos, left_quat = combine_frame_transforms(
        object_pos,
        object_quat,
        ctx["left_target_local_pos"],
        ctx["left_target_local_quat"],
    )
    right_pos, right_quat = combine_frame_transforms(
        object_pos,
        object_quat,
        ctx["right_target_local_pos"],
        ctx["right_target_local_quat"],
    )
    return left_pos, right_pos, left_quat, right_quat


def set_external_apriltag_measurements(
    env,
    tag_pos_camera: torch.Tensor,
    tag_quat_camera: torch.Tensor,
    visible: torch.Tensor | None = None,
) -> None:
    """Set deploy-time AprilTag detections expressed in each camera frame.

    Args:
        tag_pos_camera: Tensor shaped ``(num_envs, num_cameras, num_tags, 3)``.
        tag_quat_camera: Tensor shaped ``(num_envs, num_cameras, num_tags, 4)`` in wxyz.
        visible: Optional boolean/float mask shaped ``(num_envs, num_cameras, num_tags)``.
    """

    ctx = ensure_openarm_re_context(env)
    ctx["external_apriltag_measurements"] = {
        "tag_pos_camera": tag_pos_camera.to(env.device),
        "tag_quat_camera": tag_quat_camera.to(env.device),
        "visible": None if visible is None else visible.to(env.device),
    }
    ctx["actor_target_cache_step"] = -1
    ctx["actor_target_cache"] = None
    ctx["actor_object_pose_cache_step"] = -1
    ctx["actor_object_pose_cache"] = None


def clear_external_apriltag_measurements(env) -> None:
    """Clear deploy-time AprilTag detections and return to simulated measurements."""

    ctx = ensure_openarm_re_context(env)
    ctx["external_apriltag_measurements"] = None
    ctx["actor_target_cache_step"] = -1
    ctx["actor_target_cache"] = None
    ctx["actor_object_pose_cache_step"] = -1
    ctx["actor_object_pose_cache"] = None


def camera_tag_measurements_to_world(
    camera_pos_w: torch.Tensor,
    camera_quat_w: torch.Tensor,
    tag_pos_camera: torch.Tensor,
    tag_quat_camera: torch.Tensor,
    visible: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert multi-camera AprilTag detections to fused world-frame tag poses."""

    return deploy_camera_tag_measurements_to_world(
        camera_pos_w,
        camera_quat_w,
        tag_pos_camera,
        tag_quat_camera,
        visible,
    )


def _virtual_tag_poses_w(env, ctx: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate simulator tag poses used only to synthesize detector measurements.

    Ground-truth object state is confined to this virtual sensor boundary. The
    actor never consumes it directly: its grasp targets are reconstructed from
    the resulting ``T_camera_tag`` measurements through the same geometry used
    by the deployment interface.
    """

    num_tags = ctx["apriltag_local_pos"].shape[1]
    object_pos = read_data(env.object.data, "root_pos_w")[:, None, 0:3].expand(-1, num_tags, -1)
    object_quat = read_data(env.object.data, "root_quat_w")[:, None, :].expand(-1, num_tags, -1)
    tag_pos, tag_quat = _compose_pose_wxyz(
        object_pos.reshape(-1, 3),
        object_quat.reshape(-1, 4),
        ctx["apriltag_local_pos"].reshape(-1, 3),
        ctx["apriltag_local_quat"].reshape(-1, 4),
    )
    return (
        tag_pos.reshape(env.num_envs, num_tags, 3),
        tag_quat.reshape(env.num_envs, num_tags, 4),
    )


def _sample_small_rotation_quat(shape: tuple[int, ...], angle_std_deg: float, device) -> torch.Tensor:
    """Sample small random rotation quaternions in wxyz convention."""

    if angle_std_deg <= 0.0:
        quat = torch.zeros((*shape, 4), device=device)
        quat[..., 0] = 1.0
        return quat
    axis = torch.randn((*shape, 3), device=device)
    axis = axis / torch.linalg.vector_norm(axis, dim=-1, keepdim=True).clamp_min(1.0e-8)
    angle = torch.randn((*shape, 1), device=device) * torch.deg2rad(
        torch.tensor(angle_std_deg, device=device, dtype=torch.float32)
    )
    half = 0.5 * angle
    return torch.cat([torch.cos(half), axis * torch.sin(half)], dim=-1)


def apply_sim_apriltag_measurement_effects(
    env,
    ctx: dict,
    tag_pos_camera: torch.Tensor,
    tag_quat_camera: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply detector-like noise, dropout, and latency to simulated tag detections."""

    pos_noise_std = float(getattr(env.cfg, "apriltag_sim_position_noise_std", 0.0))
    rot_noise_deg = float(getattr(env.cfg, "apriltag_sim_rotation_noise_deg", 0.0))
    dropout_prob = float(getattr(env.cfg, "apriltag_sim_dropout_prob", 0.0))
    latency_steps = max(int(getattr(env.cfg, "apriltag_sim_latency_steps", 0)), 0)

    visible = torch.ones(tag_pos_camera.shape[:-1], device=env.device, dtype=tag_pos_camera.dtype)
    if pos_noise_std > 0.0:
        tag_pos_camera = tag_pos_camera + torch.randn_like(tag_pos_camera) * pos_noise_std
    if rot_noise_deg > 0.0:
        noise_quat = _sample_small_rotation_quat(tag_quat_camera.shape[:-1], rot_noise_deg, env.device)
        tag_quat_camera = _quat_mul_wxyz(tag_quat_camera, noise_quat)
    if dropout_prob > 0.0:
        visible = (torch.rand_like(visible) >= dropout_prob).to(dtype=tag_pos_camera.dtype)
        # Keep at least one tag measurement per env to avoid producing undefined object poses.
        has_any = visible.sum(dim=(1, 2)) > 0.0
        if not bool(has_any.all()):
            flat_visible = visible.view(visible.shape[0], -1)
            flat_visible[~has_any, 0] = 1.0

    if latency_steps > 0:
        history = list(ctx.get("apriltag_measurement_history", []))
        history.append(
            (
                tag_pos_camera.detach().clone(),
                tag_quat_camera.detach().clone(),
                visible.detach().clone(),
            )
        )
        max_len = latency_steps + 1
        if len(history) > max_len:
            history = history[-max_len:]
        ctx["apriltag_measurement_history"] = history
        if len(history) > latency_steps:
            tag_pos_camera, tag_quat_camera, visible = history[0]

    return tag_pos_camera, tag_quat_camera, visible


def read_apriltag_camera_poses_w(env, ctx: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute runtime optical-frame poses from FK and cached USD extrinsics."""

    camera_prims = tuple(getattr(env.cfg, "apriltag_camera_prims", ()))
    if not camera_prims:
        raise RuntimeError("actor_object_pose_source='apriltag' requires env.cfg.apriltag_camera_prims.")
    if len(ctx.get("apriltag_camera_paths_by_env", [])) != env.num_envs or any(
        len(paths) != len(camera_prims) for paths in ctx.get("apriltag_camera_paths_by_env", [])
    ):
        refresh_apriltag_targets_from_usd(env, ctx)

    if len(camera_prims) != 3:
        raise RuntimeError("OpenArmRE lift expects left wrist, right wrist, and chest cameras.")

    mount_pos = torch.stack(
        [
            read_data(env.robot.data, "body_state_w")[:, ctx["left_camera_mount_body_id"], 0:3],
            read_data(env.robot.data, "body_state_w")[:, ctx["right_camera_mount_body_id"], 0:3],
            read_data(env.robot.data, "body_state_w")[:, ctx["chest_camera_mount_body_id"], 0:3],
        ],
        dim=1,
    )
    mount_quat = torch.stack(
        [
            read_data(env.robot.data, "body_state_w")[:, ctx["left_camera_mount_body_id"], 3:7],
            read_data(env.robot.data, "body_state_w")[:, ctx["right_camera_mount_body_id"], 3:7],
            read_data(env.robot.data, "body_state_w")[:, ctx["chest_camera_mount_body_id"], 3:7],
        ],
        dim=1,
    )
    camera_pos, camera_quat = _compose_pose_wxyz(
        mount_pos.reshape(-1, 3),
        mount_quat.reshape(-1, 4),
        ctx["apriltag_camera_mount_local_pos"].reshape(-1, 3),
        ctx["apriltag_camera_mount_local_quat"].reshape(-1, 4),
    )
    return (
        camera_pos.reshape(env.num_envs, len(camera_prims), 3),
        camera_quat.reshape(env.num_envs, len(camera_prims), 4),
    )


def read_apriltag_measurements_w(
    env,
    ctx: dict,
    return_visibility: bool = False,
) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return fused world-frame AprilTag poses used by the actor-side perception path."""

    tag_names = tuple(getattr(env.cfg, "apriltag_names", ()))
    if not tag_names:
        raise RuntimeError("actor_object_pose_source='apriltag' requires env.cfg.apriltag_names.")
    if (
        "apriltag_local_pos" not in ctx
        or ctx["apriltag_local_pos"].shape[1] != len(tag_names)
        or len(ctx.get("apriltag_paths_by_env", [])) != env.num_envs
        or any(len(paths) != len(tag_names) for paths in ctx.get("apriltag_paths_by_env", []))
    ):
        refresh_apriltag_targets_from_usd(env, ctx)

    camera_pos_w, camera_quat_w = read_apriltag_camera_poses_w(env, ctx)
    external = ctx.get("external_apriltag_measurements")
    if external is not None:
        visible = external.get("visible")
        tag_pos_w, tag_quat_w = camera_tag_measurements_to_world(
            camera_pos_w,
            camera_quat_w,
            external["tag_pos_camera"],
            external["tag_quat_camera"],
            visible,
        )
        if return_visibility:
            if visible is None:
                visible_tags = torch.ones((env.num_envs, len(tag_names)), device=env.device, dtype=torch.bool)
            else:
                visible_tags = visible.to(env.device).sum(dim=1) > 0.0
            return tag_pos_w, tag_quat_w, visible_tags
        return tag_pos_w, tag_quat_w

    source = str(getattr(env.cfg, "apriltag_measurement_source", "camera_transform")).lower()
    if source in ("world_tag", "tag_world", "usd_world"):
        tag_pos = torch.zeros((env.num_envs, len(tag_names), 3), device=env.device)
        tag_quat = torch.zeros((env.num_envs, len(tag_names), 4), device=env.device)
        for env_id in range(env.num_envs):
            for tag_id, tag_path in enumerate(ctx["apriltag_paths_by_env"][env_id]):
                pos, quat = _usd_world_pose(env, tag_path)
                tag_pos[env_id, tag_id] = pos
                tag_quat[env_id, tag_id] = quat
        if return_visibility:
            visible_tags = torch.ones((env.num_envs, len(tag_names)), device=env.device, dtype=torch.bool)
            return tag_pos, tag_quat, visible_tags
        return tag_pos, tag_quat
    if source not in ("camera_transform", "camera", "deploy"):
        raise ValueError(f"Unsupported apriltag_measurement_source={source!r}.")

    # Virtual geometric detector: synthesize T_camera_tag, then discard the
    # simulator-only world tag poses before reconstructing the actor target.
    num_cameras = camera_pos_w.shape[1]
    num_tags = len(tag_names)
    tag_pos_w, tag_quat_w = _virtual_tag_poses_w(env, ctx)
    camera_pos_expanded = camera_pos_w[:, :, None, :].expand(-1, -1, num_tags, -1)
    camera_quat_expanded = camera_quat_w[:, :, None, :].expand(-1, -1, num_tags, -1)
    tag_pos_expanded = tag_pos_w[:, None, :, :].expand(-1, num_cameras, -1, -1)
    tag_quat_expanded = tag_quat_w[:, None, :, :].expand(-1, num_cameras, -1, -1)
    tag_pos_camera, tag_quat_camera = _relative_pose_wxyz(
        camera_pos_expanded.reshape(-1, 3),
        camera_quat_expanded.reshape(-1, 4),
        tag_pos_expanded.reshape(-1, 3),
        tag_quat_expanded.reshape(-1, 4),
    )
    tag_pos_camera = tag_pos_camera.reshape(env.num_envs, num_cameras, num_tags, 3)
    tag_quat_camera = tag_quat_camera.reshape(env.num_envs, num_cameras, num_tags, 4)
    tag_pos_camera, tag_quat_camera, visible = apply_sim_apriltag_measurement_effects(
        env,
        ctx,
        tag_pos_camera,
        tag_quat_camera,
    )
    tag_pos_w, tag_quat_w = camera_tag_measurements_to_world(
        camera_pos_w, camera_quat_w, tag_pos_camera, tag_quat_camera, visible
    )
    if return_visibility:
        visible_tags = visible.sum(dim=1) > 0.0
        return tag_pos_w, tag_quat_w, visible_tags
    return tag_pos_w, tag_quat_w


def estimate_object_pose_from_apriltags_w(env, ctx: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """Return a KLT pose estimate from the AprilTag rig.

    The tag local poses are cached from ``apriltag_00`` through ``apriltag_02``
    so the code knows exactly where each marker is attached on the KLT box. This
    object-pose path is kept for diagnostics and compatibility; the paper actor
    target path estimates grasp targets directly from tag poses.
    """

    tag_names = tuple(getattr(env.cfg, "apriltag_names", ()))
    if not tag_names:
        raise RuntimeError("actor_object_pose_source='apriltag' requires env.cfg.apriltag_names.")
    if "apriltag_local_pos" not in ctx or ctx["apriltag_local_pos"].shape[1] != len(tag_names):
        refresh_apriltag_targets_from_usd(env, ctx)

    tag_local_pos = ctx["apriltag_local_pos"]
    tag_local_quat = ctx["apriltag_local_quat"]
    measured_tag_pos_w, measured_tag_quat_w = read_apriltag_measurements_w(env, ctx)

    return deploy_estimate_box_pose_from_tags(
        measured_tag_pos_w,
        measured_tag_quat_w,
        tag_local_pos,
        tag_local_quat,
    )


def estimate_collision_target_poses_from_apriltags_w(
    env,
    ctx: dict,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Estimate actor grasp targets directly from visible AprilTags.

    Runtime math:

    ``T_world_tag_i * inv(T_box_tag_i) * T_box_grasp -> T_world_grasp``.

    This is the same direct tag-to-grasp path used by the real-robot deploy
    helper, so actor observations do not depend on a privileged box rigid-body
    pose when ``actor_object_pose_source='apriltag'``.
    """

    tag_names = tuple(getattr(env.cfg, "apriltag_names", ()))
    if not tag_names:
        raise RuntimeError("actor_object_pose_source='apriltag' requires env.cfg.apriltag_names.")
    if "apriltag_local_pos" not in ctx or ctx["apriltag_local_pos"].shape[1] != len(tag_names):
        refresh_apriltag_targets_from_usd(env, ctx)

    measured_tag_pos_w, measured_tag_quat_w, visible_tags = read_apriltag_measurements_w(
        env,
        ctx,
        return_visibility=True,
    )
    left_pos, left_quat, right_pos, right_quat = deploy_estimate_grip_targets_from_tags(
        measured_tag_pos_w,
        measured_tag_quat_w,
        ctx["apriltag_local_pos"],
        ctx["apriltag_local_quat"],
        ctx["left_target_local_pos"],
        ctx["left_target_local_quat"],
        ctx["right_target_local_pos"],
        ctx["right_target_local_quat"],
        visible_tags,
    )
    return left_pos, right_pos, left_quat, right_quat


def get_actor_object_pose_w(env, ctx: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the object pose source used by actor observations."""

    current_step = int(getattr(env, "common_step_counter", -1))
    cached = ctx.get("actor_object_pose_cache")
    if cached is not None and int(ctx.get("actor_object_pose_cache_step", -2)) == current_step:
        return cached

    source = str(getattr(env.cfg, "actor_object_pose_source", "ground_truth")).lower()
    if source in ("ground_truth", "gt", "object"):
        result = (read_data(env.object.data, "root_pos_w")[:, 0:3], read_data(env.object.data, "root_quat_w"))
    elif source in ("apriltag", "april_tag", "tag"):
        result = estimate_object_pose_from_apriltags_w(env, ctx)
    else:
        raise ValueError(f"Unsupported actor_object_pose_source={source!r}.")
    ctx["actor_object_pose_cache_step"] = current_step
    ctx["actor_object_pose_cache"] = result
    return result


def compute_actor_collision_target_poses_w(
    env, ctx: dict
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute actor grasp targets from the configured actor-side perception source."""

    current_step = int(getattr(env, "common_step_counter", -1))
    cached = ctx.get("actor_target_cache")
    if cached is not None and int(ctx.get("actor_target_cache_step", -2)) == current_step:
        return cached

    object_pos, object_quat = get_actor_object_pose_w(env, ctx)
    result = compute_collision_target_poses_from_object_pose_w(env, ctx, object_pos, object_quat)
    ctx["actor_target_cache_step"] = current_step
    ctx["actor_target_cache"] = result
    return result


def _closure_from_opening(env, gripper_opening: torch.Tensor) -> torch.Tensor:
    """Convert finger opening to the Paper motion-context-compatible closure scalar."""

    open_target = max(float(getattr(env.cfg, "gripper_open_target", 0.04)), 1.0e-6)
    closure = 1.0 - gripper_opening / open_target
    return torch.clamp(closure, 0.0, 1.0)


def _ee_linear_velocity_b(env, ee_body_id: int) -> torch.Tensor:
    """Return EE linear velocity in robot base frame, scaled like Paper motion-context own_obs."""

    robot_root_quat = read_data(env.robot.data, "root_quat_w")
    ee_lin_vel_w = read_data(env.robot.data, "body_state_w")[:, ee_body_id, 7:10]
    ee_lin_vel_b = quat_apply(quat_inv(robot_root_quat), ee_lin_vel_w)
    scale = max(float(getattr(env.cfg, "ee_vel_scale", 1.0)), 1.0e-6)
    return torch.clamp(ee_lin_vel_b / scale, -10.0, 10.0)


def _agent_observation(env, ctx: dict, side: str) -> torch.Tensor:
    """Build the 30D Paper motion-context-compatible per-arm own observation.

    Layout:
      0:3   ee_pos_b
      3:7   ee_quat_b
      7:8   gripper_opening
      8:15  joint_pos
      15:22 joint_vel
      22:25 target_delta_b
      25:26 target_quat_error
      26:27 closure
      27:30 ee_lin_vel_b
    """

    if side == "left":
        arm_joint_ids = ctx["left_arm_joint_ids"]
        gripper_joint_ids = ctx["left_gripper_joint_ids"]
        ee_body_id = ctx["left_ee_body_id"]
        target_pos_w, _, target_quat_w, _ = compute_actor_collision_target_poses_w(env, ctx)
    else:
        arm_joint_ids = ctx["right_arm_joint_ids"]
        gripper_joint_ids = ctx["right_gripper_joint_ids"]
        ee_body_id = ctx["right_ee_body_id"]
        _, target_pos_w, _, target_quat_w = compute_actor_collision_target_poses_w(env, ctx)

    robot_root_pos = read_data(env.robot.data, "root_pos_w")[:, 0:3]
    robot_root_quat = read_data(env.robot.data, "root_quat_w")
    ee_pos_w = read_data(env.robot.data, "body_state_w")[:, ee_body_id, 0:3]
    ee_quat_w = read_data(env.robot.data, "body_state_w")[:, ee_body_id, 3:7]

    ee_pos_b, ee_quat_b = subtract_frame_transforms(robot_root_pos, robot_root_quat, ee_pos_w, ee_quat_w)
    target_delta_b = quat_apply(quat_inv(robot_root_quat), target_pos_w - ee_pos_w)
    target_quat_error = quat_error_magnitude(ee_quat_w, target_quat_w).unsqueeze(-1)

    gripper_opening = read_data(env.robot.data, "joint_pos")[:, gripper_joint_ids].mean(dim=-1, keepdim=True)
    joint_pos = read_data(env.robot.data, "joint_pos")[:, arm_joint_ids]
    joint_vel = read_data(env.robot.data, "joint_vel")[:, arm_joint_ids]
    closure = _closure_from_opening(env, gripper_opening)
    ee_lin_vel_b = _ee_linear_velocity_b(env, ee_body_id)

    obs_parts = [
        ee_pos_b,
        ee_quat_b,
        gripper_opening,
        joint_pos,
        joint_vel,
        target_delta_b,
        target_quat_error,
        closure,
        ee_lin_vel_b,
    ]

    if getattr(env.cfg, "observe_control_state", False):
        obs_parts.append(control_state_observation(env, side, joint_pos))
    own = torch.nan_to_num(torch.cat(obs_parts, dim=-1), nan=0.0, posinf=0.0, neginf=0.0)
    extra_camera_dim = int(getattr(env.cfg, "camera_extra_observation_dim", 0))
    if extra_camera_dim:
        extra = env.get_additional_actor_camera_observation(side)
        if extra.shape != (env.num_envs, extra_camera_dim):
            raise RuntimeError("Additional camera observation has an invalid shape")
        own = torch.cat([own, extra], dim=-1)
    expected_dim = int(getattr(env, "own_observation_dim", getattr(env.cfg, "own_observation_dim", 30)))
    if own.shape[-1] != expected_dim:
        raise RuntimeError(f"OpenArmRE own_obs must be {expected_dim}D, got {own.shape[-1]}D.")
    return own


def _append_partner_message(env, own_obs: torch.Tensor, agent: str) -> torch.Tensor:
    """Append the partner communication message saved from the previous step."""

    width = int(getattr(env, "communication_feature_dim", getattr(env, "intent_feature_dim", 0)))
    horizon = int(getattr(env, "intent_horizon", getattr(env.cfg, "intent_horizon", 1)))
    flat_width = horizon * width
    if flat_width <= 0:
        return own_obs

    message_val = torch.zeros((env.num_envs, flat_width), device=env.device, dtype=own_obs.dtype)
    if getattr(env, "sharing_mode", "motion_context_share") != "no_share" and hasattr(env, "communication_buffer"):
        key = "right_intent" if agent == "left_arm" else "left_intent"
        if key in env.communication_buffer:
            message_val = env.communication_buffer[key].reshape(env.num_envs, flat_width).to(dtype=own_obs.dtype)

    actor_obs = torch.cat([own_obs, message_val], dim=-1)
    expected_dim = int(getattr(env, "actor_input_dim", getattr(env.cfg, "actor_input_dim", actor_obs.shape[-1])))
    if actor_obs.shape[-1] != expected_dim:
        raise RuntimeError(f"OpenArmRE actor obs must be {expected_dim}D, got {actor_obs.shape[-1]}D.")
    return torch.nan_to_num(actor_obs, nan=0.0, posinf=0.0, neginf=0.0)


def compute_openarm_re_observations(env, ctx: dict) -> dict[str, torch.Tensor]:
    """Return per-agent actor observations.

    none: own_obs only
    otherwise: own_obs + partner communication message
    """

    left_own = _agent_observation(env, ctx, "left")
    right_own = _agent_observation(env, ctx, "right")
    return {
        "left_arm": _append_partner_message(env, left_own, "left_arm"),
        "right_arm": _append_partner_message(env, right_own, "right_arm"),
    }


def compute_openarm_re_state(env, ctx: dict) -> torch.Tensor:
    """Centralized critic state without contact/force privileged signals."""

    left_own = _agent_observation(env, ctx, "left")
    right_own = _agent_observation(env, ctx, "right")
    object_state = torch.cat(
        [
            read_data(env.object.data, "root_pos_w")[:, 0:3],
            read_data(env.object.data, "root_quat_w"),
            read_data(env.object.data, "root_lin_vel_w"),
            read_data(env.object.data, "root_ang_vel_w"),
        ],
        dim=-1,
    )
    return torch.nan_to_num(
        torch.cat([left_own, right_own, object_state], dim=-1),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )


def _velocity_motion_intent(
    ee_lin_vel_b: torch.Tensor,
    control_dt: float,
    horizon: int,
    scale: float,
) -> torch.Tensor:
    """Deterministic short-horizon EE displacement in robot base frame."""

    return signed_motion_intent(ee_lin_vel_b, control_dt, horizon, scale)


def compute_openarm_re_motion_prediction(env, ctx: dict) -> dict[str, torch.Tensor]:
    """Return deterministic 3D base-frame signed EE motion prediction."""

    control_dt = float(getattr(env, "step_dt", float(env.cfg.sim.dt) * int(env.cfg.decimation)))
    horizon = int(getattr(env.cfg, "motion_intent_horizon", 15))
    scale = float(getattr(env.cfg, "interaction_motion_scale", 0.05))
    robot_root_quat = read_data(env.robot.data, "root_quat_w")
    left_vel_w = read_data(env.robot.data, "body_state_w")[:, ctx["left_ee_body_id"], 7:10]
    right_vel_w = read_data(env.robot.data, "body_state_w")[:, ctx["right_ee_body_id"], 7:10]
    left_vel_b = quat_apply(quat_inv(robot_root_quat), left_vel_w)
    right_vel_b = quat_apply(quat_inv(robot_root_quat), right_vel_w)
    return {
        "left_arm": torch.nan_to_num(
            _velocity_motion_intent(left_vel_b, control_dt, horizon, scale),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        ),
        "right_arm": torch.nan_to_num(
            _velocity_motion_intent(right_vel_b, control_dt, horizon, scale),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        ),
    }


def get_agent_action_dim(env) -> int:
    """Return the per-agent action width from live env action tensors."""

    actions = getattr(env, "actions", None)
    if isinstance(actions, dict):
        action = actions.get("left_arm")
        if isinstance(action, torch.Tensor):
            return int(action.shape[-1])

    action_manager = getattr(env, "action_manager", None)
    if action_manager is not None and isinstance(getattr(action_manager, "action", None), torch.Tensor):
        return int(action_manager.action.shape[-1]) // 2

    raise RuntimeError("Unable to determine per-agent action dimension.")


def get_previous_agent_action(env, agent: str, action_dim: int) -> torch.Tensor:
    """Return the previous raw action for one agent."""

    override = getattr(env, "_communication_previous_action_override", None)
    if isinstance(override, dict) and isinstance(override.get(agent), torch.Tensor):
        return override[agent]

    prev_actions = getattr(env, "prev_actions", None)
    if isinstance(prev_actions, dict):
        previous = prev_actions.get(agent)
        if isinstance(previous, torch.Tensor):
            return previous

    prev_flat = getattr(env, "_prev_action", None)
    if isinstance(prev_flat, torch.Tensor) and prev_flat.shape[0] == env.num_envs:
        start = 0 if agent == "left_arm" else int(action_dim)
        end = start + int(action_dim)
        if prev_flat.shape[-1] >= end:
            return prev_flat[:, start:end]

    return torch.zeros((env.num_envs, action_dim), device=env.device)


def _current_agent_action(env, agent: str, action_dim: int) -> torch.Tensor:
    override = getattr(env, "_communication_action_override", None)
    if isinstance(override, dict) and isinstance(override.get(agent), torch.Tensor):
        return override[agent]

    actions = getattr(env, "actions", None)
    if isinstance(actions, dict):
        current = actions.get(agent)
        if isinstance(current, torch.Tensor):
            return current

    action_manager = getattr(env, "action_manager", None)
    if action_manager is not None and isinstance(getattr(action_manager, "action", None), torch.Tensor):
        start = 0 if agent == "left_arm" else int(action_dim)
        return action_manager.action[:, start : start + int(action_dim)]

    return torch.zeros((env.num_envs, action_dim), device=env.device)


def _agent_action_rate(env, agent: str, action_dim: int) -> torch.Tensor:
    """Return mean squared action change for one agent."""

    current = _current_agent_action(env, agent, action_dim)
    previous = get_previous_agent_action(env, agent, action_dim).to(device=current.device, dtype=current.dtype)
    return (current - previous).square().mean(dim=-1)


def _full_action_variation(env, agent: str, action_dim: int) -> torch.Tensor:
    """Return RMS variation of the complete per-agent action vector."""

    current = _current_agent_action(env, agent, action_dim)
    previous = get_previous_agent_action(env, agent, action_dim)
    return full_action_variation(current, previous)


def compute_openarm_re_motion_contexts(env, ctx: dict) -> dict[str, torch.Tensor]:
    """Return the selected task-agnostic 3D context for each arm."""

    current_step = int(getattr(env, "common_step_counter", 0))
    if getattr(env, "_motion_context_cache_step", -1) == current_step:
        cached = getattr(env, "_motion_context_cache", None)
        if isinstance(cached, dict):
            return cached

    def raw_for(agent: str, ee_body_id: int, action_dim: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ee_state = read_data(env.robot.data, "body_state_w")[:, ee_body_id]
        lin_speed = torch.linalg.vector_norm(ee_state[:, 7:10], dim=-1, keepdim=True)
        ang_speed = torch.linalg.vector_norm(ee_state[:, 10:13], dim=-1, keepdim=True)
        action_variation = _full_action_variation(env, agent, action_dim)
        return lin_speed, ang_speed, action_variation

    action_dim = get_agent_action_dim(env)
    left_raw = raw_for("left_arm", ctx["left_ee_body_id"], action_dim)
    right_raw = raw_for("right_arm", ctx["right_ee_body_id"], action_dim)

    if not hasattr(env, "_motion_context_action_variation_valid"):
        env._motion_context_action_variation_valid = torch.zeros((env.num_envs, 2), device=env.device, dtype=torch.bool)
    effective_variations = []
    for side_idx, raw in enumerate((left_raw, right_raw)):
        valid = env._motion_context_action_variation_valid[:, side_idx : side_idx + 1]
        effective_variations.append(torch.where(valid, raw[2], torch.zeros_like(raw[2])))
        env._motion_context_action_variation_valid[:, side_idx : side_idx + 1] = True
    left_raw = (left_raw[0], left_raw[1], effective_variations[0])
    right_raw = (right_raw[0], right_raw[1], effective_variations[1])

    trend = update_motion_trend_context(env, left_raw, right_raw)
    ema = env._motion_context_trend_ema
    lagged = env._motion_context_trend_lagged_ema
    representation = str(getattr(env.cfg, "motion_context_representation", "trend"))
    motion_context_layout(representation)
    fixed_scale = torch.tensor(
        [
            float(getattr(env.cfg, "motion_context_lin_scale_init", 0.10)),
            float(getattr(env.cfg, "motion_context_ang_scale_init", 0.50)),
            float(getattr(env.cfg, "motion_context_action_scale_init", 0.50)),
        ],
        device=env.device,
        dtype=ema.dtype,
    ).clamp_min(1.0e-6)
    root_quat_inv = quat_inv(read_data(env.robot.data, "root_quat_w"))
    result = {}
    for side_idx, (agent, ee_body_id, raw) in enumerate(
        (("left_arm", ctx["left_ee_body_id"], left_raw), ("right_arm", ctx["right_ee_body_id"], right_raw))
    ):
        angular_velocity_w = read_data(env.robot.data, "body_state_w")[:, ee_body_id, 10:13]
        angular_velocity_b = quat_apply(root_quat_inv, angular_velocity_w)
        result[agent] = select_motion_context(
            representation,
            trend[agent],
            raw[0],
            raw[1],
            ema[:, side_idx, 2:3],
            angular_velocity_b,
            fixed_scale,
            twist_angular_scale=float(getattr(env.cfg, "motion_context_twist_ang_scale", 3.0)),
        )
    env._motion_context_debug = {
        "left_linear_speed_raw": left_raw[0].detach(),
        "left_angular_speed_raw": left_raw[1].detach(),
        "left_action_variation_raw": effective_variations[0].detach(),
        "left_linear_speed_ema": ema[:, 0, 0:1].detach(),
        "left_angular_speed_ema": ema[:, 0, 1:2].detach(),
        "left_action_variation_ema": ema[:, 0, 2:3].detach(),
        "left_trend_reference": lagged[:, 0].detach(),
        "right_linear_speed_raw": right_raw[0].detach(),
        "right_angular_speed_raw": right_raw[1].detach(),
        "right_action_variation_raw": effective_variations[1].detach(),
        "right_linear_speed_ema": ema[:, 1, 0:1].detach(),
        "right_angular_speed_ema": ema[:, 1, 1:2].detach(),
        "right_action_variation_ema": ema[:, 1, 2:3].detach(),
        "right_trend_reference": lagged[:, 1].detach(),
    }
    env._motion_context_cache = result
    env._motion_context_cache_step = current_step
    return result


def _build_current_communication_messages(
    env,
    ctx: dict,
    own_obs: dict[str, torch.Tensor] | None = None,
) -> dict[str, torch.Tensor]:
    """Return each arm's outgoing message according to communication_mode."""

    mode = str(getattr(env, "communication_mode", getattr(env.cfg, "communication_mode", "motion_context")))
    width = int(getattr(env, "communication_feature_dim", getattr(env.cfg, "communication_feature_dim", 0)))
    if width <= 0:
        zeros = torch.zeros((env.num_envs, 0), device=env.device)
        return {"left_arm": zeros, "right_arm": zeros}
    needs_motion = mode == "motion_only" or mode in MOTION_CONTEXT_MODES
    needs_context = mode == "context_only" or mode in MOTION_CONTEXT_MODES
    motion = compute_openarm_re_motion_prediction(env, ctx) if needs_motion else {}
    context = compute_openarm_re_motion_contexts(env, ctx) if needs_context else {}
    context_visible = motion_context_sharing_active(env) if needs_context else False
    if own_obs is None and mode == "full_partner_observation":
        own_obs = {
            "left_arm": _agent_observation(env, ctx, "left"),
            "right_arm": _agent_observation(env, ctx, "right"),
        }
    action_dim = get_agent_action_dim(env) if mode == "previous_action" else 0

    def pad_slot(payload: torch.Tensor) -> torch.Tensor:
        return pad_message(payload, width, mode)

    messages: dict[str, torch.Tensor] = {}
    for agent in ("left_arm", "right_arm"):
        if mode == "none":
            message = torch.zeros((env.num_envs, 0), device=env.device)
        elif mode == "motion_only":
            message = motion[agent]
        elif mode == "context_only":
            message = context[agent] if context_visible else torch.zeros_like(context[agent])
        elif mode in MOTION_CONTEXT_MODES:
            visible_context = context[agent] if context_visible else torch.zeros_like(context[agent])
            message = motion_context_payload(motion[agent], visible_context, mode)
        elif mode == "previous_action":
            message = get_previous_agent_action(env, agent, action_dim).to(device=env.device, dtype=torch.float32)
        elif mode == "full_partner_observation":
            assert own_obs is not None
            message = own_obs[agent]
        else:
            raise ValueError(f"Unsupported communication_mode={mode!r}.")
        messages[agent] = torch.nan_to_num(pad_slot(message), nan=0.0, posinf=0.0, neginf=0.0)
    return messages


def compute_openarm_re_coordination_messages(env, ctx: dict) -> dict[str, torch.Tensor]:
    """Return each arm's outgoing communication message."""

    return _build_current_communication_messages(env, ctx)


def _object_tilt(env, object_quat_w: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Return object tilt angle in degrees and normalized squared penalty."""

    if object_quat_w is None:
        object_quat_w = read_data(env.object.data, "root_quat_w")
    world_z = torch.tensor([0.0, 0.0, 1.0], device=env.device).expand(env.num_envs, -1)
    object_up = matrix_from_quat(object_quat_w)[:, :, 2]
    cos_tilt = torch.clamp((object_up * world_z).sum(dim=-1), -1.0, 1.0)
    tilt_deg = torch.rad2deg(torch.acos(cos_tilt))
    free = float(env.cfg.tilt_free_deg)
    bad = max(float(env.cfg.tilt_bad_deg), free + 1.0e-6)
    tilt_penalty = torch.clamp((tilt_deg - free) / (bad - free), 0.0, 1.0).square()
    return tilt_deg, tilt_penalty


def _quat_to_rpy_deg(quat_wxyz: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert wxyz quaternion to roll, pitch, yaw in degrees."""

    w, x, y, z = quat_wxyz.unbind(dim=-1)
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = torch.atan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (w * y - z * x)
    pitch = torch.asin(torch.clamp(sinp, -1.0, 1.0))

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = torch.atan2(siny_cosp, cosy_cosp)

    return torch.rad2deg(roll), torch.rad2deg(pitch), torch.rad2deg(yaw)


def _object_success_rpy_deg(
    env,
    ctx: dict,
    object_quat_w: torch.Tensor | None = None,
    initial_quat_w: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return object roll/pitch/yaw drift from reset pose in degrees."""

    if object_quat_w is None:
        object_quat_w = read_data(env.object.data, "root_quat_w")
    if initial_quat_w is None:
        initial_quat_w = ctx.get("initial_object_quat_w")
    if initial_quat_w is None:
        initial_quat_w = object_quat_w.clone()
        ctx["initial_object_quat_w"] = initial_quat_w
    relative_quat = _quat_mul_wxyz(object_quat_w, quat_inv(initial_quat_w))
    return _quat_to_rpy_deg(relative_quat)


def _object_success_tilt_ok(
    env,
    ctx: dict,
    object_quat_w: torch.Tensor | None = None,
    initial_quat_w: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Check local object up-axis against world vertical; keep RPY as diagnostics."""

    roll_deg, pitch_deg, yaw_deg = _object_success_rpy_deg(env, ctx, object_quat_w, initial_quat_w)
    tilt_deg, _ = _object_tilt(env, object_quat_w)
    tilt_ok = tilt_deg <= float(env.cfg.success_tilt_threshold_deg)
    return tilt_ok, tilt_deg, roll_deg, pitch_deg, yaw_deg


def _target_inside_gripper_score(
    target_pos_w: torch.Tensor,
    inner_finger_pos_w: torch.Tensor,
    outer_finger_pos_w: torch.Tensor,
    position_scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return a contact-free score for a target lying between the fingers."""

    finger_span = outer_finger_pos_w - inner_finger_pos_w
    span_len_sq = finger_span.square().sum(dim=-1, keepdim=True).clamp_min(1.0e-8)
    target_from_inner = target_pos_w - inner_finger_pos_w
    span_position = (target_from_inner * finger_span).sum(dim=-1, keepdim=True) / span_len_sq
    closest_point = inner_finger_pos_w + span_position.clamp(0.0, 1.0) * finger_span
    midline_dist = torch.linalg.vector_norm(target_pos_w - closest_point, dim=-1)

    span_position = span_position.squeeze(-1)
    inside_span = ((span_position >= 0.0) & (span_position <= 1.0)).to(midline_dist.dtype)
    centered = torch.exp(-midline_dist / max(position_scale, 1.0e-6))
    return inside_span * centered, inside_span, centered, midline_dist


def _success_grasp_validity(
    env,
    left_dist: torch.Tensor,
    right_dist: torch.Tensor,
    left_ori_err: torch.Tensor,
    right_ori_err: torch.Tensor,
    left_closure: torch.Tensor,
    right_closure: torch.Tensor,
    left_inside_span: torch.Tensor,
    right_inside_span: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Return the shared success-level bilateral grasp conditions."""

    left_ori_err_deg = torch.rad2deg(left_ori_err)
    right_ori_err_deg = torch.rad2deg(right_ori_err)
    orientation_limit_deg = float(env.cfg.success_grasp_orientation_deg)

    left_distance_ok = left_dist <= float(env.cfg.success_grasp_distance)
    right_distance_ok = right_dist <= float(env.cfg.success_grasp_distance)
    left_orientation_ok = left_ori_err_deg <= orientation_limit_deg
    right_orientation_ok = right_ori_err_deg <= orientation_limit_deg
    left_closure_ok = left_closure >= float(env.cfg.success_min_closure)
    right_closure_ok = right_closure >= float(env.cfg.success_min_closure)
    left_finger_span_ok = left_inside_span.bool()
    right_finger_span_ok = right_inside_span.bool()

    left_grasp_ok = left_distance_ok & left_orientation_ok & left_closure_ok & left_finger_span_ok
    right_grasp_ok = right_distance_ok & right_orientation_ok & right_closure_ok & right_finger_span_ok

    return {
        "left_ori_err_deg": left_ori_err_deg,
        "right_ori_err_deg": right_ori_err_deg,
        "left_distance_ok": left_distance_ok,
        "right_distance_ok": right_distance_ok,
        "left_orientation_ok": left_orientation_ok,
        "right_orientation_ok": right_orientation_ok,
        "left_closure_ok": left_closure_ok,
        "right_closure_ok": right_closure_ok,
        "left_finger_span_ok": left_finger_span_ok,
        "right_finger_span_ok": right_finger_span_ok,
        "left_grasp_ok": left_grasp_ok,
        "right_grasp_ok": right_grasp_ok,
        "grasp_distance_ok": left_distance_ok & right_distance_ok,
        "grasp_orientation_ok": left_orientation_ok & right_orientation_ok,
        "closure_ok": left_closure_ok & right_closure_ok,
        "finger_span_ok": left_finger_span_ok & right_finger_span_ok,
        "grasp_ok": left_grasp_ok & right_grasp_ok,
    }


def compute_openarm_re_rewards(env, ctx: dict) -> dict[str, torch.Tensor]:
    """Compute the shared contact-free bimanual lift reward.

    The reward deliberately contains only four task concepts:

      1. Pose: mean independent EE approach/alignment to the two grasp poses.
      2. Grasp: per-arm pose quality, finger geometry, and measured closure.
      3. Lift: object elevation gated by bilateral grasp readiness and closure.
      4. Success: one-shot bonus after a stable held lift.

    No contact/force signal, attachment estimator, temporal progress gate, or
    hand-authored stage machine is used. The strict success condition separately
    requires each target to lie between its gripper fingers.
    """

    robot = ctx["robot"]
    obj = ctx["object"]

    # AprilTag-derived actor targets are used for reward and deployment.
    left_target_w, right_target_w, left_target_q_w, right_target_q_w = compute_collision_target_poses_w(env, ctx)
    (
        left_actor_target_w,
        right_actor_target_w,
        left_actor_target_q_w,
        right_actor_target_q_w,
    ) = compute_actor_collision_target_poses_w(env, ctx)

    left_ee_pos_w = read_data(robot.data, "body_pos_w")[:, ctx["left_ee_body_id"]]
    right_ee_pos_w = read_data(robot.data, "body_pos_w")[:, ctx["right_ee_body_id"]]
    left_ee_quat_w = read_data(robot.data, "body_quat_w")[:, ctx["left_ee_body_id"]]
    right_ee_quat_w = read_data(robot.data, "body_quat_w")[:, ctx["right_ee_body_id"]]

    object_pos_gt_w = read_data(obj.data, "root_pos_w")[:, :3]
    object_quat_gt_w = read_data(obj.data, "root_quat_w")
    object_pos_w, object_quat_w = get_actor_object_pose_w(env, ctx)

    # On the first control step after reset, anchor height/tilt references to the
    # same actor-side object estimate used during the episode.
    first_step = ~ctx["reward_temporal_valid"]
    ctx["initial_actor_object_pos_w"][first_step] = object_pos_w[first_step].detach()
    ctx["initial_actor_object_quat_w"][first_step] = object_quat_w[first_step].detach()

    # 1) Bilateral pose approach. Linear angular quality avoids the vanishing
    # gradient of a squared Gaussian when the initial wrist error is large.
    left_dist = torch.linalg.vector_norm(left_ee_pos_w - left_actor_target_w, dim=-1)
    right_dist = torch.linalg.vector_norm(right_ee_pos_w - right_actor_target_w, dim=-1)
    reach_std = max(float(env.cfg.reach_std), 1.0e-6)
    left_reach = 1.0 - torch.tanh(left_dist / reach_std)
    right_reach = 1.0 - torch.tanh(right_dist / reach_std)
    left_ori_err = quat_error_magnitude(left_ee_quat_w, left_actor_target_q_w)
    right_ori_err = quat_error_magnitude(right_ee_quat_w, right_actor_target_q_w)
    left_orientation = (1.0 - left_ori_err / torch.pi).clamp(0.0, 1.0)
    right_orientation = (1.0 - right_ori_err / torch.pi).clamp(0.0, 1.0)
    # Position supplies at least half of the pose signal while orientation
    # modulates the remainder. This preserves approach learning at large angular
    # error without rewarding orientation alone while the wrist is far away.
    left_approach = left_reach * (0.5 + 0.5 * left_orientation)
    right_approach = right_reach * (0.5 + 0.5 * right_orientation)
    team_reach = 0.5 * (left_reach + right_reach)
    # Keep per-arm pose shaping independent. Bilateral coupling belongs to the
    # grasp condition, not to approach timing.
    team_approach = 0.5 * (left_approach + right_approach)

    # 2) Bilateral contact-free grasp quality. EE pose alone cannot distinguish
    # a real side grasp from closing above the box, so retain the original
    # reward's physically meaningful finger-span geometry.

    left_opening = read_data(robot.data, "joint_pos")[:, ctx["left_gripper_joint_ids"]].mean(dim=-1)
    right_opening = read_data(robot.data, "joint_pos")[:, ctx["right_gripper_joint_ids"]].mean(dim=-1)
    left_closure = _closure_from_opening(env, left_opening)
    right_closure = _closure_from_opening(env, right_opening)

    grasp_std = max(float(env.cfg.grasp_position_std), 1.0e-6)
    left_near = torch.exp(-left_dist / grasp_std)
    right_near = torch.exp(-right_dist / grasp_std)

    left_inner_finger_pos_w = read_data(robot.data, "body_pos_w")[:, ctx["left_inner_finger_body_id"]]
    left_outer_finger_pos_w = read_data(robot.data, "body_pos_w")[:, ctx["left_outer_finger_body_id"]]
    right_inner_finger_pos_w = read_data(robot.data, "body_pos_w")[:, ctx["right_inner_finger_body_id"]]
    right_outer_finger_pos_w = read_data(robot.data, "body_pos_w")[:, ctx["right_outer_finger_body_id"]]
    (
        left_target_inside_gripper,
        left_target_inside_span,
        left_target_centered_in_gripper,
        left_target_to_gripper_midline_dist,
    ) = _target_inside_gripper_score(
        left_actor_target_w,
        left_inner_finger_pos_w,
        left_outer_finger_pos_w,
        grasp_std,
    )
    (
        right_target_inside_gripper,
        right_target_inside_span,
        right_target_centered_in_gripper,
        right_target_to_gripper_midline_dist,
    ) = _target_inside_gripper_score(
        right_actor_target_w,
        right_inner_finger_pos_w,
        right_outer_finger_pos_w,
        grasp_std,
    )

    # Contact-free pose readiness uses the deployment-observable TCP target
    # pose. Finger body origins are not fingertip contact points, so their
    # segment distance is retained only as a diagnostic and strict span
    # validity check; using it here can falsely suppress closure at the rim.
    left_pose_readiness = (left_near * left_orientation).clamp(0.0, 1.0)
    right_pose_readiness = (right_near * right_orientation).clamp(0.0, 1.0)

    # Compare valid and premature closure linearly. The former square-root
    # quality created an interior optimum at partial closure whenever readiness
    # was modest. This form has no partial-closure optimum: low readiness favors
    # opening, while high readiness favors closing, without a discrete stage.
    left_grasp_feasibility = (left_pose_readiness * left_closure).clamp(0.0, 1.0)
    right_grasp_feasibility = (right_pose_readiness * right_closure).clamp(0.0, 1.0)
    # Keep the established trace keys, now reporting raw feasibility rather
    # than the removed square-root shaping value.
    left_grasp_quality = left_grasp_feasibility
    right_grasp_quality = right_grasp_feasibility
    left_premature_closure = (1.0 - left_pose_readiness) * left_closure
    right_premature_closure = (1.0 - right_pose_readiness) * right_closure
    closure_penalty_scale = float(env.cfg.premature_closure_penalty_scale)
    left_grasp = left_grasp_feasibility - closure_penalty_scale * left_premature_closure
    right_grasp = right_grasp_feasibility - closure_penalty_scale * right_premature_closure
    team_grasp = 0.5 * (left_grasp + right_grasp)

    # 3) Object-centric tilt-aware lift. The weaker arm limits object-level
    # lift credit without adding synchronization or timing shaping.
    height_delta = object_pos_w[:, 2] - ctx["initial_actor_object_pos_w"][:, 2]
    height_delta_gt = object_pos_gt_w[:, 2] - ctx["initial_object_pos_w"][:, 2]
    lift = torch.clamp(
        height_delta / max(float(env.cfg.lift_target_height), 1.0e-6),
        0.0,
        1.0,
    )

    world_z = torch.tensor([0.0, 0.0, 1.0], device=env.device).expand(env.num_envs, -1)
    object_up = quat_apply(object_quat_w, world_z)
    cos_tilt = torch.clamp(object_up[:, 2], -1.0, 1.0)
    tilt_deg = torch.rad2deg(torch.acos(cos_tilt))
    tilt_free = float(env.cfg.tilt_free_deg)
    tilt_bad = max(float(env.cfg.tilt_bad_deg), tilt_free + 1.0e-6)
    tilt_penalty = torch.clamp((tilt_deg - tilt_free) / (tilt_bad - tilt_free), 0.0, 1.0).square()
    tilt_quality = (1.0 - tilt_penalty).clamp(0.0, 1.0).square()
    lift_gate = torch.minimum(left_grasp_feasibility, right_grasp_feasibility)
    lift_quality = lift_gate * lift * tilt_quality

    # 4) Standard action-rate and joint-velocity regularization.
    action_dim = get_agent_action_dim(env)
    left_action_rate = _agent_action_rate(env, "left_arm", action_dim)
    right_action_rate = _agent_action_rate(env, "right_arm", action_dim)
    team_action_rate = 0.5 * (left_action_rate + right_action_rate)
    left_joint_vel_penalty = read_data(robot.data, "joint_vel")[:, ctx["left_arm_joint_ids"]].square().mean(dim=-1)
    right_joint_vel_penalty = read_data(robot.data, "joint_vel")[:, ctx["right_arm_joint_ids"]].square().mean(dim=-1)
    team_joint_vel_penalty = 0.5 * (left_joint_vel_penalty + right_joint_vel_penalty)

    # Success is stricter than the shaping reward, but still simple and
    # deployment-observable: height and tilt plus bilateral grasp distance,
    # orientation, closure, and finger-span validity, held for N control steps.
    height_ok = height_delta > float(env.cfg.success_height_margin)
    tilt_ok, success_tilt_deg, success_roll_deg, success_pitch_deg, success_yaw_deg = _object_success_tilt_ok(
        env,
        ctx,
        object_quat_w,
        ctx["initial_actor_object_quat_w"],
    )
    success_roll_pitch_deg = torch.maximum(success_roll_deg.abs(), success_pitch_deg.abs())
    grasp_validity = _success_grasp_validity(
        env,
        left_dist,
        right_dist,
        left_ori_err,
        right_ori_err,
        left_closure,
        right_closure,
        left_target_inside_span,
        right_target_inside_span,
    )
    grasp_distance_ok = grasp_validity["grasp_distance_ok"]
    grasp_orientation_ok = grasp_validity["grasp_orientation_ok"]
    closure_ok = grasp_validity["closure_ok"]
    finger_span_ok = grasp_validity["finger_span_ok"]
    grasp_ok = grasp_validity["grasp_ok"]
    success_now = height_ok & tilt_ok & grasp_ok
    # DirectMARLEnv evaluates termination before rewards. Use that transition's
    # result, and keep trace storage independent from the termination mask.
    terminal_success = ctx["termination_success"].clone()

    reward_reach_term = float(env.cfg.reward_reach_scale) * team_approach
    reward_grasp_term = float(env.cfg.reward_grasp_scale) * team_grasp
    reward_lift_term = float(env.cfg.reward_lift_scale) * lift_quality
    reward_success_term = float(env.cfg.reward_success_scale) * terminal_success.float()
    reward_action_term = float(env.cfg.reward_action_rate_scale) * team_action_rate
    reward_joint_vel_term = float(env.cfg.reward_joint_vel_scale) * team_joint_vel_penalty

    team_reward = (
        reward_reach_term
        + reward_grasp_term
        + reward_lift_term
        + reward_success_term
        - reward_action_term
        - reward_joint_vel_term
    )
    team_reward = torch.nan_to_num(team_reward, nan=0.0, posinf=0.0, neginf=0.0)

    # Per-arm values are diagnostics only. The policy receives team_reward.
    shared_object_reward = reward_lift_term + reward_success_term
    left_reward_raw = (
        float(env.cfg.reward_reach_scale) * left_approach
        + float(env.cfg.reward_grasp_scale) * left_grasp
        + shared_object_reward
        - float(env.cfg.reward_action_rate_scale) * left_action_rate
        - float(env.cfg.reward_joint_vel_scale) * left_joint_vel_penalty
    )
    right_reward_raw = (
        float(env.cfg.reward_reach_scale) * right_approach
        + float(env.cfg.reward_grasp_scale) * right_grasp
        + shared_object_reward
        - float(env.cfg.reward_action_rate_scale) * right_action_rate
        - float(env.cfg.reward_joint_vel_scale) * right_joint_vel_penalty
    )
    left_reward_raw = torch.nan_to_num(left_reward_raw, nan=0.0, posinf=0.0, neginf=0.0)
    right_reward_raw = torch.nan_to_num(right_reward_raw, nan=0.0, posinf=0.0, neginf=0.0)

    # Diagnostics / trace compatibility. Legacy attachment fields are fixed to
    # one because attachment is no longer a condition or reward factor.
    dual_grasp = torch.minimum(left_grasp_feasibility, right_grasp_feasibility)
    dual_grasp_quality = torch.minimum(left_grasp_quality, right_grasp_quality)
    neutral = torch.ones_like(team_reward)
    y_error = torch.abs(object_pos_w[:, 1] - ctx["target_object_xy_w"][:, 1])

    left_actor_target_pos_error = torch.linalg.vector_norm(left_actor_target_w - left_target_w, dim=-1)
    right_actor_target_pos_error = torch.linalg.vector_norm(right_actor_target_w - right_target_w, dim=-1)
    left_actor_target_rot_error = quat_error_magnitude(left_actor_target_q_w, left_target_q_w)
    right_actor_target_rot_error = quat_error_magnitude(right_actor_target_q_w, right_target_q_w)

    ctx["reward_temporal_valid"].fill_(True)
    ctx["trace_signals"] = {
        "left_grasp_hint": left_grasp.detach(),
        "right_grasp_hint": right_grasp.detach(),
        "left_grasp_quality": left_grasp_quality.detach(),
        "right_grasp_quality": right_grasp_quality.detach(),
        "left_grasp_feasibility": left_grasp_feasibility.detach(),
        "right_grasp_feasibility": right_grasp_feasibility.detach(),
        "dual_grasp_quality": dual_grasp_quality.detach(),
        "left_premature_closure": left_premature_closure.detach(),
        "right_premature_closure": right_premature_closure.detach(),
        "dual_grasp_gate": dual_grasp.detach(),
        "left_attachment": neutral.detach(),
        "right_attachment": neutral.detach(),
        "dual_attachment": neutral.detach(),
        # Legacy name retained, now reporting the actual bilateral lift gate.
        "dual_lift_gate": lift_gate.detach(),
        "lift_gate": lift_gate.detach(),
        "bilateral_grasp_quality": team_grasp.detach(),
        "object_lift_quality": lift_quality.detach(),
        "reach_progress": team_approach.detach(),  # legacy trace name: pose approach
        "height_progress": lift.detach(),  # legacy alias: now normalized height
        "tilt_aware_lift": lift_quality.detach(),
        "terminal_success": terminal_success.detach(),
        "object_tilt_deg": tilt_deg.detach(),
        "success_tilt_deg": success_tilt_deg.detach(),
        "success_roll_deg": success_roll_deg.detach(),
        "success_pitch_deg": success_pitch_deg.detach(),
        "success_roll_pitch_deg": success_roll_pitch_deg.detach(),
        "success_yaw_deg": success_yaw_deg.detach(),
        "xy_error": y_error.detach(),
        "left_reward": left_reward_raw.detach(),
        "right_reward": right_reward_raw.detach(),
        "grasp_distance_ok": grasp_distance_ok.detach(),
        "left_distance_ok": grasp_validity["left_distance_ok"].detach(),
        "right_distance_ok": grasp_validity["right_distance_ok"].detach(),
        "left_orientation_ok": grasp_validity["left_orientation_ok"].detach(),
        "right_orientation_ok": grasp_validity["right_orientation_ok"].detach(),
        "grasp_orientation_ok": grasp_orientation_ok.detach(),
        "left_ori_err_deg": grasp_validity["left_ori_err_deg"].detach(),
        "right_ori_err_deg": grasp_validity["right_ori_err_deg"].detach(),
        "left_closure_ok": grasp_validity["left_closure_ok"].detach(),
        "right_closure_ok": grasp_validity["right_closure_ok"].detach(),
        "left_finger_span_ok": grasp_validity["left_finger_span_ok"].detach(),
        "right_finger_span_ok": grasp_validity["right_finger_span_ok"].detach(),
        "left_grasp_ok": grasp_validity["left_grasp_ok"].detach(),
        "right_grasp_ok": grasp_validity["right_grasp_ok"].detach(),
        "closure_ok": closure_ok.detach(),
        "finger_span_ok": finger_span_ok.detach(),
        "grasp_ok": grasp_ok.detach(),
    }

    previous_debug_stats = dict(ctx.get("debug_stats", {}))
    ctx["debug_stats"] = {
        **previous_debug_stats,
        "team_reward": team_reward.mean().item(),
        "left_reward": left_reward_raw.mean().item(),
        "right_reward": right_reward_raw.mean().item(),
        "left_reward_raw": left_reward_raw.mean().item(),
        "right_reward_raw": right_reward_raw.mean().item(),
        "reward_reach_term": reward_reach_term.mean().item(),
        "reward_approach_term": reward_reach_term.mean().item(),
        "reward_grasp_term": reward_grasp_term.mean().item(),
        "reward_lift_term": reward_lift_term.mean().item(),
        "reward_success_term": reward_success_term.mean().item(),
        "reward_action_term": reward_action_term.mean().item(),
        "reward_joint_vel_term": reward_joint_vel_term.mean().item(),
        "paper/progress": (0.5 * team_approach + 0.5 * team_grasp).mean().item(),
        "paper/quality": lift_quality.mean().item(),
        "paper/success": terminal_success.float().mean().item(),
        "paper/terminal_success_bonus": reward_success_term.mean().item(),
        "terminal_success_bonus": reward_success_term.mean().item(),
        "paper/regularization": (reward_action_term + reward_joint_vel_term).mean().item(),
        "left_reach": left_reach.mean().item(),
        "right_reach": right_reach.mean().item(),
        "reach": team_reach.mean().item(),
        "left_approach": left_approach.mean().item(),
        "right_approach": right_approach.mean().item(),
        "approach": team_approach.mean().item(),
        "reach_progress": team_approach.mean().item(),
        "left_dist": left_dist.mean().item(),
        "right_dist": right_dist.mean().item(),
        "left_actor_dist": left_dist.mean().item(),
        "right_actor_dist": right_dist.mean().item(),
        "bottleneck_distance": torch.maximum(left_dist, right_dist).mean().item(),
        "left_actor_target_pos_error": left_actor_target_pos_error.mean().item(),
        "right_actor_target_pos_error": right_actor_target_pos_error.mean().item(),
        "left_actor_target_pos_error_max": left_actor_target_pos_error.max().item(),
        "right_actor_target_pos_error_max": right_actor_target_pos_error.max().item(),
        "left_actor_target_rot_error_deg": torch.rad2deg(left_actor_target_rot_error).mean().item(),
        "right_actor_target_rot_error_deg": torch.rad2deg(right_actor_target_rot_error).mean().item(),
        "left_target_local_x": ctx["left_target_local_pos"][:, 0].mean().item(),
        "left_target_local_y": ctx["left_target_local_pos"][:, 1].mean().item(),
        "left_target_local_z": ctx["left_target_local_pos"][:, 2].mean().item(),
        "right_target_local_x": ctx["right_target_local_pos"][:, 0].mean().item(),
        "right_target_local_y": ctx["right_target_local_pos"][:, 1].mean().item(),
        "right_target_local_z": ctx["right_target_local_pos"][:, 2].mean().item(),
        "sample_object_x": object_pos_w[0, 0].item(),
        "sample_object_y": object_pos_w[0, 1].item(),
        "sample_object_z": object_pos_w[0, 2].item(),
        "sample_object_gt_x": object_pos_gt_w[0, 0].item(),
        "sample_object_gt_y": object_pos_gt_w[0, 1].item(),
        "sample_object_gt_z": object_pos_gt_w[0, 2].item(),
        "sample_object_rot_error_deg": torch.rad2deg(quat_error_magnitude(object_quat_w, object_quat_gt_w))
        .mean()
        .item(),
        "sample_left_actor_target_x": left_actor_target_w[0, 0].item(),
        "sample_left_actor_target_y": left_actor_target_w[0, 1].item(),
        "sample_left_actor_target_z": left_actor_target_w[0, 2].item(),
        "sample_right_actor_target_x": right_actor_target_w[0, 0].item(),
        "sample_right_actor_target_y": right_actor_target_w[0, 1].item(),
        "sample_right_actor_target_z": right_actor_target_w[0, 2].item(),
        "left_orientation": left_orientation.mean().item(),
        "right_orientation": right_orientation.mean().item(),
        "orientation": (0.5 * (left_orientation + right_orientation)).mean().item(),
        "left_ori_err": left_ori_err.mean().item(),
        "right_ori_err": right_ori_err.mean().item(),
        "left_ori_err_deg": grasp_validity["left_ori_err_deg"].mean().item(),
        "right_ori_err_deg": grasp_validity["right_ori_err_deg"].mean().item(),
        "left_near_collision_target": left_near.mean().item(),
        "right_near_collision_target": right_near.mean().item(),
        "left_closure": left_closure.mean().item(),
        "right_closure": right_closure.mean().item(),
        "left_pose_readiness": left_pose_readiness.mean().item(),
        "right_pose_readiness": right_pose_readiness.mean().item(),
        "left_target_inside_gripper": left_target_inside_gripper.mean().item(),
        "right_target_inside_gripper": right_target_inside_gripper.mean().item(),
        "left_target_inside_finger_span": left_target_inside_span.mean().item(),
        "right_target_inside_finger_span": right_target_inside_span.mean().item(),
        "left_target_centered_in_gripper": left_target_centered_in_gripper.mean().item(),
        "right_target_centered_in_gripper": right_target_centered_in_gripper.mean().item(),
        "left_target_to_gripper_midline_dist": left_target_to_gripper_midline_dist.mean().item(),
        "right_target_to_gripper_midline_dist": right_target_to_gripper_midline_dist.mean().item(),
        "team_closure": torch.minimum(left_closure, right_closure).mean().item(),
        "grasp_hint": team_grasp.mean().item(),
        "left_grasp_hint": left_grasp.mean().item(),
        "right_grasp_hint": right_grasp.mean().item(),
        "left_grasp_hint_raw": left_grasp_quality.mean().item(),
        "right_grasp_hint_raw": right_grasp_quality.mean().item(),
        "left_grasp_quality": left_grasp_quality.mean().item(),
        "right_grasp_quality": right_grasp_quality.mean().item(),
        "left_grasp_feasibility": left_grasp_feasibility.mean().item(),
        "right_grasp_feasibility": right_grasp_feasibility.mean().item(),
        "dual_grasp_quality": dual_grasp_quality.mean().item(),
        "left_premature_closure": left_premature_closure.mean().item(),
        "right_premature_closure": right_premature_closure.mean().item(),
        "premature_closure_penalty_scale": closure_penalty_scale,
        "dual_grasp_gate": dual_grasp.mean().item(),
        "left_attachment": 1.0,
        "right_attachment": 1.0,
        "dual_attachment": 1.0,
        "attachment_active_ratio": 0.0,
        "dual_lift_gate": lift_gate.mean().item(),
        "lift_gate": lift_gate.mean().item(),
        "bilateral_grasp_quality": team_grasp.mean().item(),
        "object_lift_quality": lift_quality.mean().item(),
        "lift": lift.mean().item(),
        "height_progress": lift.mean().item(),
        "tilt_aware_lift": lift_quality.mean().item(),
        "object_height_delta": height_delta.mean().item(),
        "object_height_delta_min": height_delta.min().item(),
        "object_height_delta_max": height_delta.max().item(),
        "object_height_delta_gt": height_delta_gt.mean().item(),
        "object_height_estimation_error_max": (height_delta - height_delta_gt).abs().max().item(),
        "xy_error": y_error.mean().item(),
        "y_error": y_error.mean().item(),
        "object_tilt_deg": tilt_deg.mean().item(),
        "tilt_quality": tilt_quality.mean().item(),
        "success_tilt_deg": success_tilt_deg.mean().item(),
        "success_roll_deg": success_roll_deg.abs().mean().item(),
        "success_pitch_deg": success_pitch_deg.abs().mean().item(),
        "success_roll_pitch_deg": success_roll_pitch_deg.mean().item(),
        "success_yaw_deg": success_yaw_deg.abs().mean().item(),
        "tilt_penalty": (lift * tilt_penalty).mean().item(),
        "raw_tilt_penalty": tilt_penalty.mean().item(),
        "left_action_rate_penalty": left_action_rate.mean().item(),
        "right_action_rate_penalty": right_action_rate.mean().item(),
        "action_rate_penalty": team_action_rate.mean().item(),
        "left_joint_vel_penalty": left_joint_vel_penalty.mean().item(),
        "right_joint_vel_penalty": right_joint_vel_penalty.mean().item(),
        "joint_vel_penalty": team_joint_vel_penalty.mean().item(),
        "height_ok_ratio": height_ok.float().mean().item(),
        "tilt_ok_ratio": tilt_ok.float().mean().item(),
        "readiness_ok_ratio": grasp_ok.float().mean().item(),
        "attachment_ok_ratio": 1.0,
        "success_grasp_distance_ok_ratio": grasp_distance_ok.float().mean().item(),
        "left_distance_ok_ratio": grasp_validity["left_distance_ok"].float().mean().item(),
        "right_distance_ok_ratio": grasp_validity["right_distance_ok"].float().mean().item(),
        "success_grasp_orientation_deg": float(env.cfg.success_grasp_orientation_deg),
        "left_orientation_ok_ratio": grasp_validity["left_orientation_ok"].float().mean().item(),
        "right_orientation_ok_ratio": grasp_validity["right_orientation_ok"].float().mean().item(),
        "success_grasp_orientation_ok_ratio": grasp_orientation_ok.float().mean().item(),
        "left_grasp_ok_ratio": grasp_validity["left_grasp_ok"].float().mean().item(),
        "right_grasp_ok_ratio": grasp_validity["right_grasp_ok"].float().mean().item(),
        "dual_grasp_ok_ratio": grasp_ok.float().mean().item(),
        "success_closure_ok_ratio": closure_ok.float().mean().item(),
        "left_closure_ok_ratio": grasp_validity["left_closure_ok"].float().mean().item(),
        "right_closure_ok_ratio": grasp_validity["right_closure_ok"].float().mean().item(),
        "success_finger_span_ok_ratio": finger_span_ok.float().mean().item(),
        "left_finger_span_ok_ratio": grasp_validity["left_finger_span_ok"].float().mean().item(),
        "right_finger_span_ok_ratio": grasp_validity["right_finger_span_ok"].float().mean().item(),
        "success_grasp_ok_ratio": grasp_ok.float().mean().item(),
        "stable_now_ratio": success_now.float().mean().item(),
        "success_ratio": (ctx["success_hold_count"] >= int(env.cfg.hold_required_steps)).float().mean().item(),
        "strict_success_ratio": (ctx["success_hold_count"] >= int(env.cfg.hold_required_steps)).float().mean().item(),
        "hold_count_mean": ctx["success_hold_count"].float().mean().item(),
        "hold_count_p90": torch.quantile(ctx["success_hold_count"].float(), 0.90).item(),
        "hold_count_max": ctx["success_hold_count"].float().max().item(),
        "hold_required_steps": float(env.cfg.hold_required_steps),
    }

    return {"left_arm": team_reward, "right_arm": team_reward}


def compute_openarm_re_terminations(env, ctx: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return terminated, success, and invalid flags."""

    left_target_pos, right_target_pos, left_target_quat, right_target_quat = compute_actor_collision_target_poses_w(
        env, ctx
    )
    left_ee_pos = read_data(env.robot.data, "body_state_w")[:, ctx["left_ee_body_id"], 0:3]
    right_ee_pos = read_data(env.robot.data, "body_state_w")[:, ctx["right_ee_body_id"], 0:3]
    left_ee_quat = read_data(env.robot.data, "body_quat_w")[:, ctx["left_ee_body_id"]]
    right_ee_quat = read_data(env.robot.data, "body_quat_w")[:, ctx["right_ee_body_id"]]
    left_dist = torch.linalg.vector_norm(left_ee_pos - left_target_pos, dim=-1)
    right_dist = torch.linalg.vector_norm(right_ee_pos - right_target_pos, dim=-1)
    left_ori_err = quat_error_magnitude(left_ee_quat, left_target_quat)
    right_ori_err = quat_error_magnitude(right_ee_quat, right_target_quat)

    object_pos_w, object_quat_w = get_actor_object_pose_w(env, ctx)
    height_delta = object_pos_w[:, 2] - ctx["initial_actor_object_pos_w"][:, 2]
    tilt_deg, _ = _object_tilt(env, object_quat_w)
    height_ok = height_delta > float(env.cfg.success_height_margin)
    tilt_ok, _, _, _, _ = _object_success_tilt_ok(
        env,
        ctx,
        object_quat_w,
        ctx["initial_actor_object_quat_w"],
    )
    left_opening = read_data(env.robot.data, "joint_pos")[:, ctx["left_gripper_joint_ids"]].mean(dim=-1)
    right_opening = read_data(env.robot.data, "joint_pos")[:, ctx["right_gripper_joint_ids"]].mean(dim=-1)
    left_closure = _closure_from_opening(env, left_opening)
    right_closure = _closure_from_opening(env, right_opening)

    left_inside_span = _target_inside_gripper_score(
        left_target_pos,
        read_data(env.robot.data, "body_pos_w")[:, ctx["left_inner_finger_body_id"]],
        read_data(env.robot.data, "body_pos_w")[:, ctx["left_outer_finger_body_id"]],
        float(env.cfg.grasp_position_std),
    )[1]
    right_inside_span = _target_inside_gripper_score(
        right_target_pos,
        read_data(env.robot.data, "body_pos_w")[:, ctx["right_inner_finger_body_id"]],
        read_data(env.robot.data, "body_pos_w")[:, ctx["right_outer_finger_body_id"]],
        float(env.cfg.grasp_position_std),
    )[1]
    grasp_validity = _success_grasp_validity(
        env,
        left_dist,
        right_dist,
        left_ori_err,
        right_ori_err,
        left_closure,
        right_closure,
        left_inside_span,
        right_inside_span,
    )
    grasp_distance_ok = grasp_validity["grasp_distance_ok"]
    grasp_orientation_ok = grasp_validity["grasp_orientation_ok"]
    closure_ok = grasp_validity["closure_ok"]
    finger_span_ok = grasp_validity["finger_span_ok"]
    grasp_ok = grasp_validity["grasp_ok"]
    readiness_ok = grasp_ok
    attachment_ok = torch.ones_like(grasp_ok, dtype=torch.bool)
    hold_now = height_ok & tilt_ok & grasp_ok
    ctx["success_hold_count"] = torch.where(
        hold_now,
        ctx["success_hold_count"] + 1,
        torch.zeros_like(ctx["success_hold_count"]),
    )
    success = ctx["success_hold_count"] >= int(env.cfg.hold_required_steps)

    drop = height_delta < -float(env.cfg.fall_height_margin)
    far = (left_dist > float(env.cfg.max_target_distance)) | (right_dist > float(env.cfg.max_target_distance))
    tilt_fail = tilt_deg > float(env.cfg.max_tilt_deg)
    state = torch.cat(
        [
            read_data(env.object.data, "root_state_w"),
            read_data(env.robot.data, "joint_pos"),
            read_data(env.robot.data, "joint_vel"),
        ],
        dim=-1,
    )
    invalid = ~torch.isfinite(state).all(dim=-1)
    terminated = success | drop | far | tilt_fail | invalid
    ctx["termination_success"] = success
    ctx["termination_drop"] = drop
    ctx["termination_far"] = far
    ctx["termination_tilt_fail"] = tilt_fail
    ctx["termination_invalid"] = invalid
    ctx.setdefault("trace_signals", {}).update(
        {
            "termination_drop": drop.detach(),
            "termination_far": far.detach(),
            "termination_tilt_fail": tilt_fail.detach(),
            "termination_invalid": invalid.detach(),
            "left_distance_ok": grasp_validity["left_distance_ok"].detach(),
            "right_distance_ok": grasp_validity["right_distance_ok"].detach(),
            "left_orientation_ok": grasp_validity["left_orientation_ok"].detach(),
            "right_orientation_ok": grasp_validity["right_orientation_ok"].detach(),
            "grasp_orientation_ok": grasp_orientation_ok.detach(),
            "left_ori_err_deg": grasp_validity["left_ori_err_deg"].detach(),
            "right_ori_err_deg": grasp_validity["right_ori_err_deg"].detach(),
            "left_closure_ok": grasp_validity["left_closure_ok"].detach(),
            "right_closure_ok": grasp_validity["right_closure_ok"].detach(),
            "left_finger_span_ok": grasp_validity["left_finger_span_ok"].detach(),
            "right_finger_span_ok": grasp_validity["right_finger_span_ok"].detach(),
            "left_grasp_ok": grasp_validity["left_grasp_ok"].detach(),
            "right_grasp_ok": grasp_validity["right_grasp_ok"].detach(),
            "grasp_ok": grasp_ok.detach(),
        }
    )
    ctx.setdefault("debug_stats", {}).update(
        {
            "termination_drop_ratio": drop.float().mean().item(),
            "termination_far_ratio": far.float().mean().item(),
            "termination_tilt_fail_ratio": tilt_fail.float().mean().item(),
            "termination_invalid_ratio": invalid.float().mean().item(),
            "readiness_ok_ratio": readiness_ok.float().mean().item(),
            "attachment_ok_ratio": attachment_ok.float().mean().item(),
            "success_grasp_distance_ok_ratio": grasp_distance_ok.float().mean().item(),
            "left_distance_ok_ratio": grasp_validity["left_distance_ok"].float().mean().item(),
            "right_distance_ok_ratio": grasp_validity["right_distance_ok"].float().mean().item(),
            "success_grasp_orientation_deg": float(env.cfg.success_grasp_orientation_deg),
            "left_ori_err_deg": grasp_validity["left_ori_err_deg"].mean().item(),
            "right_ori_err_deg": grasp_validity["right_ori_err_deg"].mean().item(),
            "left_orientation_ok_ratio": grasp_validity["left_orientation_ok"].float().mean().item(),
            "right_orientation_ok_ratio": grasp_validity["right_orientation_ok"].float().mean().item(),
            "success_grasp_orientation_ok_ratio": grasp_orientation_ok.float().mean().item(),
            "left_grasp_ok_ratio": grasp_validity["left_grasp_ok"].float().mean().item(),
            "right_grasp_ok_ratio": grasp_validity["right_grasp_ok"].float().mean().item(),
            "dual_grasp_ok_ratio": grasp_ok.float().mean().item(),
            "success_closure_ok_ratio": closure_ok.float().mean().item(),
            "left_closure_ok_ratio": grasp_validity["left_closure_ok"].float().mean().item(),
            "right_closure_ok_ratio": grasp_validity["right_closure_ok"].float().mean().item(),
            "success_finger_span_ok_ratio": finger_span_ok.float().mean().item(),
            "left_finger_span_ok_ratio": grasp_validity["left_finger_span_ok"].float().mean().item(),
            "right_finger_span_ok_ratio": grasp_validity["right_finger_span_ok"].float().mean().item(),
            "success_grasp_ok_ratio": grasp_ok.float().mean().item(),
            "stable_now_ratio": hold_now.float().mean().item(),
            "strict_success_ratio": success.float().mean().item(),
            "success_ratio": success.float().mean().item(),
            "hold_count_mean": ctx["success_hold_count"].float().mean().item(),
            "hold_count_p90": torch.quantile(ctx["success_hold_count"].float(), 0.90).item(),
            "hold_count_max": ctx["success_hold_count"].float().max().item(),
            "hold_required_steps": float(env.cfg.hold_required_steps),
        }
    )
    return terminated, success, invalid


def collect_openarm_re_trace_signals(env, ctx: dict, env_index: int = 0) -> dict:
    """Collect real RE task signals for JSON mode-trace analysis.

    The RE baseline has no contact/force reward, so contact-like fields are
    explicit zero placeholders. Kinematic, closure, lift, and reward fields are
    recorded from the live environment to keep evaluation plots honest.
    """

    idx = int(max(0, min(int(env_index), int(env.num_envs) - 1)))
    left_own = _agent_observation(env, ctx, "left")
    right_own = _agent_observation(env, ctx, "right")
    actor_object_pos_w, _ = get_actor_object_pose_w(env, ctx)
    object_z = actor_object_pos_w[:, 2]
    object_dz = object_z - ctx["initial_actor_object_pos_w"][:, 2]
    lift = torch.clamp(object_dz / max(float(env.cfg.lift_target_height), 1.0e-6), 0.0, 1.0)

    def scalar_tensor(value, default: float = 0.0) -> torch.Tensor:
        if isinstance(value, torch.Tensor):
            tensor = value
        else:
            tensor = torch.full((env.num_envs,), float(value if value is not None else default), device=env.device)
        if tensor.ndim == 0:
            tensor = tensor.expand(env.num_envs)
        return tensor.reshape(-1)

    def env_scalar(value, default: float = 0.0) -> float:
        tensor = scalar_tensor(value, default)
        if tensor.numel() == env.num_envs:
            return float(tensor[idx].detach().cpu())
        return float(tensor.reshape(-1)[0].detach().cpu())

    def trace_scalar(name: str, default: float = 0.0) -> float:
        value = ctx.get("trace_signals", {}).get(name)
        return env_scalar(value, default) if value is not None else default

    context_debug = getattr(env, "_motion_context_last_generated_debug", None)
    if not isinstance(context_debug, dict):
        context_debug = getattr(env, "_motion_context_debug", {})

    def context_debug_scalar(name: str, default: float = 0.0) -> float:
        return env_scalar(context_debug.get(name), default)

    def obs_scalar(obs: torch.Tensor, column: int) -> float:
        return float(obs[idx, column].detach().cpu())

    def obs_vector(obs: torch.Tensor, start: int, end: int) -> list[float]:
        return obs[idx, start:end].detach().float().cpu().tolist()

    left_grip_action = 0.0
    right_grip_action = 0.0
    left_arm_action_mag = 0.0
    right_arm_action_mag = 0.0
    left_arm_action_vec = [0.0] * 7
    right_arm_action_vec = [0.0] * 7
    actions = getattr(env, "actions", {})
    if isinstance(actions, dict):
        left_action = actions.get("left_arm")
        right_action = actions.get("right_arm")
        if isinstance(left_action, torch.Tensor) and left_action.numel() > 0:
            left_grip_action = float(left_action[idx, 7].detach().cpu()) if left_action.shape[-1] > 7 else 0.0
            left_arm_action_vec = left_action[idx, :7].detach().float().cpu().tolist()
            left_arm_action_mag = float(torch.linalg.vector_norm(left_action[idx, :7]).detach().cpu())
        if isinstance(right_action, torch.Tensor) and right_action.numel() > 0:
            right_grip_action = float(right_action[idx, 7].detach().cpu()) if right_action.shape[-1] > 7 else 0.0
            right_arm_action_vec = right_action[idx, :7].detach().float().cpu().tolist()
            right_arm_action_mag = float(torch.linalg.vector_norm(right_action[idx, :7]).detach().cpu())

    left_closure = obs_scalar(left_own, 26)
    right_closure = obs_scalar(right_own, 26)
    motion_context = getattr(env, "_motion_context_last_generated", None)
    if not isinstance(motion_context, dict):
        # Before the first handoff after reset there is no generated context.
        # Trace reads must not advance EMA/history or action-variation validity.
        motion_context = {
            "left_arm": left_own.new_zeros((env.num_envs, 3)),
            "right_arm": right_own.new_zeros((env.num_envs, 3)),
        }
    left_context = motion_context["left_arm"][idx].detach().float().cpu().tolist()
    right_context = motion_context["right_arm"][idx].detach().float().cpu().tolist()
    height_ok = float(object_dz[idx].detach().cpu()) > float(env.cfg.success_height_margin)
    tilt_ok = trace_scalar("success_tilt_deg") <= float(env.cfg.success_tilt_threshold_deg)
    dual_grasp_ok = trace_scalar("grasp_ok") > 0.5
    attachment_ok = True  # legacy trace field; attachment is not used by this task reward
    hold_ok = int(ctx["success_hold_count"][idx].detach().cpu()) > 0
    strict_success = int(ctx["success_hold_count"][idx].detach().cpu()) >= int(env.cfg.hold_required_steps)

    calibration = motion_context_calibration_diagnostics(env)
    return {
        "success": bool(strict_success),
        "object_z": float(object_z[idx].detach().cpu()),
        "object_dz": float(object_dz[idx].detach().cpu()),
        "left_ee_pos_b": obs_vector(left_own, 0, 3),
        "right_ee_pos_b": obs_vector(right_own, 0, 3),
        "left_ee_quat_b": obs_vector(left_own, 3, 7),
        "right_ee_quat_b": obs_vector(right_own, 3, 7),
        "left_ee_vel": read_data(env.robot.data, "body_state_w")[idx, ctx["left_ee_body_id"], 7:10]
        .detach()
        .float()
        .cpu()
        .tolist(),
        "right_ee_vel": read_data(env.robot.data, "body_state_w")[idx, ctx["right_ee_body_id"], 7:10]
        .detach()
        .float()
        .cpu()
        .tolist(),
        "left_motion_context": left_context,
        "right_motion_context": right_context,
        # Legacy trend-specific trace fields are intentionally zero for other
        # representations. The actual payload is in *_motion_context.
        "left_translation_trend": left_context[0] if env.cfg.motion_context_representation == "trend" else 0.0,
        "left_rotation_trend": left_context[1] if env.cfg.motion_context_representation == "trend" else 0.0,
        "left_stability_trend": left_context[2] if env.cfg.motion_context_representation == "trend" else 0.0,
        "left_action_smoothness": left_context[2] if env.cfg.motion_context_representation == "trend" else 0.0,
        "right_translation_trend": right_context[0] if env.cfg.motion_context_representation == "trend" else 0.0,
        "right_rotation_trend": right_context[1] if env.cfg.motion_context_representation == "trend" else 0.0,
        "right_stability_trend": right_context[2] if env.cfg.motion_context_representation == "trend" else 0.0,
        "right_action_smoothness": right_context[2] if env.cfg.motion_context_representation == "trend" else 0.0,
        "left_linear_speed_raw": context_debug_scalar("left_linear_speed_raw"),
        "left_angular_speed_raw": context_debug_scalar("left_angular_speed_raw"),
        "left_action_variation_raw": context_debug_scalar("left_action_variation_raw"),
        "left_action_variation_ema": context_debug_scalar("left_action_variation_ema"),
        "right_linear_speed_raw": context_debug_scalar("right_linear_speed_raw"),
        "right_angular_speed_raw": context_debug_scalar("right_angular_speed_raw"),
        "right_action_variation_raw": context_debug_scalar("right_action_variation_raw"),
        "right_action_variation_ema": context_debug_scalar("right_action_variation_ema"),
        "context_linear_scale": float(env._motion_context_running_scale[0].detach().cpu()),
        "context_angular_scale": float(env._motion_context_running_scale[1].detach().cpu()),
        "context_action_variation_scale": float(env._motion_context_running_scale[2].detach().cpu()),
        **calibration,
        "left_target_dist": float(torch.linalg.vector_norm(left_own[idx, 22:25]).detach().cpu()),
        "right_target_dist": float(torch.linalg.vector_norm(right_own[idx, 22:25]).detach().cpu()),
        "left_target_quat_error": obs_scalar(left_own, 25),
        "right_target_quat_error": obs_scalar(right_own, 25),
        "left_gripper_opening": obs_scalar(left_own, 7),
        "right_gripper_opening": obs_scalar(right_own, 7),
        "left_prev_gripper_action": env_scalar(
            ctx.get("left_prev_action", torch.zeros((env.num_envs, 8), device=env.device))[:, 7]
        ),
        "right_prev_gripper_action": env_scalar(
            ctx.get("right_prev_action", torch.zeros((env.num_envs, 8), device=env.device))[:, 7]
        ),
        "left_closure": left_closure,
        "right_closure": right_closure,
        "left_ee_lin_vel": obs_vector(left_own, 27, 30),
        "right_ee_lin_vel": obs_vector(right_own, 27, 30),
        "left_contact_min": 0.0,
        "right_contact_min": 0.0,
        "left_force_min": 0.0,
        "right_force_min": 0.0,
        "left_grip_score": trace_scalar("left_grasp_hint"),
        "right_grip_score": trace_scalar("right_grasp_hint"),
        "left_lift_reward": trace_scalar("tilt_aware_lift"),
        "right_lift_reward": trace_scalar("tilt_aware_lift"),
        "object_tilt_deg": trace_scalar("object_tilt_deg"),
        "success_tilt_deg": trace_scalar("success_tilt_deg"),
        "success_roll_deg": trace_scalar("success_roll_deg"),
        "success_pitch_deg": trace_scalar("success_pitch_deg"),
        "success_roll_pitch_deg": trace_scalar("success_roll_pitch_deg"),
        "success_yaw_deg": trace_scalar("success_yaw_deg"),
        "goal_error": trace_scalar("xy_error"),
        "xy_error": trace_scalar("xy_error"),
        "hprog": float(lift[idx].detach().cpu()),
        "left_arm_action_magnitude": left_arm_action_mag,
        "right_arm_action_magnitude": right_arm_action_mag,
        "left_arm_action": left_arm_action_vec,
        "right_arm_action": right_arm_action_vec,
        "left_action_grip": left_grip_action,
        "right_action_grip": right_grip_action,
        "height_ok": bool(height_ok),
        "tilt_ok": bool(tilt_ok),
        "left_grasp_ok": bool(trace_scalar("left_grasp_ok") > 0.5),
        "right_grasp_ok": bool(trace_scalar("right_grasp_ok") > 0.5),
        "dual_grasp_ok": bool(dual_grasp_ok),
        "left_distance_ok": bool(trace_scalar("left_distance_ok") > 0.5),
        "right_distance_ok": bool(trace_scalar("right_distance_ok") > 0.5),
        "left_orientation_ok": bool(trace_scalar("left_orientation_ok") > 0.5),
        "right_orientation_ok": bool(trace_scalar("right_orientation_ok") > 0.5),
        "grasp_orientation_ok": bool(trace_scalar("grasp_orientation_ok") > 0.5),
        "left_ori_err_deg": trace_scalar("left_ori_err_deg"),
        "right_ori_err_deg": trace_scalar("right_ori_err_deg"),
        "left_closure_ok": bool(trace_scalar("left_closure_ok") > 0.5),
        "right_closure_ok": bool(trace_scalar("right_closure_ok") > 0.5),
        "left_finger_span_ok": bool(trace_scalar("left_finger_span_ok") > 0.5),
        "right_finger_span_ok": bool(trace_scalar("right_finger_span_ok") > 0.5),
        "left_span_ok": bool(trace_scalar("left_finger_span_ok") > 0.5),
        "right_span_ok": bool(trace_scalar("right_finger_span_ok") > 0.5),
        "success_hold_count": int(ctx["success_hold_count"][idx].detach().cpu()),
        "bilateral_grasp_quality": trace_scalar("bilateral_grasp_quality"),
        "attachment_ok": bool(attachment_ok),
        "left_attachment": 1.0,
        "right_attachment": 1.0,
        "dual_attachment": 1.0,
        "reach_progress": trace_scalar("reach_progress"),
        "height_progress": trace_scalar("height_progress"),
        "hold_ok": bool(hold_ok),
        "strict_success": bool(strict_success),
        "drop": bool(trace_scalar("termination_drop") > 0.0),
        "far": bool(trace_scalar("termination_far") > 0.0),
        "tilt_fail": bool(trace_scalar("termination_tilt_fail") > 0.0),
        "invalid": bool(trace_scalar("termination_invalid") > 0.0),
        "secure_bi_min": trace_scalar("dual_grasp_gate"),
        "secure_bi_sqrt": trace_scalar("dual_grasp_quality"),
        "grasp_imbalance": abs(trace_scalar("left_grasp_hint") - trace_scalar("right_grasp_hint")),
        "reward_gap": abs(trace_scalar("left_reward") - trace_scalar("right_reward")),
        "left_right_closure_gap": abs(left_closure - right_closure),
        "left_right_contact_gap": 0.0,
    }


def reset_openarm_re_context(
    env,
    ctx: dict,
    env_ids: torch.Tensor,
    object_state: torch.Tensor | None = None,
) -> None:
    """Reset per-episode reference state for selected envs."""

    # T_box_tag, T_box_grasp, and T_mount_camera are fixed asset extrinsics.
    # They are loaded once when the context is created and remain valid across
    # randomized object/root resets, so avoid expensive per-episode USD reads.
    if object_state is None:
        reset_pos_w = read_data(env.object.data, "root_pos_w")[env_ids, 0:3]
        reset_quat_w = read_data(env.object.data, "root_quat_w")[env_ids]
    else:
        reset_pos_w = object_state[:, 0:3]
        reset_quat_w = object_state[:, 3:7]
    ctx["initial_object_pos_w"][env_ids] = reset_pos_w
    ctx["initial_object_quat_w"][env_ids] = reset_quat_w
    ctx["initial_actor_object_pos_w"][env_ids] = reset_pos_w
    ctx["initial_actor_object_quat_w"][env_ids] = reset_quat_w
    ctx["target_object_xy_w"][env_ids] = reset_pos_w[:, 0:2]
    ctx["success_hold_count"][env_ids] = 0
    ctx["reward_temporal_valid"][env_ids] = False
    for value in ctx.get("trace_signals", {}).values():
        if isinstance(value, torch.Tensor) and value.ndim > 0 and value.shape[0] == env.num_envs:
            value[env_ids] = 0
    ctx["apriltag_measurement_history"] = []
    ctx["actor_target_cache_step"] = -1
    ctx["actor_target_cache"] = None
    ctx["actor_object_pose_cache_step"] = -1
    ctx["actor_object_pose_cache"] = None
    env._motion_context_cache_step = -1
    env._motion_context_cache = None
    if hasattr(env, "_motion_context_action_variation_valid"):
        env._motion_context_action_variation_valid[env_ids] = False
    reset_motion_trend_state(env, env_ids)
