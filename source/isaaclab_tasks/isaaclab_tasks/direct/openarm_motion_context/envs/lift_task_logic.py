# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Contact-free cooperative OpenArm bimanual KLT lift task logic.

Reward:
    per-arm reach
    + per-arm orientation
    + kinematic grasp readiness
    + dual-grasp-gated tilt-aware lift quality
    + lifted object-center tracking
    - action-rate penalty
    - joint-velocity penalty

No contact sensor, force threshold, privileged force signal, learned mode, or
auxiliary-intent reward is used here. The same reward is used for every paper
communication baseline; only the actor communication input changes.
"""

from __future__ import annotations

import torch

from isaaclab.utils.math import (
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
from ..communication.message_builder import pad_message
from ..communication.motion import signed_motion_intent
from ..communication.motion_context import motion_context_from_raw


def _resolve_env_path(path_template: str, env_id: int) -> str:
    """Resolve an Isaac Lab env-regex path for one cloned environment."""

    env_ns = f"/World/envs/env_{env_id}"
    return (
        str(path_template)
        .replace("{ENV_REGEX_NS}", env_ns)
        .replace("/World/envs/env_.*/", f"{env_ns}/")
    )


def _resolve_robot_child_path(env, child_path: str, env_id: int) -> str:
    """Resolve a robot-local child path for one cloned environment."""

    robot_root = _resolve_env_path(env.scene["robot"].cfg.prim_path, env_id)
    child_path = str(child_path).strip("/")
    return f"{robot_root}/{child_path}"


def _usd_authored_local_pose(env, prim_path: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Read a USD collision target prim's authored local pose in wxyz convention.

    The current KLT asset uses collision-box prims such as ``Collision/Cube``
    and ``Collision/Cube_01`` as object-local grasp targets.
    Reading world poses and re-projecting through the simulated RigidObject can
    drift during clone/reset synchronization, which corrupts the reach target.
    """

    import omni.usd  # type: ignore
    from pxr import Usd, UsdGeom  # type: ignore

    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise RuntimeError(f"OpenArmRE collision target prim not found: {prim_path}")
    xformable = UsdGeom.Xformable(prim)
    local_tf = xformable.GetLocalTransformation(Usd.TimeCode.Default())
    local_tf = local_tf[0] if isinstance(local_tf, tuple) else local_tf
    translation = local_tf.ExtractTranslation()
    rotation = local_tf.ExtractRotationQuat()
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

    import omni.usd  # type: ignore
    from pxr import Usd, UsdGeom  # type: ignore

    stage = omni.usd.get_context().get_stage()
    root = stage.GetPrimAtPath(root_path)
    prim = stage.GetPrimAtPath(prim_path)
    if not root.IsValid():
        raise RuntimeError(f"OpenArmRE root prim not found: {root_path}")
    if not prim.IsValid():
        raise RuntimeError(f"OpenArmRE child prim not found: {prim_path}")

    root_tf = UsdGeom.Xformable(root).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    prim_tf = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    rel_tf = prim_tf * root_tf.GetInverse()
    translation = rel_tf.ExtractTranslation()
    rotation = rel_tf.ExtractRotationQuat()
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

    import omni.usd  # type: ignore
    from pxr import Usd, UsdGeom  # type: ignore

    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise RuntimeError(f"OpenArmRE runtime prim not found: {prim_path}")

    world_tf = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    translation = world_tf.ExtractTranslation()
    rotation = world_tf.ExtractRotationQuat()
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

    import omni.usd  # type: ignore

    stage = omni.usd.get_context().get_stage()
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


def _apply_collision_target_pose_offset(env, pos: torch.Tensor, quat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
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
    left_camera_mount_body_ids, _ = robot.find_bodies("openarm_left_link7", preserve_order=True)
    right_camera_mount_body_ids, _ = robot.find_bodies("openarm_right_link7", preserve_order=True)
    chest_camera_mount_body_ids, _ = robot.find_bodies("openarm_body_link", preserve_order=True)
    left_inner_finger_body_ids, _ = robot.find_bodies("openarm_left_left_finger", preserve_order=True)
    left_outer_finger_body_ids, _ = robot.find_bodies("openarm_left_right_finger", preserve_order=True)
    right_inner_finger_body_ids, _ = robot.find_bodies("openarm_right_left_finger", preserve_order=True)
    right_outer_finger_body_ids, _ = robot.find_bodies("openarm_right_right_finger", preserve_order=True)

    if not left_arm_joint_ids or not right_arm_joint_ids:
        raise RuntimeError("OpenArmRE could not resolve left/right arm joints.")
    if not left_gripper_joint_ids or not right_gripper_joint_ids:
        raise RuntimeError("OpenArmRE could not resolve left/right gripper joints.")
    if not left_ee_body_ids or not right_ee_body_ids:
        raise RuntimeError("OpenArmRE could not resolve left/right EE TCP bodies.")
    if not left_camera_mount_body_ids or not right_camera_mount_body_ids or not chest_camera_mount_body_ids:
        raise RuntimeError("OpenArmRE could not resolve camera mount bodies.")
    if not (
        left_inner_finger_body_ids
        and left_outer_finger_body_ids
        and right_inner_finger_body_ids
        and right_outer_finger_body_ids
    ):
        raise RuntimeError("OpenArmRE could not resolve left/right finger bodies for kinematic grasp hint.")

    ctx = {
        "robot": env.robot,
        "object": env.object,
        "left_arm_joint_ids": left_arm_joint_ids,
        "right_arm_joint_ids": right_arm_joint_ids,
        "left_gripper_joint_ids": left_gripper_joint_ids,
        "right_gripper_joint_ids": right_gripper_joint_ids,
        "left_ee_body_id": left_ee_body_ids[0],
        "right_ee_body_id": right_ee_body_ids[0],
        "left_camera_mount_body_id": left_camera_mount_body_ids[0],
        "right_camera_mount_body_id": right_camera_mount_body_ids[0],
        "chest_camera_mount_body_id": chest_camera_mount_body_ids[0],
        "left_inner_finger_body_id": left_inner_finger_body_ids[0],
        "left_outer_finger_body_id": left_outer_finger_body_ids[0],
        "right_inner_finger_body_id": right_inner_finger_body_ids[0],
        "right_outer_finger_body_id": right_outer_finger_body_ids[0],
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
        "initial_object_pos_w": env.object.data.root_pos_w[:, 0:3].clone(),
        "initial_object_quat_w": env.object.data.root_quat_w.clone(),
        "target_object_xy_w": env.object.data.root_pos_w[:, 0:2].clone(),
        "success_hold_count": torch.zeros(env.num_envs, device=env.device, dtype=torch.long),
        "debug_stats": {},
    }
    env._openarm_re_ctx = ctx
    refresh_collision_targets_from_usd(env, ctx)
    refresh_apriltag_targets_from_usd(env, ctx)
    return ctx


def resolve_collision_target_paths(env, env_id: int) -> tuple[str, str]:
    """Return runtime USD paths for the left/right KLT collision target prims."""

    return (
        _resolve_env_path(env.cfg.left_collision_target_prim, env_id),
        _resolve_env_path(env.cfg.right_collision_target_prim, env_id),
    )


def refresh_collision_targets_from_usd(env, ctx: dict, env_ids: torch.Tensor | None = None) -> None:
    """Refresh object-local grasp targets from USD-authored collision transforms.

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
            f"z+{float(getattr(env.cfg, 'collision_target_z_offset', 0.0)):.4f}m, "
            f"xrot+{float(getattr(env.cfg, 'collision_target_x_rot_offset_deg', 0.0)):.1f}deg"
        )
        print(f"  left_target_local={ctx['left_target_local_pos'][0].detach().cpu().tolist()}")
        print(f"  right_target_local={ctx['right_target_local_pos'][0].detach().cpu().tolist()}")
        print(f"  object_pos={env.object.data.root_pos_w[0, 0:3].detach().cpu().tolist()}")
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
        ctx["_printed_apriltag_debug"] = True


def compute_collision_target_poses_w(env, ctx: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Attach the cached object-local collision targets to the current object pose."""

    object_pos = env.object.data.root_pos_w[:, 0:3]
    object_quat = env.object.data.root_quat_w
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


def clear_external_apriltag_measurements(env) -> None:
    """Clear deploy-time AprilTag detections and return to simulated measurements."""

    ctx = ensure_openarm_re_context(env)
    ctx["external_apriltag_measurements"] = None
    ctx["actor_target_cache_step"] = -1
    ctx["actor_target_cache"] = None


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
    object_pos = env.object.data.root_pos_w[:, None, 0:3].expand(-1, num_tags, -1)
    object_quat = env.object.data.root_quat_w[:, None, :].expand(-1, num_tags, -1)
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
    if (
        len(ctx.get("apriltag_camera_paths_by_env", [])) != env.num_envs
        or any(len(paths) != len(camera_prims) for paths in ctx.get("apriltag_camera_paths_by_env", []))
    ):
        refresh_apriltag_targets_from_usd(env, ctx)

    if len(camera_prims) != 3:
        raise RuntimeError("OpenArmRE lift expects left wrist, right wrist, and chest cameras.")

    mount_pos = torch.stack(
        [
            env.robot.data.body_state_w[:, ctx["left_camera_mount_body_id"], 0:3],
            env.robot.data.body_state_w[:, ctx["right_camera_mount_body_id"], 0:3],
            env.robot.data.body_state_w[:, ctx["chest_camera_mount_body_id"], 0:3],
        ],
        dim=1,
    )
    mount_quat = torch.stack(
        [
            env.robot.data.body_state_w[:, ctx["left_camera_mount_body_id"], 3:7],
            env.robot.data.body_state_w[:, ctx["right_camera_mount_body_id"], 3:7],
            env.robot.data.body_state_w[:, ctx["chest_camera_mount_body_id"], 3:7],
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
    tag_pos_w, tag_quat_w = camera_tag_measurements_to_world(camera_pos_w, camera_quat_w, tag_pos_camera, tag_quat_camera, visible)
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

    source = str(getattr(env.cfg, "actor_object_pose_source", "ground_truth")).lower()
    if source in ("ground_truth", "gt", "object"):
        return env.object.data.root_pos_w[:, 0:3], env.object.data.root_quat_w
    if source in ("apriltag", "april_tag", "tag"):
        return estimate_object_pose_from_apriltags_w(env, ctx)
    raise ValueError(f"Unsupported actor_object_pose_source={source!r}.")


def compute_actor_collision_target_poses_w(env, ctx: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute actor grasp targets from the configured actor-side perception source."""

    current_step = int(getattr(env, "common_step_counter", -1))
    cached = ctx.get("actor_target_cache")
    if cached is not None and int(ctx.get("actor_target_cache_step", -2)) == current_step:
        return cached

    source = str(getattr(env.cfg, "actor_object_pose_source", "ground_truth")).lower()
    if source in ("apriltag", "april_tag", "tag"):
        result = estimate_collision_target_poses_from_apriltags_w(env, ctx)
    else:
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

    robot_root_quat = env.robot.data.root_quat_w
    ee_lin_vel_w = env.robot.data.body_state_w[:, ee_body_id, 7:10]
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

    robot_root_pos = env.robot.data.root_pos_w[:, 0:3]
    robot_root_quat = env.robot.data.root_quat_w
    ee_pos_w = env.robot.data.body_state_w[:, ee_body_id, 0:3]
    ee_quat_w = env.robot.data.body_state_w[:, ee_body_id, 3:7]

    ee_pos_b, ee_quat_b = subtract_frame_transforms(robot_root_pos, robot_root_quat, ee_pos_w, ee_quat_w)
    target_delta_b = quat_apply(quat_inv(robot_root_quat), target_pos_w - ee_pos_w)
    target_quat_error = quat_error_magnitude(ee_quat_w, target_quat_w).unsqueeze(-1)

    gripper_opening = env.robot.data.joint_pos[:, gripper_joint_ids].mean(dim=-1, keepdim=True)
    joint_pos = env.robot.data.joint_pos[:, arm_joint_ids]
    joint_vel = env.robot.data.joint_vel[:, arm_joint_ids]
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

    own = torch.nan_to_num(torch.cat(obs_parts, dim=-1), nan=0.0, posinf=0.0, neginf=0.0)
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
            env.object.data.root_pos_w[:, 0:3],
            env.object.data.root_quat_w,
            env.object.data.root_lin_vel_w,
            env.object.data.root_ang_vel_w,
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
    robot_root_quat = env.robot.data.root_quat_w
    left_vel_w = env.robot.data.body_state_w[:, ctx["left_ee_body_id"], 7:10]
    right_vel_w = env.robot.data.body_state_w[:, ctx["right_ee_body_id"], 7:10]
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


def _arm_action_change_norm(env, agent: str, action_dim: int) -> torch.Tensor:
    """Return ||a_arm(t) - a_arm(t-1)|| for one arm."""

    current = _current_agent_action(env, agent, action_dim)
    current_arm = torch.clamp(current[..., :7], -1.0, 1.0)
    previous = get_previous_agent_action(env, agent, action_dim)[..., :7]
    previous = torch.clamp(previous, -1.0, 1.0).to(device=current_arm.device, dtype=current_arm.dtype)
    return torch.linalg.vector_norm(current_arm - previous, dim=-1, keepdim=True)


def compute_openarm_re_motion_contexts(env, ctx: dict) -> dict[str, torch.Tensor]:
    """Return task-agnostic proprioceptive motion context c_t for each arm.

    c_t = [linear_activity, angular_activity, action_smoothness].
    """

    current_step = int(getattr(env, "common_step_counter", 0))
    if getattr(env, "_motion_context_cache_step", -1) == current_step:
        cached = getattr(env, "_motion_context_cache", None)
        if isinstance(cached, dict):
            return cached

    def raw_for(agent: str, ee_body_id: int, action_dim: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ee_state = env.robot.data.body_state_w[:, ee_body_id]
        lin_speed = torch.linalg.vector_norm(ee_state[:, 7:10], dim=-1, keepdim=True)
        ang_speed = torch.linalg.vector_norm(ee_state[:, 10:13], dim=-1, keepdim=True)
        action_change = _arm_action_change_norm(env, agent, action_dim)
        return lin_speed, ang_speed, action_change

    action_dim = get_agent_action_dim(env)
    left_raw = raw_for("left_arm", ctx["left_ee_body_id"], action_dim)
    right_raw = raw_for("right_arm", ctx["right_ee_body_id"], action_dim)

    if not hasattr(env, "_motion_context_running_scale"):
        env._motion_context_running_scale = torch.tensor(
            [
                float(getattr(env.cfg, "motion_context_lin_scale_init", 0.10)),
                float(getattr(env.cfg, "motion_context_ang_scale_init", 0.50)),
                float(getattr(env.cfg, "motion_context_action_scale_init", 0.50)),
            ],
            device=env.device,
            dtype=torch.float32,
        ).clamp_min(1.0e-6)
        env._motion_context_scale_update_step = -1
        env._motion_context_scale_frozen = False

    update_enabled = bool(
        getattr(env, "motion_context_update_scale", getattr(env.cfg, "motion_context_update_scale", True))
    )
    freeze_after = int(getattr(env.cfg, "motion_context_freeze_after_steps", 10000))
    training = bool(getattr(env, "training", True))
    scale_frozen = bool(getattr(env, "_motion_context_scale_frozen", False))
    if current_step >= freeze_after:
        scale_frozen = True
        env._motion_context_scale_frozen = True
    should_update_scale = update_enabled and training and not scale_frozen
    if should_update_scale and getattr(env, "_motion_context_scale_update_step", -1) != current_step:
        beta = float(getattr(env.cfg, "motion_context_scale_beta", 0.99))
        beta = min(max(beta, 0.0), 0.9999)
        q = float(getattr(env.cfg, "motion_context_scale_percentile", 0.90))
        q = min(max(q, 0.0), 1.0)
        batch_scale = torch.stack(
            [
                torch.quantile(torch.cat([left_raw[0], right_raw[0]], dim=0).flatten(), q),
                torch.quantile(torch.cat([left_raw[1], right_raw[1]], dim=0).flatten(), q),
                torch.quantile(torch.cat([left_raw[2], right_raw[2]], dim=0).flatten(), q),
            ]
        ).detach().clamp_min(1.0e-6)
        env._motion_context_running_scale.mul_(beta).add_(batch_scale, alpha=1.0 - beta)
        env._motion_context_scale_update_step = current_step

    scale = env._motion_context_running_scale.clamp_min(1.0e-6)
    norm_max = max(float(getattr(env.cfg, "motion_context_norm_max", 1.5)), 1.0e-6)

    def context_from_raw(raw: tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> torch.Tensor:
        return motion_context_from_raw(raw[0], raw[1], raw[2], scale, norm_max)

    result = {
        "left_arm": context_from_raw(left_raw),
        "right_arm": context_from_raw(right_raw),
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
    needs_motion = mode in ("motion_only", "motion_context")
    needs_context = mode in ("context_only", "motion_context")
    motion = compute_openarm_re_motion_prediction(env, ctx) if needs_motion else {}
    context = compute_openarm_re_motion_contexts(env, ctx) if needs_context else {}
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
            message = context[agent]
        elif mode == "motion_context":
            message = torch.cat([motion[agent], context[agent]], dim=-1)
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


def _object_tilt(env) -> tuple[torch.Tensor, torch.Tensor]:
    """Return object tilt angle in degrees and normalized squared penalty."""

    world_z = torch.tensor([0.0, 0.0, 1.0], device=env.device).expand(env.num_envs, -1)
    object_up = matrix_from_quat(env.object.data.root_quat_w)[:, :, 2]
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


def _object_success_rpy_deg(env, ctx: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return object roll/pitch/yaw drift from reset pose in degrees."""

    initial_quat = ctx.get("initial_object_quat_w")
    if initial_quat is None:
        initial_quat = env.object.data.root_quat_w.clone()
        ctx["initial_object_quat_w"] = initial_quat
    relative_quat = _quat_mul_wxyz(env.object.data.root_quat_w, quat_inv(initial_quat))
    return _quat_to_rpy_deg(relative_quat)


def _object_success_tilt_ok(env, ctx: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Check success orientation with separate roll/pitch and yaw thresholds."""

    roll_deg, pitch_deg, yaw_deg = _object_success_rpy_deg(env, ctx)
    roll_pitch_limit = float(env.cfg.success_roll_pitch_threshold)
    yaw_limit = float(env.cfg.success_yaw_threshold)
    tilt_ok = (
        (roll_deg.abs() < roll_pitch_limit)
        & (pitch_deg.abs() < roll_pitch_limit)
        & (yaw_deg.abs() < yaw_limit)
    )
    return tilt_ok, roll_deg, pitch_deg, yaw_deg


def _target_inside_gripper_score(
    env,
    target_pos_w: torch.Tensor,
    inner_finger_pos_w: torch.Tensor,
    outer_finger_pos_w: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Score whether the collision target is inside the gripper opening.

    The check is purely kinematic and contact-free:
      1. project target onto the line from inner to outer finger;
      2. score whether the projection is inside the finger span;
      3. score whether the target is centered near the finger-to-finger line.
    """

    finger_to_finger_vec = outer_finger_pos_w - inner_finger_pos_w
    finger_span_len_sq = finger_to_finger_vec.square().sum(dim=-1, keepdim=True).clamp_min(1.0e-8)
    target_from_inner = target_pos_w - inner_finger_pos_w
    target_span_position = (target_from_inner * finger_to_finger_vec).sum(dim=-1, keepdim=True) / finger_span_len_sq
    target_span_position_clamped = target_span_position.clamp(0.0, 1.0)
    closest_point_in_gripper = inner_finger_pos_w + target_span_position_clamped * finger_to_finger_vec
    target_distance_from_gripper_midline = torch.linalg.vector_norm(target_pos_w - closest_point_in_gripper, dim=-1)

    target_span_position = target_span_position.squeeze(-1)
    inside_finger_span_score = ((target_span_position >= 0.0) & (target_span_position <= 1.0)).to(
        dtype=target_distance_from_gripper_midline.dtype
    )

    center_distance_scale = 0.08
    centered_between_fingers_score = torch.exp(-target_distance_from_gripper_midline / center_distance_scale)
    target_inside_gripper_score = inside_finger_span_score * centered_between_fingers_score
    return (
        target_inside_gripper_score,
        inside_finger_span_score,
        centered_between_fingers_score,
        target_distance_from_gripper_midline,
    )


def _gripper_close_command_signal(env, side: str, reference: torch.Tensor) -> torch.Tensor:
    """Return whether the policy/environment is commanding the gripper closed.

    Actual finger joint closure can stay small when the box wall is already
    between the fingers and physically blocks further motion. For grasp reward,
    use the commanded close state first, falling back to the signed gripper
    action if the latch is unavailable.
    """

    latch_name = "_left_gripper_closed" if side == "left" else "_right_gripper_closed"
    latched = getattr(env, latch_name, None)
    if isinstance(latched, torch.Tensor):
        return latched.to(device=reference.device, dtype=reference.dtype).reshape(reference.shape)

    actions = getattr(env, "actions", {})
    agent = "left_arm" if side == "left" else "right_arm"
    if isinstance(actions, dict):
        action = actions.get(agent)
        if isinstance(action, torch.Tensor) and action.shape[-1] > 7:
            grip_action = action[:, 7].to(device=reference.device, dtype=reference.dtype)
            eps = float(getattr(env.cfg, "gripper_switch_epsilon", 0.02))
            return torch.clamp((-grip_action - eps) / max(1.0 - eps, 1.0e-6), 0.0, 1.0)

    return torch.zeros_like(reference)


def compute_openarm_re_rewards(env, ctx: dict) -> dict[str, torch.Tensor]:
    """Compute the shared contact-free RE reward for both arms.

    Reward terms:
      reach + orientation + weak grasp hint + lift + lifted-center tracking
      - tilt - action-rate - joint-velocity.
    """

    robot = ctx["robot"]
    obj = ctx["object"]

    left_target_w, right_target_w, left_target_q_w, right_target_q_w = compute_collision_target_poses_w(env, ctx)

    left_ee_pos_w = robot.data.body_pos_w[:, ctx["left_ee_body_id"]]
    right_ee_pos_w = robot.data.body_pos_w[:, ctx["right_ee_body_id"]]
    left_ee_quat_w = robot.data.body_quat_w[:, ctx["left_ee_body_id"]]
    right_ee_quat_w = robot.data.body_quat_w[:, ctx["right_ee_body_id"]]

    object_pos_w = obj.data.root_pos_w[:, :3]
    object_quat_w = obj.data.root_quat_w

    # 1. Reach reward: each EE approaches its USD-authored collision target.
    left_dist = torch.linalg.vector_norm(left_ee_pos_w - left_target_w, dim=-1)
    right_dist = torch.linalg.vector_norm(right_ee_pos_w - right_target_w, dim=-1)
    reach_std = max(float(env.cfg.reach_std), 1.0e-6)
    left_reach = 1.0 - torch.tanh(left_dist / reach_std)
    right_reach = 1.0 - torch.tanh(right_dist / reach_std)

    # 2. Orientation reward: each TCP orientation matches its collision target.
    left_ori_err = quat_error_magnitude(left_ee_quat_w, left_target_q_w)
    right_ori_err = quat_error_magnitude(right_ee_quat_w, right_target_q_w)
    ori_std = max(float(env.cfg.orientation_std), 1.0e-6)
    left_orientation = 1.0 - torch.tanh(left_ori_err / ori_std)
    right_orientation = 1.0 - torch.tanh(right_ori_err / ori_std)

    # 3. Grasp hint: contact-free kinematic grasp readiness.
    # A hint is paid only when the TCP is near the authored collision target, the
    # TCP orientation is aligned, the target sits between the two fingers, and
    # the gripper is actually closing around that target.
    left_opening = robot.data.joint_pos[:, ctx["left_gripper_joint_ids"]].mean(dim=-1)
    right_opening = robot.data.joint_pos[:, ctx["right_gripper_joint_ids"]].mean(dim=-1)
    open_pos = float(env.cfg.gripper_open_target)
    close_pos = float(env.cfg.gripper_close_target)
    denom = max(open_pos - close_pos, 1.0e-6)
    left_closure = torch.clamp((open_pos - left_opening) / denom, 0.0, 1.0)
    right_closure = torch.clamp((open_pos - right_opening) / denom, 0.0, 1.0)
    left_inner_finger_pos_w = robot.data.body_pos_w[:, ctx["left_inner_finger_body_id"]]
    left_outer_finger_pos_w = robot.data.body_pos_w[:, ctx["left_outer_finger_body_id"]]
    right_inner_finger_pos_w = robot.data.body_pos_w[:, ctx["right_inner_finger_body_id"]]
    right_outer_finger_pos_w = robot.data.body_pos_w[:, ctx["right_outer_finger_body_id"]]
    (
        left_target_inside_gripper,
        left_target_inside_span,
        left_target_centered_in_gripper,
        left_target_to_gripper_midline_dist,
    ) = _target_inside_gripper_score(
        env,
        left_target_w,
        left_inner_finger_pos_w,
        left_outer_finger_pos_w,
    )
    (
        right_target_inside_gripper,
        right_target_inside_span,
        right_target_centered_in_gripper,
        right_target_to_gripper_midline_dist,
    ) = _target_inside_gripper_score(
        env,
        right_target_w,
        right_inner_finger_pos_w,
        right_outer_finger_pos_w,
    )
    near_distance_scale = 0.08
    left_near_collision_target = torch.exp(-left_dist / near_distance_scale)
    right_near_collision_target = torch.exp(-right_dist / near_distance_scale)
    left_close_command = _gripper_close_command_signal(env, "left", left_dist)
    right_close_command = _gripper_close_command_signal(env, "right", right_dist)
    left_close_signal = left_close_command
    right_close_signal = right_close_command
    left_grasp_hint_raw = (
        left_near_collision_target
        * left_orientation
        * left_target_inside_gripper
        * left_close_signal
    )
    right_grasp_hint_raw = (
        right_near_collision_target
        * right_orientation
        * right_target_inside_gripper
        * right_close_signal
    )
    left_grasp_hint = torch.sqrt(left_grasp_hint_raw.clamp_min(0.0))
    right_grasp_hint = torch.sqrt(right_grasp_hint_raw.clamp_min(0.0))

    # Object lift/goal rewards require the weaker learned grasp hint and the
    # weaker measured closure, keeping grasp readiness and hold tightness
    # separated in the cooperative lift gate.
    dual_grasp_gate = torch.minimum(left_grasp_hint, right_grasp_hint).detach()
    team_closure = torch.minimum(left_closure, right_closure)
    raw_lift_gate = dual_grasp_gate * team_closure
    dual_lift_gate = torch.sqrt(raw_lift_gate.clamp_min(0.0)).detach()

    # 4. Lift reward: object height progress from its reset height.
    object_z = object_pos_w[:, 2]
    initial_z = ctx["initial_object_pos_w"][:, 2]
    height_delta = object_z - initial_z
    lift = torch.clamp(height_delta / max(float(env.cfg.lift_target_height), 1.0e-6), 0.0, 1.0)

    # 5. Center/goal reward: once lifted, preserve the reset Y center.
    y_error = torch.abs(object_pos_w[:, 1] - ctx["target_object_xy_w"][:, 1])
    lift_gate = (dual_lift_gate * lift).detach()
    goal = lift_gate * (1.0 - torch.tanh(y_error / max(float(env.cfg.goal_std), 1.0e-6)))

    # 6. Tilt-aware lift quality. Tilt is folded into lift reward instead of
    # being applied as a separate negative reward term.
    world_z = torch.tensor([0.0, 0.0, 1.0], device=env.device).expand(env.num_envs, -1)
    object_up = quat_apply(object_quat_w, world_z)
    cos_tilt = torch.clamp(object_up[:, 2], -1.0, 1.0)
    tilt_deg = torch.rad2deg(torch.acos(cos_tilt))
    tilt_free = float(env.cfg.tilt_free_deg)
    tilt_bad = float(env.cfg.tilt_bad_deg)
    tilt_range = max(tilt_bad - tilt_free, 1.0e-6)
    tilt_penalty = torch.clamp((tilt_deg - tilt_free) / tilt_range, 0.0, 1.0).square()
    _, _, object_yaw_deg = _object_success_rpy_deg(env, ctx)
    object_yaw_error_deg = object_yaw_deg.abs()
    yaw_free = float(env.cfg.success_yaw_threshold)
    yaw_bad = max(float(env.cfg.tilt_bad_deg), yaw_free + 1.0e-6)
    yaw_penalty = torch.clamp((object_yaw_error_deg - yaw_free) / (yaw_bad - yaw_free), 0.0, 1.0).square()
    tilt_penalty = torch.clamp(tilt_penalty + 0.25 * yaw_penalty, 0.0, 1.0)
    tilt_quality = (1.0 - tilt_penalty).clamp(0.0, 1.0).square()
    tilt_aware_lift = dual_lift_gate * lift * tilt_quality

    # 7. Smoothness penalties.
    action_dim = get_agent_action_dim(env)
    left_action_rate = _agent_action_rate(env, "left_arm", action_dim)
    right_action_rate = _agent_action_rate(env, "right_arm", action_dim)

    left_joint_vel = robot.data.joint_vel[:, ctx["left_arm_joint_ids"]]
    right_joint_vel = robot.data.joint_vel[:, ctx["right_arm_joint_ids"]]
    left_joint_vel_penalty = left_joint_vel.square().mean(dim=-1)
    right_joint_vel_penalty = right_joint_vel.square().mean(dim=-1)

    team_reach = 0.5 * (left_reach + right_reach)
    team_orientation = 0.5 * (left_orientation + right_orientation)
    team_grasp_hint = 0.5 * (left_grasp_hint + right_grasp_hint)
    team_action_rate = 0.5 * (left_action_rate + right_action_rate)
    team_joint_vel_penalty = 0.5 * (left_joint_vel_penalty + right_joint_vel_penalty)

    height_ok = height_delta > float(env.cfg.success_height_margin)
    tilt_ok, success_roll_deg, success_pitch_deg, success_yaw_deg = _object_success_tilt_ok(env, ctx)
    success_roll_pitch_deg = torch.maximum(success_roll_deg.abs(), success_pitch_deg.abs())
    success_now = height_ok & tilt_ok

    shared_object_reward = (
        + float(env.cfg.reward_lift_scale) * tilt_aware_lift
        + float(env.cfg.reward_goal_scale) * goal
        + float(env.cfg.reward_stability_scale) * success_now.float()
    )
    left_reward_raw = (
        float(env.cfg.reward_reach_scale) * left_reach
        + float(env.cfg.reward_orientation_scale) * left_orientation
        + float(env.cfg.reward_grasp_scale) * left_grasp_hint
        + shared_object_reward
        - float(env.cfg.reward_action_rate_scale) * left_action_rate
        - float(env.cfg.reward_joint_vel_scale) * left_joint_vel_penalty
    )
    right_reward_raw = (
        float(env.cfg.reward_reach_scale) * right_reach
        + float(env.cfg.reward_orientation_scale) * right_orientation
        + float(env.cfg.reward_grasp_scale) * right_grasp_hint
        + shared_object_reward
        - float(env.cfg.reward_action_rate_scale) * right_action_rate
        - float(env.cfg.reward_joint_vel_scale) * right_joint_vel_penalty
    )
    left_reward_raw = torch.nan_to_num(left_reward_raw, nan=0.0, posinf=0.0, neginf=0.0)
    right_reward_raw = torch.nan_to_num(right_reward_raw, nan=0.0, posinf=0.0, neginf=0.0)

    team_reward = (
        float(env.cfg.reward_reach_scale) * team_reach
        + float(env.cfg.reward_orientation_scale) * team_orientation
        + float(env.cfg.reward_grasp_scale) * team_grasp_hint
        + shared_object_reward
        - float(env.cfg.reward_action_rate_scale) * team_action_rate
        - float(env.cfg.reward_joint_vel_scale) * team_joint_vel_penalty
    )
    team_reward = torch.nan_to_num(team_reward, nan=0.0, posinf=0.0, neginf=0.0)

    # Paper grouping is debug-only: the actual scalar reward above keeps the
    # incremental reward terms and weights unchanged.
    progress = 0.5 * team_reach + 0.5 * team_grasp_hint
    center_quality = 1.0 - torch.tanh(y_error / max(float(env.cfg.goal_std), 1.0e-6))
    quality = tilt_aware_lift * (0.8 + 0.2 * center_quality)
    regularization = (
        float(env.cfg.reward_action_rate_scale) * team_action_rate
        + float(env.cfg.reward_joint_vel_scale) * team_joint_vel_penalty
    )

    ctx["debug_stats"] = {
        "team_reward": team_reward.mean().item(),
        "left_reward": left_reward_raw.mean().item(),
        "right_reward": right_reward_raw.mean().item(),
        "left_reward_raw": left_reward_raw.mean().item(),
        "right_reward_raw": right_reward_raw.mean().item(),
        "paper/progress": progress.mean().item(),
        "paper/quality": quality.mean().item(),
        "paper/success": success_now.float().mean().item(),
        "paper/stability_bonus": (
            float(env.cfg.reward_stability_scale) * success_now.float()
        ).mean().item(),
        "stability_bonus": (
            float(env.cfg.reward_stability_scale) * success_now.float()
        ).mean().item(),
        "paper/regularization": regularization.mean().item(),
        "left_reach": left_reach.mean().item(),
        "right_reach": right_reach.mean().item(),
        "reach": team_reach.mean().item(),
        "left_dist": left_dist.mean().item(),
        "right_dist": right_dist.mean().item(),
        "left_orientation": left_orientation.mean().item(),
        "right_orientation": right_orientation.mean().item(),
        "orientation": team_orientation.mean().item(),
        "left_ori_err": left_ori_err.mean().item(),
        "right_ori_err": right_ori_err.mean().item(),
        "grasp_hint": team_grasp_hint.mean().item(),
        "left_grasp_hint": left_grasp_hint.mean().item(),
        "right_grasp_hint": right_grasp_hint.mean().item(),
        "left_grasp_hint_raw": left_grasp_hint_raw.mean().item(),
        "right_grasp_hint_raw": right_grasp_hint_raw.mean().item(),
        "dual_grasp_gate": dual_grasp_gate.mean().item(),
        "dual_lift_gate": dual_lift_gate.mean().item(),
        "left_near_collision_target": left_near_collision_target.mean().item(),
        "right_near_collision_target": right_near_collision_target.mean().item(),
        "left_close_command": left_close_command.mean().item(),
        "right_close_command": right_close_command.mean().item(),
        "left_close_signal": left_close_signal.mean().item(),
        "right_close_signal": right_close_signal.mean().item(),
        "left_closure": left_closure.mean().item(),
        "right_closure": right_closure.mean().item(),
        "team_closure": team_closure.mean().item(),
        "left_target_inside_gripper": left_target_inside_gripper.mean().item(),
        "right_target_inside_gripper": right_target_inside_gripper.mean().item(),
        "left_target_inside_finger_span": left_target_inside_span.mean().item(),
        "right_target_inside_finger_span": right_target_inside_span.mean().item(),
        "left_target_centered_between_fingers": left_target_centered_in_gripper.mean().item(),
        "right_target_centered_between_fingers": right_target_centered_in_gripper.mean().item(),
        "left_target_to_gripper_midline_dist": left_target_to_gripper_midline_dist.mean().item(),
        "right_target_to_gripper_midline_dist": right_target_to_gripper_midline_dist.mean().item(),
        "lift": lift.mean().item(),
        "tilt_aware_lift": tilt_aware_lift.mean().item(),
        "object_height_delta": height_delta.mean().item(),
        "goal": goal.mean().item(),
        "xy_error": y_error.mean().item(),
        "y_error": y_error.mean().item(),
        "object_tilt_deg": tilt_deg.mean().item(),
        "tilt_quality": tilt_quality.mean().item(),
        "success_roll_deg": success_roll_deg.abs().mean().item(),
        "success_pitch_deg": success_pitch_deg.abs().mean().item(),
        "success_roll_pitch_deg": success_roll_pitch_deg.mean().item(),
        "success_yaw_deg": success_yaw_deg.abs().mean().item(),
        "raw_yaw_penalty": yaw_penalty.mean().item(),
        "tilt_penalty": (lift_gate * tilt_penalty).mean().item(),
        "raw_tilt_penalty": tilt_penalty.mean().item(),
        "left_action_rate_penalty": left_action_rate.mean().item(),
        "right_action_rate_penalty": right_action_rate.mean().item(),
        "action_rate_penalty": team_action_rate.mean().item(),
        "left_joint_vel_penalty": left_joint_vel_penalty.mean().item(),
        "right_joint_vel_penalty": right_joint_vel_penalty.mean().item(),
        "joint_vel_penalty": team_joint_vel_penalty.mean().item(),
        "height_ok_ratio": height_ok.float().mean().item(),
        "tilt_ok_ratio": tilt_ok.float().mean().item(),
        "stable_now_ratio": success_now.float().mean().item(),
        "success_ratio": ((ctx["success_hold_count"] >= int(env.cfg.hold_required_steps))).float().mean().item(),
        "strict_success_ratio": ((ctx["success_hold_count"] >= int(env.cfg.hold_required_steps))).float().mean().item(),
        "hold_count_mean": ctx["success_hold_count"].float().mean().item(),
        "hold_count_p90": torch.quantile(ctx["success_hold_count"].float(), 0.90).item(),
        "hold_count_max": ctx["success_hold_count"].float().max().item(),
        "hold_required_steps": float(env.cfg.hold_required_steps),
    }
    return {"left_arm": team_reward, "right_arm": team_reward}


def compute_openarm_re_terminations(env, ctx: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return terminated, success, and invalid flags."""

    left_target_pos, right_target_pos, left_target_quat, right_target_quat = compute_collision_target_poses_w(env, ctx)
    left_ee_pos = env.robot.data.body_state_w[:, ctx["left_ee_body_id"], 0:3]
    right_ee_pos = env.robot.data.body_state_w[:, ctx["right_ee_body_id"], 0:3]
    left_dist = torch.linalg.vector_norm(left_ee_pos - left_target_pos, dim=-1)
    right_dist = torch.linalg.vector_norm(right_ee_pos - right_target_pos, dim=-1)

    height_delta = env.object.data.root_pos_w[:, 2] - ctx["initial_object_pos_w"][:, 2]
    tilt_deg, _ = _object_tilt(env)
    height_ok = height_delta > float(env.cfg.success_height_margin)
    tilt_ok, _, _, _ = _object_success_tilt_ok(env, ctx)
    hold_now = height_ok & tilt_ok
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
            env.object.data.root_state_w,
            env.robot.data.joint_pos,
            env.robot.data.joint_vel,
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
    ctx.setdefault("debug_stats", {}).update(
        {
            "termination_drop_ratio": drop.float().mean().item(),
            "termination_far_ratio": far.float().mean().item(),
            "termination_tilt_fail_ratio": tilt_fail.float().mean().item(),
            "termination_invalid_ratio": invalid.float().mean().item(),
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
    log = ctx.get("debug_stats", {})
    object_z = env.object.data.root_pos_w[:, 2]
    object_dz = object_z - ctx["initial_object_pos_w"][:, 2]
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

    def log_scalar(name: str, default: float = 0.0) -> float:
        return env_scalar(log.get(name, default), default)

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
    motion_context = compute_openarm_re_motion_contexts(env, ctx)
    left_context = motion_context["left_arm"][idx].detach().float().cpu().tolist()
    right_context = motion_context["right_arm"][idx].detach().float().cpu().tolist()
    height_ok = float(object_dz[idx].detach().cpu()) > float(env.cfg.success_height_margin)
    tilt_ok = (
        log_scalar("success_roll_deg") < float(env.cfg.success_roll_pitch_threshold)
        and log_scalar("success_pitch_deg") < float(env.cfg.success_roll_pitch_threshold)
        and log_scalar("success_yaw_deg") < float(env.cfg.success_yaw_threshold)
    )
    dual_grasp_ok = log_scalar("dual_lift_gate") > 0.0
    hold_ok = int(ctx["success_hold_count"][idx].detach().cpu()) > 0
    strict_success = int(ctx["success_hold_count"][idx].detach().cpu()) >= int(env.cfg.hold_required_steps)

    return {
        "success": bool(strict_success),
        "object_z": float(object_z[idx].detach().cpu()),
        "object_dz": float(object_dz[idx].detach().cpu()),
        "left_ee_pos_b": obs_vector(left_own, 0, 3),
        "right_ee_pos_b": obs_vector(right_own, 0, 3),
        "left_ee_quat_b": obs_vector(left_own, 3, 7),
        "right_ee_quat_b": obs_vector(right_own, 3, 7),
        "left_ee_vel": env.robot.data.body_state_w[idx, ctx["left_ee_body_id"], 7:10].detach().float().cpu().tolist(),
        "right_ee_vel": env.robot.data.body_state_w[idx, ctx["right_ee_body_id"], 7:10].detach().float().cpu().tolist(),
        "left_motion_context": left_context,
        "right_motion_context": right_context,
        "left_linear_activity": left_context[0],
        "left_angular_activity": left_context[1],
        "left_action_smoothness": left_context[2],
        "right_linear_activity": right_context[0],
        "right_angular_activity": right_context[1],
        "right_action_smoothness": right_context[2],
        "left_target_dist": float(torch.linalg.vector_norm(left_own[idx, 22:25]).detach().cpu()),
        "right_target_dist": float(torch.linalg.vector_norm(right_own[idx, 22:25]).detach().cpu()),
        "left_target_quat_error": obs_scalar(left_own, 25),
        "right_target_quat_error": obs_scalar(right_own, 25),
        "left_gripper_opening": obs_scalar(left_own, 7),
        "right_gripper_opening": obs_scalar(right_own, 7),
        "left_prev_gripper_action": env_scalar(ctx.get("left_prev_action", torch.zeros((env.num_envs, 8), device=env.device))[:, 7]),
        "right_prev_gripper_action": env_scalar(ctx.get("right_prev_action", torch.zeros((env.num_envs, 8), device=env.device))[:, 7]),
        "left_closure": left_closure,
        "right_closure": right_closure,
        "left_ee_lin_vel": obs_vector(left_own, 27, 30),
        "right_ee_lin_vel": obs_vector(right_own, 27, 30),
        "left_contact_min": 0.0,
        "right_contact_min": 0.0,
        "left_force_min": 0.0,
        "right_force_min": 0.0,
        "left_grip_score": log_scalar("left_grasp_hint"),
        "right_grip_score": log_scalar("right_grasp_hint"),
        "left_lift_reward": log_scalar("tilt_aware_lift"),
        "right_lift_reward": log_scalar("tilt_aware_lift"),
        "object_tilt_deg": log_scalar("object_tilt_deg"),
        "success_roll_deg": log_scalar("success_roll_deg"),
        "success_pitch_deg": log_scalar("success_pitch_deg"),
        "success_roll_pitch_deg": log_scalar("success_roll_pitch_deg"),
        "success_yaw_deg": log_scalar("success_yaw_deg"),
        "goal_error": log_scalar("xy_error"),
        "xy_error": log_scalar("xy_error"),
        "hprog": float(lift[idx].detach().cpu()),
        "left_arm_action_magnitude": left_arm_action_mag,
        "right_arm_action_magnitude": right_arm_action_mag,
        "left_arm_action": left_arm_action_vec,
        "right_arm_action": right_arm_action_vec,
        "left_action_grip": left_grip_action,
        "right_action_grip": right_grip_action,
        "height_ok": bool(height_ok),
        "tilt_ok": bool(tilt_ok),
        "left_grasp_ok": bool(log_scalar("left_grasp_hint") > 0.0),
        "right_grasp_ok": bool(log_scalar("right_grasp_hint") > 0.0),
        "dual_grasp_ok": bool(dual_grasp_ok),
        "hold_ok": bool(hold_ok),
        "strict_success": bool(strict_success),
        "drop": bool(log_scalar("termination_drop_ratio") > 0.0),
        "far": bool(log_scalar("termination_far_ratio") > 0.0),
        "tilt_fail": bool(log_scalar("termination_tilt_fail_ratio") > 0.0),
        "invalid": bool(log_scalar("termination_invalid_ratio") > 0.0),
        "secure_bi_min": log_scalar("dual_lift_gate"),
        "secure_bi_sqrt": log_scalar("dual_lift_gate"),
        "grasp_imbalance": abs(log_scalar("left_grasp_hint") - log_scalar("right_grasp_hint")),
        "reward_gap": abs(log_scalar("left_reward") - log_scalar("right_reward")),
        "left_right_closure_gap": abs(left_closure - right_closure),
        "left_right_contact_gap": 0.0,
    }


def reset_openarm_re_context(env, ctx: dict, env_ids: torch.Tensor) -> None:
    """Reset per-episode reference state for selected envs."""

    # T_box_tag, T_box_grasp, and T_mount_camera are fixed asset extrinsics.
    # They are loaded once when the context is created and remain valid across
    # randomized object/root resets, so avoid expensive per-episode USD reads.
    ctx["initial_object_pos_w"][env_ids] = env.object.data.root_pos_w[env_ids, 0:3]
    ctx["initial_object_quat_w"][env_ids] = env.object.data.root_quat_w[env_ids]
    ctx["target_object_xy_w"][env_ids] = env.object.data.root_pos_w[env_ids, 0:2]
    ctx["success_hold_count"][env_ids] = 0
    ctx["apriltag_measurement_history"] = []
    ctx["actor_target_cache_step"] = -1
    ctx["actor_target_cache"] = None
    env._motion_context_cache_step = -1
    env._motion_context_cache = None
