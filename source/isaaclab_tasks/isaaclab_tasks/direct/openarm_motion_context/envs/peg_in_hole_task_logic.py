# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""OpenArm bimanual peg-in-hole task logic.

The policy interface intentionally stays compatible with the box-lift RE task:
each arm still receives a 30D own observation plus the selected partner
communication message. The task-specific part is the peg/hole frame
calculation and the Factory-style keypoint reward.
"""

from __future__ import annotations

import torch

from isaaclab_tasks.direct.openarm_motion_context.legacy_math import (
    combine_frame_transforms,
    matrix_from_quat,
    quat_apply,
    quat_inv,
    quat_mul,
    subtract_frame_transforms,
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
from .peg_reward_shaping import axis_alignment_quality, insertion_settle_quality, target_depth_quality
from .peg_grasp_geometry import physical_grasp_validity
from .peg_termination_geometry import side_wall_penetration
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


def _first_valid_usd_path(paths: tuple[str, ...]) -> str:
    """Return the first existing USD path, or the first candidate for a clear error."""

    from isaaclab.sim.utils import get_current_stage

    stage = get_current_stage()
    for path in paths:
        if stage.GetPrimAtPath(path).IsValid():
            return path
    return paths[0]


def _matrix_to_pose_wxyz(env, matrix) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert a USD matrix transform to torch position and wxyz quaternion."""

    translation = matrix.ExtractTranslation()
    rotation = matrix.ExtractRotationQuat()
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


def _usd_pose_relative_to_root(env, prim_path: str, root_path: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Read a USD child frame pose relative to the simulated rigid root.

    The peg/hole assets can put RigidBodyAPI on ``peg_usd`` / ``hole_usd``
    while semantic frames live below ``peg_usd/peg/...`` and
    ``hole_usd/hole/...``. This computes the nested child transform in the
    rigid body's local frame instead of assuming the child is a direct parented
    local transform.
    """

    from isaaclab.sim.utils import get_current_stage
    from pxr import Usd, UsdGeom  # type: ignore

    stage = get_current_stage()
    prim = stage.GetPrimAtPath(prim_path)
    root = stage.GetPrimAtPath(root_path)
    if not prim.IsValid():
        raise RuntimeError(f"OpenArm peg-in-hole target prim not found: {prim_path}")
    if not root.IsValid():
        raise RuntimeError(f"OpenArm peg-in-hole rigid root prim not found: {root_path}")
    time = Usd.TimeCode.Default()
    prim_world = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(time)
    root_world = UsdGeom.Xformable(root).ComputeLocalToWorldTransform(time)
    local_tf = prim_world * root_world.GetInverse()
    return _matrix_to_pose_wxyz(env, local_tf)


def ensure_openarm_re_context(env) -> dict:
    """Create and cache ids, peg/hole frames, hold counters, and debug stats."""

    if hasattr(env, "_openarm_re_ctx"):
        return env._openarm_re_ctx

    robot = env.robot

    left_arm_joint_ids, _ = robot.find_joints("openarm_left_joint[1-7]", preserve_order=True)
    right_arm_joint_ids, _ = robot.find_joints("openarm_right_joint[1-7]", preserve_order=True)
    left_gripper_joint_ids, _ = robot.find_joints("openarm_left_finger_joint.*", preserve_order=True)
    right_gripper_joint_ids, _ = robot.find_joints("openarm_right_finger_joint.*", preserve_order=True)
    left_ee_body_ids, _ = robot.find_bodies("openarm_left_ee_tcp", preserve_order=True)
    right_ee_body_ids, _ = robot.find_bodies("openarm_right_ee_tcp", preserve_order=True)
    left_wrist_body_ids, _ = robot.find_bodies("openarm_left_link7", preserve_order=True)
    right_wrist_body_ids, _ = robot.find_bodies("openarm_right_link7", preserve_order=True)
    left_inner_finger_body_ids, _ = robot.find_bodies("openarm_left_left_finger", preserve_order=True)
    left_outer_finger_body_ids, _ = robot.find_bodies("openarm_left_right_finger", preserve_order=True)
    right_inner_finger_body_ids, _ = robot.find_bodies("openarm_right_left_finger", preserve_order=True)
    right_outer_finger_body_ids, _ = robot.find_bodies("openarm_right_right_finger", preserve_order=True)

    if not left_arm_joint_ids or not right_arm_joint_ids:
        raise RuntimeError("OpenArmPegInHole could not resolve left/right arm joints.")
    if not left_gripper_joint_ids or not right_gripper_joint_ids:
        raise RuntimeError("OpenArmPegInHole could not resolve left/right gripper joints.")
    if not left_ee_body_ids or not right_ee_body_ids:
        raise RuntimeError("OpenArmPegInHole could not resolve left/right EE TCP bodies.")
    if not left_wrist_body_ids or not right_wrist_body_ids:
        raise RuntimeError("OpenArmPegInHole could not resolve left/right wrist (link7) bodies.")
    if not (
        left_inner_finger_body_ids
        and left_outer_finger_body_ids
        and right_inner_finger_body_ids
        and right_outer_finger_body_ids
    ):
        raise RuntimeError("OpenArmPegInHole could not resolve left/right finger bodies.")

    ctx = {
        "robot": env.robot,
        "peg": env.peg,
        "hole": env.hole,
        "object": env.peg,
        "left_arm_joint_ids": left_arm_joint_ids,
        "right_arm_joint_ids": right_arm_joint_ids,
        "left_gripper_joint_ids": left_gripper_joint_ids,
        "right_gripper_joint_ids": right_gripper_joint_ids,
        "left_ee_body_id": left_ee_body_ids[0],
        "right_ee_body_id": right_ee_body_ids[0],
        "left_wrist_body_id": left_wrist_body_ids[0],
        "right_wrist_body_id": right_wrist_body_ids[0],
        "left_inner_finger_body_id": left_inner_finger_body_ids[0],
        "left_outer_finger_body_id": left_outer_finger_body_ids[0],
        "right_inner_finger_body_id": right_inner_finger_body_ids[0],
        "right_outer_finger_body_id": right_outer_finger_body_ids[0],
        "peg_tip_local_pos": torch.zeros((env.num_envs, 3), device=env.device),
        "peg_tip_local_quat": torch.zeros((env.num_envs, 4), device=env.device),
        "peg_grip_local_pos": torch.zeros((env.num_envs, 3), device=env.device),
        "peg_grip_local_quat": torch.zeros((env.num_envs, 4), device=env.device),
        "hole_grip_local_pos": torch.zeros((env.num_envs, 3), device=env.device),
        "hole_grip_local_quat": torch.zeros((env.num_envs, 4), device=env.device),
        "hole_entrance_local_pos": torch.zeros((env.num_envs, 3), device=env.device),
        "hole_entrance_local_quat": torch.zeros((env.num_envs, 4), device=env.device),
        "hole_bottom_local_pos": torch.zeros((env.num_envs, 3), device=env.device),
        "hole_bottom_local_quat": torch.zeros((env.num_envs, 4), device=env.device),
        "hole_depth": torch.zeros(env.num_envs, device=env.device),
        "peg_marker_length": torch.zeros(env.num_envs, device=env.device),
        "success_hold_count": torch.zeros(env.num_envs, device=env.device, dtype=torch.long),
        "debug_stats": {},
    }
    env._openarm_re_ctx = ctx
    refresh_peg_hole_frames_from_usd(env, ctx)
    return ctx


def resolve_peg_hole_frame_paths(env, env_id: int) -> tuple[str, str, str, str, str]:
    """Return runtime USD paths for peg_tip, peg_grip, hole_grip, hole_entrance, hole_bottom."""

    hole_grip_path = _resolve_env_path(env.cfg.hole_grip_prim, env_id)
    hole_grip_path = _first_valid_usd_path(
        (
            hole_grip_path,
            hole_grip_path.replace("/hole_grip_point", "/hole_grap_point"),
            hole_grip_path.replace("/hole_grip_point", "/hole_grasp_point"),
        )
    )
    return (
        _resolve_env_path(env.cfg.peg_tip_prim, env_id),
        _resolve_env_path(env.cfg.peg_grip_prim, env_id),
        hole_grip_path,
        _resolve_env_path(env.cfg.hole_entrance_prim, env_id),
        _resolve_env_path(env.cfg.hole_bottom_prim, env_id),
    )


def refresh_peg_hole_frames_from_usd(env, ctx: dict, env_ids: torch.Tensor | None = None) -> None:
    """Refresh peg/hole child frame local poses from USD-authored transforms."""

    from isaaclab_newton.physics import NewtonCfg

    newton = isinstance(env.cfg.sim.physics, NewtonCfg)
    if newton and env.cfg.scene.clone_cfg.clone_combinations:
        raise ValueError("OpenArm Peg semantic-frame loading currently requires homogeneous Newton clones")

    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)

    for env_id_tensor in env_ids:
        env_id = int(env_id_tensor.item())
        # Newton can replicate native bodies without authoring every USD child.
        # These are geometry-local constants, not the source world's dynamic
        # pose. Homogeneous clones therefore read the authored prototype.
        frame_env_id = 0 if newton else env_id
        peg_tip_path, peg_grip_path, hole_grip_path, hole_entrance_path, hole_bottom_path = (
            resolve_peg_hole_frame_paths(env, frame_env_id)
        )
        peg_root_path = _resolve_env_path(env.peg.cfg.prim_path, frame_env_id)
        hole_root_path = _resolve_env_path(env.hole.cfg.prim_path, frame_env_id)
        peg_tip_pos, peg_tip_quat = _usd_pose_relative_to_root(env, peg_tip_path, peg_root_path)
        peg_grip_pos, peg_grip_quat = _usd_pose_relative_to_root(env, peg_grip_path, peg_root_path)
        hole_grip_pos, hole_grip_quat = _usd_pose_relative_to_root(env, hole_grip_path, hole_root_path)
        hole_entrance_pos, hole_entrance_quat = _usd_pose_relative_to_root(env, hole_entrance_path, hole_root_path)
        hole_bottom_pos, hole_bottom_quat = _usd_pose_relative_to_root(env, hole_bottom_path, hole_root_path)

        ctx["peg_tip_local_pos"][env_id] = peg_tip_pos
        ctx["peg_tip_local_quat"][env_id] = peg_tip_quat
        ctx["peg_grip_local_pos"][env_id] = peg_grip_pos
        ctx["peg_grip_local_quat"][env_id] = peg_grip_quat
        ctx["hole_grip_local_pos"][env_id] = hole_grip_pos
        ctx["hole_grip_local_quat"][env_id] = hole_grip_quat
        ctx["hole_entrance_local_pos"][env_id] = hole_entrance_pos
        ctx["hole_entrance_local_quat"][env_id] = hole_entrance_quat
        ctx["hole_bottom_local_pos"][env_id] = hole_bottom_pos
        ctx["hole_bottom_local_quat"][env_id] = hole_bottom_quat
        ctx["hole_depth"][env_id] = torch.linalg.vector_norm(hole_bottom_pos - hole_entrance_pos)
        ctx["peg_marker_length"][env_id] = torch.linalg.vector_norm(peg_tip_pos - peg_grip_pos)

    if not bool(ctx.get("_printed_target_debug", False)):
        frames = compute_peg_hole_frames_w(env, ctx)
        paths = resolve_peg_hole_frame_paths(env, 0)
        print("[OpenArmPegInHole Target Debug]")
        print(f"  peg_tip_path={paths[0]}")
        print(f"  peg_grip_path={paths[1]}")
        print(f"  hole_grip_path={paths[2]}")
        print(f"  hole_entrance_path={paths[3]}")
        print(f"  hole_bottom_path={paths[4]}")
        print(f"  peg_root={env.scene['peg'].cfg.prim_path}")
        print(f"  hole_root={env.scene['hole'].cfg.prim_path}")
        print(f"  peg_tip_w={frames['peg_tip_w'][0].detach().cpu().tolist()}")
        print(f"  peg_grip_w={frames['peg_grip_w'][0].detach().cpu().tolist()}")
        print(f"  hole_entrance_w={frames['hole_entrance_w'][0].detach().cpu().tolist()}")
        print(f"  hole_bottom_w={frames['hole_bottom_w'][0].detach().cpu().tolist()}")
        print(f"  usd_hole_depth={ctx['hole_depth'][0].item():.5f}m")
        print(f"  usd_peg_marker_length={ctx['peg_marker_length'][0].item():.5f}m")
        ctx["_printed_target_debug"] = True


def _normalize_vec(vec: torch.Tensor, eps: float = 1.0e-8) -> torch.Tensor:
    return vec / torch.linalg.vector_norm(vec, dim=-1, keepdim=True).clamp_min(eps)


def compute_peg_hole_frames_w(env, ctx: dict) -> dict[str, torch.Tensor]:
    """Attach cached peg/hole child frames to current rigid body poses."""

    peg_pos = read_data(env.peg.data, "root_pos_w")[:, 0:3]
    peg_quat = read_data(env.peg.data, "root_quat_w")
    hole_pos = read_data(env.hole.data, "root_pos_w")[:, 0:3]
    hole_quat = read_data(env.hole.data, "root_quat_w")

    peg_tip_w, peg_tip_q_w = combine_frame_transforms(
        peg_pos, peg_quat, ctx["peg_tip_local_pos"], ctx["peg_tip_local_quat"]
    )
    peg_grip_w, peg_grip_q_w = combine_frame_transforms(
        peg_pos, peg_quat, ctx["peg_grip_local_pos"], ctx["peg_grip_local_quat"]
    )
    hole_entrance_w, hole_entrance_q_w = combine_frame_transforms(
        hole_pos, hole_quat, ctx["hole_entrance_local_pos"], ctx["hole_entrance_local_quat"]
    )
    hole_bottom_w, hole_bottom_q_w = combine_frame_transforms(
        hole_pos, hole_quat, ctx["hole_bottom_local_pos"], ctx["hole_bottom_local_quat"]
    )

    peg_axis_w = _normalize_vec(peg_tip_w - peg_grip_w)
    hole_axis_w = _normalize_vec(hole_bottom_w - hole_entrance_w)

    return {
        "peg_tip_w": peg_tip_w,
        "peg_tip_q_w": peg_tip_q_w,
        "peg_axis_w": peg_axis_w,
        "peg_grip_w": peg_grip_w,
        "peg_grip_q_w": peg_grip_q_w,
        "hole_entrance_w": hole_entrance_w,
        "hole_entrance_q_w": hole_entrance_q_w,
        "hole_bottom_w": hole_bottom_w,
        "hole_bottom_q_w": hole_bottom_q_w,
        "hole_axis_w": hole_axis_w,
    }


def compute_peg_in_hole_metrics(env, ctx: dict) -> dict[str, torch.Tensor]:
    """Compute insertion geometry metrics from peg_tip and hole frames."""

    frames = compute_peg_hole_frames_w(env, ctx)
    tip_to_entrance = frames["peg_tip_w"] - frames["hole_entrance_w"]
    entrance_to_tip = frames["hole_entrance_w"] - frames["peg_tip_w"]
    insertion_depth = (tip_to_entrance * frames["hole_axis_w"]).sum(dim=-1)
    lateral_vec = tip_to_entrance - insertion_depth.unsqueeze(-1) * frames["hole_axis_w"]
    lateral_error = torch.linalg.vector_norm(lateral_vec, dim=-1)
    axis_alignment = torch.clamp((frames["peg_axis_w"] * frames["hole_axis_w"]).sum(dim=-1), -1.0, 1.0)
    local_x = torch.tensor([1.0, 0.0, 0.0], device=env.device).expand(env.num_envs, -1)
    peg_ref = quat_apply(frames["peg_tip_q_w"], local_x)
    hole_ref = quat_apply(frames["hole_bottom_q_w"], local_x)
    peg_ref = _normalize_vec(
        peg_ref - (peg_ref * frames["hole_axis_w"]).sum(dim=-1, keepdim=True) * frames["hole_axis_w"]
    )
    hole_ref = _normalize_vec(
        hole_ref - (hole_ref * frames["hole_axis_w"]).sum(dim=-1, keepdim=True) * frames["hole_axis_w"]
    )
    hole_ref_y = _normalize_vec(torch.linalg.cross(frames["hole_axis_w"], hole_ref, dim=-1))
    twist_angle = torch.atan2(
        (peg_ref * hole_ref_y).sum(dim=-1),
        (peg_ref * hole_ref).sum(dim=-1),
    )
    symmetry_order = int(getattr(env.cfg, "rotational_symmetry_order", 1))
    if symmetry_order <= 0:
        twist_alignment = torch.ones_like(axis_alignment)
        twist_error = torch.zeros_like(axis_alignment)
    else:
        twist_alignment = torch.cos(float(symmetry_order) * twist_angle)
        twist_error = torch.abs(
            torch.atan2(
                torch.sin(float(symmetry_order) * twist_angle),
                torch.cos(float(symmetry_order) * twist_angle),
            )
        ) / float(symmetry_order)
    tip_dist = torch.linalg.vector_norm(tip_to_entrance, dim=-1)
    hole_depth = torch.linalg.vector_norm(frames["hole_bottom_w"] - frames["hole_entrance_w"], dim=-1).clamp_min(1.0e-6)
    depth_limit = float(getattr(env.cfg, "insertion_depth_limit", 0.0))
    if depth_limit > 0.0:
        target_depth = hole_depth.clamp(max=depth_limit)
        insertion_target_w = frames["hole_entrance_w"] + target_depth.unsqueeze(-1) * frames["hole_axis_w"]
    else:
        target_depth = hole_depth
        insertion_target_w = frames["hole_bottom_w"]
    success_depth = target_depth * float(getattr(env.cfg, "success_depth_ratio", 0.90))
    cosine = axis_alignment.clamp_min(1.0e-6)
    transverse_axis = frames["peg_axis_w"] - axis_alignment.unsqueeze(-1) * frames["hole_axis_w"]
    entrance_offset = lateral_vec - (insertion_depth.clamp_min(0.0) / cosine).unsqueeze(-1) * transverse_axis
    # Bound the entire inserted centerline and the tilted cylinder's elliptical
    # section, rather than accepting a centered tip with its shaft through a wall.
    radial_envelope = torch.maximum(lateral_error, torch.linalg.vector_norm(entrance_offset, dim=-1))
    radial_envelope += float(getattr(env.cfg, "peg_collision_radius", 0.0)) * (1.0 / cosine - 1.0)
    return {
        **frames,
        "tip_dist": tip_dist,
        "tip_to_entrance": tip_to_entrance,
        "entrance_to_tip": entrance_to_tip,
        "insertion_depth": insertion_depth,
        "lateral_vec": lateral_vec,
        "lateral_error": lateral_error,
        "axis_alignment": axis_alignment,
        "twist_alignment": twist_alignment,
        "twist_error": twist_error,
        "hole_depth": hole_depth,
        "target_insertion_depth": target_depth,
        "insertion_target_w": insertion_target_w,
        "shaft_radial_envelope": radial_envelope,
        "success_depth": success_depth,
    }


def _insertion_hold_condition(env, metrics: dict[str, torch.Tensor]) -> torch.Tensor:
    """Share one seating predicate across termination, rewards and traces."""
    hold = (
        (metrics["lateral_error"] < float(env.cfg.success_lateral_threshold))
        & (metrics["axis_alignment"] > float(env.cfg.success_axis_threshold))
        & (metrics["twist_alignment"] > float(env.cfg.success_axis_threshold))
        & (metrics["insertion_depth"] > metrics["success_depth"])
    )
    overrun = float(getattr(env.cfg, "success_depth_overrun_tolerance", 0.0))
    if overrun > 0.0:
        hold &= metrics["insertion_depth"] <= metrics["target_insertion_depth"] + overrun
    clearance = float(getattr(env.cfg, "success_radial_clearance", 0.0))
    if clearance > 0.0:
        hold &= metrics["shaft_radial_envelope"] <= clearance
    if bool(getattr(env.cfg, "require_contact_grasp", False)):
        hold &= physical_grasp_validity(env, env._ctx)
    return hold


def _smooth_insertion_gate(
    env,
    axis_alignment: torch.Tensor,
    lateral_error: torch.Tensor,
    twist_alignment: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return smooth readiness gates for activating insertion rewards/targets."""

    axis_gate = torch.sigmoid((axis_alignment - 0.90) / 0.03)
    lateral_center = 1.25 * float(env.cfg.success_lateral_threshold)
    lateral_width = max(0.25 * float(env.cfg.success_lateral_threshold), 1.0e-4)
    lateral_gate = torch.sigmoid((lateral_center - lateral_error) / lateral_width)
    twist_gate = torch.sigmoid((twist_alignment - 0.90) / 0.03)
    insertion_gate = axis_gate * lateral_gate * twist_gate
    return axis_gate, lateral_gate, twist_gate, insertion_gate


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
      3:7   relative semantic-frame quaternion error
      7:8   gripper_opening
      8:15  joint_pos
      15:22 joint_vel
      22:25 target_delta_b
      25:26 target_quat_error
      26:27 closure
      27:30 ee_lin_vel_b
    """

    metrics = compute_peg_in_hole_metrics(env, ctx)
    pre_clearance = float(getattr(env.cfg, "preinsert_clearance", 0.03))
    pre_tip_w = metrics["hole_entrance_w"] - pre_clearance * metrics["hole_axis_w"]
    _, _, _, insertion_gate = _smooth_insertion_gate(
        env,
        metrics["axis_alignment"],
        metrics["lateral_error"],
        metrics["twist_alignment"],
    )
    target_w = (1.0 - insertion_gate).unsqueeze(-1) * pre_tip_w + insertion_gate.unsqueeze(-1) * metrics[
        "insertion_target_w"
    ]
    if side == "left":
        arm_joint_ids = ctx["left_arm_joint_ids"]
        gripper_joint_ids = ctx["left_gripper_joint_ids"]
        ee_body_id = ctx["left_ee_body_id"]
        target_delta_w = target_w - metrics["peg_tip_w"]
        current_semantic_quat = metrics["peg_tip_q_w"]
        target_semantic_quat = metrics["hole_bottom_q_w"]
    else:
        arm_joint_ids = ctx["right_arm_joint_ids"]
        gripper_joint_ids = ctx["right_gripper_joint_ids"]
        ee_body_id = ctx["right_ee_body_id"]
        target_delta_w = metrics["peg_tip_w"] - target_w
        current_semantic_quat = metrics["hole_bottom_q_w"]
        target_semantic_quat = metrics["peg_tip_q_w"]

    robot_root_pos = read_data(env.robot.data, "root_pos_w")[:, 0:3]
    robot_root_quat = read_data(env.robot.data, "root_quat_w")
    ee_pos_w = read_data(env.robot.data, "body_state_w")[:, ee_body_id, 0:3]
    ee_quat_w = read_data(env.robot.data, "body_state_w")[:, ee_body_id, 3:7]

    ee_pos_b, _ = subtract_frame_transforms(robot_root_pos, robot_root_quat, ee_pos_w, ee_quat_w)
    relative_target_quat = quat_mul(quat_inv(current_semantic_quat), target_semantic_quat)
    # q and -q describe the same rotation. A fixed hemisphere avoids an
    # artificial sign discontinuity in the policy observation.
    relative_target_quat = torch.where(
        relative_target_quat[:, :1] < 0.0,
        -relative_target_quat,
        relative_target_quat,
    )
    target_delta_b = quat_apply(quat_inv(robot_root_quat), target_delta_w)
    axis_error = torch.acos(torch.clamp(metrics["axis_alignment"], -1.0, 1.0))
    target_quat_error = torch.maximum(axis_error, metrics["twist_error"]).unsqueeze(-1)

    gripper_opening = read_data(env.robot.data, "joint_pos")[:, gripper_joint_ids].mean(dim=-1, keepdim=True)
    joint_pos = read_data(env.robot.data, "joint_pos")[:, arm_joint_ids]
    joint_vel = read_data(env.robot.data, "joint_vel")[:, arm_joint_ids]
    closure = _closure_from_opening(env, gripper_opening)
    ee_lin_vel_b = _ee_linear_velocity_b(env, ee_body_id)

    obs_parts = [
        ee_pos_b,
        relative_target_quat,
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
    peg_state = torch.cat(
        [
            read_data(env.peg.data, "root_pos_w")[:, 0:3],
            read_data(env.peg.data, "root_quat_w"),
            read_data(env.peg.data, "root_lin_vel_w"),
            read_data(env.peg.data, "root_ang_vel_w"),
        ],
        dim=-1,
    )
    hole_state = torch.cat(
        [
            read_data(env.hole.data, "root_pos_w")[:, 0:3],
            read_data(env.hole.data, "root_quat_w"),
            read_data(env.hole.data, "root_lin_vel_w"),
            read_data(env.hole.data, "root_ang_vel_w"),
        ],
        dim=-1,
    )
    return torch.nan_to_num(
        torch.cat([left_own, right_own, peg_state, hole_state], dim=-1),
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


def _object_tilt(env) -> tuple[torch.Tensor, torch.Tensor]:
    """Return object tilt angle in degrees and normalized squared penalty."""

    world_z = torch.tensor([0.0, 0.0, 1.0], device=env.device).expand(env.num_envs, -1)
    object_up = matrix_from_quat(read_data(env.object.data, "root_quat_w"))[:, :, 2]
    cos_tilt = torch.clamp((object_up * world_z).sum(dim=-1), -1.0, 1.0)
    tilt_deg = torch.rad2deg(torch.acos(cos_tilt))
    free = float(env.cfg.tilt_free_deg)
    bad = max(float(env.cfg.tilt_bad_deg), free + 1.0e-6)
    tilt_penalty = torch.clamp((tilt_deg - free) / (bad - free), 0.0, 1.0).square()
    return tilt_deg, tilt_penalty


def _distance_reward(x: torch.Tensor, std: float) -> torch.Tensor:
    """Bounded distance reward with a directly interpretable length scale."""

    return torch.exp(-x / max(float(std), 1.0e-6))


def _compute_factory_keypoint_reward(
    env, metrics: dict[str, torch.Tensor]
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute staged keypoint rewards while preserving old trace field names."""

    num_keypoints = int(getattr(env.cfg, "num_keypoints", 4))
    keypoint_spacing = float(getattr(env.cfg, "keypoint_spacing", 0.005))
    offsets = torch.arange(num_keypoints, device=env.device, dtype=torch.float32) * keypoint_spacing
    offsets = offsets.view(1, num_keypoints, 1)

    peg_tip = metrics["peg_tip_w"].unsqueeze(1)
    peg_axis = metrics["peg_axis_w"].unsqueeze(1)
    hole_axis = metrics["hole_axis_w"].unsqueeze(1)

    keypoints_held = peg_tip - offsets * peg_axis
    keypoints_insert = metrics["insertion_target_w"].unsqueeze(1) - offsets * hole_axis
    pre_clearance = float(getattr(env.cfg, "preinsert_clearance", 0.03))
    pre_tip = metrics["hole_entrance_w"] - pre_clearance * metrics["hole_axis_w"]
    keypoints_pre = pre_tip.unsqueeze(1) - offsets * hole_axis

    pre_dist_per_point = torch.linalg.vector_norm(keypoints_held - keypoints_pre, dim=-1)
    insert_dist_per_point = torch.linalg.vector_norm(keypoints_held - keypoints_insert, dim=-1)
    pre_dist = pre_dist_per_point.mean(dim=-1)
    insert_dist = insert_dist_per_point.mean(dim=-1)
    r_preinsert = _distance_reward(pre_dist, float(getattr(env.cfg, "preinsert_std", 0.04)))
    r_insert_pose = _distance_reward(insert_dist, float(getattr(env.cfg, "insert_pose_std", 0.02)))

    return insert_dist, {
        "r_keypoint_baseline": r_preinsert,
        "r_keypoint_coarse": r_insert_pose,
        "r_keypoint_fine": r_insert_pose,
        "r_preinsert": r_preinsert,
        "r_insert_pose": r_insert_pose,
        "preinsert_dist": pre_dist,
        "insert_pose_dist": insert_dist,
        "keypoint_dist_per_point": insert_dist_per_point,
        "preinsert_dist_per_point": pre_dist_per_point,
        "insert_pose_dist_per_point": insert_dist_per_point,
        "keypoints_held": keypoints_held,
        "keypoints_fixed": keypoints_insert,
        "keypoints_pre": keypoints_pre,
        "keypoints_insert": keypoints_insert,
        "keypoint_offsets": offsets.reshape(-1),
    }


def compute_openarm_re_rewards(env, ctx: dict) -> dict[str, torch.Tensor]:
    """Compute shared cooperative peg-in-hole reward for both arms."""

    metrics = compute_peg_in_hole_metrics(env, ctx)
    tip_dist = metrics["tip_dist"]
    lateral_error = metrics["lateral_error"]
    axis_alignment = metrics["axis_alignment"]
    twist_alignment = metrics["twist_alignment"]
    insertion_depth = metrics["insertion_depth"]

    hold_now = _insertion_hold_condition(env, metrics)

    # Termination already advanced the hold counter for this transition.
    success_now = ctx["termination_success"].clone()
    keypoint_dist, keypoint_rewards = _compute_factory_keypoint_reward(env, metrics)
    r_preinsert = keypoint_rewards["r_preinsert"]
    r_insert_pose = keypoint_rewards["r_insert_pose"]
    keypoint_dist_per_point = keypoint_rewards["keypoint_dist_per_point"]
    keypoints_held = keypoint_rewards["keypoints_held"]
    keypoints_fixed = keypoint_rewards["keypoints_fixed"]
    keypoint_axis_held = _normalize_vec(keypoints_held[:, -1] - keypoints_held[:, 0])
    keypoint_axis_fixed = _normalize_vec(keypoints_fixed[:, -1] - keypoints_fixed[:, 0])
    keypoint_axis_alignment = torch.clamp((keypoint_axis_held * keypoint_axis_fixed).sum(dim=-1), -1.0, 1.0)
    held_k0_error = torch.linalg.vector_norm(keypoints_held[:, 0] - metrics["peg_tip_w"], dim=-1)
    fixed_k0_error = torch.linalg.vector_norm(keypoints_fixed[:, 0] - metrics["insertion_target_w"], dim=-1)
    held_spacing = torch.linalg.vector_norm(keypoints_held[:, 1:] - keypoints_held[:, :-1], dim=-1)
    fixed_spacing = torch.linalg.vector_norm(keypoints_fixed[:, 1:] - keypoints_fixed[:, :-1], dim=-1)
    axis_gate, lateral_gate, twist_gate, insertion_gate = _smooth_insertion_gate(
        env,
        axis_alignment,
        lateral_error,
        twist_alignment,
    )
    depth_progress = target_depth_quality(
        insertion_depth,
        metrics["target_insertion_depth"],
        float(getattr(env.cfg, "depth_overrun_std", 0.0)),
    )
    axis_error_deg = torch.rad2deg(torch.acos(torch.clamp(axis_alignment, -1.0, 1.0)))
    twist_error_deg = torch.rad2deg(metrics["twist_error"])
    # Disable the pre-insertion attractor as the insertion objective becomes
    # active; otherwise the two stages pull the peg tip in opposite directions.
    r_preinsert_stage = (1.0 - insertion_gate) * r_preinsert
    r_insert_pose_gated = insertion_gate * r_insert_pose
    r_depth_gated = insertion_gate * depth_progress
    settle_scale = float(getattr(env.cfg, "reward_insert_settle_scale", 0.0))
    if settle_scale > 0.0:
        body_vel_w = read_data(env.robot.data, "body_state_w")[:, :, 7:10]
        left_speed = torch.linalg.vector_norm(body_vel_w[:, ctx["left_ee_body_id"]], dim=-1)
        right_speed = torch.linalg.vector_norm(body_vel_w[:, ctx["right_ee_body_id"]], dim=-1)
        ee_speed = 0.5 * (left_speed + right_speed)
        r_insert_settle = insertion_gate * insertion_settle_quality(
            axis_alignment,
            depth_progress,
            ee_speed,
            float(env.cfg.success_axis_threshold),
            float(env.cfg.insert_settle_axis_width),
            float(env.cfg.insert_settle_speed_std),
        )
    else:
        r_insert_settle = torch.zeros_like(insertion_gate)

    # Keypoint distance can be small even with a tilted peg. Give the actor
    # an alignment gradient near pre-insertion; zero scale preserves v4.
    axis_quality = axis_alignment_quality(axis_alignment, float(getattr(env.cfg, "alignment_shaping_power", 1.0)))
    r_axis_alignment = r_preinsert_stage * axis_quality
    # Decouple lateral convergence from insertion depth. The pre-insertion
    # distance alone can improve while the peg remains off-axis; this term
    # retains a gradient near the mouth without rewarding a misaligned axis.
    r_lateral_alignment = (
        (1.0 - insertion_gate) * _distance_reward(lateral_error, float(env.cfg.lateral_alignment_std)) * axis_quality
    )

    action_dim = get_agent_action_dim(env)
    left_action_rate = _agent_action_rate(env, "left_arm", action_dim)
    right_action_rate = _agent_action_rate(env, "right_arm", action_dim)
    action_penalty = 0.5 * (left_action_rate + right_action_rate)

    # Positive preparation bonuses disappear as the insertion gate opens.
    # Referencing those bounded qualities to their maxima makes preparation
    # deficits nonpositive, removing that artificial reward loss on alignment.
    # This is a distinct, opt-in reward experiment; success geometry is shared.
    preinsert_shaping_offset = torch.zeros_like(insertion_gate)
    if bool(getattr(env.cfg, "use_preinsert_deficit_reward", False)):
        preparation_maximum = (
            float(env.cfg.reward_preinsert_scale)
            + float(env.cfg.reward_axis_alignment_scale)
            + float(env.cfg.reward_lateral_alignment_scale)
        )
        preinsert_shaping_offset = -(1.0 - insertion_gate) * preparation_maximum

    team_reward = (
        float(env.cfg.reward_preinsert_scale) * r_preinsert_stage
        + float(env.cfg.reward_axis_alignment_scale) * r_axis_alignment
        + float(env.cfg.reward_lateral_alignment_scale) * r_lateral_alignment
        + float(env.cfg.reward_insert_pose_scale) * r_insert_pose_gated
        + float(env.cfg.reward_depth_scale) * r_depth_gated
        + preinsert_shaping_offset
        + settle_scale * r_insert_settle
        + float(env.cfg.reward_success_scale) * success_now.float()
        - float(env.cfg.reward_action_rate_scale) * action_penalty
    )
    team_reward = torch.nan_to_num(team_reward, nan=0.0, posinf=0.0, neginf=0.0)

    # Paper logs only: compact blocks for presentation/analysis. These do not
    # change the scalar training reward above.
    progress = r_preinsert_stage + r_depth_gated
    quality = r_insert_pose_gated
    regularization = float(env.cfg.reward_action_rate_scale) * action_penalty

    previous_debug_stats = dict(ctx.get("debug_stats", {}))
    ctx["debug_stats"] = {
        **previous_debug_stats,
        "team_reward": team_reward.mean().item(),
        "left_reward": team_reward.mean().item(),
        "right_reward": team_reward.mean().item(),
        "left_reward_raw": team_reward.mean().item(),
        "right_reward_raw": team_reward.mean().item(),
        "paper/progress": progress.mean().item(),
        "paper/quality": quality.mean().item(),
        "paper/preinsert": r_preinsert_stage.mean().item(),
        "preinsert_shaping_offset": preinsert_shaping_offset.mean().item(),
        "paper/insert_pose": r_insert_pose_gated.mean().item(),
        "paper/depth": r_depth_gated.mean().item(),
        "paper/insert_settle": r_insert_settle.mean().item(),
        "paper/success": success_now.float().mean().item(),
        "paper/hold_now": hold_now.float().mean().item(),
        "paper/regularization": regularization.mean().item(),
        "peg_hole/tip_dist": tip_dist.mean().item(),
        "peg_hole/lateral_error": lateral_error.mean().item(),
        "peg_hole/axis_alignment": axis_alignment.mean().item(),
        "peg_hole/axis_error_deg": axis_error_deg.mean().item(),
        "peg_hole/twist_alignment": twist_alignment.mean().item(),
        "peg_hole/twist_error_deg": twist_error_deg.mean().item(),
        "peg_hole/insertion_depth": insertion_depth.mean().item(),
        "peg_hole/success_rate": success_now.float().mean().item(),
        "peg_hole/hold_now_rate": hold_now.float().mean().item(),
        "peg_hole/keypoint_num": float(int(env.cfg.num_keypoints)),
        "peg_hole/keypoint_spacing": float(env.cfg.keypoint_spacing),
        "peg_hole/target_insertion_depth": metrics["target_insertion_depth"].mean().item(),
        "peg_hole/shaft_radial_envelope": metrics["shaft_radial_envelope"].mean().item(),
        "peg_hole/success_lateral_threshold": float(env.cfg.success_lateral_threshold),
        "peg_hole/success_axis_threshold": float(env.cfg.success_axis_threshold),
        "peg_hole/success_depth_threshold": metrics["success_depth"].mean().item(),
        "success_lateral_threshold": float(env.cfg.success_lateral_threshold),
        "success_axis_threshold": float(env.cfg.success_axis_threshold),
        "success_depth_threshold": metrics["success_depth"].mean().item(),
        "peg_hole/keypoint_dist": keypoint_dist.mean().item(),
        "peg_hole/keypoint_dist_0": keypoint_dist_per_point[:, 0].mean().item(),
        "peg_hole/keypoint_dist_last": keypoint_dist_per_point[:, -1].mean().item(),
        "peg_hole/keypoint_axis_alignment": keypoint_axis_alignment.mean().item(),
        "peg_hole/preinsert_dist": keypoint_rewards["preinsert_dist"].mean().item(),
        "peg_hole/insert_pose_dist": keypoint_rewards["insert_pose_dist"].mean().item(),
        "peg_hole/held_k0_error": held_k0_error.mean().item(),
        "peg_hole/fixed_k0_error": fixed_k0_error.mean().item(),
        "peg_hole/held_keypoint_spacing": held_spacing.mean().item(),
        "peg_hole/fixed_keypoint_spacing": fixed_spacing.mean().item(),
        "peg_hole/r_keypoint_baseline": r_preinsert_stage.mean().item(),
        "peg_hole/r_keypoint_coarse": r_insert_pose_gated.mean().item(),
        "peg_hole/r_keypoint_fine": r_depth_gated.mean().item(),
        "peg_hole/r_keypoint_baseline_raw": r_preinsert.mean().item(),
        "peg_hole/r_keypoint_coarse_raw": r_insert_pose.mean().item(),
        "peg_hole/r_keypoint_fine_raw": depth_progress.mean().item(),
        "peg_hole/r_preinsert": r_preinsert_stage.mean().item(),
        "peg_hole/r_lateral_alignment": r_lateral_alignment.mean().item(),
        "peg_hole/r_insert_pose": r_insert_pose_gated.mean().item(),
        "peg_hole/r_depth": r_depth_gated.mean().item(),
        "peg_hole/r_insert_settle": r_insert_settle.mean().item(),
        "peg_hole/baseline_axis_gate": torch.ones_like(axis_gate).mean().item(),
        "peg_hole/axis_gate": axis_gate.mean().item(),
        "peg_hole/lateral_gate": lateral_gate.mean().item(),
        "peg_hole/twist_gate": twist_gate.mean().item(),
        "peg_hole/insertion_gate": insertion_gate.mean().item(),
        "peg_hole/depth_progress": depth_progress.mean().item(),
        "peg_hole/fixed_insert_obs": 1.0,
        "left_closure": 1.0,
        "right_closure": 1.0,
        "tip_dist": tip_dist.mean().item(),
        "lateral_error": lateral_error.mean().item(),
        "axis_alignment": axis_alignment.mean().item(),
        "axis_error_deg": axis_error_deg.mean().item(),
        "twist_alignment": twist_alignment.mean().item(),
        "twist_error_deg": twist_error_deg.mean().item(),
        "insertion_depth": insertion_depth.mean().item(),
        "success_ratio": (ctx["success_hold_count"] >= int(env.cfg.hold_required_steps)).float().mean().item(),
        "strict_success_ratio": (ctx["success_hold_count"] >= int(env.cfg.hold_required_steps)).float().mean().item(),
        "hold_count_mean": ctx["success_hold_count"].float().mean().item(),
        "hold_count_p90": torch.quantile(ctx["success_hold_count"].float(), 0.90).item(),
        "hold_count_max": ctx["success_hold_count"].float().max().item(),
        "hold_required_steps": float(env.cfg.hold_required_steps),
        "keypoint_num": float(int(env.cfg.num_keypoints)),
        "keypoint_spacing": float(env.cfg.keypoint_spacing),
        "target_insertion_depth": metrics["target_insertion_depth"].mean().item(),
        "keypoint_dist": keypoint_dist.mean().item(),
        "keypoint_dist_0": keypoint_dist_per_point[:, 0].mean().item(),
        "keypoint_dist_last": keypoint_dist_per_point[:, -1].mean().item(),
        "keypoint_axis_alignment": keypoint_axis_alignment.mean().item(),
        "preinsert_dist": keypoint_rewards["preinsert_dist"].mean().item(),
        "insert_pose_dist": keypoint_rewards["insert_pose_dist"].mean().item(),
        "held_k0_error": held_k0_error.mean().item(),
        "fixed_k0_error": fixed_k0_error.mean().item(),
        "held_keypoint_spacing": held_spacing.mean().item(),
        "fixed_keypoint_spacing": fixed_spacing.mean().item(),
        "keypoint_reward_baseline": r_preinsert_stage.mean().item(),
        "keypoint_reward_coarse": r_insert_pose_gated.mean().item(),
        "keypoint_reward_fine": r_depth_gated.mean().item(),
        "keypoint_reward_baseline_raw": r_preinsert.mean().item(),
        "keypoint_reward_coarse_raw": r_insert_pose.mean().item(),
        "keypoint_reward_fine_raw": depth_progress.mean().item(),
        "preinsert_reward": r_preinsert_stage.mean().item(),
        "insert_pose_reward": r_insert_pose_gated.mean().item(),
        "depth_reward": r_depth_gated.mean().item(),
        "baseline_axis_gate": torch.ones_like(axis_gate).mean().item(),
        "axis_gate": axis_gate.mean().item(),
        "lateral_gate": lateral_gate.mean().item(),
        "twist_gate": twist_gate.mean().item(),
        "insertion_gate": insertion_gate.mean().item(),
        "depth_progress": depth_progress.mean().item(),
        "action_rate_penalty": action_penalty.mean().item(),
    }
    return {"left_arm": team_reward, "right_arm": team_reward}


def compute_openarm_re_terminations(env, ctx: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return terminated, success, and invalid flags."""

    metrics = compute_peg_in_hole_metrics(env, ctx)
    hold_now = _insertion_hold_condition(env, metrics)
    ctx["success_hold_count"] = torch.where(
        hold_now,
        ctx["success_hold_count"] + 1,
        torch.zeros_like(ctx["success_hold_count"]),
    )
    success = ctx["success_hold_count"] >= int(env.cfg.hold_required_steps)

    far = metrics["tip_dist"] > float(env.cfg.max_tip_distance)
    side_wall_invalid = side_wall_penetration(
        metrics["insertion_depth"],
        metrics["lateral_error"],
        metrics["tip_dist"],
        depth_threshold=float(getattr(env.cfg, "wall_penetration_depth_threshold", 0.0)),
        lateral_threshold=float(getattr(env.cfg, "wall_penetration_lateral_threshold", 0.015)),
        max_tip_distance=float(getattr(env.cfg, "wall_penetration_max_tip_distance", float("inf"))),
    )
    state = torch.cat(
        [
            read_data(env.peg.data, "root_state_w"),
            read_data(env.hole.data, "root_state_w"),
            read_data(env.robot.data, "joint_pos"),
            read_data(env.robot.data, "joint_vel"),
        ],
        dim=-1,
    )
    invalid = (~torch.isfinite(state).all(dim=-1)) | side_wall_invalid
    ctx["termination_success"] = success
    ctx["termination_far"] = far
    ctx["termination_side_wall"] = side_wall_invalid
    ctx["termination_invalid"] = invalid
    ctx.setdefault("debug_stats", {}).update(
        {
            "strict_success_ratio": success.float().mean().item(),
            "success_ratio": success.float().mean().item(),
            "hold_count_mean": ctx["success_hold_count"].float().mean().item(),
            "hold_count_p90": torch.quantile(ctx["success_hold_count"].float(), 0.90).item(),
            "hold_count_max": ctx["success_hold_count"].float().max().item(),
            "hold_required_steps": float(env.cfg.hold_required_steps),
            "peg_hole/far_ratio": far.float().mean().item(),
            "peg_hole/side_wall_penetration_rate": side_wall_invalid.float().mean().item(),
        }
    )
    terminated = success | far | invalid
    return terminated, success, invalid


def collect_openarm_re_trace_signals(env, ctx: dict, env_index: int = 0) -> dict:
    """Collect real RE task signals for JSON mode-trace analysis.

    Kinematic, insertion, and reward fields are recorded from the live
    environment to keep evaluation plots honest.
    """

    idx = int(max(0, min(int(env_index), int(env.num_envs) - 1)))
    left_own = _agent_observation(env, ctx, "left")
    right_own = _agent_observation(env, ctx, "right")
    log = ctx.get("debug_stats", {})
    metrics = compute_peg_in_hole_metrics(env, ctx)

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

    context_debug = getattr(env, "_motion_context_last_generated_debug", None)
    if not isinstance(context_debug, dict):
        context_debug = getattr(env, "_motion_context_debug", {})

    def context_debug_scalar(name: str, default: float = 0.0) -> float:
        return env_scalar(context_debug.get(name), default)

    def obs_scalar(obs: torch.Tensor, column: int) -> float:
        return float(obs[idx, column].detach().cpu())

    def obs_vector(obs: torch.Tensor, start: int, end: int) -> list[float]:
        return obs[idx, start:end].detach().float().cpu().tolist()

    left_arm_action_mag = 0.0
    right_arm_action_mag = 0.0
    left_arm_action_vec = [0.0] * 7
    right_arm_action_vec = [0.0] * 7
    actions = getattr(env, "actions", {})
    if isinstance(actions, dict):
        left_action = actions.get("left_arm")
        right_action = actions.get("right_arm")
        if isinstance(left_action, torch.Tensor) and left_action.numel() > 0:
            left_arm_action_vec = left_action[idx, :7].detach().float().cpu().tolist()
            left_arm_action_mag = float(torch.linalg.vector_norm(left_action[idx, :7]).detach().cpu())
        if isinstance(right_action, torch.Tensor) and right_action.numel() > 0:
            right_arm_action_vec = right_action[idx, :7].detach().float().cpu().tolist()
            right_arm_action_mag = float(torch.linalg.vector_norm(right_action[idx, :7]).detach().cpu())

    left_closure = obs_scalar(left_own, 26)
    right_closure = obs_scalar(right_own, 26)
    motion_context = getattr(env, "_motion_context_last_generated", None)
    if not isinstance(motion_context, dict):
        motion_context = compute_openarm_re_motion_contexts(env, ctx)
    left_context = motion_context["left_arm"][idx].detach().float().cpu().tolist()
    right_context = motion_context["right_arm"][idx].detach().float().cpu().tolist()
    keypoint_dist, keypoint_rewards = _compute_factory_keypoint_reward(env, metrics)
    keypoint_dist_per_point = keypoint_rewards["keypoint_dist_per_point"]
    keypoints_held = keypoint_rewards["keypoints_held"]
    keypoints_fixed = keypoint_rewards["keypoints_fixed"]
    keypoint_offsets = keypoint_rewards["keypoint_offsets"]
    axis_gate, lateral_gate, twist_gate, insertion_gate = _smooth_insertion_gate(
        env,
        metrics["axis_alignment"],
        metrics["lateral_error"],
        metrics["twist_alignment"],
    )
    depth_progress = torch.clamp(
        metrics["insertion_depth"] / metrics["target_insertion_depth"],
        0.0,
        1.0,
    )
    inserted_ok = bool(_insertion_hold_condition(env, metrics)[idx])
    hold_ok = int(ctx["success_hold_count"][idx].detach().cpu()) > 0
    strict_success = int(ctx["success_hold_count"][idx].detach().cpu()) >= int(env.cfg.hold_required_steps)

    calibration = motion_context_calibration_diagnostics(env)
    return {
        "success": bool(strict_success),
        "object_z": float(metrics["peg_tip_w"][idx, 2].detach().cpu()),
        "object_dz": float(metrics["insertion_depth"][idx].detach().cpu()),
        "peg_tip_w": metrics["peg_tip_w"][idx].detach().float().cpu().tolist(),
        "hole_entrance_w": metrics["hole_entrance_w"][idx].detach().float().cpu().tolist(),
        "hole_bottom_w": metrics["hole_bottom_w"][idx].detach().float().cpu().tolist(),
        "peg_axis_w": metrics["peg_axis_w"][idx].detach().float().cpu().tolist(),
        "hole_axis_w": metrics["hole_axis_w"][idx].detach().float().cpu().tolist(),
        "tip_dist": float(metrics["tip_dist"][idx].detach().cpu()),
        "lateral_error": float(metrics["lateral_error"][idx].detach().cpu()),
        "axis_alignment": float(metrics["axis_alignment"][idx].detach().cpu()),
        "twist_alignment": float(metrics["twist_alignment"][idx].detach().cpu()),
        "twist_error_deg": float(torch.rad2deg(metrics["twist_error"][idx]).detach().cpu()),
        "insertion_depth": float(metrics["insertion_depth"][idx].detach().cpu()),
        "target_insertion_depth": float(metrics["target_insertion_depth"][idx].detach().cpu()),
        "shaft_radial_envelope": float(metrics["shaft_radial_envelope"][idx].detach().cpu()),
        "keypoint_num": int(env.cfg.num_keypoints),
        "keypoint_spacing": float(env.cfg.keypoint_spacing),
        "keypoint_offsets": keypoint_offsets.detach().float().cpu().tolist(),
        "keypoint_dist": float(keypoint_dist[idx].detach().cpu()),
        "preinsert_dist": float(keypoint_rewards["preinsert_dist"][idx].detach().cpu()),
        "insert_pose_dist": float(keypoint_rewards["insert_pose_dist"][idx].detach().cpu()),
        "keypoint_dist_per_point": keypoint_dist_per_point[idx].detach().float().cpu().tolist(),
        "keypoints_held_w": keypoints_held[idx].detach().float().cpu().tolist(),
        "keypoints_target_w": keypoints_fixed[idx].detach().float().cpu().tolist(),
        "keypoint_reward_baseline": float(
            ((1.0 - insertion_gate) * keypoint_rewards["r_preinsert"])[idx].detach().cpu()
        ),
        "keypoint_reward_coarse": float((insertion_gate * keypoint_rewards["r_insert_pose"])[idx].detach().cpu()),
        "keypoint_reward_fine": float((insertion_gate * depth_progress)[idx].detach().cpu()),
        "preinsert_reward": float(((1.0 - insertion_gate) * keypoint_rewards["r_preinsert"])[idx].detach().cpu()),
        "insert_pose_reward": float((insertion_gate * keypoint_rewards["r_insert_pose"])[idx].detach().cpu()),
        "depth_reward": float((insertion_gate * depth_progress)[idx].detach().cpu()),
        "axis_gate": float(axis_gate[idx].detach().cpu()),
        "lateral_gate": float(lateral_gate[idx].detach().cpu()),
        "twist_gate": float(twist_gate[idx].detach().cpu()),
        "insertion_gate": float(insertion_gate[idx].detach().cpu()),
        "depth_progress": float(depth_progress[idx].detach().cpu()),
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
        # These legacy trace fields have trend semantics only. Keep them
        # neutral for other payloads; *_motion_context records actual values.
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
        "left_prev_gripper_action": 0.0,
        "right_prev_gripper_action": 0.0,
        "left_closure": left_closure,
        "right_closure": right_closure,
        "left_ee_lin_vel": obs_vector(left_own, 27, 30),
        "right_ee_lin_vel": obs_vector(right_own, 27, 30),
        "object_tilt_deg": 0.0,
        "goal_error": float(metrics["lateral_error"][idx].detach().cpu()),
        "xy_error": float(metrics["lateral_error"][idx].detach().cpu()),
        "hprog": float(
            torch.clamp(
                metrics["insertion_depth"][idx] / metrics["target_insertion_depth"][idx],
                0.0,
                1.0,
            )
            .detach()
            .cpu()
        ),
        "left_arm_action_magnitude": left_arm_action_mag,
        "right_arm_action_magnitude": right_arm_action_mag,
        "left_arm_action": left_arm_action_vec,
        "right_arm_action": right_arm_action_vec,
        "height_ok": bool(inserted_ok),
        "tilt_ok": bool(metrics["axis_alignment"][idx] > float(env.cfg.success_axis_threshold)),
        "hold_ok": bool(hold_ok),
        "success_hold_count": int(ctx["success_hold_count"][idx].detach().cpu()),
        "strict_success": bool(strict_success),
        "reward_gap": abs(log_scalar("left_reward") - log_scalar("right_reward")),
        "left_right_closure_gap": abs(left_closure - right_closure),
    }


def reset_openarm_re_context(env, ctx: dict, env_ids: torch.Tensor) -> None:
    """Reset per-episode reference state for selected envs."""

    refresh_peg_hole_frames_from_usd(env, ctx, env_ids)
    ctx["success_hold_count"][env_ids] = 0
    env._motion_context_cache_step = -1
    env._motion_context_cache = None
    if hasattr(env, "_motion_context_action_variation_valid"):
        env._motion_context_action_variation_valid[env_ids] = False
    reset_motion_trend_state(env, env_ids)
