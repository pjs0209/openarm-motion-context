# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Deploy-side AprilTag pose fusion for the Paper OpenArm lift task.

This module contains no Isaac Sim dependency. The real robot process should
provide:

* ``T_base_camera`` from robot FK/TF and calibrated camera extrinsics.
* ``T_camera_tag`` from the AprilTag detector.
* ``T_box_tag`` and ``T_box_grip`` from the same USD/calibration values used in
  simulation.

The output is the same actor-facing target representation used during training:
grip target position delta in robot-base frame and EE-to-target quaternion error.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


EPS = 1.0e-8


def normalize_quat_wxyz(quat: torch.Tensor) -> torch.Tensor:
    """Normalize quaternions in wxyz convention."""

    return quat / torch.linalg.vector_norm(quat, dim=-1, keepdim=True).clamp_min(EPS)


def quat_inv_wxyz(quat: torch.Tensor) -> torch.Tensor:
    """Invert unit quaternions in wxyz convention."""

    out = quat.clone()
    out[..., 1:] = -out[..., 1:]
    return normalize_quat_wxyz(out)


def quat_mul_wxyz(q0: torch.Tensor, q1: torch.Tensor) -> torch.Tensor:
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
    return normalize_quat_wxyz(out)


def quat_apply_wxyz(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    """Rotate vectors by quaternions in wxyz convention."""

    q_vec = quat[..., 1:]
    q_w = quat[..., 0:1]
    t = 2.0 * torch.cross(q_vec, vec, dim=-1)
    return vec + q_w * t + torch.cross(q_vec, t, dim=-1)


def compose_pose(
    parent_pos: torch.Tensor,
    parent_quat: torch.Tensor,
    child_pos: torch.Tensor,
    child_quat: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compose ``T_parent_child`` after ``T_world_parent``."""

    pos = parent_pos + quat_apply_wxyz(parent_quat, child_pos)
    quat = quat_mul_wxyz(parent_quat, child_quat)
    return pos, quat


def invert_pose(pos: torch.Tensor, quat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Invert a pose."""

    inv_quat = quat_inv_wxyz(quat)
    inv_pos = -quat_apply_wxyz(inv_quat, pos)
    return inv_pos, inv_quat


def relative_pose(
    root_pos: torch.Tensor,
    root_quat: torch.Tensor,
    child_pos: torch.Tensor,
    child_quat: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Express a child/world pose in the root frame."""

    inv_root_pos, inv_root_quat = invert_pose(root_pos, root_quat)
    return compose_pose(inv_root_pos, inv_root_quat, child_pos, child_quat)


def quat_angle_error_wxyz(q0: torch.Tensor, q1: torch.Tensor) -> torch.Tensor:
    """Return unsigned angular error between two wxyz quaternions in radians."""

    q0 = normalize_quat_wxyz(q0)
    q1 = normalize_quat_wxyz(q1)
    dot = torch.abs((q0 * q1).sum(dim=-1)).clamp(max=1.0)
    return 2.0 * torch.acos(dot)


def _weighted_quat_average(quats: torch.Tensor, weights: torch.Tensor, dim: int) -> torch.Tensor:
    """Average quaternions after aligning signs to the first weighted sample."""

    reference_indices = weights.argmax(dim=dim, keepdim=True)
    gather_indices = reference_indices.unsqueeze(-1).expand(*reference_indices.shape, 4)
    quat_ref = torch.gather(quats, dim, gather_indices)
    quat_sign = torch.where(
        (quats * quat_ref).sum(dim=-1, keepdim=True) < 0.0,
        -torch.ones_like(quats[..., 0:1]),
        torch.ones_like(quats[..., 0:1]),
    )
    denom = weights.sum(dim=dim).clamp_min(EPS).unsqueeze(-1)
    return normalize_quat_wxyz((quats * quat_sign * weights.unsqueeze(-1)).sum(dim=dim) / denom)


def fuse_camera_tag_detections_to_base(
    camera_pos_base: torch.Tensor,
    camera_quat_base: torch.Tensor,
    tag_pos_camera: torch.Tensor,
    tag_quat_camera: torch.Tensor,
    visible: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert and fuse multi-camera tag detections into base-frame tag poses.

    Shapes:
        camera_pos_base: ``(C, 3)`` or ``(B, C, 3)``
        camera_quat_base: ``(C, 4)`` or ``(B, C, 4)``
        tag_pos_camera: ``(C, T, 3)`` or ``(B, C, T, 3)``
        tag_quat_camera: ``(C, T, 4)`` or ``(B, C, T, 4)``
        visible: optional ``(C, T)`` or ``(B, C, T)``

    Returns:
        tag_pos_base: ``(T, 3)`` or ``(B, T, 3)``
        tag_quat_base: ``(T, 4)`` or ``(B, T, 4)``
    """

    squeeze_batch = camera_pos_base.dim() == 2
    if squeeze_batch:
        camera_pos_base = camera_pos_base.unsqueeze(0)
        camera_quat_base = camera_quat_base.unsqueeze(0)
        tag_pos_camera = tag_pos_camera.unsqueeze(0)
        tag_quat_camera = tag_quat_camera.unsqueeze(0)
        visible = None if visible is None else visible.unsqueeze(0)

    batch_size, num_cameras, num_tags, _ = tag_pos_camera.shape
    camera_quat_expanded = camera_quat_base[:, :, None, :].expand(-1, -1, num_tags, -1)
    tag_pos_base_all = camera_pos_base[:, :, None, :] + quat_apply_wxyz(
        camera_quat_expanded.reshape(-1, 4),
        tag_pos_camera.reshape(-1, 3),
    ).reshape(batch_size, num_cameras, num_tags, 3)
    tag_quat_base_all = quat_mul_wxyz(
        camera_quat_expanded.reshape(-1, 4),
        tag_quat_camera.reshape(-1, 4),
    ).reshape(batch_size, num_cameras, num_tags, 4)

    if visible is None:
        weights = torch.ones((batch_size, num_cameras, num_tags), device=tag_pos_camera.device)
    else:
        weights = visible.to(device=tag_pos_camera.device, dtype=tag_pos_camera.dtype).clamp_min(0.0)
    denom = weights.sum(dim=1).clamp_min(EPS).unsqueeze(-1)

    tag_pos_base = (tag_pos_base_all * weights.unsqueeze(-1)).sum(dim=1) / denom
    tag_quat_base = _weighted_quat_average(tag_quat_base_all, weights, dim=1)

    if squeeze_batch:
        return tag_pos_base.squeeze(0), tag_quat_base.squeeze(0)
    return tag_pos_base, tag_quat_base


def estimate_box_pose_from_tags(
    tag_pos_base: torch.Tensor,
    tag_quat_base: torch.Tensor,
    tag_pos_box: torch.Tensor,
    tag_quat_box: torch.Tensor,
    visible_tags: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Estimate ``T_base_box`` from base-frame tag poses and known ``T_box_tag``."""

    squeeze_batch = tag_pos_base.dim() == 2
    if squeeze_batch:
        tag_pos_base = tag_pos_base.unsqueeze(0)
        tag_quat_base = tag_quat_base.unsqueeze(0)
        tag_pos_box = tag_pos_box.unsqueeze(0)
        tag_quat_box = tag_quat_box.unsqueeze(0)
        visible_tags = None if visible_tags is None else visible_tags.unsqueeze(0)
    elif tag_pos_box.dim() == 2:
        tag_pos_box = tag_pos_box.unsqueeze(0).expand(tag_pos_base.shape[0], -1, -1)
        tag_quat_box = tag_quat_box.unsqueeze(0).expand(tag_pos_base.shape[0], -1, -1)

    box_pos_estimates = []
    box_quat_estimates = []
    for tag_id in range(tag_pos_base.shape[1]):
        box_to_tag_inv_pos, box_to_tag_inv_quat = invert_pose(tag_pos_box[:, tag_id], tag_quat_box[:, tag_id])
        box_pos, box_quat = compose_pose(
            tag_pos_base[:, tag_id],
            tag_quat_base[:, tag_id],
            box_to_tag_inv_pos,
            box_to_tag_inv_quat,
        )
        box_pos_estimates.append(box_pos)
        box_quat_estimates.append(box_quat)

    box_pos_by_tag = torch.stack(box_pos_estimates, dim=1)
    box_quat_by_tag = torch.stack(box_quat_estimates, dim=1)

    if visible_tags is None:
        weights = torch.ones(box_pos_by_tag.shape[:2], device=box_pos_by_tag.device)
    else:
        weights = visible_tags.to(device=box_pos_by_tag.device, dtype=box_pos_by_tag.dtype).clamp_min(0.0)
    denom = weights.sum(dim=1).clamp_min(EPS).unsqueeze(-1)

    box_pos = (box_pos_by_tag * weights.unsqueeze(-1)).sum(dim=1) / denom
    box_quat = _weighted_quat_average(box_quat_by_tag, weights, dim=1)

    if squeeze_batch:
        return box_pos.squeeze(0), box_quat.squeeze(0)
    return box_pos, box_quat


def compute_grip_targets_from_box_pose(
    box_pos_base: torch.Tensor,
    box_quat_base: torch.Tensor,
    left_grip_pos_box: torch.Tensor,
    left_grip_quat_box: torch.Tensor,
    right_grip_pos_box: torch.Tensor,
    right_grip_quat_box: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Attach left/right grip targets to an estimated box pose."""

    if box_pos_base.dim() == 2 and left_grip_pos_box.dim() == 1:
        left_grip_pos_box = left_grip_pos_box.unsqueeze(0).expand(box_pos_base.shape[0], -1)
        left_grip_quat_box = left_grip_quat_box.unsqueeze(0).expand(box_pos_base.shape[0], -1)
        right_grip_pos_box = right_grip_pos_box.unsqueeze(0).expand(box_pos_base.shape[0], -1)
        right_grip_quat_box = right_grip_quat_box.unsqueeze(0).expand(box_pos_base.shape[0], -1)

    left_pos, left_quat = compose_pose(box_pos_base, box_quat_base, left_grip_pos_box, left_grip_quat_box)
    right_pos, right_quat = compose_pose(box_pos_base, box_quat_base, right_grip_pos_box, right_grip_quat_box)
    return left_pos, left_quat, right_pos, right_quat


def compute_tag_to_grip_offsets(
    tag_pos_box: torch.Tensor,
    tag_quat_box: torch.Tensor,
    grip_pos_box: torch.Tensor,
    grip_quat_box: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``T_tag_grip`` from known ``T_box_tag`` and ``T_box_grip``."""

    if tag_pos_box.dim() == 2 and grip_pos_box.dim() == 1:
        grip_pos_box = grip_pos_box.unsqueeze(0).expand(tag_pos_box.shape[0], -1)
        grip_quat_box = grip_quat_box.unsqueeze(0).expand(tag_pos_box.shape[0], -1)
    tag_inv_pos, tag_inv_quat = invert_pose(tag_pos_box, tag_quat_box)
    return compose_pose(tag_inv_pos, tag_inv_quat, grip_pos_box, grip_quat_box)


def estimate_grip_pose_from_tags(
    tag_pos_base: torch.Tensor,
    tag_quat_base: torch.Tensor,
    tag_pos_box: torch.Tensor,
    tag_quat_box: torch.Tensor,
    grip_pos_box: torch.Tensor,
    grip_quat_box: torch.Tensor,
    visible_tags: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Estimate one grasp pose directly from visible AprilTags.

    This avoids reconstructing the full box pose for the actor path. Each tag
    contributes ``T_base_tag * T_tag_grip`` and the visible estimates are fused.
    """

    squeeze_batch = tag_pos_base.dim() == 2
    if squeeze_batch:
        tag_pos_base = tag_pos_base.unsqueeze(0)
        tag_quat_base = tag_quat_base.unsqueeze(0)
        visible_tags = None if visible_tags is None else visible_tags.unsqueeze(0)
    batch = tag_pos_base.shape[0]
    if tag_pos_box.dim() == 2:
        tag_pos_box = tag_pos_box.unsqueeze(0)
        tag_quat_box = tag_quat_box.unsqueeze(0)
    if tag_pos_box.shape[0] == 1 and batch > 1:
        tag_pos_box = tag_pos_box.expand(batch, -1, -1)
        tag_quat_box = tag_quat_box.expand(batch, -1, -1)
    if grip_pos_box.dim() == 1:
        grip_pos_box = grip_pos_box.unsqueeze(0)
        grip_quat_box = grip_quat_box.unsqueeze(0)
    if grip_pos_box.shape[0] == 1 and batch > 1:
        grip_pos_box = grip_pos_box.expand(batch, -1)
        grip_quat_box = grip_quat_box.expand(batch, -1)
    if tag_pos_box.shape[:2] != tag_pos_base.shape[:2]:
        raise ValueError(
            f"Tag geometry shape {tuple(tag_pos_box.shape)} does not match detections {tuple(tag_pos_base.shape)}"
        )
    if grip_pos_box.shape != (batch, 3) or grip_quat_box.shape != (batch, 4):
        raise ValueError(
            f"Grip pose must broadcast to [{batch}, 3]/[{batch}, 4], got "
            f"{tuple(grip_pos_box.shape)}/{tuple(grip_quat_box.shape)}"
        )

    tag_to_grip_pos, tag_to_grip_quat = compute_tag_to_grip_offsets(
        tag_pos_box.reshape(-1, 3),
        tag_quat_box.reshape(-1, 4),
        grip_pos_box[:, None, :].expand(-1, tag_pos_box.shape[1], -1).reshape(-1, 3),
        grip_quat_box[:, None, :].expand(-1, tag_quat_box.shape[1], -1).reshape(-1, 4),
    )
    tag_to_grip_pos = tag_to_grip_pos.reshape(tag_pos_box.shape[0], tag_pos_box.shape[1], 3)
    tag_to_grip_quat = tag_to_grip_quat.reshape(tag_quat_box.shape[0], tag_quat_box.shape[1], 4)

    grip_pos_by_tag, grip_quat_by_tag = compose_pose(
        tag_pos_base.reshape(-1, 3),
        tag_quat_base.reshape(-1, 4),
        tag_to_grip_pos.reshape(-1, 3),
        tag_to_grip_quat.reshape(-1, 4),
    )
    grip_pos_by_tag = grip_pos_by_tag.reshape(tag_pos_base.shape[0], tag_pos_base.shape[1], 3)
    grip_quat_by_tag = grip_quat_by_tag.reshape(tag_quat_base.shape[0], tag_quat_base.shape[1], 4)

    if visible_tags is None:
        weights = torch.ones(grip_pos_by_tag.shape[:2], device=grip_pos_by_tag.device)
    else:
        weights = visible_tags.to(device=grip_pos_by_tag.device, dtype=grip_pos_by_tag.dtype).clamp_min(0.0)
    denom = weights.sum(dim=1).clamp_min(EPS).unsqueeze(-1)

    grip_pos = (grip_pos_by_tag * weights.unsqueeze(-1)).sum(dim=1) / denom
    grip_quat = _weighted_quat_average(grip_quat_by_tag, weights, dim=1)

    if squeeze_batch:
        return grip_pos.squeeze(0), grip_quat.squeeze(0)
    return grip_pos, grip_quat


def estimate_grip_targets_from_tags(
    tag_pos_base: torch.Tensor,
    tag_quat_base: torch.Tensor,
    tag_pos_box: torch.Tensor,
    tag_quat_box: torch.Tensor,
    left_grip_pos_box: torch.Tensor,
    left_grip_quat_box: torch.Tensor,
    right_grip_pos_box: torch.Tensor,
    right_grip_quat_box: torch.Tensor,
    visible_tags: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Estimate left/right grasp poses directly from base-frame tag poses."""

    left_pos, left_quat = estimate_grip_pose_from_tags(
        tag_pos_base,
        tag_quat_base,
        tag_pos_box,
        tag_quat_box,
        left_grip_pos_box,
        left_grip_quat_box,
        visible_tags,
    )
    right_pos, right_quat = estimate_grip_pose_from_tags(
        tag_pos_base,
        tag_quat_base,
        tag_pos_box,
        tag_quat_box,
        right_grip_pos_box,
        right_grip_quat_box,
        visible_tags,
    )
    return left_pos, left_quat, right_pos, right_quat


def compute_actor_target_from_grip(
    ee_pos_base: torch.Tensor,
    ee_quat_base: torch.Tensor,
    grip_pos_base: torch.Tensor,
    grip_quat_base: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return actor target fields: base-frame position delta and quaternion error."""

    target_delta_base = grip_pos_base - ee_pos_base
    target_quat_error = quat_angle_error_wxyz(ee_quat_base, grip_quat_base).unsqueeze(-1)
    return target_delta_base, target_quat_error


@dataclass
class AprilTagDeployPoseProvider:
    """Stateful deploy helper for KLT pose and grip target estimation."""

    tag_pos_box: torch.Tensor
    tag_quat_box: torch.Tensor
    left_grip_pos_box: torch.Tensor
    left_grip_quat_box: torch.Tensor
    right_grip_pos_box: torch.Tensor
    right_grip_quat_box: torch.Tensor

    def estimate_box_pose(
        self,
        camera_pos_base: torch.Tensor,
        camera_quat_base: torch.Tensor,
        tag_pos_camera: torch.Tensor,
        tag_quat_camera: torch.Tensor,
        visible: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Estimate KLT pose from FK/TF camera poses and detector measurements."""

        tag_pos_base, tag_quat_base = fuse_camera_tag_detections_to_base(
            camera_pos_base,
            camera_quat_base,
            tag_pos_camera,
            tag_quat_camera,
            visible,
        )
        visible_tags = None if visible is None else visible.sum(dim=-2) > 0.0
        return estimate_box_pose_from_tags(
            tag_pos_base,
            tag_quat_base,
            self.tag_pos_box.to(tag_pos_base.device),
            self.tag_quat_box.to(tag_pos_base.device),
            visible_tags,
        )

    def compute_grip_targets(
        self,
        camera_pos_base: torch.Tensor,
        camera_quat_base: torch.Tensor,
        tag_pos_camera: torch.Tensor,
        tag_quat_camera: torch.Tensor,
        visible: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Estimate left/right grip targets directly from visible AprilTags."""

        tag_pos_base, tag_quat_base = fuse_camera_tag_detections_to_base(
            camera_pos_base,
            camera_quat_base,
            tag_pos_camera,
            tag_quat_camera,
            visible,
        )
        visible_tags = None if visible is None else visible.sum(dim=-2) > 0.0
        return estimate_grip_targets_from_tags(
            tag_pos_base,
            tag_quat_base,
            self.tag_pos_box.to(tag_pos_base.device),
            self.tag_quat_box.to(tag_pos_base.device),
            self.left_grip_pos_box.to(tag_pos_base.device),
            self.left_grip_quat_box.to(tag_pos_base.device),
            self.right_grip_pos_box.to(tag_pos_base.device),
            self.right_grip_quat_box.to(tag_pos_base.device),
            visible_tags,
        )

    def compute_grip_targets_via_box_pose(
        self,
        camera_pos_base: torch.Tensor,
        camera_quat_base: torch.Tensor,
        tag_pos_camera: torch.Tensor,
        tag_quat_camera: torch.Tensor,
        visible: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compatibility path: estimate KLT pose, then attach grip targets."""

        box_pos, box_quat = self.estimate_box_pose(
            camera_pos_base,
            camera_quat_base,
            tag_pos_camera,
            tag_quat_camera,
            visible,
        )
        return compute_grip_targets_from_box_pose(
            box_pos,
            box_quat,
            self.left_grip_pos_box.to(box_pos.device),
            self.left_grip_quat_box.to(box_pos.device),
            self.right_grip_pos_box.to(box_pos.device),
            self.right_grip_quat_box.to(box_pos.device),
        )
