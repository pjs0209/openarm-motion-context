"""Example deploy-side AprilTag pose conversion for OpenArm Paper Lift.

This script is intentionally SDK-agnostic. Replace the placeholder tensors with:

* camera poses from robot FK/TF:
  ``T_base_camera_left/right/chest``
* AprilTag detector outputs:
  ``T_camera_tag`` for apriltag_00 through apriltag_02
* calibrated USD/local constants:
  ``T_box_tag`` and ``T_box_grip``

For a ROS2 end-to-end bridge that subscribes to AprilTag detections, reads TF,
loads the policy checkpoint, and publishes incremental joint commands, use:

``python -m isaaclab_tasks.direct.openarm_motion_context.deploy.deploy_node``
"""

from __future__ import annotations

import torch

from isaaclab_tasks.direct.openarm_motion_context.perception.apriltag_geometry import (
    AprilTagDeployPoseProvider,
    compute_actor_target_from_grip,
)


def main() -> None:
    device = torch.device("cpu")

    # Fill these six tensors from the USD debug print or a calibration file.
    tag_pos_box = torch.zeros((3, 3), device=device)
    tag_quat_box = torch.tensor([[1.0, 0.0, 0.0, 0.0]] * 3, device=device)
    left_grip_pos_box = torch.zeros((3,), device=device)
    left_grip_quat_box = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device)
    right_grip_pos_box = torch.zeros((3,), device=device)
    right_grip_quat_box = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device)

    provider = AprilTagDeployPoseProvider(
        tag_pos_box=tag_pos_box,
        tag_quat_box=tag_quat_box,
        left_grip_pos_box=left_grip_pos_box,
        left_grip_quat_box=left_grip_quat_box,
        right_grip_pos_box=right_grip_pos_box,
        right_grip_quat_box=right_grip_quat_box,
    )

    # From robot FK/TF and calibrated camera extrinsics: one pose per camera.
    camera_pos_base = torch.zeros((3, 3), device=device)
    camera_quat_base = torch.tensor([[1.0, 0.0, 0.0, 0.0]] * 3, device=device)

    # From AprilTag detector: camera-frame pose for each camera/tag pair.
    tag_pos_camera = torch.zeros((3, 3, 3), device=device)
    tag_quat_camera = torch.tensor([[[1.0, 0.0, 0.0, 0.0]] * 3] * 3, device=device)
    visible = torch.ones((3, 3), device=device)

    left_pos_base, left_quat_base, right_pos_base, right_quat_base = provider.compute_grip_targets(
        camera_pos_base,
        camera_quat_base,
        tag_pos_camera,
        tag_quat_camera,
        visible,
    )

    # Example actor target for the left arm. Replace with FK EE pose.
    left_ee_pos_base = torch.zeros((3,), device=device)
    left_ee_quat_base = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device)
    target_delta_base, target_quat_error = compute_actor_target_from_grip(
        left_ee_pos_base,
        left_ee_quat_base,
        left_pos_base,
        left_quat_base,
    )

    print("left_pos_base:", left_pos_base)
    print("right_pos_base:", right_pos_base)
    print("left target_delta_base:", target_delta_base)
    print("left target_quat_error:", target_quat_error)


if __name__ == "__main__":
    main()
