# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Public migration boundary: preserve trained WXYZ actor inputs on native3."""

from types import MethodType, SimpleNamespace

import pytest
import torch
import warp as wp

from isaaclab.utils.warp import ProxyArray

from isaaclab_tasks.direct.openarm_motion_context import legacy_math as m
from isaaclab_tasks.direct.openarm_motion_context.envs.ball_camera import ball_semantic_mask
from isaaclab_tasks.direct.openarm_motion_context.legacy_api import read_data


def test_signed_camera_rotation_retains_all_old_local_channels_and_uses_base_axes(monkeypatch):
    """Opposite tilts and antipodal quaternions must retain direction without losing proprioception."""
    from isaaclab_tasks.direct.openarm_motion_context.camera_pose.envs import (
        OpenArmCameraRotationLiftEnv,
        OpenArmCameraRotationLiftEnvCfg,
    )
    from isaaclab_tasks.direct.openarm_motion_context.envs import lift_task_logic as logic
    from isaaclab_tasks.direct.openarm_motion_context.envs.lift_env import OpenArmReIncrementalBimanualLiftEnvCfg

    angle = torch.tensor([torch.pi / 12, -torch.pi / 12, 0.0, torch.pi / 12])
    target = torch.stack((torch.cos(angle / 2), torch.sin(angle / 2), torch.zeros(4), torch.zeros(4)), dim=-1)
    target[-1] *= -1  # Same physical rotation, opposite quaternion representative.
    monkeypatch.setattr(logic, "get_actor_object_pose_w", lambda env, ctx: (torch.zeros(4, 3), target))
    bodies = torch.zeros(4, 2, 13)
    bodies[:, 0, 6] = 1  # Native XYZW identity.
    bodies[:, 1, 3] = torch.sin(torch.tensor(torch.pi / 24))
    bodies[:, 1, 6] = torch.cos(torch.tensor(torch.pi / 24))
    joints = torch.arange(72.0).reshape(4, 18) / 100
    root_quat = torch.tensor([[0.0, 0.0, 2**-0.5, 2**-0.5]]).expand(4, -1)
    env = SimpleNamespace(
        num_envs=4,
        device="cpu",
        common_step_counter=0,
        robot=SimpleNamespace(
            data=SimpleNamespace(
                body_state_w=bodies,
                root_pos_w=torch.zeros(4, 3),
                root_quat_w=root_quat,
                joint_pos=joints,
                joint_vel=joints / 10,
            )
        ),
    )
    ctx = {
        "left_arm_joint_ids": list(range(7)),
        "right_arm_joint_ids": list(range(9, 16)),
        "left_gripper_joint_ids": [7, 8],
        "right_gripper_joint_ids": [16, 17],
        "left_ee_body_id": 0,
        "right_ee_body_id": 1,
    }
    for side in ("left", "right"):
        ctx[f"{side}_target_local_pos"] = torch.zeros(4, 3)
        ctx[f"{side}_target_local_quat"] = torch.tensor([[1.0, 0.0, 0.0, 0.0]]).expand(4, -1)
    env._ctx = ctx
    env.cfg = OpenArmReIncrementalBimanualLiftEnvCfg()
    original = {side: logic._agent_observation(env, ctx, side) for side in ("left", "right")}
    env.cfg = OpenArmCameraRotationLiftEnvCfg()
    env.get_additional_actor_camera_observation = MethodType(
        OpenArmCameraRotationLiftEnv.get_additional_actor_camera_observation, env
    )
    expected_y = {"left": [-1 / 12, 1 / 12, 0.0, -1 / 12], "right": [0.0, 1 / 6, 1 / 12, 0.0]}
    for side in ("left", "right"):
        actual = logic._agent_observation(env, ctx, side)
        assert actual.shape == (4, 33)
        torch.testing.assert_close(actual[:, :30], original[side])
        expected = torch.zeros(4, 3)
        expected[:, 1] = torch.tensor(expected_y[side])
        torch.testing.assert_close(actual[:, 30:], expected, rtol=0, atol=1e-6)


def test_old_policy_quaternion_rotates_and_composes_known_frames():
    q = torch.tensor([[2**-0.5, 0.0, 0.0, 2**-0.5]])  # saved WXYZ, +90deg about Z
    x = torch.tensor([[1.0, 0.0, 0.0]])
    torch.testing.assert_close(m.quat_apply(q, x), torch.tensor([[0.0, 1.0, 0.0]]), atol=1e-6, rtol=0)
    p, combined = m.combine_frame_transforms(torch.tensor([[1.0, 2.0, 3.0]]), q, x, q)
    torch.testing.assert_close(p, torch.tensor([[1.0, 3.0, 3.0]]), atol=1e-6, rtol=0)
    assert abs(float(combined[0, 0])) < 1e-6 and abs(float(combined[0, 3]) - 1) < 1e-6
    back, back_q = m.subtract_frame_transforms(torch.tensor([[1.0, 2.0, 3.0]]), q, p, combined)
    torch.testing.assert_close(back, x, atol=1e-6, rtol=0)
    torch.testing.assert_close(back_q, q, atol=1e-6, rtol=0)


def test_native_proxy_pose_is_old_actor_order_without_mutating_simulator_buffer():
    wp.init()
    native = wp.array(
        [[1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0, 0.1, 0.2, 0.3, 0.0, 0.0, 0.0]], dtype=wp.float32, device="cpu"
    )
    data = SimpleNamespace(root_state_w=ProxyArray(native))
    actual = read_data(data, "root_state_w")
    torch.testing.assert_close(actual[0, :7], torch.tensor([1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0]))
    torch.testing.assert_close(actual[0, 7:10], torch.tensor([0.1, 0.2, 0.3]))
    torch.testing.assert_close(data.root_state_w.torch[0, 3:7], torch.tensor([0.0, 0.0, 0.0, 1.0]))


@pytest.mark.parametrize(
    ("pixels", "labels"),
    [
        ([0, 3, 0, 3], {"0": {"class": "BACKGROUND"}, "3": {"class": "ball"}}),
        # Recorded Isaac Sim6.2 int32 buffer + RGBA metadata, even with colorize=False.
        (
            [-16777216, -16518367, -16777216, -16518367],
            {"(0, 0, 0, 255)": {"class": "UNLABELLED"}, "(33, 243, 3, 255)": {"class": "ball"}},
        ),
    ],
)
def test_camera_mask_selects_ball_in_legacy_and_native_segmentation(pixels, labels):
    image = torch.tensor(pixels, dtype=torch.int32).reshape(1, 2, 2, 1)
    mask = ball_semantic_mask(image, {"idToLabels": labels})
    assert mask.flatten().tolist() == [False, True, False, True]
