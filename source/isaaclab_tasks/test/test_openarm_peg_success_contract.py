# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Physical seating must use the same reachable target for rewards and success."""

import math
from types import SimpleNamespace

import pytest
import torch

from isaaclab_tasks.direct.openarm_motion_context.envs import peg_in_hole_task_logic as logic
from isaaclab_tasks.direct.openarm_motion_context.envs.peg_contact_grasp import OpenArmContactCirclePegAndHoleEnv
from isaaclab_tasks.utils.hydra import resolve_task_config


def test_contact_seating_does_not_succeed_with_released_or_slipped_parts(monkeypatch):
    """Identical insertion geometry requires two physically retained grasps."""
    n = 3
    tip = torch.tensor([[0.0, 0.0, -0.019]]).expand(n, -1)
    axis = torch.tensor([[0.0, 0.0, -1.0]]).expand(n, -1)
    identity = torch.tensor([[1.0, 0.0, 0.0, 0.0]]).expand(n, -1)
    frames = {
        "peg_tip_w": tip,
        "peg_grip_w": tip - 0.08 * axis,
        "peg_axis_w": axis,
        "hole_entrance_w": torch.zeros(n, 3),
        "hole_bottom_w": 0.025 * axis,
        "hole_axis_w": axis,
        "peg_tip_q_w": identity,
        "hole_bottom_q_w": identity,
    }
    monkeypatch.setattr(logic, "compute_peg_hole_frames_w", lambda env, ctx: frames)
    state = torch.zeros(n, 13)
    state[:, 6] = 1.0
    data = SimpleNamespace(root_state_w=state, root_pos_w=state[:, :3], root_quat_w=state[:, 3:7])
    tcp = torch.zeros(n, 2, 3)
    tcp[2, 0, 0] = 0.05  # third peg has slipped away from its own gripper
    cfg = SimpleNamespace(
        rotational_symmetry_order=0,
        success_depth_ratio=0.9,
        insertion_depth_limit=0.020,
        success_radial_clearance=0.0025,
        peg_collision_radius=0.028,
        success_depth_overrun_tolerance=0.001,
        success_lateral_threshold=0.0025,
        success_axis_threshold=0.97,
        hold_required_steps=45,
        max_tip_distance=1.0,
        require_contact_grasp=True,
        grasp_success_distance=0.020,
        grasp_drop_distance=0.040,
        possible_agents=("left_arm", "right_arm"),
        episode_length_s=30.0,
        sim=SimpleNamespace(dt=1 / 120),
        decimation=2,
    )
    env = object.__new__(OpenArmContactCirclePegAndHoleEnv)
    env.cfg, env.sim, env.scene = cfg, SimpleNamespace(device="cpu"), SimpleNamespace(num_envs=n)
    env.peg, env.hole = SimpleNamespace(data=data), SimpleNamespace(data=data)
    env.robot = SimpleNamespace(
        data=SimpleNamespace(body_pos_w=tcp, joint_pos=torch.zeros(n, 18), joint_vel=torch.zeros(n, 18))
    )
    env._left_gripper_closed = torch.tensor([[True], [False], [True]])
    env._right_gripper_closed = torch.ones(n, 1, dtype=torch.bool)
    env.episode_length_buf = torch.zeros(n, dtype=torch.long)
    env._ctx = {
        "success_hold_count": torch.zeros(n, dtype=torch.long),
        "peg_grip_local_pos": torch.zeros(n, 3),
        "hole_grip_local_pos": torch.zeros(n, 3),
        "left_ee_body_id": 0,
        "right_ee_body_id": 1,
    }
    for _ in range(45):
        terminated, truncated = env._get_dones()
    assert env._ctx["termination_success"].tolist() == [True, False, False]
    # Dropped parts must be reported as physical failures, not just generic
    # invalid insertion geometry in the existing evaluation trace protocol.
    assert env._ctx["termination_drop"].tolist() == [False, False, True]
    assert terminated["left_arm"].tolist() == [True, False, True]
    assert not truncated["left_arm"].any()


def test_deficit_reward_does_not_prefer_waiting_outside_the_aperture(monkeypatch):
    """Improving alignment or seating must improve the opt-in geometric reward."""
    cfg, _ = resolve_task_config("Isaac-OpenArm-PegInHole-Circle-Seated-v0", "skrl_mappo_cfg_entry_point", overrides=[])
    cfg.use_preinsert_deficit_reward = True
    lateral = torch.cat((torch.linspace(0.0, 0.030, 121), torch.zeros(101)))
    depth = torch.cat((torch.full((121,), -0.030), torch.linspace(-0.030, 0.020, 101)))
    n = len(lateral)
    tip = torch.stack((lateral, torch.zeros(n), -depth), dim=-1)
    axis = torch.tensor([[0.0, 0.0, -1.0]]).expand(n, -1)
    identity = torch.tensor([[1.0, 0.0, 0.0, 0.0]]).expand(n, -1)
    frames = {
        "peg_tip_w": tip,
        "peg_grip_w": tip - 0.08 * axis,
        "peg_axis_w": axis,
        "hole_entrance_w": torch.zeros(n, 3),
        "hole_bottom_w": 0.025 * axis,
        "hole_axis_w": axis,
        "peg_tip_q_w": identity,
        "hole_bottom_q_w": identity,
    }
    # Supply only geometry; the production metrics, reward composition and
    # strict success predicate remain in the tested path.
    monkeypatch.setattr(logic, "compute_peg_hole_frames_w", lambda env, ctx: frames)
    env = SimpleNamespace(
        cfg=cfg,
        num_envs=n,
        device="cpu",
        actions={agent: torch.zeros(n, width) for agent, width in cfg.action_spaces.items()},
    )
    ctx = {
        "termination_success": torch.zeros(n, dtype=torch.bool),
        "success_hold_count": torch.zeros(n, dtype=torch.long),
    }
    rewards = logic.compute_openarm_re_rewards(env, ctx)
    assert torch.all(torch.diff(rewards["left_arm"][:121]) < 0), "Off-axis waiting earns more than alignment"
    assert torch.all(torch.diff(rewards["left_arm"][121:]) > 0), "Axial approach loses geometric reward"
    torch.testing.assert_close(rewards["left_arm"], rewards["right_arm"])
    assert not ctx["termination_success"].any()
    assert not ctx["success_hold_count"].any()


@pytest.mark.parametrize(
    "limit,lateral,depth,angle,expected",
    [
        (0.0, 0.0, 0.019, 0.0, False),
        (0.020, 0.0, 0.019, 0.0, True),
        (0.020, 0.003, 0.019, 0.0, False),
        (0.020, 0.0, 0.023, 0.0, False),
        (0.020, 0.0, 0.019, 0.15, False),
        (0.020, 0.0, 0.019, 0.10, True),
    ],
)
def test_reachable_seating_target_and_whole_shaft_clearance(monkeypatch, limit, lateral, depth, angle, expected):
    tip = torch.tensor([[lateral, 0.0, -depth]])
    axis = torch.tensor([[math.sin(angle), 0.0, -math.cos(angle)]])
    frames = {
        "peg_tip_w": tip,
        "peg_grip_w": tip - 0.08 * axis,
        "peg_axis_w": axis,
        "hole_entrance_w": torch.zeros(1, 3),
        "hole_bottom_w": torch.tensor([[0.0, 0.0, -0.025]]),
        "hole_axis_w": torch.tensor([[0.0, 0.0, -1.0]]),
        "peg_tip_q_w": torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        "hole_bottom_q_w": torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
    }
    # Only the physics pose provider is replaced. All goal geometry, reward
    # keypoints, hold accumulation and termination use the production code.
    monkeypatch.setattr(logic, "compute_peg_hole_frames_w", lambda env, ctx: frames)
    state = torch.zeros(1, 13)
    state[:, 6] = 1.0  # native XYZW identity
    cfg = SimpleNamespace(
        rotational_symmetry_order=0,
        success_depth_ratio=0.9,
        insertion_depth_limit=limit,
        success_radial_clearance=0.0025 if limit else 0.0,
        peg_collision_radius=0.028 if limit else 0.0,
        success_depth_overrun_tolerance=0.001 if limit else 0.0,
        success_lateral_threshold=0.0025 if limit else 0.012,
        success_axis_threshold=0.97,
        hold_required_steps=45,
        max_tip_distance=1.0,
    )
    env = SimpleNamespace(
        cfg=cfg,
        device="cpu",
        num_envs=1,
        peg=SimpleNamespace(data=SimpleNamespace(root_state_w=state)),
        hole=SimpleNamespace(data=SimpleNamespace(root_state_w=state)),
        robot=SimpleNamespace(data=SimpleNamespace(joint_pos=torch.zeros(1, 18), joint_vel=torch.zeros(1, 18))),
    )
    ctx = {"success_hold_count": torch.zeros(1, dtype=torch.long)}
    for step in range(45):
        _, success, invalid = logic.compute_openarm_re_terminations(env, ctx)
        assert not invalid[0]
        assert bool(success[0]) == (expected and step == 44)
    metrics = logic.compute_peg_in_hole_metrics(env, ctx)
    assert float(metrics["hole_depth"][0]) == pytest.approx(0.025)
    _, rewards = logic._compute_factory_keypoint_reward(env, metrics)
    expected_target = -0.020 if limit else -0.025
    assert float(rewards["keypoints_fixed"][0, 0, 2]) == pytest.approx(expected_target)
