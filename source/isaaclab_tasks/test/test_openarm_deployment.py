# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Deployment must preserve the observation and normalization consumed by the trained actor."""

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml
from skrl.resources.preprocessors.torch import RunningStandardScaler

from isaaclab_tasks.direct.openarm_motion_context.deploy.contract import apriltag_detector_parameters
from isaaclab_tasks.direct.openarm_motion_context.deploy.deploy_node import OpenArmPaperReLiftDeployNode
from isaaclab_tasks.direct.openarm_motion_context.deploy.observation import ObservationNormalizer
from isaaclab_tasks.direct.openarm_motion_context.deploy.robot_state import ArmState, JointSnapshot
from isaaclab_tasks.direct.openarm_motion_context.deploy.safety import joint_position_limits, message_is_fresh
from isaaclab_tasks.direct.openarm_motion_context.envs.gripper_control import update_gripper_command
from isaaclab_tasks.direct.openarm_motion_context.envs.lift_task_logic import _agent_observation


def test_deploy_normalization_matches_training_including_clipping_and_zero_variance():
    scaler = RunningStandardScaler(size=3, device="cpu")
    scaler.running_mean = torch.tensor([1.0, 0.0, -2.0], dtype=torch.float64)
    scaler.running_variance = torch.tensor([4.0, 0.0, 1.0e-20], dtype=torch.float64)
    scaler.current_count.fill_(100)
    checkpoint = {"left_arm": {"state_preprocessor": scaler.state_dict()}}
    normalizer = ObservationNormalizer(checkpoint, "left_arm", 3, device=torch.device("cpu"))
    states = torch.tensor([[15.0, 1.0e-10, -1.9999999], [-15.0, -1.0, -2.0]])
    assert normalizer.loaded
    torch.testing.assert_close(normalizer(states), scaler(states, train=False))
    checkpoint["left_arm"]["state_preprocessor"]["running_variance"][0] = -1
    assert not ObservationNormalizer(checkpoint, "left_arm", 3, device=torch.device("cpu")).loaded


def test_detector_exports_metric_sizes_and_distinct_tf_children_for_every_camera():
    path = Path(__file__).parents[1] / "isaaclab_tasks/direct/openarm_motion_context/deploy/configs/lift_real.yaml"
    config = yaml.safe_load(path.read_text())
    frames = []
    for camera in config["frames"]["cameras"]:
        tag = apriltag_detector_parameters(config, camera)["/**"]["ros__parameters"]["tag"]
        assert tag["ids"] == [0, 1, 2]
        assert tag["sizes"] == [0.04, 0.016, 0.016]
        frames.extend(tag["frames"])
    assert len(set(frames)) == 9


def test_live_urdf_limits_follow_controller_order_and_reject_invalid_data():
    urdf = (
        '<robot><joint name="b"><limit lower="-2" upper="3"/></joint>'
        '<joint name="a"><limit lower="0" upper="1"/></joint></robot>'
    )
    assert joint_position_limits(urdf, ["a", "b"]) == ([0.0, -2.0], [1.0, 3.0])
    for invalid in (urdf.replace('lower="0"', 'lower="nan"'), "<robot>", "<robot/>"):
        with pytest.raises(ValueError):
            joint_position_limits(invalid, ["a"])
    assert message_is_fresh(10.0, 9.9, 0.25)
    assert not message_is_fresh(10.0, 9.0, 0.25)
    assert not message_is_fresh(10.0, 10.1, 0.25)


def test_ros_joint_state_and_base_frame_observation_match_simulation():
    # A translated, 90-degree-rotated robot root distinguishes world/base
    # conventions. Native simulator buffers are XYZW; policy/ROS bridge is WXYZ.
    half = 2**-0.5
    quat = torch.tensor([[0.0, 0.0, half, half]])
    body_state = torch.zeros(1, 1, 13)
    body_state[0, 0, :3] = torch.tensor([1.1, 2.2, 3.3])
    body_state[0, 0, 3:7] = quat[0]
    body_state[0, 0, 7:10] = torch.tensor([-0.3, 0.2, 0.1])
    positions = torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.02, 0.02]])
    velocities = torch.tensor([[0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0, 0.0]])
    robot = SimpleNamespace(
        data=SimpleNamespace(
            root_pos_w=torch.tensor([[1.0, 2.0, 3.0]]),
            root_quat_w=quat,
            body_state_w=body_state,
            joint_pos=positions,
            joint_vel=velocities,
        )
    )
    cfg = SimpleNamespace(
        gripper_open_target=0.04, ee_vel_scale=2.0, camera_extra_observation_dim=0, own_observation_dim=30
    )
    env = SimpleNamespace(robot=robot, cfg=cfg, common_step_counter=3, num_envs=1, device="cpu")
    target = torch.tensor([[0.9, 2.3, 3.3]])
    target_quat = torch.tensor([[half, 0.0, 0.0, half]])
    ctx = {
        "left_arm_joint_ids": list(range(7)),
        "left_gripper_joint_ids": [7, 8],
        "left_ee_body_id": 0,
        "actor_target_cache_step": 3,
        "actor_target_cache": (target, target, target_quat, target_quat),
    }
    simulated = _agent_observation(env, ctx, "left")[0]
    node = object.__new__(OpenArmPaperReLiftDeployNode)
    node.device = torch.device("cpu")
    node.cfg = {"action": {"gripper_open_target": 0.04}, "policy": {"ee_vel_scale": 2.0}}
    node.left_arm_joints = [f"joint{i}" for i in range(7)]
    node.left_gripper_joints = ["finger"]
    node.last_joint_state = JointSnapshot(
        1.0,
        SimpleNamespace(
            name=node.left_arm_joints + ["finger"],
            position=positions[0, :8].tolist(),
            velocity=velocities[0, :8].tolist(),
        ),
    )
    node.arm_state = {"left_arm": ArmState(q_target=torch.zeros(7), ee_lin_vel=torch.tensor([0.2, 0.3, 0.1]))}
    deployed = node._own_obs(
        "left_arm",
        torch.tensor([0.2, -0.1, 0.3]),
        torch.tensor([1.0, 0.0, 0.0, 0.0]),
        torch.tensor([0.3, 0.1, 0.3]),
        torch.tensor([1.0, 0.0, 0.0, 0.0]),
    )
    assert deployed.shape == (30,)
    torch.testing.assert_close(deployed, simulated, atol=1.0e-6, rtol=1.0e-6)


@pytest.mark.parametrize("speed", [0.0, 0.2])
def test_deployment_gripper_trajectory_matches_control_ticks_and_reset(speed):
    node = object.__new__(OpenArmPaperReLiftDeployNode)
    node.device, node.dt = torch.device("cpu"), 1 / 60
    node.left_default = torch.zeros(7)
    node.left_arm_joints, node.left_gripper_joints = [f"joint{i}" for i in range(7)], ["finger"]
    node.cfg = {"action": {"gripper_target_speed_limit": speed, "reset_grace_steps": 2}}
    node._joint_values = lambda names: (torch.zeros(len(names)), torch.zeros(len(names)))
    for _ in range(2):
        node.arm_state = {"left_arm": ArmState(q_target=torch.zeros(7))}
        expected = torch.full((1, 2), 0.04)
        closed = False
        for tick, command in enumerate([-1.0] * 20 + [0.0] * 3 + [1.0] * 15 + [-1.0] * 20):
            node.control_steps = tick
            if command:
                closed = command < 0
            if tick < 2:
                closed = False
            update_gripper_command(
                expected,
                torch.tensor([[closed]]),
                torch.full_like(expected, 0.04),
                torch.zeros_like(expected),
                speed * node.dt,
            )
            _, actual = node._apply_incremental_action("left_arm", torch.tensor([0.0] * 7 + [command]))
            torch.testing.assert_close(actual, expected[0, :1])
