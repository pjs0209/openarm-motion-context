# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""A policy action must have the same effect regardless of physics decimation ownership."""

import runpy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from isaaclab_tasks.direct.openarm_motion_context.envs.arm_reference_rate import install_arm_reference_rate_limit
from isaaclab_tasks.direct.openarm_motion_context.envs.arm_target_bounds import install_arm_target_bounds
from isaaclab_tasks.direct.openarm_motion_context.envs.lift_env import OpenArmReBimanualLiftEnv
from isaaclab_tasks.direct.openarm_motion_context.envs.peg_contact_grasp import OpenArmContactCirclePegAndHoleEnv
from isaaclab_tasks.direct.openarm_motion_context.envs.peg_in_hole_env import (
    OpenArmFixedPegAndHoleEnv,
    OpenArmPegInHoleEnv,
)

_ROOT = Path(__file__).resolve().parents[3]
_SkrlWrapper = runpy.run_path(str(_ROOT / "compat/skrl_1_4_3/skrl/envs/wrappers/torch/base.py"))["MultiAgentEnvWrapper"]


class PositionCommands:
    """Capture actuator writes without supplying any task-side control behavior."""

    def __init__(self):
        self.position = torch.zeros(1, 18)

    def set_position_index(self, *, value, joint_ids, env_ids=None):
        self.position[:, joint_ids] = value


@pytest.mark.parametrize(
    "env_type,gripper_speed,target_error_bound,arm_rate_scale,physical_target_bound",
    [
        (OpenArmReBimanualLiftEnv, 0.0, 0.0, 0.0, 0.0),
        (OpenArmPegInHoleEnv, 0.0, 0.0, 0.0, 0.0),
        (OpenArmFixedPegAndHoleEnv, 0.0, 0.0, 0.0, 0.0),
        (OpenArmReBimanualLiftEnv, 0.2, 0.0, 0.0, 0.0),
        (OpenArmPegInHoleEnv, 0.2, 0.0, 0.0, 0.0),
        (OpenArmContactCirclePegAndHoleEnv, 0.2, 0.02, 0.0, 0.0),
        (OpenArmReBimanualLiftEnv, 0.2, 0.0, 1.0, 0.0),
        (OpenArmReBimanualLiftEnv, 0.2, 0.0, 0.0, 0.02),
    ],
)
def test_policy_targets_are_held_during_physics_substeps(
    env_type, gripper_speed, target_error_bound, arm_rate_scale, physical_target_bound
):
    env = object.__new__(env_type)
    fixed = env_type is OpenArmFixedPegAndHoleEnv
    width = 7 if fixed else 8
    env.sim, env.scene = SimpleNamespace(device="cpu"), SimpleNamespace(num_envs=1)
    env.cfg = SimpleNamespace(
        action_scale=0.7,
        intent_task_label="openarm_lift",
        action_step_scale=0.04,
        gripper_switch_epsilon=0.02,
        gripper_target_speed_limit=gripper_speed,
        sim=SimpleNamespace(dt=1 / 120),
        decimation=2,
        fixed_peg_and_hole=False,
        fixed_hole=False,
        use_runtime_fixed_joints=True,
        max_joint_target_error_rad=target_error_bound,
        fixed_peg_gripper_open_target=0.01,
        fixed_hole_gripper_open_target=0.02,
    )
    commands = PositionCommands()
    env.robot = SimpleNamespace(
        data=SimpleNamespace(
            joint_pos=torch.zeros(1, 18),
            joint_vel=torch.full((1, 18), 3.0),
            joint_vel_limits=torch.tensor([[1.2] * 7 + [2.4] * 7 + [0.2] * 4]),
            joint_pos_limits=torch.tensor([[[-0.1, 0.015]] * 7 + [[-0.1, 0.1]] * 11]),
        ),
        actuators=SimpleNamespace(target_command=commands),
    )
    env.left_arm_joint_ids, env.right_arm_joint_ids = list(range(7)), list(range(7, 14))
    env.left_gripper_joint_ids, env.right_gripper_joint_ids = [14, 15], [16, 17]
    for side in ("left", "right"):
        setattr(env, f"{side}_arm_default_joint_pos", torch.zeros(1, 7))
        setattr(env, f"_{side}_arm_incremental_target", torch.zeros(1, 7))
        setattr(env, f"_{side}_gripper_closed", torch.zeros(1, 1, dtype=torch.bool))
        setattr(env, f"{side}_gripper_open_command", torch.full((1, 2), 0.04))
        setattr(env, f"{side}_gripper_close_command", torch.zeros(1, 2))
        setattr(env, f"_{side}_gripper_position_command", torch.full((1, 2), 0.04))
    env._reset_grace_left = torch.tensor([0 if gripper_speed else 20])
    env._ctx = {}
    env.action_manager = SimpleNamespace(action=torch.zeros(1, 2 * width))
    env._prev_action, env._smoothed_action = torch.zeros(1, 2 * width), torch.zeros(1, 2 * width)
    # Preserve latent Gaussian commands for PPO; only the task's execution
    # copy may be clipped to the physical action interval.
    actions = {agent: torch.full((1, width), 3.0) for agent in ("left_arm", "right_arm")}
    if gripper_speed:
        for action in actions.values():
            action[:, 7] = -3
    if arm_rate_scale:
        # The training runner passes a delegating skrl wrapper. Install on
        # that real boundary, then step the underlying task as env.step does.
        install_arm_reference_rate_limit(_SkrlWrapper(env), arm_rate_scale)
    if physical_target_bound:
        install_arm_target_bounds(_SkrlWrapper(env), physical_target_bound)
    for _ in range(2):
        env._pre_physics_step(actions)
        env._apply_action()
        first_write = commands.position.clone()
        env._apply_action()
        torch.testing.assert_close(commands.position, first_write)
    for action in actions.values():
        torch.testing.assert_close(action[:, :7], torch.full((1, 7), 3.0))
        if not fixed:
            assert float(action[0, 7]) == (-3.0 if gripper_speed else 3.0)
    expected = min(0.08, target_error_bound) if target_error_bound else 0.08
    if arm_rate_scale:
        # Two ticks: slow joints advance 2 * 1.2/60, fast joints retain the
        # original two 0.04 increments. Physics substeps add no extra motion.
        torch.testing.assert_close(commands.position[0, :7], torch.full((7,), 0.04))
        torch.testing.assert_close(commands.position[0, 7:14], torch.full((7,), 0.08))
        # Rate limiting commands must not overwrite actual simulator state,
        # even if measured velocity already exceeds the nominal limit.
        torch.testing.assert_close(env.robot.data.joint_pos, torch.zeros(1, 18))
        torch.testing.assert_close(env.robot.data.joint_vel, torch.full((1, 18), 3.0))
    elif physical_target_bound:
        # Left joints meet their authored stop before the tracking-error cap;
        # right joints reach the error cap inside their wider physical range.
        torch.testing.assert_close(commands.position[0, :7], torch.full((7,), 0.015))
        torch.testing.assert_close(commands.position[0, 7:14], torch.full((7,), 0.02))
        torch.testing.assert_close(env.robot.data.joint_pos, torch.zeros(1, 18))
        torch.testing.assert_close(env.robot.data.joint_vel, torch.full((1, 18), 3.0))
    else:
        torch.testing.assert_close(commands.position[0, :14], torch.full((14,), expected))
    if not fixed:
        assert env._reset_grace_left.item() == (0 if gripper_speed else 18)
    if gripper_speed:
        # Two policy ticks at 60 Hz, irrespective of the four actuator writes.
        torch.testing.assert_close(commands.position[0, 14:], torch.full((4,), 0.04 - 0.2 / 30))
    if physical_target_bound:
        # A solver excursion can make the error ball and legal reference
        # interval disjoint. Recover toward a legal target without teleporting
        # the actual joints or silently commanding beyond their hard stops.
        env.robot.data.joint_pos[0, :7] = -0.2
        env.robot.data.joint_pos[0, 7:14] = 0.2
        measured = env.robot.data.joint_pos.clone()
        env._pre_physics_step(actions)
        env._apply_action()
        torch.testing.assert_close(commands.position[0, :7], torch.full((7,), -0.1))
        torch.testing.assert_close(commands.position[0, 7:14], torch.full((7,), 0.1))
        torch.testing.assert_close(env.robot.data.joint_pos, measured)
