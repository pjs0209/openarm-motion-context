# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The optional training reset mixture must leave the final interval and evaluation original."""

from types import SimpleNamespace

import pytest
import torch
from isaaclab_newton.physics import NewtonCfg

from isaaclab_tasks.direct.openarm_motion_context.models.grasp_reset_curriculum import (
    _warm_start_probability,
    install_grasp_reset_curriculum,
)
from isaaclab_tasks.direct.openarm_motion_context.train.trainer import MotionContextTrainer


def test_gripper_initialization_preserves_arm_outputs_and_restores_learned_checkpoint(monkeypatch):
    """The opt-in prior initializes the Gaussian, rather than forcing actions during replay."""
    from gymnasium.spaces import Box

    from isaaclab_tasks.direct.openarm_motion_context.models import grasp_reset_exploration as exploration
    from isaaclab_tasks.direct.openarm_motion_context.models.actor import MotionContextPolicyModel

    policy = MotionContextPolicyModel(
        observation_space=Box(-1.0, 1.0, (66,)),
        action_space=Box(-1.0, 1.0, (8,)),
        device="cpu",
        hidden_sizes=[8, 8],
        own_observation_dim=33,
        motion_intent_dim=3,
        motion_context_dim=3,
        communication_feature_dim=33,
        initial_log_std=-1.0,
    )
    states = torch.linspace(-1.0, 1.0, 66 * 4).reshape(4, 66)
    before, std, _ = policy.compute({"states": states})
    before, std = before.detach().clone(), std.detach().clone()
    agent = SimpleNamespace(models={"left_arm": {"policy": policy}})
    # The real policy forward and checkpoint restore own the asserted behavior;
    # replace only simulator-dependent construction of the unchanged parent.
    monkeypatch.setattr(exploration, "build_grasp_reset_curriculum_mappo_agent", lambda env, cfg: agent)
    exploration.build_grasp_exploration_mappo_agent(None, {"runtime": {"gripper_initial_mean": -0.2}})
    actual, log_std, _ = policy.compute({"states": states})
    torch.testing.assert_close(actual[:, :7], before[:, :7], rtol=0, atol=0)
    torch.testing.assert_close(log_std, std, rtol=0, atol=0)
    torch.testing.assert_close(actual[:, 7], torch.full((4,), -0.2), rtol=0, atol=1e-7)
    actual[:, 7].sum().backward()
    assert policy.gripper_action_head[-1].weight.grad.abs().max() > 0
    with torch.no_grad():
        policy.gripper_action_head[-1].bias.fill_(0.37)
    learned = {key: value.detach().clone() for key, value in policy.state_dict().items()}
    with torch.no_grad():
        policy.gripper_action_head[-1].bias.zero_()
    policy.load_state_dict(learned)
    restored, _, _ = policy.compute({"states": states})
    torch.testing.assert_close(restored[:, 7], torch.full((4,), 0.37), rtol=0, atol=1e-7)
    with pytest.raises(ValueError, match="finite"):
        exploration.build_grasp_exploration_mappo_agent(None, {"runtime": {"gripper_initial_mean": float("nan")}})


def test_warm_reset_schedule_accounts_for_ongoing_episodes_and_disables_evaluation():
    # Six-step episodes must finish by40, so new warm starts stop at34.
    # The middle of the fade from10 to34 is22, independently of episode count.
    schedule = {0: 1.0, 10: 1.0, 22: 0.5, 34: 0.0, 40: 0.0}
    for step, expected in schedule.items():
        assert _warm_start_probability(step, True, 10, 40, 6) == pytest.approx(expected)
        assert _warm_start_probability(step, False, 10, 40, 6) == 0.0
    with pytest.raises(ValueError, match="normal_from - episode_steps"):
        _warm_start_probability(0, True, 10, 16, 6)


def test_curriculum_activation_reaches_the_task_through_the_trainer_wrapper():
    """A newly installed task flag must receive real trainer train/eval updates."""
    limits = torch.tensor([[[-4.0, 4.0]] * 18])
    task = SimpleNamespace(
        cfg=SimpleNamespace(
            intent_task_label="openarm_lift", action_scale=1.8, sim=SimpleNamespace(physics=NewtonCfg())
        ),
        own_observation_dim=33,
        num_envs=1,
        device="cpu",
        max_episode_length=600,
        robot=SimpleNamespace(data=SimpleNamespace(joint_pos_limits=limits)),
        left_arm_joint_ids=list(range(7)),
        right_arm_joint_ids=list(range(9, 16)),
        left_arm_default_joint_pos=torch.zeros(1, 7),
        right_arm_default_joint_pos=torch.zeros(1, 7),
        _reset_idx=lambda ids: None,
        _get_rewards=lambda: {},
        get_paper_motion_context_trace_signals=lambda index=0: {},
    )
    install_grasp_reset_curriculum(task)
    assert hasattr(task, "training") and task.training is False
    trainer = SimpleNamespace(env=SimpleNamespace(unwrapped=task))
    MotionContextTrainer._set_env_attr(trainer, "training", True)
    assert task.training is True
    MotionContextTrainer._set_env_attr(trainer, "training", False)
    assert task.training is False
