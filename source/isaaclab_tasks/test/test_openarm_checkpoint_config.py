# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Checkpoint playback must preserve the saved policy/environment contract."""

from pathlib import Path

import pytest
import yaml

from isaaclab.utils.io import dump_yaml

from isaaclab_tasks.direct.openarm_motion_context.deploy.contract import restore_training_contract
from isaaclab_tasks.direct.openarm_motion_context.train.checkpoint_config import restore_checkpoint_configs
from isaaclab_tasks.utils.hydra import resolve_task_config


@pytest.fixture
def recorded_run(tmp_path):
    env, agent = resolve_task_config("Isaac-OpenArm-Lift-v0", "skrl_mappo_cfg_entry_point", overrides=[])
    env.seed = agent["seed"] = 123
    env.communication_mode = "none"
    env.motion_context_trend_horizon = 13
    env.action_step_scale = 0.08
    env.reset_grace_steps = 10
    env.premature_closure_penalty_scale = 0.25
    env.gripper_target_speed_limit = 0.2
    env.sim.physics.solver_cfg.njmax = 9000
    agent["models"]["policy"]["partner_message_embed_dim"] = 24
    agent["models"]["policy"]["clip_actions"] = True
    dump_yaml(str(tmp_path / "params/env.yaml"), env)
    dump_yaml(str(tmp_path / "params/agent.yaml"), agent)
    return str(tmp_path / "checkpoints/agent_60000.pt")


@pytest.mark.parametrize("records_gravity", [True, False])
def test_playback_restores_saved_contract_and_explicit_overrides_even_when_equal_to_defaults(
    recorded_run, records_gravity
):
    if not records_gravity:
        config_path = Path(recorded_run).parents[1] / "params/env.yaml"
        config = yaml.load(config_path.read_text(), Loader=yaml.FullLoader)
        del config["scene"]["robot"]["spawn"]["newton_gravity_compensation"]
        config_path.write_text(yaml.dump(config))
    env, agent = resolve_task_config(
        "Isaac-OpenArm-Lift-v0",
        "skrl_mappo_cfg_entry_point",
        overrides=["env.communication_mode=motion_context", "agent.seed=42"],
    )
    restored, restored_agent, training_seed = restore_checkpoint_configs(
        recorded_run, env, agent, ["env.communication_mode=motion_context", "agent.seed=42"]
    )
    assert restored.motion_context_trend_horizon == 13
    assert restored.action_step_scale == 0.08
    assert restored.reset_grace_steps == 10
    assert restored.premature_closure_penalty_scale == 0.25
    assert restored.scene.robot.spawn.newton_gravity_compensation == (1.0 if records_gravity else 0.0)
    assert restored.sim.physics.solver_cfg.njmax == 9000
    assert restored.communication_mode == "motion_context"
    assert restored_agent["models"]["policy"]["partner_message_embed_dim"] == 24
    assert restored_agent["models"]["policy"]["clip_actions"] is True
    assert restored_agent["seed"] == 42
    assert training_seed == 123
    # Restoring a run must not corrupt the caller's independently resolved config.
    assert env.seed is None
    assert agent["models"]["policy"]["partner_message_embed_dim"] == 32
    assert agent["models"]["policy"]["clip_actions"] is False


def test_playback_rejects_implicit_backend_change_but_allows_explicit_migration(recorded_run):
    env, agent = resolve_task_config(
        "Isaac-OpenArm-Lift-v0", "skrl_mappo_cfg_entry_point", overrides=["physics=isaacsim_physx"]
    )
    with pytest.raises(ValueError, match="Checkpoint physics.*differs from selected physics"):
        restore_checkpoint_configs(recorded_run, env, agent, [])
    restored, _, _ = restore_checkpoint_configs(recorded_run, env, agent, ["physics=isaacsim_physx"])
    assert restored.sim.physics.friction_correlation_distance == 0.00625
    assert restored.motion_context_trend_horizon == 13


def test_playback_requires_recorded_configs_instead_of_silent_defaults(tmp_path):
    with pytest.raises(FileNotFoundError, match="Checkpoint configuration is missing"):
        restore_checkpoint_configs(str(tmp_path / "checkpoints/agent.pt"), None, {}, [])


@pytest.mark.parametrize("recorded_deficit", [None, True])
def test_peg_playback_restores_legacy_geometry_and_recorded_reward_variant(tmp_path, recorded_deficit):
    old_env, old_agent = resolve_task_config(
        "Isaac-OpenArm-PegInHole-Circle-V13-v0", "skrl_mappo_cfg_entry_point", overrides=[]
    )
    old_env.seed = 42
    old_env.use_preinsert_deficit_reward = bool(recorded_deficit)
    dump_yaml(str(tmp_path / "params/env.yaml"), old_env)
    dump_yaml(str(tmp_path / "params/agent.yaml"), old_agent)
    path = tmp_path / "params/env.yaml"
    config = yaml.load(path.read_text(), Loader=yaml.FullLoader)
    for name in (
        "insertion_depth_limit",
        "success_radial_clearance",
        "peg_collision_radius",
        "success_depth_overrun_tolerance",
        "success_protocol",
    ):
        del config[name]
    if recorded_deficit is None:
        del config["use_preinsert_deficit_reward"]
    path.write_text(yaml.dump(config))
    current, agent = resolve_task_config(
        "Isaac-OpenArm-PegInHole-Circle-Seated-v0", "skrl_mappo_cfg_entry_point", overrides=[]
    )
    current.use_preinsert_deficit_reward = True
    restored, _, _ = restore_checkpoint_configs(str(tmp_path / "checkpoints/agent.pt"), current, agent, [])
    assert restored.insertion_depth_limit == 0.0
    assert restored.success_radial_clearance == 0.0
    assert restored.success_protocol == "legacy_marker_depth_v1"
    assert restored.success_depth_threshold == pytest.approx(0.0225)
    assert restored.use_preinsert_deficit_reward is bool(recorded_deficit)
    assert current.insertion_depth_limit == 0.020


def test_deployment_restores_recorded_policy_contract_without_importing_simulator(recorded_run):
    path = Path(recorded_run)
    path.parent.mkdir()
    path.touch()
    template = Path(__file__).parents[1] / "isaaclab_tasks/direct/openarm_motion_context/deploy/configs/lift_real.yaml"
    cfg = yaml.safe_load(template.read_text())
    cfg["checkpoint"] = str(path)
    cfg["control_rate_hz"] = 30.0
    restored = restore_training_contract(cfg)
    assert restored["control_rate_hz"] == 60.0
    assert restored["communication"]["context_trend_horizon"] == 13
    assert restored["communication"]["mode"] == "none"
    assert restored["policy"]["partner_intent_embed_dim"] == 24
    assert restored["action"]["gripper_target_speed_limit"] == 0.2
    assert restored["topics"] == cfg["topics"]
    assert cfg["control_rate_hz"] == 30.0
