# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Plain common builders and actual model evidence for the four-message study."""

import hashlib
import json
from pathlib import Path

from .actor import MotionContextMAPPO, MotionContextPolicyModel
from .control_state import build_control_state_grasp_exploration_agent, build_control_state_mappo_agent


def build_main_comparison_agent(env, experiment_cfg):
    """Keep task-specific common controls; never select a message-specific model."""
    task = getattr(env, "unwrapped", env)
    runtime = experiment_cfg["runtime"]
    profile = runtime["main_comparison_task"]
    allowed = {"agent_builder", "trainer", "main_comparison_task"}
    if profile == "lift":
        allowed.update(
            {
                "arm_target_error_rad",
                "grasp_curriculum_warm_until",
                "grasp_curriculum_normal_from",
                "gripper_initial_mean",
            }
        )
        builder = build_control_state_grasp_exploration_agent
    elif profile == "peg":
        builder = build_control_state_mappo_agent
    else:
        raise ValueError("main_comparison_task must be lift or peg")
    if set(runtime) - allowed:
        raise ValueError(f"Undeclared main-comparison runtime options: {set(runtime) - allowed}")
    if (task.cfg.communication_mode, task.cfg.sharing_mode, task.cfg.motion_context_representation) not in {
        ("motion_context", "no_share", "trend"),
        ("motion_only", "intent_share", "trend"),
        ("motion_context", "intent_share", "signed_twist"),
        ("motion_context", "intent_share", "trend"),
    }:
        raise ValueError("Main comparison requires one of the four declared message conditions")
    agent = builder(env, experiment_cfg)
    if type(agent) is not MotionContextMAPPO:
        raise ValueError("Main comparison requires the common MotionContextMAPPO update")
    models = {}
    for uid, modules in agent.models.items():
        policy = modules["policy"]
        if type(policy) is not MotionContextPolicyModel or not policy.uses_context or policy.gripper_partner_local_only:
            raise ValueError("Main comparison requires the same unmodified partner encoder and action heads")
        models[uid] = {}
        for role, model in modules.items():
            digest = hashlib.sha256()
            for name, value in model.state_dict().items():
                digest.update(name.encode())
                digest.update(value.detach().cpu().contiguous().numpy().tobytes())
            models[uid][role] = {
                "class": f"{type(model).__module__}.{type(model).__name__}",
                "state_shapes": {name: list(value.shape) for name, value in model.state_dict().items()},
                "parameter_count": sum(value.numel() for value in model.parameters()),
                "initial_state_sha256": digest.hexdigest(),
            }
    evidence = {
        "profile": "common_actor_critic_own_control_state/v1",
        "task": profile,
        "message": {
            key: getattr(task.cfg, key)
            for key in ("communication_mode", "sharing_mode", "motion_context_representation")
        },
        "own_observation_dim": task.cfg.own_observation_dim,
        "partner_slot_dim": task.cfg.communication_feature_dim,
        "control_dt_s": task.cfg.sim.dt * task.cfg.decimation,
        "trend_horizon_steps": task.cfg.motion_context_trend_horizon,
        "trend_horizon_s": task.cfg.motion_context_trend_horizon * task.cfg.sim.dt * task.cfg.decimation,
        "models": models,
    }
    # Playback uses an existing log directory. Preserve the training evidence.
    log_dir = getattr(task.cfg, "log_dir", None)
    path = Path(log_dir) / "main_comparison_model.json" if log_dir else None
    if path is not None and not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x") as stream:
            json.dump(evidence, stream, indent=2)
            stream.write("\n")
    return agent
