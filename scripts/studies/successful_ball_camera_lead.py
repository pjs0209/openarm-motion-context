"""Opt-in 74D camera-lead candidate from the audited successful 68D Ball actor.

The extra XY lead/valid channels use rendered camera measurements only. The
original four camera channels and six Motion Trend communication values remain.
This is a developmental transfer candidate, not evidence of Ball success.
"""

import hashlib
import json
from pathlib import Path

import torch

from scripts.studies import ball_policy_transfer
from scripts.studies import camera_lead_observation
from scripts.studies.camera_lead_transfer import expand_source
from scripts.studies.raw_reference_actor import build_raw_reference_mappo_agent


SOURCE = (
    Path(__file__).resolve().parents[2]
    / "logs/motion_context_analysis"
    / "ball_successful_teacher_transfer_v1/common_actor.pt"
)


def verified_expanded_source(path=SOURCE, *, device="cpu"):
    """Verify the old teacher chain before expanding only policy/actor scaler."""
    path = Path(path).resolve()
    sidecar = path.with_suffix(".json")
    source_sha = hashlib.sha256(path.read_bytes()).hexdigest()
    provenance = json.loads(sidecar.read_bytes())
    original = Path(provenance.get("source_checkpoint", "")).resolve()
    if (
        provenance.get("kind") != ball_policy_transfer.SOURCE_KIND
        or provenance.get("derived_sha256") != source_sha
        or provenance.get("copied") != ["policy", "state_preprocessor"]
        or provenance.get("source_sha256") != ball_policy_transfer.SOURCE_SHA256
        or not original.is_file()
        or hashlib.sha256(original.read_bytes()).hexdigest() != ball_policy_transfer.SOURCE_SHA256
    ):
        raise ValueError("Audited successful camera68 actor source required")
    payload = torch.load(path, map_location=device, weights_only=True)
    expanded = expand_source(payload)
    return expanded, {
        "kind": "successful_ball_camera68_to_lead74_v1",
        "source": str(path),
        "source_sha256": source_sha,
        "source_provenance_sha256": hashlib.sha256(sidecar.read_bytes()).hexdigest(),
        "copied": ["policy", "state_preprocessor"],
        "fresh": ["value", "optimizer", "shared_state_preprocessor", "value_preprocessor"],
        "new_feature_initialization": "zero policy weights, neutral scaler with count 1",
    }


def install_actor_transfer(agent, path=SOURCE):
    expanded, provenance = verified_expanded_source(path, device=agent.device)
    if set(expanded) != set(agent.possible_agents) or set(expanded) != {"left_arm", "right_arm"}:
        raise ValueError("Both matching arms required")
    validated = {}
    for uid in agent.possible_agents:
        modules = agent.checkpoint_modules[uid]
        if set(expanded[uid]) != {"policy", "state_preprocessor"}:
            raise ValueError("Only actor and actor normalizer may be transferred")
        validated[uid] = {}
        for name in ("policy", "state_preprocessor"):
            target = modules[name].state_dict()
            source = expanded[uid][name]
            if set(source) != set(target) or any(
                value.shape != target[key].shape or not torch.isfinite(value).all() for key, value in source.items()
            ):
                raise ValueError("Exact finite camera-lead actor/normalizer architecture required")
            validated[uid][name] = {key: value.to(target[key]).clone() for key, value in source.items()}
        scaler = validated[uid]["state_preprocessor"]
        if (
            scaler["running_mean"].shape != (74,)
            or scaler["running_variance"].shape != (74,)
            or scaler["current_count"].shape != (74,)
            or (scaler["running_variance"] < 0).any()
            or (scaler["current_count"] < 1).any()
        ):
            raise ValueError("Valid featurewise camera74 actor scaler required")
    for uid in agent.possible_agents:
        for name in ("policy", "state_preprocessor"):
            agent.checkpoint_modules[uid][name].load_state_dict(validated[uid][name], strict=True)
    return provenance


def build_successful_ball_camera_lead_mappo_agent(env, experiment_cfg):
    from isaaclab_tasks.direct.openarm_motion_context.envs.ball_center_goal import (
        install_centered_hold_goal,
        unwrap_ball_task,
    )

    task = unwrap_ball_task(env)
    cfg = task.cfg
    expected = {
        "intent_task_label": "openarm_lift_with_ball",
        "own_observation_dim": 37,
        "actor_input_dim": 74,
        "camera_extra_observation_dim": 7,
        "state_space": 87,
        "actor_object_pose_source": "apriltag",
        "apriltag_measurement_source": "camera_transform",
        "ball_reward_profile": "center_penalty",
        "ball_center_penalty_full_error": 0.25,
        "hold_required_steps": 60,
        "ball_center_success_threshold": 0.05,
        "communication_mode": "motion_context",
        "motion_context_representation": "trend",
        "motion_context_trend_horizon": 5,
    }
    if any(getattr(cfg, name, None) != value for name, value in expected.items()) or cfg.sharing_mode not in (
        "intent_share",
        "no_share",
    ):
        raise ValueError("Exact rendered camera-lead74 TrendH5 centered-held Ball task required")
    camera_lead_observation.install_environment(task)
    install_centered_hold_goal(task)
    agent = build_raw_reference_mappo_agent(env, experiment_cfg)
    for uid in agent.possible_agents:
        modules = agent.checkpoint_modules[uid]
        camera_lead_observation.install_policy_marker(modules["policy"])
        scaler = modules["state_preprocessor"]
        if scaler.running_mean.shape != (74,) or scaler.current_count.ndim != 0:
            raise ValueError("Standard scalar-count camera74 actor scaler required")
        scaler.current_count = torch.ones_like(scaler.running_mean)
    provenance = install_actor_transfer(agent)
    agent._successful_ball_camera_lead_warmstart = provenance
    print(
        "[BALL_CAMERA_LEAD_WARMSTART] "
        + provenance["source_sha256"]
        + " trained camera68 copied, lead columns zero; critic_optimizer_fresh",
        flush=True,
    )
    return agent
