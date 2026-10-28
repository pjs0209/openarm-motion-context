"""Camera-only fresh normalization counts; opt-in, unchanged 34D actor input.

skrl's variance update broadcasts counts per feature. Keep old Lift history
for the existing 30D blocks, but assign new camera channels one neutral prior
sample instead of millions of nonexistent camera samples.
"""

import torch


def expand_warmstart(payload):
    result = {}
    for uid, checkpoint in payload.items():
        policy = dict(checkpoint["policy"])
        for key in ("own_backbone.0.weight", "partner_intent_encoder.0.weight"):
            value = policy[key]
            if value.ndim != 2 or value.shape[1] != 30 or not torch.isfinite(value).all():
                raise ValueError("Requires original 30D common Lift actor")
            policy[key] = torch.cat((value, value.new_zeros(value.shape[0], 4)), dim=1)
        old = checkpoint["state_preprocessor"]
        if set(old) != {"running_mean", "running_variance", "current_count"}:
            raise ValueError("Unexpected scaler state keys")
        count = old["current_count"]
        if count.numel() != 1 or not torch.isfinite(count).all() or float(count) < 1:
            raise ValueError("Requires valid scalar Lift historical count")
        scaler = {}
        for key in ("running_mean", "running_variance"):
            value = old[key]
            if (
                value.shape != (60,)
                or not torch.isfinite(value).all()
                or (key == "running_variance" and (value < 0).any())
            ):
                raise ValueError("Invalid original Lift scaler")
            expanded = value.new_full((68,), 1.0 if key == "running_variance" else 0.0)
            expanded[:30], expanded[34:64] = value[:30], value[30:]
            scaler[key] = expanded
        expanded_count = count.new_ones(68)
        expanded_count[:30] = count
        expanded_count[34:64] = count
        scaler["current_count"] = expanded_count
        result[uid] = {**checkpoint, "policy": policy, "state_preprocessor": scaler}
    return result


def install_vector_counts(agent):
    for uid in agent.possible_agents:
        scaler = agent.checkpoint_modules[uid]["state_preprocessor"]
        if scaler.running_mean.shape != (68,) or scaler.current_count.ndim != 0:
            raise ValueError("Fresh-camera scaler requires the standard 68D Ball actor scaler")
        scaler.current_count = torch.ones_like(scaler.running_mean)


def build_fresh_camera_center_goal_mappo_agent(env, experiment_cfg):
    from isaaclab_tasks.direct.openarm_motion_context.models.ball_center_goal_matched import (
        build_center_goal_matched_mappo_agent,
    )

    agent = build_center_goal_matched_mappo_agent(env, experiment_cfg)
    install_vector_counts(agent)
    print("[INFO] fresh camera scaler v1: per-feature counts; inherited proprioception history, fresh camera history")
    return agent
