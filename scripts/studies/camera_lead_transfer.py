"""Unlaunched common Lift30/camera34 -> camera-lead37 source transformation.

Preserve all learned tensors, including sigma and control guard buffers. Append
zero camera columns to both encoders; retain the original scaler blocks.
Existing34D camera/vector history is preserved; only three new features get
fresh counts. Legacy30D sources get seven new features and fresh peer counts.
This function does not create/overwrite an artifact or initialize a live agent.
"""

import torch

from scripts.studies.camera_lead_observation import MARKER


def expand_source(payload):
    if set(payload) != {"left_arm", "right_arm"}:
        raise ValueError("Both common Lift actors required")
    result = {}
    source_width = None
    for uid, checkpoint in payload.items():
        if set(checkpoint) != {"policy", "state_preprocessor"}:
            raise ValueError("Actor-only source; critics and optimizer must remain fresh")
        old_policy = checkpoint["policy"]
        if MARKER in old_policy or any(
            not isinstance(v, torch.Tensor) or not torch.isfinite(v).all() for v in old_policy.values()
        ):
            raise ValueError("Finite original Lift policy tensors required")
        policy = {key: value.clone() for key, value in old_policy.items()}
        width = policy["own_backbone.0.weight"].shape[-1]
        if width not in (30, 34) or source_width is not None and width != source_width:
            raise ValueError("Both actors must share the original30D or camera34D common layout")
        source_width = width
        for key in ("own_backbone.0.weight", "partner_intent_encoder.0.weight"):
            value = policy[key]
            if value.ndim != 2 or value.shape[1] != width or not value.is_floating_point():
                raise ValueError("Matching original common own/partner encoders required")
            policy[key] = torch.cat((value, value.new_zeros((value.shape[0], 37 - width))), dim=1)
        policy[MARKER] = policy["own_backbone.0.weight"].new_tensor(1.0)
        old = checkpoint["state_preprocessor"]
        if (
            set(old) != {"running_mean", "running_variance", "current_count"}
            or any(not isinstance(v, torch.Tensor) or not torch.isfinite(v).all() for v in old.values())
            or old["current_count"].shape != (torch.Size([]) if width == 30 else torch.Size([68]))
            or (old["current_count"] < 1).any()
        ):
            raise ValueError("Finite original scalar30D or vector34D historical counts required")
        scaler = {}
        for key in ("running_mean", "running_variance"):
            value = old[key]
            if (
                value.shape != (2 * width,)
                or not value.is_floating_point()
                or (key == "running_variance" and (value < 0).any())
            ):
                raise ValueError("Original common featurewise scaler required")
            expanded = value.new_full((74,), 1.0 if key == "running_variance" else 0.0)
            expanded[:width], expanded[37 : 37 + width] = value[:width], value[width:]
            scaler[key] = expanded
        count = old["current_count"]
        expanded_count = count.new_ones(74)
        if width == 30:
            expanded_count[:30], expanded_count[37:67] = count, count
            expanded_count[37:43] = 1.0  # Newly activated peer features in legacy30D sources.
        else:
            expanded_count[:34], expanded_count[37:71] = count[:34], count[34:]
        scaler["current_count"] = expanded_count
        result[uid] = {"policy": policy, "state_preprocessor": scaler}
    return result
