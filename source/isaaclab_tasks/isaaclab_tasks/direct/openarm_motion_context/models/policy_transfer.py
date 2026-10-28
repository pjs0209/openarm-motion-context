"""Actor-only Lift initialization for the rendered-camera WithBall task."""

from __future__ import annotations

import hashlib
from pathlib import Path

import torch


def expand_policy_state(source: dict, target: dict) -> dict:
    """Preserve the 30D actor and append four zero-weight camera columns."""
    if source.keys() != target.keys():
        raise ValueError("Transfer requires identical actor architecture")
    result = {}
    for key, value in source.items():
        expected = target[key]
        if value.shape == expected.shape:
            result[key] = value.to(expected).clone()
        elif key in ("own_backbone.0.weight", "partner_intent_encoder.0.weight"):
            if value.ndim != 2 or expected.shape != (value.shape[0], 34) or value.shape[1] != 30:
                raise ValueError(f"Unsupported actor input expansion: {key}")
            padded = torch.zeros_like(expected)
            padded[:, :30] = value.to(expected)
            result[key] = padded
        else:
            raise ValueError(f"Unsupported actor shape change: {key}")
    return result


def expand_actor_scaler(source: dict, target: dict) -> dict:
    """Keep both 30D observation blocks and insert neutral camera statistics."""
    if source.keys() != target.keys():
        raise ValueError("Transfer scaler layout mismatch")
    result = {}
    for key, value in source.items():
        expected = target[key]
        if value.shape == expected.shape:
            result[key] = value.to(expected).clone()
        elif key in ("running_mean", "running_variance") and value.shape == (60,) and expected.shape == (68,):
            padded = torch.full_like(expected, 1.0 if key == "running_variance" else 0.0)
            padded[:30] = value[:30].to(expected)
            padded[34:64] = value[30:].to(expected)
            result[key] = padded
        else:
            raise ValueError(f"Unsupported actor scaler expansion: {key}")
    return result


def initialize_from_lift(agent, checkpoint_path: str) -> dict:
    """Initialize actor/scaler only, keeping the new critic and optimizer fresh."""
    path = Path(checkpoint_path).resolve()
    source = torch.load(path, map_location=agent.device, weights_only=True)
    if set(source) != set(agent.possible_agents):
        raise ValueError("Transfer must include both matching agents")
    states = {}
    # Validate both agents before mutating either one.
    for uid in agent.possible_agents:
        modules = agent.checkpoint_modules[uid]
        states[uid] = {
            "policy": expand_policy_state(source[uid]["policy"], modules["policy"].state_dict()),
            "state_preprocessor": expand_actor_scaler(
                source[uid]["state_preprocessor"], modules["state_preprocessor"].state_dict()
            ),
        }
    for uid, payload in states.items():
        for name, state in payload.items():
            agent.checkpoint_modules[uid][name].load_state_dict(state)
    return {
        "kind": "lift_actor_warmstart_with_ball",
        "source_checkpoint": str(path),
        "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "copied": ["policy", "state_preprocessor"],
        "fresh": ["value", "optimizer", "shared_state_preprocessor", "value_preprocessor"],
        "camera_weights": "zero appended columns; trainable during WithBall learning",
        "initial_timestep": 0,
    }
