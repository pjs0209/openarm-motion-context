"""Actor-only transfer of a trained 68D rendered-camera Ball policy.

This is distinct from the existing 30D Lift-to-Ball expansion: camera
weights are already trained and are copied exactly. The critic, optimizer,
and their preprocessors remain fresh. A later, separately registered study
must select this initializer explicitly; current studies are unaffected.
"""

import hashlib
import json
from pathlib import Path

import torch


SOURCE_KIND = "successful_ball_full6d_actor_with_vector_counts_v1"
SOURCE_SHA256 = "648e528b8762062a731b7668ffa8da86c8e9e6d9f7321898cc7b8c947f1d2b0e"


def initialize_from_ball(agent, checkpoint_path):
    path = Path(checkpoint_path).resolve()
    sidecar = path.with_suffix(".json")
    source_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    provenance = json.loads(sidecar.read_bytes())
    original = Path(provenance.get("source_checkpoint", "")).resolve()
    if (
        provenance.get("kind") != SOURCE_KIND
        or provenance.get("derived_sha256") != source_hash
        or provenance.get("copied") != ["policy", "state_preprocessor"]
        or provenance.get("source_sha256") != SOURCE_SHA256
        or not original.is_file()
        or hashlib.sha256(original.read_bytes()).hexdigest() != SOURCE_SHA256
    ):
        raise ValueError("Audited 68D camera-Ball actor artifact required")
    source = torch.load(path, map_location=agent.device, weights_only=True)
    if set(source) != set(agent.possible_agents) or set(source) != {"left_arm", "right_arm"}:
        raise ValueError("Both matching arm actors required")

    validated = {}
    for uid in agent.possible_agents:
        modules = agent.checkpoint_modules[uid]
        if set(source[uid]) != {"policy", "state_preprocessor"}:
            raise ValueError("Only actor and actor normalizer may be transferred")
        policy = source[uid]["policy"]
        scaler = source[uid]["state_preprocessor"]
        target_policy = modules["policy"].state_dict()
        target_scaler = modules["state_preprocessor"].state_dict()
        if (
            set(policy) != set(target_policy)
            or any(
                value.shape != target_policy[name].shape or not torch.isfinite(value).all()
                for name, value in policy.items()
            )
            or set(scaler) != set(target_scaler)
            or any(
                value.shape != target_scaler[name].shape or not torch.isfinite(value).all()
                for name, value in scaler.items()
            )
            or scaler["running_mean"].shape != (68,)
            or scaler["running_variance"].shape != (68,)
            or scaler["current_count"].shape != (68,)
            or (scaler["running_variance"] < 0).any()
            or (scaler["current_count"] < 1).any()
        ):
            raise ValueError("Exact finite 68D actor/normalizer architecture required")
        validated[uid] = {
            "policy": {name: value.to(target_policy[name]).clone() for name, value in policy.items()},
            "state_preprocessor": {name: value.to(target_scaler[name]).clone() for name, value in scaler.items()},
        }

    # Both arms are validated before either live module changes.
    for uid in agent.possible_agents:
        for name, payload in validated[uid].items():
            agent.checkpoint_modules[uid][name].load_state_dict(payload, strict=True)
    return {
        "kind": "camera_ball_actor_warmstart_v1",
        "source_checkpoint": str(path),
        "source_sha256": source_hash,
        "source_provenance_sha256": hashlib.sha256(sidecar.read_bytes()).hexdigest(),
        "copied": ["policy", "state_preprocessor"],
        "fresh": ["value", "optimizer", "shared_state_preprocessor", "value_preprocessor"],
        "camera_weights": "trained 68D camera columns copied exactly; no zero expansion",
        "initial_timestep": 0,
        "limits": "Source is Full6D-selected and achieved original Lift, not strict centered-ball, success.",
    }
