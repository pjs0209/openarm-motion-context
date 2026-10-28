# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Restore the recorded OpenArm policy/environment contract before evaluation."""

from copy import deepcopy
from pathlib import Path

from isaaclab.utils.io import load_yaml


def restore_checkpoint_configs(checkpoint: str, env_cfg, agent_cfg: dict, overrides: list[str]):
    """Restore saved configs, retaining explicit scalar overrides and physics selection.

    The caller first resolves the task and CLI overrides through Isaac Lab.
    Checkpoint YAML must come from a trusted training run. Incompatible schemas
    and implicit backend changes fail before simulator startup, not silently
    fall back to today's defaults. World count, device and evaluation seed are
    applied by the playback entry point afterwards.

    Returns:
        Restored environment, agent config and original training seed.
    """
    params = Path(checkpoint).resolve().parent.parent / "params"
    missing = [str(params / name) for name in ("env.yaml", "agent.yaml") if not (params / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"Checkpoint configuration is missing: {missing}. "
            "Use --use_current_config only for an explicitly configured migration/diagnostic replay."
        )
    saved_env = load_yaml(str(params / "env.yaml"))
    saved_agent = load_yaml(str(params / "agent.yaml"))
    # Before the explicit MJWarp field, these Newton runs received no body
    # compensation despite the legacy PhysX disableGravity flag. Reproduce that
    # mechanical model unless the caller explicitly requests the repaired one.
    saved_env["scene"]["robot"]["spawn"].setdefault("newton_gravity_compensation", 0.0)
    legacy_options = {
        "gripper_target_speed_limit": 0.0,
        "premature_closure_penalty_scale": 1.0,
        "use_preinsert_deficit_reward": False,
        "require_contact_grasp": False,
        "insertion_depth_limit": 0.0,
        "success_radial_clearance": 0.0,
        "peg_collision_radius": 0.0,
        "success_depth_overrun_tolerance": 0.0,
        "success_protocol": "legacy_marker_depth_v1",
    }
    for name, value in legacy_options.items():
        if hasattr(env_cfg, name):
            saved_env.setdefault(name, value)
    training_seed = saved_agent.get("seed")
    resolved = {"env": env_cfg.to_dict(), "agent": agent_cfg}
    saved = {"env": saved_env, "agent": saved_agent}
    physics_selected = False
    for token in overrides:
        key = token.split("=", 1)[0]
        if key in ("presets", "renderer"):
            raise ValueError(
                "Preset/renderer changes require --use_current_config and an explicit replay configuration."
            )
        if key == "physics":
            # Preset selection replaces the entire physics block, as in training.
            saved_env["sim"]["physics"] = deepcopy(resolved["env"]["sim"]["physics"])
            physics_selected = True
        elif key.startswith(("env.", "agent.")):
            section, *path = key.split(".")
            source, destination = resolved[section], saved[section]
            for part in path[:-1]:
                source = source[part]
                destination = destination.setdefault(part, {})
            destination[path[-1]] = deepcopy(source[path[-1]])
        elif key.startswith(("+", "~")):
            raise ValueError(
                "Config additions/deletions require --use_current_config and an explicit replay configuration."
            )

    if not physics_selected:
        recorded_physics = saved_env.get("sim", {}).get("physics", {}).get("class_type")
        current_physics = resolved["env"]["sim"]["physics"]["class_type"]
        if recorded_physics != current_physics:
            raise ValueError(
                f"Checkpoint physics {recorded_physics!r} differs from selected physics {current_physics!r}. "
                "Select the recorded physics preset explicitly, or use --use_current_config for a migration probe."
            )
    restored_env = env_cfg.copy()
    # The generic config updater rejects None -> int for an initially unset seed.
    restored_env.seed = saved_env.get("seed")
    restored_env.from_dict(saved_env)
    return restored_env, saved_agent, training_seed
