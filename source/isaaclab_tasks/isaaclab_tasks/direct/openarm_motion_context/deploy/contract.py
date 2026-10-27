# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Read policy-facing settings from a trusted training run without starting Isaac Sim."""

from copy import deepcopy
from pathlib import Path

import yaml


def restore_training_contract(config: dict) -> dict:
    """Restore the Lift policy contract; retain ROS topics and physical calibration.

    A checkpoint's own ``params`` files own model dimensions, preprocessing,
    control timing and communication settings. Calibration and ROS transport
    belong to the deployment configuration. Simulation-specific Ball perception
    and the physically welded Peg variant require separate robot task adapters.
    """
    cfg = deepcopy(config)
    checkpoint = Path(cfg["checkpoint"]).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    params = checkpoint.parent.parent / "params"
    with (params / "env.yaml").open() as stream:
        env = yaml.load(stream, Loader=yaml.FullLoader)
    with (params / "agent.yaml").open() as stream:
        agent = yaml.load(stream, Loader=yaml.FullLoader)
    if env["intent_task_label"] != "openarm_lift" or env["own_observation_dim"] != 30:
        raise ValueError(
            "This ROS task adapter supports Lift 30D observations; "
            "Ball and welded Peg require their own perception/grasp adapters"
        )
    if env["motion_context_representation"] != "trend":
        raise ValueError("This deployment adapter requires a Trend checkpoint")
    if env["action_spaces"] != {"left_arm": 8, "right_arm": 8}:
        raise ValueError("Lift deployment requires 8 actions per arm")
    cfg["checkpoint"] = str(checkpoint)
    cfg["control_rate_hz"] = 1.0 / (float(env["sim"]["dt"]) * int(env["decimation"]))
    action = cfg.setdefault("action", {})
    for key in (
        "action_scale",
        "action_step_scale",
        "gripper_open_target",
        "gripper_close_target",
        "gripper_switch_epsilon",
        "reset_grace_steps",
    ):
        action[key] = env[key]
    action["command_time_from_start"] = 1.0 / cfg["control_rate_hz"]
    comm = cfg.setdefault("communication", {})
    comm["mode"] = env["communication_mode"]
    comm["representation"] = env["motion_context_representation"]
    for suffix in ("horizon", "beta", "linear_epsilon", "angular_epsilon", "action_epsilon"):
        comm[f"context_trend_{suffix}"] = env[f"motion_context_trend_{suffix}"]
    for key in ("motion_intent_horizon", "interaction_motion_scale"):
        comm[key] = env[key]
    model = agent["models"]["policy"]
    policy = cfg.setdefault("policy", {})
    policy.update(
        own_observation_dim=env["own_observation_dim"],
        communication_feature_dim=env["communication_feature_dim"],
        hidden_sizes=model["network"][0]["layers"],
        partner_intent_embed_dim=model["partner_message_embed_dim"],
        motion_intent_dim=env["motion_intent_dim"],
        motion_context_dim=env["motion_context_dim"],
        ee_vel_scale=env["ee_vel_scale"],
    )
    preprocessor = agent["agent"]["state_preprocessor"]
    if preprocessor not in (None, "RunningStandardScaler"):
        raise ValueError(f"Unsupported checkpoint state preprocessor: {preprocessor}")
    policy["use_checkpoint_state_normalizer"] = preprocessor is not None
    normalizer = agent["agent"].get("state_preprocessor_kwargs") or {}
    policy["normalizer_epsilon"] = normalizer.get("epsilon", 1.0e-8)
    policy["normalizer_clip_threshold"] = normalizer.get("clip_threshold", 5.0)
    defaults = env["scene"]["robot"]["init_state"]["joint_pos"]
    for side in ("left_arm", "right_arm"):
        expected = [f"openarm_{side.removesuffix('_arm')}_joint{i}" for i in range(1, 8)]
        if cfg["joints"][side] != expected:
            raise ValueError(f"Unexpected {side} controller joint order")
        cfg.setdefault("default_joint_pos", {})[side] = [defaults[name] for name in expected]
    cfg["training_contract"] = {
        "params_dir": str(params),
        "training_seed": agent["seed"],
        "physics": env["sim"]["physics"]["class_type"],
        "quaternion_order": "wxyz",
        "action_update": "once_per_policy_step",
        "partner_message_delay_steps": 1,
        "robot_gravity_disabled": env["scene"]["robot"]["spawn"]["rigid_props"]["disable_gravity"],
        "actuators": env["scene"]["robot"]["actuators"],
    }
    return cfg


def apriltag_detector_parameters(config: dict, camera: str) -> dict:
    """Build apriltag_ros parameters with camera-specific TF names and per-tag metric sizes."""
    tag = config["apriltag"]
    ids, sizes = tag["ids"], tag["detector_edge_sizes_m"]
    if len(ids) != len(sizes) or any(size is None or float(size) <= 0 for size in sizes):
        raise ValueError("Every tag needs its own positive black-border edge size in metres")
    frames = tag["frames_by_camera"][camera]
    if len(frames) != len(ids):
        raise ValueError("Tag frame and ID counts differ")
    return {
        "/**": {
            "ros__parameters": {
                "family": tag["family"],
                "size": float(sizes[0]),
                "image_transport": "raw",
                "max_hamming": 0,
                "qos_profile": "sensor_data",
                "pose_estimation_method": "pnp",
                "tag": {"ids": ids, "frames": frames, "sizes": [float(value) for value in sizes]},
            }
        }
    }
