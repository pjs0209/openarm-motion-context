# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Separate Lift variants that retain the direction of camera rotation error."""

import gymnasium as gym


def mappo_cfg():
    """Reuse the common MAPPO settings with versioned camera-pose tracing."""
    from pathlib import Path

    from isaaclab.utils.io import load_yaml

    cfg = load_yaml(str(Path(__file__).resolve().parents[1] / "train/configs/lift_mappo.yaml"))
    cfg["runtime"]["agent_builder"] = f"{__name__}.runtime:build_camera_pose_mappo_agent"
    cfg["runtime"]["trainer"] = f"{__name__}.runtime:CameraPoseMotionContextTrainer"
    return cfg


for task, suffix in (("Lift", "Lift"), ("LiftWithBall", "LiftWithBall")):
    gym.register(
        id=f"Isaac-OpenArm-{task}-RotationObs-v0",
        entry_point=f"{__name__}.envs:OpenArmCameraRotation{suffix}Env",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.envs:OpenArmCameraRotation{suffix}EnvCfg",
            "skrl_mappo_cfg_entry_point": f"{__name__}:mappo_cfg",
        },
    )
