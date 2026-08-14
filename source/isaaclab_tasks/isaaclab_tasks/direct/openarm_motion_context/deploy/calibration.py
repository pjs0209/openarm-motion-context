# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Export deploy calibration from the OpenArm Lift USD/task geometry.

The export intentionally uses the same task-logic path as training:

* KLT-local ``apriltag_00`` through ``apriltag_02`` poses
* KLT-local collision grasp targets after the configured z/rotation offset
* camera and robot frame names copied from the deploy YAML template

Run this with Isaac Lab so the USD stage is available.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", type=str, default="Isaac-OpenArm-Lift-v0", help="OpenArm lift task name")
parser.add_argument(
    "--template",
    type=str,
    default="source/isaaclab_tasks/isaaclab_tasks/direct/openarm_motion_context/deploy/configs/lift_real.yaml",
    help="Deploy YAML template to update",
)
parser.add_argument(
    "--output",
    type=str,
    default="source/isaaclab_tasks/isaaclab_tasks/direct/openarm_motion_context/deploy/configs/lift_real_calibrated.yaml",
    help="Output calibrated deploy YAML",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
import yaml

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab_tasks.direct.openarm_motion_context.envs.lift_task_logic import _usd_relative_pose


_CAMERA_MOUNT_PRIMS = {
    "left_wrist": "openarm_left_link7",
    "right_wrist": "openarm_right_link7",
    "chest": "openarm_body_link",
}


def _round_nested(values, digits: int = 8):
    if isinstance(values, torch.Tensor):
        values = values.detach().cpu().tolist()
    if isinstance(values, list):
        return [_round_nested(v, digits) for v in values]
    if isinstance(values, tuple):
        return [_round_nested(v, digits) for v in values]
    return round(float(values), digits)


def _load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def main() -> None:
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1, use_fabric=False)
    env = gym.make(args_cli.task, cfg=env_cfg)
    try:
        env.reset()
        task_env = env.unwrapped
        ctx = task_env._ctx
        required = (
            "apriltag_local_pos",
            "apriltag_local_quat",
            "left_target_local_pos",
            "left_target_local_quat",
            "right_target_local_pos",
            "right_target_local_quat",
        )
        missing = [name for name in required if name not in ctx]
        if missing:
            raise RuntimeError(f"Task context did not provide calibration tensors: {missing}")

        template_path = Path(args_cli.template)
        data = _load_yaml(str(template_path)) if template_path.exists() else {}
        data.setdefault("apriltag", {})
        data.setdefault("grip_targets", {})
        data.setdefault("camera_extrinsics_usd", {})
        data["apriltag"]["ids"] = [0, 1, 2]
        data["apriltag"]["tag_pos_box"] = _round_nested(ctx["apriltag_local_pos"][0])
        data["apriltag"]["tag_quat_box"] = _round_nested(ctx["apriltag_local_quat"][0])
        data["grip_targets"]["left_pos_box"] = _round_nested(ctx["left_target_local_pos"][0])
        data["grip_targets"]["left_quat_box"] = _round_nested(ctx["left_target_local_quat"][0])
        data["grip_targets"]["right_pos_box"] = _round_nested(ctx["right_target_local_pos"][0])
        data["grip_targets"]["right_quat_box"] = _round_nested(ctx["right_target_local_quat"][0])

        camera_paths = ctx.get("apriltag_camera_paths_by_env", [[]])[0]
        if len(camera_paths) != len(_CAMERA_MOUNT_PRIMS):
            raise RuntimeError(
                f"Expected {len(_CAMERA_MOUNT_PRIMS)} runtime camera prims, got {len(camera_paths)}"
            )
        robot_root = "/World/envs/env_0/Robot"
        for (camera_name, mount_prim), camera_path in zip(_CAMERA_MOUNT_PRIMS.items(), camera_paths):
            mount_path = f"{robot_root}/{mount_prim}"
            pos, quat = _usd_relative_pose(task_env, mount_path, camera_path)
            data["camera_extrinsics_usd"][camera_name] = {
                "mount_prim": mount_prim,
                "camera_prim": str(camera_path).removeprefix(f"{robot_root}/"),
                "pos_mount_camera": _round_nested(pos),
                "quat_mount_camera": _round_nested(quat),
            }

        output_path = Path(args_cli.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, sort_keys=False)
        print(f"[INFO] Exported deploy calibration to: {os.path.abspath(output_path)}")
        print("[INFO] AprilTag local positions:", data["apriltag"]["tag_pos_box"])
        print("[INFO] Left grip local position:", data["grip_targets"]["left_pos_box"])
        print("[INFO] Right grip local position:", data["grip_targets"]["right_pos_box"])
        print("[INFO] USD mount-to-color-optical transforms (wxyz):")
        for camera_name, extrinsic in data["camera_extrinsics_usd"].items():
            print(
                f"  {camera_name}: {extrinsic['mount_prim']} -> {extrinsic['camera_prim']} "
                f"pos={extrinsic['pos_mount_camera']} quat={extrinsic['quat_mount_camera']}"
            )
    finally:
        env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
