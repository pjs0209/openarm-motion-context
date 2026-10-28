"""Opt-in WithBall reward for early box sway preceding ball-center loss.

Keep native Lift reward, center penalty, camera observations, actor, and
centered-held success exactly as in raw_camera_ball_reference. The only new
term is a bounded penalty for box XY displacement from its reset pose.
Ground-truth box pose is used for reward only; no extra actor observation.
"""

from types import MethodType
import importlib

import torch

from scripts.studies.raw_camera_ball_reference import build_raw_camera_ball_reference_mappo_agent
from isaaclab_tasks.direct.openarm_motion_context.legacy_api import (
    read_data,
    write_root_pose,
    set_joint_position,
    tensor,
)


PENALTY_SCALE = 2.0
FREE_DISPLACEMENT_M = 0.005
FULL_DISPLACEMENT_M = 0.05


def lateral_penalty(current_xy, reset_xy):
    if current_xy.shape != reset_xy.shape or current_xy.ndim != 2 or current_xy.shape[1] != 2:
        raise ValueError("Matching [environment,2] box XY states required")
    if not torch.isfinite(current_xy).all() or not torch.isfinite(reset_xy).all():
        raise ValueError("Finite box XY states required")
    displacement = torch.linalg.vector_norm(current_xy - reset_xy, dim=-1)
    severity = ((displacement - FREE_DISPLACEMENT_M) / (FULL_DISPLACEMENT_M - FREE_DISPLACEMENT_M)).clamp(0, 1)
    return -PENALTY_SCALE * severity, displacement


def install_lateral_penalty(task):
    if getattr(task, "_ball_lateral_penalty_installed", False):
        raise ValueError("Ball lateral penalty already installed")
    if (
        task.cfg.ball_reward_profile != "center_penalty"
        or float(task.cfg.ball_center_penalty_full_error) != 0.25
        or getattr(task, "_ball_goal_variant", None) != "centered_hold_v1"
    ):
        raise ValueError("Exact centered-hold center-penalty Ball task required")
    original = task._get_rewards

    def rewards(self):
        result = original()
        penalty, displacement = lateral_penalty(
            read_data(self.object.data, "root_pos_w")[:, :2], self._ctx["target_object_xy_w"]
        )
        if any(value.shape != penalty.shape for value in result.values()):
            raise ValueError("Expected one team reward per environment")
        result = {uid: value + penalty for uid, value in result.items()}
        logs = self.extras.setdefault("log", {})
        logs["reward_ball_lateral_retention"] = float(penalty.mean())
        logs["box_xy_displacement"] = float(displacement.mean())
        for key in (
            "team_reward",
            "left_reward",
            "right_reward",
            "left_reward_raw",
            "right_reward_raw",
            "left_reward_total",
            "right_reward_total",
        ):
            if key in logs:
                logs[key] += float(penalty.mean())
        return result

    task._get_rewards = MethodType(rewards, task)
    task._ball_lateral_penalty_installed = True


def build_ball_lateral_retention_mappo_agent(env, experiment_cfg):
    raw_agent = build_raw_camera_ball_reference_mappo_agent(env, experiment_cfg)
    goal = importlib.import_module("isaaclab_tasks.direct.openarm_motion_context.envs.ball_center_goal")
    task = goal.unwrap_ball_task(env)
    install_lateral_penalty(task)
    return raw_agent
