"""Versioned matched builder: identical stricter Ball goal for both methods."""

from .matched_communication import build_matched_communication_mappo_agent
from ..envs.ball_center_goal import install_centered_hold_goal, unwrap_ball_task


def build_center_goal_matched_mappo_agent(env, experiment_cfg):
    task = unwrap_ball_task(env)
    install_centered_hold_goal(task)
    agent = build_matched_communication_mappo_agent(env, experiment_cfg)
    print("[INFO] Ball goal variant=centered_hold_v1; original Lift predicates AND central ball held for N steps")
    return agent
