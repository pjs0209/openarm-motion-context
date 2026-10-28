"""Opt-in 0.75-scaled camera-lead variant; never changes frozen studies.

The factor was selected from offline old-teacher camera traces, not strict
centered-ball success. Future policy evaluation must use untouched seeds.
"""

from types import MethodType

import torch

from scripts.studies.successful_ball_camera_lead import (
    build_successful_ball_camera_lead_mappo_agent,
)


LEAD_SCALE = 0.75
MARKER = "_successful_ball_camera_lead_scaled_v1"


def install_scaled_lead(task):
    if getattr(task, MARKER, False):
        raise ValueError("Scaled camera lead already installed")
    original = task.get_additional_actor_camera_observation

    def observation(self, side):
        delivered = original(side)
        if delivered.ndim != 2 or delivered.shape[1] != 7 or not torch.isfinite(delivered).all():
            raise ValueError("Finite seven-channel rendered camera lead required")
        scaled = delivered.clone()
        scaled[:, 4:6] *= LEAD_SCALE
        return scaled

    task.get_additional_actor_camera_observation = MethodType(observation, task)
    setattr(task, MARKER, True)


def build_successful_ball_camera_lead_scaled_mappo_agent(env, experiment_cfg):
    from isaaclab_tasks.direct.openarm_motion_context.envs.ball_center_goal import unwrap_ball_task

    agent = build_successful_ball_camera_lead_mappo_agent(env, experiment_cfg)
    task = unwrap_ball_task(env)
    install_scaled_lead(task)
    for uid in agent.possible_agents:
        policy = agent.checkpoint_modules[uid]["policy"]
        policy.register_buffer(MARKER, policy.own_backbone[0].weight.new_tensor(LEAD_SCALE))

        def validate(module, state, prefix, *args):
            value = state.get(prefix + MARKER)
            if value is None or value.numel() != 1 or float(value) != LEAD_SCALE:
                raise ValueError("Scaled camera-lead checkpoint identity missing or changed")

        policy.register_load_state_dict_pre_hook(validate)
    print(f"[BALL_CAMERA_LEAD_SCALE] {LEAD_SCALE:.2f}; offline-calibrated, not held-out policy result", flush=True)
    return agent
