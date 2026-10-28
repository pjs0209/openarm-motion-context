"""Opt-in camera-Ball reward combining post-lift retention and center margin.

Both terms are added to the unchanged native Lift reward and strict
centered-held success predicate. Simulator ball position is reward-only; the
actor still receives the rendered depth/semantic-segmentation estimate.
This candidate reuses the same audited camera68 actor as the post-lift study.
"""

import torch

from scripts.studies.ball_postlift_center_retention import (
    build_ball_postlift_center_retention_mappo_agent,
)
from scripts.studies.ball_prelift_center_margin import install_prelift_center_margin


MARKER = "_ball_postlift_margin_combo_reward_version"


def build_ball_postlift_margin_combo_mappo_agent(env, experiment_cfg):
    """Install the bounded pre-lift margin on the successful post-lift base."""
    from isaaclab_tasks.direct.openarm_motion_context.envs.ball_center_goal import unwrap_ball_task

    agent = build_ball_postlift_center_retention_mappo_agent(env, experiment_cfg)
    task = unwrap_ball_task(env)
    install_prelift_center_margin(task)
    for modules in agent.models.values():
        critic = modules["value"]
        critic.register_buffer(MARKER, torch.tensor(1.0, device=critic.device))

        def validate(module, incompatible):
            if MARKER in incompatible.missing_keys or float(getattr(module, MARKER)) != 1.0:
                raise ValueError("Ball post-lift margin-combo critic identity missing or changed")

        critic.register_load_state_dict_post_hook(validate)
    print("[BALL_POSTLIFT_MARGIN_COMBO] postlift retention plus prelift 3.5cm margin", flush=True)
    return agent
