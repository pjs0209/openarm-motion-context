"""Opt-in V2 camera-Ball warm-start with bounded partner action correction.

Hold V2 reward, centered-held goal, camera observation and common camera68
teacher fixed; change only the matched Full6D/NoShare policy action heads.
This is a prepared branch, not a trained or validated result.
"""

import torch

from scripts.studies.ball_early_center_preservation_v2 import (
    build_ball_early_center_preservation_v2_mappo_agent,
)
from scripts.studies.lift_orientation_bounded_partner_v1 import (
    BOUND,
    install_bounded_partner,
)


MARKER = "_ball_v2_bounded_partner_v1_actor_version"


def build_ball_v2_bounded_partner_v1_mappo_agent(env, experiment_cfg):
    if (
        getattr(env, "communication_mode", None) != "motion_context"
        or getattr(env, "sharing_mode", None) not in ("intent_share", "no_share")
        or experiment_cfg["models"]["policy"].get("clip_actions") is not False
        or float(experiment_cfg["models"]["policy"].get("partner_action_max_ratio", 0.5)) != 0.5
    ):
        raise ValueError("Exact matched camera-Ball raw-action package required")
    agent = build_ball_early_center_preservation_v2_mappo_agent(env, experiment_cfg)
    for modules in agent.models.values():
        install_bounded_partner(modules["policy"])
        critic = modules["value"]
        critic.register_buffer(MARKER, torch.tensor(1.0, device=critic.device))

        def validate(module, incompatible):
            if MARKER in incompatible.missing_keys or float(getattr(module, MARKER)) != 1.0:
                raise ValueError("Ball V2 bounded-partner actor identity missing")

        critic.register_load_state_dict_post_hook(validate)
    print(f"[BALL_V2_BOUNDED_PARTNER_V1] warm-start V2; partner correction ±{BOUND}", flush=True)
    return agent
