"""Unqueued reference-family candidates; no mean/noise/control caps added.

The historically successful matched builder is retained. Require explicit
raw Gaussian sampling so PPO scores its actual pre-physical-clipping samples.
An independent builder adds only the staged actor-normalizer update schedule.
Neither builder is GPU-tested or selected by any frozen queue.
"""

import importlib

from scripts.studies import rollout_frozen_actor_normalization as normalization


GUARDS = (
    "_likelihood_mean_guard_version",
    "_arm_noise_guard_version",
    "_all_action_noise_guard_version",
    "_gripper_mean_guard_version",
    "_partner_residual_guard_version",
)


def build_raw_reference_mappo_agent(env, experiment_cfg):
    if (
        experiment_cfg["models"]["policy"].get("clip_actions") is not False
        or experiment_cfg["agent"].get("clip_predicted_values") is not True
    ):
        raise ValueError("Explicit raw Gaussian actions and original project value clipping required")
    matched = importlib.import_module("isaaclab_tasks.direct.openarm_motion_context.models.matched_communication")
    agent = matched.build_matched_communication_mappo_agent(env, experiment_cfg)
    for modules in agent.models.values():
        policy = modules["policy"]
        if getattr(policy, "_g_clip_actions", None) is not False or any(hasattr(policy, key) for key in GUARDS):
            raise ValueError("Unbounded original matched policy without experimental guards required")
    return agent


def build_rollout_frozen_raw_reference_mappo_agent(env, experiment_cfg):
    agent = build_raw_reference_mappo_agent(env, experiment_cfg)
    normalization.install(agent)
    return agent
