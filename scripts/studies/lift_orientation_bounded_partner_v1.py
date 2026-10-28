"""Opt-in Lift orientation-retention actor with bounded partner correction.

This keeps the native Lift task, orientation-retention reward, and original
matched encoder/optimizer package. It bounds only the partner-induced arm
and gripper mean corrections to ±0.25 before physical action clipping.
Apply identically to Full6D and trained NoShare; this is a prospective
architecture contrast, not a replacement for the frozen reference runs.
"""

from torch import nn

from scripts.studies.lift_orientation_retention import build_lift_orientation_retention_mappo_agent


BOUND = 0.25


def _head_classes():
    from isaaclab_tasks.direct.openarm_motion_context.models.full6d_bounded_residual import (
        BoundedPartnerArmHead,
        BoundedPartnerGripperHead,
    )

    return BoundedPartnerArmHead, BoundedPartnerGripperHead


def install_bounded_partner(policy) -> None:
    BoundedPartnerArmHead, BoundedPartnerGripperHead = _head_classes()

    if (
        getattr(policy, "communication_mode", None) != "motion_context"
        or not getattr(policy, "uses_context", False)
        or type(policy.arm_action_head) is not nn.Linear
        or type(policy.gripper_action_head) is not nn.Sequential
        or policy.arm_action_head.in_features != policy.actor_action_input_dim
        or policy.actor_action_input_dim <= policy.policy_feature_dim
    ):
        raise ValueError("Requires the unmodified matched 6D policy")
    policy.arm_action_head = BoundedPartnerArmHead(policy.arm_action_head, policy.policy_feature_dim, BOUND)
    policy.gripper_action_head = BoundedPartnerGripperHead(
        policy.gripper_action_head,
        policy.policy_feature_dim,
        policy.actor_action_input_dim,
        BOUND,
    )

    def validate_markers(module, incompatible):
        expected = ["arm_action_head._residual_bound_marker", "gripper_action_head._residual_bound_marker"]
        if any(name in incompatible.missing_keys for name in expected):
            raise ValueError("Lift bounded-partner checkpoint identity missing")
        for name in expected:
            owner = module.arm_action_head if name.startswith("arm_") else module.gripper_action_head
            if float(owner._residual_bound_marker) != BOUND:
                raise ValueError("Lift bounded-partner checkpoint bound changed")

    policy.register_load_state_dict_post_hook(validate_markers)


def build_lift_orientation_bounded_partner_v1_mappo_agent(env, experiment_cfg):
    if (
        env.communication_mode != "motion_context"
        or env.sharing_mode not in ("intent_share", "no_share")
        or experiment_cfg["models"]["policy"].get("clip_actions") is not False
        or float(experiment_cfg["models"]["policy"].get("partner_action_max_ratio", 0.5)) != 0.5
    ):
        raise ValueError("Exact matched 6D/NoShare raw-action package required")
    agent = build_lift_orientation_retention_mappo_agent(env, experiment_cfg)
    for modules in agent.models.values():
        install_bounded_partner(modules["policy"])
    print("[LIFT_BOUNDED_PARTNER_V1] both methods: max partner correction ±0.25", flush=True)
    return agent
