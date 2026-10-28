"""Opt-in Full6D pilot: communicate motion/trend to arms, not gripper heads.

The actor slot, encoder, local backbone, action heads and MAPPO settings are
otherwise identical to the matched communication builder. A separately trained
No Share arm must use this builder as its matched architectural control.
"""

from .matched_communication import build_matched_communication_mappo_agent


def build_matched_gripper_local_mappo_agent(env, experiment_cfg):
    if env.communication_mode != "motion_context" or env.sharing_mode not in ("intent_share", "no_share"):
        raise ValueError("Gripper-local pilot requires Full6D or matched No Share")
    agent = build_matched_communication_mappo_agent(env, experiment_cfg)
    for modules in agent.models.values():
        policy = modules["policy"]
        if policy.gripper_action_head is None:
            raise ValueError("Gripper-local pilot requires a gripper action head")
        policy.gripper_partner_local_only = True
    print("[INFO] gripper-local pilot: partner 6D reaches arm head but not gripper head")
    return agent
