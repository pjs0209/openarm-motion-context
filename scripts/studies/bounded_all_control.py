"""Combined next candidate: bounded arm correction, all means and all sigma."""

from scripts.studies import bounded_peer_control as control
from scripts.studies import gripper_mean_prototype as gripper


def build_bounded_all_control_mappo_agent(env, experiment_cfg):
    agent = control.build_bounded_peer_control_mappo_agent(env, experiment_cfg)
    for modules in agent.models.values():
        gripper.install(modules["policy"])
    return agent


def build_fresh_camera_bounded_all_control_mappo_agent(env, experiment_cfg):
    agent = control.build_fresh_camera_bounded_peer_control_mappo_agent(env, experiment_cfg)
    for modules in agent.models.values():
        gripper.install(modules["policy"])
    return agent
