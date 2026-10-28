# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Checkpoint identity for matched own-controller-state observation experiments."""

import torch

from ..envs.control_state_observation import CONTROL_STATE_DIM
from .actor import build_motion_context_mappo_agent
from .grasp_reset_exploration import build_grasp_exploration_mappo_agent

_MARKER = "_own_control_state_observation_version"


def _validate_checkpoint(module, state, prefix, *args):
    actual = state.get(prefix + _MARKER)
    expected = getattr(module, _MARKER)
    if actual is None or not torch.equal(actual.detach().cpu(), expected.detach().cpu()):
        raise ValueError("Own controller-state observation checkpoint protocol mismatch")


def _check_profile(env):
    task = getattr(env, "unwrapped", env)
    if not task.cfg.observe_control_state or task.cfg.control_state_observation_dim != CONTROL_STATE_DIM:
        raise ValueError("Controller-state builder requires observe_control_state=True and the exact declared layout")


def _mark(agent):
    for modules in agent.models.values():
        policy = modules["policy"]
        policy.register_buffer(_MARKER, policy.arm_action_head.weight.new_tensor([1.0, CONTROL_STATE_DIM]))
        policy.register_load_state_dict_pre_hook(_validate_checkpoint)
    return agent


def build_control_state_mappo_agent(env, experiment_cfg):
    """Use the unchanged MAPPO builder with the optional marked observation layout."""
    _check_profile(env)
    return _mark(build_motion_context_mappo_agent(env, experiment_cfg))


def build_control_state_grasp_exploration_agent(env, experiment_cfg):
    """Keep the Lift grasp-exploration protocol; add only own controller inputs."""
    _check_profile(env)
    return _mark(build_grasp_exploration_mappo_agent(env, experiment_cfg))
