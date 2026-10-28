# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Checkpoint and trace identity for the optional signed camera-rotation tasks."""

import torch

from ..models.arm_target_control import build_target_bounded_mappo_agent
from ..train.trainer import MotionContextTrainer
from .envs import PROFILE

_MARKER = "_camera_rotation_observation_version"


def _validate_profile(module, state, prefix, *args):
    value = state.get(prefix + _MARKER)
    if value is None or not torch.equal(value.detach().cpu(), getattr(module, _MARKER).detach().cpu()):
        raise ValueError("Camera-rotation observation checkpoint protocol mismatch")


def build_camera_pose_mappo_agent(env, experiment_cfg):
    """Use the same Gaussian actor and physical target guard with a marked layout."""
    task = getattr(env, "unwrapped", env)
    if task.cfg.local_pose_observation_profile != PROFILE or task.own_observation_dim - getattr(
        task.cfg, "control_state_observation_dim", 0
    ) not in (33, 37):
        raise ValueError("Camera-rotation builder requires its registered observation task")
    agent = build_target_bounded_mappo_agent(env, experiment_cfg)
    for modules in agent.models.values():
        policy = modules["policy"]
        policy.register_buffer(_MARKER, policy.arm_action_head.weight.new_tensor(1.0))
        policy.register_load_state_dict_pre_hook(_validate_profile)
    return agent


class CameraPoseMotionContextTrainer(MotionContextTrainer):
    """Trace appended camera features at the same pre-action time as the policy."""

    @staticmethod
    def _new_mode_trace():
        trace = MotionContextTrainer._new_mode_trace()
        trace.update(
            local_observation_profile=PROFILE,
            camera_rotation_error_normalization="rotation_vector_in_base_axes_divided_by_pi",
            left_camera_rotation_error_b=[],
            right_camera_rotation_error_b=[],
            left_policy_camera_rotation_error_b=[],
            right_policy_camera_rotation_error_b=[],
        )
        return trace

    def _append_mode_trace(self, outputs, rewards, infos, trace_signals=None, policy_states=None):
        before = len(self._mode_trace["left_motion_context"])
        super()._append_mode_trace(outputs, rewards, infos, trace_signals, policy_states)
        if len(self._mode_trace["left_motion_context"]) == before:
            return
        if trace_signals is None or policy_states is None:
            raise RuntimeError("Camera-rotation traces require the policy's pre-action signals")
        own_dim = int(self._resolve_env_attr("own_observation_dim"))
        index = max(0, min(int(self.mode_trace_env_index), self.env.num_envs - 1))
        for side in ("left", "right"):
            key = f"{side}_camera_rotation_error_b"
            self._mode_trace[key].append(trace_signals[key])
            delivered = self._env_vector(policy_states[f"{side}_arm"][..., own_dim - 3 : own_dim], index, 3)
            if not torch.allclose(torch.tensor(delivered), torch.tensor(trace_signals[key]), rtol=0, atol=1e-6):
                raise RuntimeError(f"{side} camera-rotation trace differs from the actual pre-action actor input")
            self._mode_trace[f"{side}_policy_camera_rotation_error_b"].append(delivered)
