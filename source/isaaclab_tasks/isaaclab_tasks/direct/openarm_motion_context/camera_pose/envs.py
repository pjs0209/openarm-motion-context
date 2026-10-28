# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Append signed camera rotation error without removing existing local features.

The three added channels are a shortest rotation vector in robot-base axes,
divided by pi. Targets come from the existing AprilTag pose reconstruction;
no simulator object quaternion is read here. The partner payload remains the
same six proprioceptive motion/trend channels, with a correspondingly padded
slot. These are distinct task/observation protocols, not legacy-checkpoint
adapters or evidence of communication superiority.
"""

import math

import torch

from isaaclab.utils import configclass

from ..envs.lift_env import OpenArmReIncrementalBimanualLiftEnv, OpenArmReIncrementalBimanualLiftEnvCfg
from ..envs.lift_task_logic import compute_actor_collision_target_poses_w
from ..envs.lift_with_ball_env import OpenArmLiftWithBallEnv, OpenArmLiftWithBallEnvCfg
from ..legacy_api import read_data
from ..legacy_math import quat_apply, quat_inv, quat_mul

PROFILE = "camera_rotation_vector_v1"


def camera_rotation_error_b(env, side):
    """Return camera target rotation error in base axes, normalized by pi [N,3]."""
    index = {"left": 2, "right": 3}[side]
    target = compute_actor_collision_target_poses_w(env, env._ctx)[index]
    actual = read_data(env.robot.data, "body_state_w")[:, env._ctx[f"{side}_ee_body_id"], 3:7]
    delta = quat_mul(target, quat_inv(actual))
    delta = torch.where(delta[:, :1] < 0, -delta, delta)
    xyz = delta[:, 1:]
    length = torch.linalg.vector_norm(xyz, dim=-1, keepdim=True)
    angle = 2 * torch.atan2(length, delta[:, :1].clamp_min(0))
    vector = xyz * angle / length.clamp_min(1e-8)
    base = quat_apply(quat_inv(read_data(env.robot.data, "root_quat_w")), vector)
    return (base / math.pi).clamp(-1, 1)


def _check_camera_pose_source(cfg):
    if cfg.actor_object_pose_source != "apriltag" or cfg.apriltag_measurement_source != "camera_transform":
        raise ValueError("Camera-rotation observations require the AprilTag camera-transform pose path")
    if cfg.local_pose_observation_profile != PROFILE:
        raise ValueError("Camera-rotation observation protocol mismatch")


def _rotation_trace(env, signals, env_index):
    index = max(0, min(int(env_index), env.num_envs - 1))
    for side in ("left", "right"):
        signals[f"{side}_camera_rotation_error_b"] = camera_rotation_error_b(env, side)[index].detach().cpu().tolist()
    return signals


@configclass
class OpenArmCameraRotationLiftEnvCfg(OpenArmReIncrementalBimanualLiftEnvCfg):
    """Original30 local channels plus3 camera-rotation channels; actor66/critic79."""

    camera_extra_observation_dim = 3
    local_pose_observation_profile: str = PROFILE

    def __post_init__(self):
        super().__post_init__()
        _check_camera_pose_source(self)


class OpenArmCameraRotationLiftEnv(OpenArmReIncrementalBimanualLiftEnv):
    """Preserve all original proprioception and append signed camera rotation."""

    cfg: OpenArmCameraRotationLiftEnvCfg

    def get_additional_actor_camera_observation(self, side):
        return camera_rotation_error_b(self, side)

    def get_paper_motion_context_trace_signals(self, env_index=0):
        return _rotation_trace(self, super().get_paper_motion_context_trace_signals(env_index), env_index)


@configclass
class OpenArmCameraRotationLiftWithBallEnvCfg(OpenArmLiftWithBallEnvCfg):
    """Retain30 proprioception+4 rendered-ball channels, append3; actor74/critic87."""

    camera_extra_observation_dim = 7
    local_pose_observation_profile: str = PROFILE

    def __post_init__(self):
        super().__post_init__()
        _check_camera_pose_source(self)


class OpenArmCameraRotationLiftWithBallEnv(OpenArmLiftWithBallEnv):
    """Keep rendered depth/mask ball sensing and append AprilTag rotation error."""

    cfg: OpenArmCameraRotationLiftWithBallEnvCfg

    def get_additional_actor_camera_observation(self, side):
        ball = super().get_additional_actor_camera_observation(side)
        return torch.cat((ball, camera_rotation_error_b(self, side)), dim=-1)

    def get_paper_motion_context_trace_signals(self, env_index=0):
        signals = super().get_paper_motion_context_trace_signals(env_index)
        # Preserve the existing ball-camera field's four-channel semantics.
        signals["ball_camera_observation"] = signals["ball_camera_observation"][:4]
        return _rotation_trace(self, signals, env_index)
