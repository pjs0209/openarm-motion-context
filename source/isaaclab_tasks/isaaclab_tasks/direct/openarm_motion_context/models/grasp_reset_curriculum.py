# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Optional Lift reset curriculum with the original evaluation distribution.

Warm episodes start with open jaws at the canonical box grasp poses. Only
reset writes robot/object states; all subsequent grasping uses physics. Warm
episodes fade out early enough that every episode in the configured final
training interval starts from the original randomized reset distribution.
Evaluation never draws curriculum randomness or changes the original reset.
Observations, rewards, success conditions and the six-channel message are
unchanged relative to the signed camera-rotation Lift profile.
"""

from types import MethodType

import torch

from ..camera_pose.runtime import CameraPoseMotionContextTrainer, build_camera_pose_mappo_agent
from ..envs.lift_task_logic import reset_openarm_re_context
from ..legacy_api import read_data, set_joint_position, write_root_pose

PROFILE = "lift_open_jaw_reset_curriculum_v1"
_MARKER = "_grasp_reset_curriculum_protocol"
# Canonical grasp FK targets from the physical Lift path. These are reset
# poses, not policy labels or an object-following controller.
_WARM_ARM_POSITIONS = {
    "left": (-0.7404261231, -0.4015008807, 0.5062908530, 0.9834080935, -0.6183286309, 0.1556064487, 1.5688555241),
    "right": (0.7387738228, 0.3987355530, -0.4996838868, 0.9834084511, 0.6121079326, -0.1529811323, -1.5707325935),
}


def _warm_start_probability(step, training, warm_until, normal_from, episode_steps):
    """Fade warm resets before the last possible warm episode can reach normal_from."""
    fade_end = normal_from - episode_steps
    if not 0 <= warm_until < fade_end or episode_steps <= 0:
        raise ValueError("Reset curriculum requires warm_until < normal_from - episode_steps")
    if not training:
        return 0.0
    return min(1.0, max(0.0, (fade_end - step) / (fade_end - warm_until)))


def _probability(task):
    return _warm_start_probability(
        task.common_step_counter,
        bool(getattr(task, "training", False)),
        task._grasp_curriculum_warm_until,
        task._grasp_curriculum_normal_from,
        int(task.max_episode_length),
    )


def _reset_idx(self, env_ids):
    result = self._grasp_curriculum_original_reset_idx(env_ids)
    if env_ids is None:
        env_ids = self.robot._ALL_INDICES
    env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
    self._grasp_curriculum_warm_mask[env_ids] = False
    probability = _probability(self)
    if not len(env_ids) or probability == 0.0:
        return result
    warm_ids = env_ids if probability == 1.0 else env_ids[torch.rand(len(env_ids), device=self.device) < probability]
    if not len(warm_ids):
        return result

    joints = read_data(self.robot.data, "default_joint_pos")[warm_ids].clone()
    for side in ("left", "right"):
        arm_ids = getattr(self, f"{side}_arm_joint_ids")
        fingers = getattr(self, f"{side}_gripper_joint_ids")
        joints[:, arm_ids] = self._grasp_curriculum_arm_positions[side]
        joints[:, fingers] = self.cfg.gripper_open_target
        getattr(self, f"_{side}_arm_incremental_target")[warm_ids] = joints[:, arm_ids]
        set_joint_position(self.robot, joints[:, arm_ids], joint_ids=arm_ids, env_ids=warm_ids)
        set_joint_position(self.robot, joints[:, fingers], joint_ids=fingers, env_ids=warm_ids)
    self.robot.write_joint_state_to_sim_index(position=joints, velocity=torch.zeros_like(joints), env_ids=warm_ids)

    state = read_data(self.object.data, "default_root_state")[warm_ids].clone()
    state[:, :3] = self.scene.env_origins[warm_ids] + self.object_authored_local_pos
    state[:, 3:7] = self.object_authored_local_rot
    state[:, 7:] = 0.0
    write_root_pose(self.object, state[:, :7], env_ids=warm_ids)
    self.object.write_root_velocity_to_sim_index(root_velocity=state[:, 7:], env_ids=warm_ids)
    reset_openarm_re_context(self, self._ctx, warm_ids, object_state=state)
    self._grasp_curriculum_warm_mask[warm_ids] = True
    return result


def _get_rewards(self):
    rewards = self._grasp_curriculum_original_get_rewards()
    self.extras["log"].update(
        {
            "curriculum/warm_reset_probability": _probability(self),
            "curriculum/active_warm_episode_fraction": self._grasp_curriculum_warm_mask.float().mean().item(),
        }
    )
    return rewards


def _trace_signals(self, env_index=0):
    signals = self._grasp_curriculum_original_trace_signals(env_index)
    index = max(0, min(int(env_index), self.num_envs - 1))
    signals["reset_curriculum_warm_start"] = bool(self._grasp_curriculum_warm_mask[index])
    return signals


def install_grasp_reset_curriculum(env, warm_until=10000, normal_from=40000):
    """Install a training-only reset mixture on the signed camera-rotation Lift.

    The first ``warm_until`` control steps use canonical open-jaw starts.
    Their probability fades to zero one episode horizon before ``normal_from``.
    Thus no warm episode remains at ``normal_from`` without forced resets.
    Evaluation retains the original randomized reset and its RNG consumption.
    """
    from isaaclab_newton.physics import NewtonCfg

    task = getattr(env, "unwrapped", env)
    if hasattr(task, "_grasp_curriculum_profile"):
        raise ValueError("Lift reset curriculum is already installed")
    if (
        task.cfg.intent_task_label != "openarm_lift"
        or task.own_observation_dim - getattr(task.cfg, "control_state_observation_dim", 0) != 33
    ):
        raise ValueError("Reset curriculum requires the signed camera-rotation plain Lift task")
    if not isinstance(task.cfg.sim.physics, NewtonCfg):
        raise ValueError("This reset curriculum is validated only for the Newton Lift profile")
    if any(not isinstance(v, int) or isinstance(v, bool) for v in (warm_until, normal_from)):
        raise ValueError("Reset curriculum boundaries must be integer control steps")
    _warm_start_probability(0, False, warm_until, normal_from, int(task.max_episode_length))
    limits = read_data(task.robot.data, "joint_pos_limits")
    positions = {}
    for side in ("left", "right"):
        ids = getattr(task, f"{side}_arm_joint_ids")
        default = getattr(task, f"{side}_arm_default_joint_pos")
        target = limits.new_tensor(_WARM_ARM_POSITIONS[side])
        lower = torch.maximum(limits[:, ids, 0], default - task.cfg.action_scale)
        upper = torch.minimum(limits[:, ids, 1], default + task.cfg.action_scale)
        if (target < lower).any() or (target > upper).any():
            raise ValueError("Canonical warm poses must lie within physical and configured arm action bounds")
        positions[side] = target
    task._grasp_curriculum_profile = PROFILE
    # The trainer updates every existing wrapper/task attribute. Expose this
    # on the task before its first reset, rather than only on the RL wrapper.
    task.training = bool(getattr(task, "training", False))
    task._grasp_curriculum_warm_until = warm_until
    task._grasp_curriculum_normal_from = normal_from
    task._grasp_curriculum_arm_positions = positions
    task._grasp_curriculum_warm_mask = torch.zeros(task.num_envs, dtype=torch.bool, device=task.device)
    task._grasp_curriculum_original_reset_idx = task._reset_idx
    task._grasp_curriculum_original_get_rewards = task._get_rewards
    task._grasp_curriculum_original_trace_signals = task.get_paper_motion_context_trace_signals
    task._reset_idx = MethodType(_reset_idx, task)
    task._get_rewards = MethodType(_get_rewards, task)
    task.get_paper_motion_context_trace_signals = MethodType(_trace_signals, task)


def _validate_checkpoint(module, state, prefix, *args):
    value = state.get(prefix + _MARKER)
    if value is None or not torch.equal(value.detach().cpu(), getattr(module, _MARKER).detach().cpu()):
        raise ValueError("Lift reset curriculum checkpoint protocol mismatch")


def build_grasp_reset_curriculum_mappo_agent(env, experiment_cfg):
    """Keep the camera-pose Gaussian MAPPO policy and add reset curriculum identity."""
    runtime = experiment_cfg.get("runtime", {})
    warm_until = runtime.get("grasp_curriculum_warm_until", 10000)
    normal_from = runtime.get("grasp_curriculum_normal_from", 40000)
    agent = build_camera_pose_mappo_agent(env, experiment_cfg)
    install_grasp_reset_curriculum(env, warm_until, normal_from)
    for modules in agent.models.values():
        policy = modules["policy"]
        policy.register_buffer(_MARKER, policy.arm_action_head.weight.new_tensor([1.0, warm_until, normal_from]))
        policy.register_load_state_dict_pre_hook(_validate_checkpoint)
    print(f"[INFO] Lift reset curriculum: warm_until={warm_until}, all_normal_from={normal_from}; eval always original")
    return agent


class GraspResetCurriculumTrainer(CameraPoseMotionContextTrainer):
    """Record actual episode starts separately from the unchanged partner payload."""

    @staticmethod
    def _new_mode_trace():
        trace = CameraPoseMotionContextTrainer._new_mode_trace()
        trace.update(reset_curriculum_profile=PROFILE, reset_curriculum_warm_start=[])
        return trace

    def _append_mode_trace(self, outputs, rewards, infos, trace_signals=None, policy_states=None):
        before = len(self._mode_trace["left_motion_context"])
        super()._append_mode_trace(outputs, rewards, infos, trace_signals, policy_states)
        if len(self._mode_trace["left_motion_context"]) != before:
            if trace_signals is None:
                raise RuntimeError("Reset curriculum tracing requires pre-action episode-start signals")
            self._mode_trace["reset_curriculum_warm_start"].append(trace_signals["reset_curriculum_warm_start"])
