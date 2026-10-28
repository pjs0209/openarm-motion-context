# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Free Circle parts retained by gripper contact, without attachment constraints.

This is a pregrasp insertion task: reset places the two parts between the jaws.
Every subsequent object pose and velocity comes from the physics solver. Opening
the fingers may release a part; neither reward nor observation code repositions it.
"""

import torch

from isaaclab.utils import configclass

from ..legacy_api import read_data, set_joint_position, write_root_pose
from ..legacy_math import quat_apply, quat_inv, quat_mul
from .peg_circle_seated_cfg import OpenArmFixedCirclePegAndHoleSeatedEnvCfg
from .peg_grasp_geometry import grasp_point_errors, physical_grasp_validity
from .peg_in_hole_env import OpenArmFixedPegAndHoleEnv, OpenArmPegInHoleEnv


@configclass
class OpenArmContactCirclePegAndHoleEnvCfg(OpenArmFixedCirclePegAndHoleSeatedEnvCfg):
    """The CAD seating geometry with two free parts and eight actions per arm."""

    action_spaces = {"left_arm": 8, "right_arm": 8}
    arm_only_actions = False
    fixed_objects_to_grippers = False
    use_runtime_fixed_joints = False
    fixed_peg_and_hole = False
    fixed_hole = False
    require_contact_grasp: bool = True
    success_protocol: str = "circle_contact_grasp_seating_v1"
    reset_grace_steps = 0
    # The authored driven finger range is 0..44 mm. Reset at the nominal
    # 80 mm part width; the extra opening clearance permits physical release.
    gripper_open_target = 0.044
    gripper_target_speed_limit = 0.2
    pregrasp_finger_position: float = 0.040
    grasp_success_distance: float = 0.020
    grasp_drop_distance: float = 0.040
    # Retaining the initial physical grasp must earn positive preparation
    # reward. Losing the part is a separate terminal failure.
    reward_grasp_retention_scale: float = 6.0
    reward_grasp_drop_scale: float = 10.0
    # Nonterminal reward is at most 6 + 6 + 10 = 22. With the MAPPO
    # discount .99, 3000 exceeds 22/(1-.99), so avoiding successful
    # termination cannot outperform success by collecting shaping forever.
    reward_success_scale: float = 3000.0
    use_preinsert_deficit_reward: bool = True

    def __post_init__(self):
        super().__post_init__()
        self.experimental_reset_fixed_objects_to_tcp = False
        self.experimental_preserve_authored_joint_frames_on_reset = False
        if (
            self.use_runtime_fixed_joints
            or self.fixed_objects_to_grippers
            or self.fixed_hole
            or self.fixed_peg_and_hole
        ):
            raise ValueError("Contact-grasp Circle requires two free objects, without welds or root pinning")
        if self.action_spaces != {"left_arm": 8, "right_arm": 8}:
            raise ValueError("Contact-grasp Circle requires eight actions per arm; welded checkpoints are incompatible")


class OpenArmContactCirclePegAndHoleEnv(OpenArmPegInHoleEnv):
    """Initialize a pregrasp once on reset, then let jaw contact support both parts."""

    cfg: OpenArmContactCirclePegAndHoleEnvCfg

    def __init__(self, cfg, render_mode=None, **kwargs):
        # Deliberately use the free-object base, not the fixed-object subclass
        # or its Newton builder hook that creates loop welds.
        super().__init__(cfg, render_mode=render_mode, **kwargs)
        self._reset_idx(None)
        self.sim.forward()
        self.scene.update(dt=self.physics_dt)

    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        joints = read_data(self.robot.data, "joint_pos")[env_ids].clone()
        velocities = read_data(self.robot.data, "joint_vel")[env_ids].clone()
        for side in ("left", "right"):
            ids = getattr(self, f"{side}_gripper_joint_ids")
            joints[:, ids] = self.cfg.pregrasp_finger_position
            velocities[:, ids] = 0.0
            getattr(self, f"_{side}_gripper_closed")[env_ids] = True
            getattr(self, f"_{side}_gripper_position_command")[env_ids] = self.cfg.pregrasp_finger_position
            set_joint_position(self.robot, joints[:, ids], joint_ids=ids, env_ids=env_ids)
        self.robot.write_joint_state_to_sim_index(position=joints, velocity=velocities, env_ids=env_ids)
        self._reset_grace_left[env_ids] = 0
        self.sim.forward()
        self.scene.update(dt=self.physics_dt)
        self._place_pregrasp_objects(env_ids)
        # Native step auto-resets before the evaluator reads terminal flags.
        # Preserve the completed transition like the base task does; the next
        # _get_dones call overwrites both masks before computing rewards.
        for name in ("termination_grasp_lost", "termination_drop"):
            self._ctx.setdefault(name, torch.zeros(self.num_envs, device=self.device, dtype=torch.bool))
        self.sim.forward()
        self.scene.update(dt=self.physics_dt)

    def _place_pregrasp_objects(self, env_ids):
        """Place free roots once at reset; no step/reward/observation caller exists."""
        state = read_data(self.robot.data, "body_state_w")
        # Reuse only the pure quaternion utility, never attachment methods.
        rotate_between = OpenArmFixedPegAndHoleEnv._quat_between_vectors
        left_outward = (
            state[env_ids, self._ctx["left_ee_body_id"], :3] - state[env_ids, self._ctx["left_wrist_body_id"], :3]
        )
        left_outward = torch.nn.functional.normalize(left_outward, dim=-1)
        for side, role in (("left", "peg"), ("right", "hole")):
            tcp = state[env_ids, self._ctx[f"{side}_ee_body_id"]]
            grip_position = self._ctx[f"{role}_grip_local_pos"][env_ids]
            grip_rotation = self._ctx[f"{role}_grip_local_quat"][env_ids]
            if role == "peg":
                axis = self._ctx["peg_tip_local_pos"][env_ids] - grip_position
            else:
                axis = self._ctx["hole_bottom_local_pos"][env_ids] - self._ctx["hole_entrance_local_pos"][env_ids]
            axis = torch.nn.functional.normalize(axis, dim=-1)
            nominal = quat_mul(tcp[:, 3:7], quat_inv(grip_rotation))
            rotation = quat_mul(rotate_between(quat_apply(nominal, axis), left_outward), nominal)
            position = tcp[:, :3] - quat_apply(rotation, grip_position)
            part = getattr(self, role)
            write_root_pose(part, torch.cat((position, rotation), dim=-1), env_ids=env_ids)
            part.write_root_velocity_to_sim_index(
                root_velocity=torch.zeros((len(env_ids), 6), device=self.device), env_ids=env_ids
            )

    def _pre_physics_step(self, actions):
        """Keep the configured target-error guard when adding gripper actions."""
        super()._pre_physics_step(actions)
        bound = float(self.cfg.max_joint_target_error_rad)
        if bound <= 0.0:
            return
        joints = read_data(self.robot.data, "joint_pos")
        for side in ("left", "right"):
            current = joints[:, getattr(self, f"{side}_arm_joint_ids")]
            target = getattr(self, f"_{side}_arm_incremental_target")
            default = getattr(self, f"{side}_arm_default_joint_pos")
            target.clamp_(min=current - bound, max=current + bound)
            target.clamp_(min=default - self.cfg.action_scale, max=default + self.cfg.action_scale)
            error = target - current
            self._ctx.setdefault("debug_stats", {}).update(
                {
                    f"{side}_joint_target_error_norm": error.norm(dim=-1).mean().item(),
                    f"{side}_joint_target_error_abs_max": error.abs().max().item(),
                }
            )

    def _get_dones(self):
        terminated, truncated = super()._get_dones()
        left, right = grasp_point_errors(self, self._ctx)
        lost = torch.maximum(left, right) > self.cfg.grasp_drop_distance
        self._ctx["termination_grasp_lost"] = lost
        # Use the existing evaluator's physical-drop category as well as the
        # task-specific flag, so a slipped grasp is not only labeled invalid.
        self._ctx["termination_drop"] = lost
        self._ctx["termination_invalid"] |= lost
        self._ctx.setdefault("debug_stats", {}).update(
            {
                "physical_grasp/left_error_m": left.mean().item(),
                "physical_grasp/right_error_m": right.mean().item(),
                "physical_grasp/lost_ratio": lost.float().mean().item(),
                "physical_grasp/peg_height": read_data(self.peg.data, "root_pos_w")[:, 2].mean().item(),
                "physical_grasp/hole_height": read_data(self.hole.data, "root_pos_w")[:, 2].mean().item(),
            }
        )
        return {agent: done | lost for agent, done in terminated.items()}, truncated

    def _get_rewards(self):
        rewards = super()._get_rewards()
        left, right = grasp_point_errors(self, self._ctx)
        retention = 0.5 * (
            torch.exp(-left / self.cfg.grasp_success_distance) * self._left_gripper_closed[:, 0]
            + torch.exp(-right / self.cfg.grasp_success_distance) * self._right_gripper_closed[:, 0]
        )
        lost = self._ctx.get("termination_grasp_lost", torch.zeros_like(left, dtype=torch.bool))
        extra = self.cfg.reward_grasp_retention_scale * retention - self.cfg.reward_grasp_drop_scale * lost.float()
        stats = self._ctx["debug_stats"]
        stats.update(
            {
                "physical_grasp/retention_reward": retention.mean().item(),
                "physical_grasp/drop_penalty": lost.float().mean().item(),
                "physical_grasp/friction_only": 1.0,
                "peg_hole/fixed_insert_obs": 0.0,
            }
        )
        for name in ("team_reward", "left_reward", "right_reward", "left_reward_raw", "right_reward_raw"):
            stats[name] += extra.mean().item()
        for side in ("left", "right"):
            opening = read_data(self.robot.data, "joint_pos")[:, getattr(self, f"{side}_gripper_joint_ids")].mean(-1)
            stats[f"{side}_closure"] = (1.0 - opening / self.cfg.gripper_open_target).clamp(0, 1).mean().item()
        self.extras["log"] = dict(stats)
        return {agent: value + extra for agent, value in rewards.items()}

    def get_paper_motion_context_trace_signals(self, env_index=0):
        """Record retained grasps independently of the old welded-task labels."""
        signals = super().get_paper_motion_context_trace_signals(env_index)
        index = max(0, min(int(env_index), self.num_envs - 1))
        left, right = grasp_point_errors(self, self._ctx)
        for side, error in (("left", left), ("right", right)):
            closed = getattr(self, f"_{side}_gripper_closed")[index, 0]
            signals[f"{side}_grasp_error_m"] = float(error[index])
            signals[f"{side}_gripper_command_closed"] = bool(closed)
            signals[f"{side}_grasp_ok"] = bool(closed & (error[index] <= self.cfg.grasp_success_distance))
            signals[f"{side}_prev_gripper_action"] = float(self.actions[f"{side}_arm"][index, 7])
        signals["dual_grasp_ok"] = bool(physical_grasp_validity(self, self._ctx)[index])
        return signals
