# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""OpenArm bimanual peg-in-hole DirectMARL task with motion-context sharing."""

from __future__ import annotations

from collections.abc import Sequence
import os
from types import SimpleNamespace

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import DirectMARLEnv, DirectMARLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply, quat_inv, quat_mul

from .peg_in_hole_task_logic import (
    collect_openarm_re_trace_signals,
    compute_peg_in_hole_metrics,
    compute_openarm_re_motion_prediction,
    compute_openarm_re_observations,
    compute_openarm_re_rewards,
    compute_openarm_re_state,
    compute_openarm_re_terminations,
    ensure_openarm_re_context,
    reset_openarm_re_context,
)
from .robot_cfg import OPENARM_ENVIRONMENT_ASSET_DIR, OPEN_ARM_HIGH_PD_CFG


@configclass
class OpenArmPegInHoleSceneCfg(InteractiveSceneCfg):
    """Peg-in-hole scene using USD-authored peg and hole rigid bodies."""

    replicate_physics = True

    robot: ArticulationCfg = OPEN_ARM_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    robot.spawn.activate_contact_sensors = False

    environment = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Scene",
        spawn=UsdFileCfg(
            usd_path=os.path.join(
                OPENARM_ENVIRONMENT_ASSET_DIR, "peg_in_hole", "openarm_env_peg_in_hole.usd"
            )
        ),
    )

    peg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Scene/Environment/peg_usd/peg",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.28), rot=(1.0, 0.0, 0.0, 0.0)),
        spawn=None,
    )

    hole = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Scene/Environment/hole_usd/hole",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.08, 0.0, 0.28), rot=(1.0, 0.0, 0.0, 0.0)),
        spawn=None,
    )


@configclass
class OpenArmPegInHoleEnvCfg(DirectMARLEnvCfg):
    """Config for the OpenArm bimanual peg-in-hole task."""

    decimation = 2
    # 1000 policy steps at dt=1/120 and decimation=2.
    episode_length_s = 1000.0 / 60.0

    possible_agents = ["left_arm", "right_arm"]
    action_spaces = {"left_arm": 8, "right_arm": 8}

    # Proprioceptive Motion Context Sharing policy interface.
    # communication_mode:
    #   none                     -> own_obs 30D + zero slot 30D
    #   motion_only              -> own_obs 30D + [signed EE motion 3D, zeros 27D]
    #   context_only             -> own_obs 30D + [motion context 3D, zeros 27D]
    #   motion_context           -> own_obs 30D + [motion 3D, context 3D, zeros 24D]
    #   previous_action          -> own_obs 30D + [previous partner action 8D, zeros 22D]
    #   full_partner_observation -> own_obs 30D + partner own_obs 30D
    communication_mode = "motion_context"
    communication_feature_dim = 30
    sharing_mode = "motion_context_share"
    intent_horizon = 1
    intent_stride = 1
    motion_intent_horizon = 15
    interaction_motion_scale = 0.05
    motion_intent_dim = 3
    # Compatibility name for the shared Paper motion-context agent. In the paper tasks
    # these 3 slots are direct motion context, not learned latent modes.
    motion_context_dim = 3
    intent_feature_dim = 30
    actor_partner_intent_dim = 30
    motion_context_lin_scale_init = 0.10
    motion_context_ang_scale_init = 0.50
    motion_context_action_scale_init = 0.50
    motion_context_scale_beta = 0.99
    motion_context_scale_percentile = 0.90
    motion_context_norm_max = 1.5
    motion_context_update_scale = True
    motion_context_freeze_after_steps = 10000
    base_own_obs_dim = 26
    gripper_extra_obs_dim = 4
    use_gripper_proprio_obs = True
    own_observation_dim = 30
    actor_input_dim = 60
    intent_task_label = "openarm_peg_in_hole"
    observation_spaces = {"left_arm": 60, "right_arm": 60}
    state_space = 86

    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120,
        render_interval=decimation,
        physx=PhysxCfg(
            bounce_threshold_velocity=0.2,
            enable_ccd=True,
            enable_external_forces_every_iteration=True,
            min_position_iteration_count=16,
            min_velocity_iteration_count=1,
            solve_articulation_contact_last=True,
            gpu_max_rigid_contact_count=2**22,
            gpu_max_rigid_patch_count=2**22,
            gpu_found_lost_aggregate_pairs_capacity=2**25,
            gpu_total_aggregate_pairs_capacity=2**23,
        ),
    )
    scene: OpenArmPegInHoleSceneCfg = OpenArmPegInHoleSceneCfg(num_envs=64, env_spacing=3.0)

    # Action.
    # Arm actions are accumulated into joint-position targets.
    action_scale = 0.8
    action_step_scale = 0.005
    gripper_open_target = 0.04
    gripper_close_target = 0.0
    gripper_switch_epsilon = 0.05
    ee_vel_scale = 1.0
    use_gripper_close_prior = False
    reset_grace_steps = 20

    # USD-authored peg/hole frames. These must exist at runtime.
    peg_tip_prim = "{ENV_REGEX_NS}/Scene/Environment/peg_usd/peg/peg_tip"
    peg_grip_prim = "{ENV_REGEX_NS}/Scene/Environment/peg_usd/peg/peg_grip_point"
    hole_grip_prim = "{ENV_REGEX_NS}/Scene/Environment/hole_usd/hole/hole_grip_point"
    hole_entrance_prim = "{ENV_REGEX_NS}/Scene/Environment/hole_usd/hole/hole_top"
    hole_bottom_prim = "{ENV_REGEX_NS}/Scene/Environment/hole_usd/hole/hole_bottom"

    # Staged peg-in-hole reward.
    num_keypoints = 11
    keypoint_spacing = 0.005
    preinsert_clearance = 0.03
    preinsert_std = 0.02
    insert_pose_std = 0.02
    reward_preinsert_scale = 2.0
    reward_insert_pose_scale = 6.0
    reward_depth_scale = 10.0
    reward_success_scale = 30.0
    target_insertion_depth = 0.05
    reward_action_rate_scale = 1.0e-3

    # Reset.
    fixed_hole = False
    fixed_peg_and_hole = False

    # Termination / success.
    success_lateral_threshold = 0.012
    success_axis_threshold = 0.97
    success_depth_threshold = 0.040
    wall_penetration_depth_threshold = 0.010
    wall_penetration_lateral_threshold = 0.025
    hold_required_steps = 45
    max_tip_distance = 0.60

    def __post_init__(self):
        """Synchronize derived dimensions and simulation render interval."""

        self.scene.replicate_physics = True
        self.sim.render_interval = self.decimation
        self.sim.physx.friction_correlation_distance = 0.00625

        valid_modes = {
            "none",
            "motion_only",
            "context_only",
            "motion_context",
            "previous_action",
            "full_partner_observation",
        }
        if self.communication_mode not in valid_modes:
            raise ValueError(f"Unsupported communication_mode={self.communication_mode!r}. Valid modes: {sorted(valid_modes)}")
        if int(self.motion_intent_dim) != 3 or int(self.motion_context_dim) != 3:
            raise ValueError("OpenArmPegInHole paper communication requires motion 3D + context 3D.")

        self.own_observation_dim = self.base_own_obs_dim + (
            self.gripper_extra_obs_dim if self.use_gripper_proprio_obs else 0
        )
        if self.own_observation_dim != 30:
            raise ValueError(f"OpenArmPegInHole own_observation_dim must be 30, got {self.own_observation_dim}.")
        self.communication_feature_dim = int(self.own_observation_dim)
        # Compatibility aliases for shared scripts/checkpoint metadata.
        self.intent_feature_dim = self.communication_feature_dim
        self.actor_partner_intent_dim = self.communication_feature_dim
        self.actor_input_dim = self.own_observation_dim + self.intent_horizon * self.communication_feature_dim
        self.observation_spaces = {
            "left_arm": self.actor_input_dim,
            "right_arm": self.actor_input_dim,
        }
        self.state_space = 2 * self.own_observation_dim + 26


class OpenArmPegInHoleEnv(DirectMARLEnv):
    """OpenArm bimanual peg-in-hole task preserving the Paper motion-context policy interface."""

    cfg: OpenArmPegInHoleEnvCfg

    def __init__(self, cfg: OpenArmPegInHoleEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.robot = self.scene.articulations["robot"]
        self.peg = self.scene.rigid_objects["peg"]
        self.hole = self.scene.rigid_objects["hole"]
        # Backward-compatible alias for shared Paper motion-context helper code.
        self.object = self.peg

        self.communication_mode = str(self.cfg.communication_mode)
        self.sharing_mode = str(self.cfg.sharing_mode)
        self.intent_horizon = int(self.cfg.intent_horizon)
        self.intent_stride = int(self.cfg.intent_stride)
        self.communication_feature_dim = int(self.cfg.communication_feature_dim)
        self.intent_feature_dim = int(self.cfg.intent_feature_dim)
        self.motion_intent_dim = int(self.cfg.motion_intent_dim)
        self.motion_context_dim = int(self.cfg.motion_context_dim)
        self.own_observation_dim = int(self.cfg.own_observation_dim)
        self.base_own_obs_dim = int(self.cfg.base_own_obs_dim)
        self.gripper_extra_obs_dim = int(self.cfg.gripper_extra_obs_dim)
        self.use_gripper_proprio_obs = bool(self.cfg.use_gripper_proprio_obs)
        self.actor_partner_intent_dim = int(self.cfg.actor_partner_intent_dim)
        self.actor_input_dim = int(self.cfg.actor_input_dim)
        self.intent_task_label = str(self.cfg.intent_task_label)
        self.motion_context_update_scale = bool(self.cfg.motion_context_update_scale)
        self.motion_context_freeze_after_steps = int(self.cfg.motion_context_freeze_after_steps)
        expected_actor_dim = self.own_observation_dim + self.intent_horizon * self.communication_feature_dim
        if (
            self.own_observation_dim != 30
            or self.intent_feature_dim != self.communication_feature_dim
            or self.actor_partner_intent_dim != self.communication_feature_dim
            or self.actor_input_dim != expected_actor_dim
        ):
            raise ValueError(
                "OpenArmPegInHole paper communication dimensions are inconsistent: "
                f"own={self.own_observation_dim}, communication={self.communication_feature_dim}, "
                f"intent_alias={self.intent_feature_dim}, actor={self.actor_input_dim}, "
                f"expected actor={expected_actor_dim}."
            )

        self._ctx = ensure_openarm_re_context(self)
        self.left_arm_joint_ids = self._ctx["left_arm_joint_ids"]
        self.right_arm_joint_ids = self._ctx["right_arm_joint_ids"]
        self.left_gripper_joint_ids = self._ctx["left_gripper_joint_ids"]
        self.right_gripper_joint_ids = self._ctx["right_gripper_joint_ids"]

        self.left_arm_default_joint_pos = self.robot.data.default_joint_pos[:, self.left_arm_joint_ids].clone()
        self.right_arm_default_joint_pos = self.robot.data.default_joint_pos[:, self.right_arm_joint_ids].clone()

        self.left_gripper_open_command = torch.full(
            (self.num_envs, len(self.left_gripper_joint_ids)),
            float(self.cfg.gripper_open_target),
            device=self.device,
        )
        self.right_gripper_open_command = torch.full(
            (self.num_envs, len(self.right_gripper_joint_ids)),
            float(self.cfg.gripper_open_target),
            device=self.device,
        )
        self.left_gripper_close_command = torch.full(
            (self.num_envs, len(self.left_gripper_joint_ids)),
            float(self.cfg.gripper_close_target),
            device=self.device,
        )
        self.right_gripper_close_command = torch.full(
            (self.num_envs, len(self.right_gripper_joint_ids)),
            float(self.cfg.gripper_close_target),
            device=self.device,
        )

        self._left_gripper_closed = torch.zeros((self.num_envs, 1), device=self.device, dtype=torch.bool)
        self._right_gripper_closed = torch.zeros((self.num_envs, 1), device=self.device, dtype=torch.bool)
        self._reset_grace_left = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)

        self.actions = {
            agent: torch.zeros((self.num_envs, self.cfg.action_spaces[agent]), device=self.device)
            for agent in self.cfg.possible_agents
        }
        self.action_manager = SimpleNamespace(action=torch.zeros((self.num_envs, 16), device=self.device))
        self._prev_action = torch.zeros((self.num_envs, 16), device=self.device)
        self._smoothed_action = torch.zeros((self.num_envs, 16), device=self.device)
        self._left_arm_incremental_target = self.robot.data.joint_pos[:, self.left_arm_joint_ids].clone()
        self._right_arm_incremental_target = self.robot.data.joint_pos[:, self.right_arm_joint_ids].clone()

        self.communication_enabled = True
        self.communication_buffer: dict[str, torch.Tensor] = {
            "left_intent": torch.zeros(
                (self.num_envs, self.intent_horizon, self.communication_feature_dim),
                device=self.device,
                dtype=torch.float32,
            ),
            "right_intent": torch.zeros(
                (self.num_envs, self.intent_horizon, self.communication_feature_dim),
                device=self.device,
                dtype=torch.float32,
            ),
        }
        self.intent_buffer = self.communication_buffer

        env_origin0 = self.scene.env_origins[0]
        self.peg_authored_local_pos = (self.peg.data.root_pos_w[0, 0:3] - env_origin0).clone()
        self.peg_authored_local_rot = self.peg.data.root_quat_w[0].clone()
        self.hole_authored_local_pos = (self.hole.data.root_pos_w[0, 0:3] - env_origin0).clone()
        self.hole_authored_local_rot = self.hole.data.root_quat_w[0].clone()

        print("[INFO] OpenArm PegInHole motion-context task initialized")
        print(f"[INFO] communication_mode={self.communication_mode}")
        print(f"[INFO] communication_sharing={self.sharing_mode}")
        print(f"[INFO] own_observation_dim={self.own_observation_dim}")
        print(f"[INFO] communication_feature_dim={self.communication_feature_dim}")
        print(f"[INFO] actor_input_dim={self.actor_input_dim}")
        print("[INFO] 6D motion-context message = base-frame signed EE motion 3D + proprioceptive context 3D")
        print("[INFO] motion context = linear_activity, angular_activity, action_smoothness")
        print("[INFO] reward=preinsert + gated_insert_pose + gated_depth + success - action_rate")
        print(
            "[INFO] keypoints: pre target=hole_entrance-front pose, insert target=hole_bottom pose "
            f"num={self.cfg.num_keypoints} spacing={self.cfg.keypoint_spacing:.4f}m "
            f"pre_clearance={self.cfg.preinsert_clearance:.4f}m target_depth={self.cfg.target_insertion_depth:.4f}m"
        )
        print("[INFO] contact/force/auxiliary-intent rewards are not used")
        print(
            "[INFO] incremental arm targets enabled "
            f"step_scale={self.cfg.action_step_scale} action_scale={self.cfg.action_scale}"
        )

    def _pin_authored_peg_hole_roots(self, env_ids: torch.Tensor | None = None, peg: bool = False, hole: bool = False) -> None:
        """Keep selected peg/hole rigid roots at their authored USD poses."""

        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        origins = self.scene.env_origins[env_ids]

        if hole:
            hole_state = self.hole.data.default_root_state[env_ids].clone()
            hole_state[:, 0:3] = origins + self.hole_authored_local_pos.unsqueeze(0)
            hole_state[:, 3:7] = self.hole_authored_local_rot.unsqueeze(0)
            hole_state[:, 7:] = 0.0
            self.hole.write_root_pose_to_sim(hole_state[:, :7], env_ids=env_ids)
            self.hole.write_root_velocity_to_sim(hole_state[:, 7:], env_ids=env_ids)

        if peg:
            peg_state = self.peg.data.default_root_state[env_ids].clone()
            peg_state[:, 0:3] = origins + self.peg_authored_local_pos.unsqueeze(0)
            peg_state[:, 3:7] = self.peg_authored_local_rot.unsqueeze(0)
            peg_state[:, 7:] = 0.0
            self.peg.write_root_pose_to_sim(peg_state[:, :7], env_ids=env_ids)
            self.peg.write_root_velocity_to_sim(peg_state[:, 7:], env_ids=env_ids)

    def set_pending_intents(
        self,
        predicted_intents: dict[str, torch.Tensor],
        policy_states: dict[str, torch.Tensor],
    ) -> None:
        """Compatibility hook: store current communication messages for the next step."""

        del predicted_intents, policy_states
        if self.communication_feature_dim == 0 or not getattr(self, "communication_enabled", True):
            self.communication_buffer["left_intent"].zero_()
            self.communication_buffer["right_intent"].zero_()
            return

        current_messages = self._current_communication_message_dict()
        for agent, key in (("left_arm", "left_intent"), ("right_arm", "right_intent")):
            message = current_messages[agent].detach().reshape(
                self.num_envs,
                self.intent_horizon,
                self.communication_feature_dim,
            )
            self.communication_buffer[key] = torch.nan_to_num(
                message,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            ).to(device=self.device, dtype=torch.float32)
        self.intent_buffer = self.communication_buffer

    def get_current_motion_intents(self) -> dict[str, torch.Tensor]:
        """Return deterministic base-frame signed EE motion prediction for each arm."""

        from .peg_in_hole_task_logic import compute_openarm_re_motion_prediction

        return compute_openarm_re_motion_prediction(self, self._ctx)

    def get_paper_motion_context_trace_signals(self, env_index: int = 0) -> dict:
        """Return per-step scalar/vector signals for Paper motion-context trace analysis."""

        return collect_openarm_re_trace_signals(self, self._ctx, env_index)

    def _current_intent_feature_dict(self) -> dict[str, torch.Tensor]:
        """Backward-compatible alias for the current communication message."""

        return self._current_communication_message_dict()

    def _current_communication_message_dict(self) -> dict[str, torch.Tensor]:
        """Expose z_i according to communication_mode."""

        from .peg_in_hole_task_logic import compute_openarm_re_coordination_messages

        return compute_openarm_re_coordination_messages(self, self._ctx)

    def _setup_scene(self):
        """Create only a dome light; robot/object are declared in the scene cfg."""

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: dict[str, torch.Tensor]) -> None:
        """Clamp raw policy actions without smoothing."""

        self._prev_action.copy_(self.action_manager.action)
        raw_actions = {agent: torch.clamp(action, -1.0, 1.0) for agent, action in actions.items()}
        action_tensor = torch.cat([raw_actions["left_arm"], raw_actions["right_arm"]], dim=-1)
        self._smoothed_action.copy_(action_tensor)
        self.action_manager.action = action_tensor
        self.actions = raw_actions

    def _apply_action(self) -> None:
        """Accumulate arm joint targets and convert signed gripper commands to targets."""

        step_scale = float(self.cfg.action_step_scale)
        debug = self._ctx.setdefault("debug_stats", {})
        debug["action_scale_mean"] = step_scale
        debug["fine_control_ratio"] = 1.0

        left_arm_action = self.actions["left_arm"][:, :7]
        right_arm_action = self.actions["right_arm"][:, :7]
        target_limit = float(self.cfg.action_scale)
        self._left_arm_incremental_target = torch.clamp(
            self._left_arm_incremental_target + step_scale * left_arm_action,
            min=self.left_arm_default_joint_pos - target_limit,
            max=self.left_arm_default_joint_pos + target_limit,
        )
        self._right_arm_incremental_target = torch.clamp(
            self._right_arm_incremental_target + step_scale * right_arm_action,
            min=self.right_arm_default_joint_pos - target_limit,
            max=self.right_arm_default_joint_pos + target_limit,
        )
        left_arm_target = self._left_arm_incremental_target
        right_arm_target = self._right_arm_incremental_target
        left_gripper_action = self.actions["left_arm"][:, 7]
        right_gripper_action = self.actions["right_arm"][:, 7]
        left_current_joint_pos = self.robot.data.joint_pos[:, self.left_arm_joint_ids]
        right_current_joint_pos = self.robot.data.joint_pos[:, self.right_arm_joint_ids]
        left_target_error = left_arm_target - left_current_joint_pos
        right_target_error = right_arm_target - right_current_joint_pos
        debug.update(
            {
                "left_action_abs_max": left_arm_action.abs().amax().item(),
                "right_action_abs_max": right_arm_action.abs().amax().item(),
                "left_action_mean_abs": left_arm_action.abs().mean().item(),
                "right_action_mean_abs": right_arm_action.abs().mean().item(),
                "left_action_saturation_ratio": (left_arm_action.abs() > 0.95).float().mean().item(),
                "right_action_saturation_ratio": (right_arm_action.abs() > 0.95).float().mean().item(),
                "left_gripper_action_mean": left_gripper_action.mean().item(),
                "right_gripper_action_mean": right_gripper_action.mean().item(),
                "left_gripper_action_abs_mean": left_gripper_action.abs().mean().item(),
                "right_gripper_action_abs_mean": right_gripper_action.abs().mean().item(),
                "left_joint_target_error_norm": torch.linalg.vector_norm(left_target_error, dim=-1).mean().item(),
                "right_joint_target_error_norm": torch.linalg.vector_norm(right_target_error, dim=-1).mean().item(),
                "left_joint_target_error_abs_max": left_target_error.abs().amax().item(),
                "right_joint_target_error_abs_max": right_target_error.abs().amax().item(),
            }
        )

        eps = float(self.cfg.gripper_switch_epsilon)
        left_set_close = self.actions["left_arm"][:, 7:8] < -eps
        right_set_close = self.actions["right_arm"][:, 7:8] < -eps
        left_set_open = self.actions["left_arm"][:, 7:8] > eps
        right_set_open = self.actions["right_arm"][:, 7:8] > eps

        self._left_gripper_closed = torch.where(left_set_close, torch.ones_like(self._left_gripper_closed), self._left_gripper_closed)
        self._right_gripper_closed = torch.where(right_set_close, torch.ones_like(self._right_gripper_closed), self._right_gripper_closed)
        self._left_gripper_closed = torch.where(left_set_open, torch.zeros_like(self._left_gripper_closed), self._left_gripper_closed)
        self._right_gripper_closed = torch.where(right_set_open, torch.zeros_like(self._right_gripper_closed), self._right_gripper_closed)

        in_reset_grace = self._reset_grace_left > 0
        if in_reset_grace.any():
            self._left_gripper_closed[in_reset_grace] = False
            self._right_gripper_closed[in_reset_grace] = False
            self._reset_grace_left[in_reset_grace] -= 1

        left_gripper_target = torch.where(
            self._left_gripper_closed,
            self.left_gripper_close_command,
            self.left_gripper_open_command,
        )
        right_gripper_target = torch.where(
            self._right_gripper_closed,
            self.right_gripper_close_command,
            self.right_gripper_open_command,
        )

        self.robot.set_joint_position_target(left_arm_target, joint_ids=self.left_arm_joint_ids)
        self.robot.set_joint_position_target(right_arm_target, joint_ids=self.right_arm_joint_ids)
        self.robot.set_joint_position_target(left_gripper_target, joint_ids=self.left_gripper_joint_ids)
        self.robot.set_joint_position_target(right_gripper_target, joint_ids=self.right_gripper_joint_ids)

        if bool(getattr(self.cfg, "fixed_peg_and_hole", False)):
            self._pin_authored_peg_hole_roots(peg=True, hole=True)
        elif bool(getattr(self.cfg, "fixed_hole", False)):
            self._pin_authored_peg_hole_roots(peg=False, hole=True)

    def _get_observations(self) -> dict[str, torch.Tensor]:
        """Return Paper motion-context-compatible actor observations."""

        self.extras["communication_messages"] = self._current_communication_message_dict()
        self.extras["intent_features"] = self.extras["communication_messages"]
        return compute_openarm_re_observations(self, self._ctx)

    def _get_states(self) -> torch.Tensor:
        """Return centralized critic state without contact/force signals."""

        return compute_openarm_re_state(self, self._ctx)

    def _get_rewards(self) -> dict[str, torch.Tensor]:
        """Return shared cooperative simple reward."""

        rewards = compute_openarm_re_rewards(self, self._ctx)
        self.extras["log"] = dict(self._ctx.get("debug_stats", {}))
        return rewards

    def _get_dones(self) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """Return task termination and timeout dictionaries."""

        terminated_tensor, _, _ = compute_openarm_re_terminations(self, self._ctx)
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return (
            {agent: terminated_tensor for agent in self.cfg.possible_agents},
            {agent: time_out for agent in self.cfg.possible_agents},
        )

    def _reset_idx(self, env_ids: Sequence[int] | torch.Tensor | None):
        """Reset robot first, force grippers open, then place the object at authored pose."""

        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        if not isinstance(env_ids, torch.Tensor):
            env_ids = torch.tensor(env_ids, device=self.device, dtype=torch.long)

        super()._reset_idx(env_ids)

        default_root_state = self.robot.data.default_root_state[env_ids].clone()
        default_root_state[:, 0:3] += self.scene.env_origins[env_ids]
        self.robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids=env_ids)
        self.robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids=env_ids)

        default_joint_pos = self.robot.data.default_joint_pos[env_ids].clone()
        default_joint_vel = self.robot.data.default_joint_vel[env_ids].clone()
        default_joint_pos[:, self.left_gripper_joint_ids] = float(self.cfg.gripper_open_target)
        default_joint_pos[:, self.right_gripper_joint_ids] = float(self.cfg.gripper_open_target)
        default_joint_vel[:] = 0.0
        self.robot.write_joint_state_to_sim(default_joint_pos, default_joint_vel, env_ids=env_ids)

        self.robot.set_joint_position_target(
            default_joint_pos[:, self.left_arm_joint_ids],
            joint_ids=self.left_arm_joint_ids,
            env_ids=env_ids,
        )
        self.robot.set_joint_position_target(
            default_joint_pos[:, self.right_arm_joint_ids],
            joint_ids=self.right_arm_joint_ids,
            env_ids=env_ids,
        )
        self.robot.set_joint_position_target(
            default_joint_pos[:, self.left_gripper_joint_ids],
            joint_ids=self.left_gripper_joint_ids,
            env_ids=env_ids,
        )
        self.robot.set_joint_position_target(
            default_joint_pos[:, self.right_gripper_joint_ids],
            joint_ids=self.right_gripper_joint_ids,
            env_ids=env_ids,
        )

        self._pin_authored_peg_hole_roots(env_ids=env_ids, peg=True, hole=True)

        for agent in self.cfg.possible_agents:
            self.actions[agent][env_ids] = 0.0
        self.action_manager.action[env_ids] = 0.0
        self._prev_action[env_ids] = 0.0
        self._smoothed_action[env_ids] = 0.0
        self._left_arm_incremental_target[env_ids] = self.robot.data.joint_pos[env_ids][:, self.left_arm_joint_ids]
        self._right_arm_incremental_target[env_ids] = self.robot.data.joint_pos[env_ids][:, self.right_arm_joint_ids]
        self._left_gripper_closed[env_ids] = False
        self._right_gripper_closed[env_ids] = False
        self._reset_grace_left[env_ids] = int(self.cfg.reset_grace_steps)
        self.communication_buffer["left_intent"][env_ids] = 0.0
        self.communication_buffer["right_intent"][env_ids] = 0.0
        self.intent_buffer = self.communication_buffer

        reset_openarm_re_context(self, self._ctx, env_ids)

    def reset(self, seed: int | None = None, options: dict | None = None):
        """Refresh extras log after reset."""

        observations, extras = super().reset(seed=seed, options=options)
        self.extras["log"] = dict(self._ctx.get("debug_stats", {}))
        return observations, extras


@configclass
class OpenArmFixedPegAndHoleEnvCfg(OpenArmPegInHoleEnvCfg):
    """Peg-in-hole config with peg and hole kinematically attached to the grippers."""

    action_spaces = {"left_arm": 7, "right_arm": 7}
    fixed_peg_and_hole = False
    fixed_hole = False
    fixed_objects_to_grippers = True
    reset_grace_steps = 0
    fixed_peg_gripper_open_target = 0.004
    fixed_hole_gripper_open_target = 0.010
    use_runtime_fixed_joints = True
    intent_task_label = "openarm_peg_in_hole"


class OpenArmFixedPegAndHoleEnv(OpenArmPegInHoleEnv):
    """OpenArm peg-in-hole task with peg/hole attached to left/right grippers."""

    cfg: OpenArmFixedPegAndHoleEnvCfg

    def __init__(self, cfg: OpenArmFixedPegAndHoleEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self.actions = {
            agent: torch.zeros((self.num_envs, self.cfg.action_spaces[agent]), device=self.device)
            for agent in self.cfg.possible_agents
        }
        self.action_manager = SimpleNamespace(action=torch.zeros((self.num_envs, 14), device=self.device))
        self._prev_action = torch.zeros((self.num_envs, 14), device=self.device)
        self._smoothed_action = torch.zeros((self.num_envs, 14), device=self.device)
        self._runtime_fixed_joints_created = False
        self._attach_objects_to_grippers()
        if bool(self.cfg.use_runtime_fixed_joints):
            self._create_runtime_fixed_joints()

    def _fixed_gripper_targets(
        self, env_ids: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return fixed visual/hold gripper targets: left peg, right hole."""

        if env_ids is None:
            left_shape = (self.num_envs, len(self.left_gripper_joint_ids))
            right_shape = (self.num_envs, len(self.right_gripper_joint_ids))
        else:
            left_shape = (len(env_ids), len(self.left_gripper_joint_ids))
            right_shape = (len(env_ids), len(self.right_gripper_joint_ids))
        left_target = torch.full(left_shape, float(self.cfg.fixed_peg_gripper_open_target), device=self.device)
        right_target = torch.full(right_shape, float(self.cfg.fixed_hole_gripper_open_target), device=self.device)
        return left_target, right_target

    def _attach_objects_to_grippers(self, env_ids: torch.Tensor | None = None) -> None:
        """Kinematically attach peg/hole grip frames to the left/right EEs."""

        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)

        left_ee_pos = self.robot.data.body_state_w[env_ids, self._ctx["left_ee_body_id"], 0:3]
        left_ee_quat = self.robot.data.body_state_w[env_ids, self._ctx["left_ee_body_id"], 3:7]
        right_ee_pos = self.robot.data.body_state_w[env_ids, self._ctx["right_ee_body_id"], 0:3]
        right_ee_quat = self.robot.data.body_state_w[env_ids, self._ctx["right_ee_body_id"], 3:7]

        peg_grip_local_pos = self._ctx["peg_grip_local_pos"][env_ids]
        peg_grip_local_quat = self._ctx["peg_grip_local_quat"][env_ids]
        peg_root_quat = quat_mul(left_ee_quat, quat_inv(peg_grip_local_quat))
        peg_root_pos = left_ee_pos - quat_apply(peg_root_quat, peg_grip_local_pos)

        hole_grip_local_pos = self._ctx["hole_grip_local_pos"][env_ids]
        hole_grip_local_quat = self._ctx["hole_grip_local_quat"][env_ids]
        hole_root_quat = quat_mul(right_ee_quat, quat_inv(hole_grip_local_quat))
        hole_root_pos = right_ee_pos - quat_apply(hole_root_quat, hole_grip_local_pos)

        peg_state = self.peg.data.default_root_state[env_ids].clone()
        peg_state[:, 0:3] = peg_root_pos
        peg_state[:, 3:7] = peg_root_quat
        peg_state[:, 7:] = 0.0
        self.peg.write_root_pose_to_sim(peg_state[:, :7], env_ids=env_ids)
        self.peg.write_root_velocity_to_sim(peg_state[:, 7:], env_ids=env_ids)

        hole_state = self.hole.data.default_root_state[env_ids].clone()
        hole_state[:, 0:3] = hole_root_pos
        hole_state[:, 3:7] = hole_root_quat
        hole_state[:, 7:] = 0.0
        self.hole.write_root_pose_to_sim(hole_state[:, :7], env_ids=env_ids)
        self.hole.write_root_velocity_to_sim(hole_state[:, 7:], env_ids=env_ids)

    def _rigid_body_prim_path(self, env_id: int, root_path_template: str, body_name: str) -> str:
        """Resolve a rigid body prim path under an env root by body name."""

        import omni.usd  # type: ignore
        from pxr import UsdPhysics  # type: ignore

        env_ns = f"/World/envs/env_{env_id}"
        root_path = (
            str(root_path_template)
            .replace("{ENV_REGEX_NS}", env_ns)
            .replace("/World/envs/env_.*/", f"{env_ns}/")
        )
        candidate = f"{root_path}/{body_name}"
        stage = omni.usd.get_context().get_stage()
        prim = stage.GetPrimAtPath(candidate)
        if prim.IsValid() and prim.HasAPI(UsdPhysics.RigidBodyAPI):
            return candidate

        root = stage.GetPrimAtPath(root_path)
        if not root.IsValid():
            raise RuntimeError(f"Runtime fixed joint root prim not found: {root_path}")
        for prim in stage.Traverse():
            path = prim.GetPath().pathString
            if path.startswith(root_path + "/") and path.split("/")[-1] == body_name and prim.HasAPI(UsdPhysics.RigidBodyAPI):
                return path
        raise RuntimeError(f"Runtime fixed joint rigid body prim not found: root={root_path}, body={body_name}")

    @staticmethod
    def _resolve_env_prim_path(path_template: str, env_id: int) -> str:
        env_ns = f"/World/envs/env_{env_id}"
        return (
            str(path_template)
            .replace("{ENV_REGEX_NS}", env_ns)
            .replace("/World/envs/env_.*/", f"{env_ns}/")
        )

    @staticmethod
    def _gf_vec3_from_tensor(vec: torch.Tensor):
        from pxr import Gf  # type: ignore

        values = vec.detach().cpu().tolist()
        return Gf.Vec3f(float(values[0]), float(values[1]), float(values[2]))

    @staticmethod
    def _gf_quat_from_tensor(quat_wxyz: torch.Tensor):
        from pxr import Gf  # type: ignore

        values = quat_wxyz.detach().cpu().tolist()
        return Gf.Quatf(float(values[0]), Gf.Vec3f(float(values[1]), float(values[2]), float(values[3])))

    def _define_fixed_joint(
        self,
        joint_path: str,
        body0_path: str,
        body1_path: str,
        local_pos0,
        local_rot0,
        local_pos1,
        local_rot1,
    ) -> None:
        """Define or update one USD Physics fixed joint."""

        import omni.usd  # type: ignore
        from pxr import Sdf, UsdPhysics  # type: ignore

        stage = omni.usd.get_context().get_stage()
        parent_path = "/".join(str(joint_path).split("/")[:-1])
        if parent_path:
            stage.DefinePrim(parent_path, "Xform")
        joint = UsdPhysics.FixedJoint.Define(stage, joint_path)
        joint.CreateBody0Rel().SetTargets([Sdf.Path(body0_path)])
        joint.CreateBody1Rel().SetTargets([Sdf.Path(body1_path)])
        joint.CreateLocalPos0Attr(local_pos0)
        joint.CreateLocalRot0Attr(local_rot0)
        joint.CreateLocalPos1Attr(local_pos1)
        joint.CreateLocalRot1Attr(local_rot1)
        if hasattr(joint, "CreateCollisionEnabledAttr"):
            joint.CreateCollisionEnabledAttr(False)

    def _create_runtime_fixed_joints(self) -> None:
        """Create fixed joints from gripper TCP bodies to peg/hole rigid bodies."""

        if self._runtime_fixed_joints_created:
            return

        from pxr import Gf  # type: ignore

        left_body_name = self.robot.body_names[self._ctx["left_ee_body_id"]]
        right_body_name = self.robot.body_names[self._ctx["right_ee_body_id"]]
        identity_pos = Gf.Vec3f(0.0, 0.0, 0.0)
        identity_rot = Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0))
        robot_root_template = self.robot.cfg.prim_path
        peg_root_template = self.peg.cfg.prim_path
        hole_root_template = self.hole.cfg.prim_path

        for env_id in range(self.num_envs):
            env_ns = f"/World/envs/env_{env_id}"
            left_body_path = self._rigid_body_prim_path(env_id, robot_root_template, left_body_name)
            right_body_path = self._rigid_body_prim_path(env_id, robot_root_template, right_body_name)
            peg_body_path = self._resolve_env_prim_path(peg_root_template, env_id)
            hole_body_path = self._resolve_env_prim_path(hole_root_template, env_id)

            peg_grip_local_pos = self._ctx["peg_grip_local_pos"][env_id]
            peg_grip_local_quat = self._ctx["peg_grip_local_quat"][env_id]
            hole_grip_local_pos = self._ctx["hole_grip_local_pos"][env_id]
            hole_grip_local_quat = self._ctx["hole_grip_local_quat"][env_id]
            self._define_fixed_joint(
                f"{env_ns}/RuntimeFixedJoints/LeftPegFixedJoint",
                left_body_path,
                peg_body_path,
                identity_pos,
                identity_rot,
                self._gf_vec3_from_tensor(peg_grip_local_pos),
                self._gf_quat_from_tensor(peg_grip_local_quat),
            )
            self._define_fixed_joint(
                f"{env_ns}/RuntimeFixedJoints/RightHoleFixedJoint",
                right_body_path,
                hole_body_path,
                identity_pos,
                identity_rot,
                self._gf_vec3_from_tensor(hole_grip_local_pos),
                self._gf_quat_from_tensor(hole_grip_local_quat),
            )

        self._runtime_fixed_joints_created = True
        print("[INFO] Runtime fixed joints created for fixed peg-in-hole objects.")

    def _pre_physics_step(self, actions: dict[str, torch.Tensor]) -> None:
        """Clamp fixed-task arm-only actions."""

        self._prev_action.copy_(self.action_manager.action)
        raw_actions = {agent: torch.clamp(action, -1.0, 1.0) for agent, action in actions.items()}
        action_tensor = torch.cat([raw_actions["left_arm"], raw_actions["right_arm"]], dim=-1)
        self._smoothed_action.copy_(action_tensor)
        self.action_manager.action = action_tensor
        self.actions = raw_actions

    def _apply_action(self) -> None:
        """Apply 7D arm actions while holding fixed gripper openings."""

        base_step_scale = float(self.cfg.action_step_scale)
        step_scale = torch.full((self.num_envs, 1), base_step_scale, device=self.device)
        debug = self._ctx.setdefault("debug_stats", {})
        try:
            metrics = compute_peg_in_hole_metrics(self, self._ctx)
            near_entrance = (
                (metrics["tip_dist"] < 0.03)
                & (metrics["lateral_error"] < 0.05)
                & (metrics["axis_alignment"] > 0.7)
                & (metrics["insertion_depth"] > -float(self.cfg.target_insertion_depth))
            )
            fine_step_scale = 0.002
            step_scale = torch.where(
                near_entrance.unsqueeze(-1),
                torch.full_like(step_scale, fine_step_scale),
                step_scale,
            )
            debug["fine_control_ratio"] = near_entrance.float().mean().item()
        except Exception:
            debug["fine_control_ratio"] = 0.0
        debug["action_scale_mean"] = step_scale.mean().item()

        left_arm_action = self.actions["left_arm"]
        right_arm_action = self.actions["right_arm"]
        target_limit = float(self.cfg.action_scale)
        left_current_joint_pos = self.robot.data.joint_pos[:, self.left_arm_joint_ids]
        right_current_joint_pos = self.robot.data.joint_pos[:, self.right_arm_joint_ids]
        self._left_arm_incremental_target = torch.clamp(
            self._left_arm_incremental_target + step_scale * left_arm_action,
            min=self.left_arm_default_joint_pos - target_limit,
            max=self.left_arm_default_joint_pos + target_limit,
        )
        self._right_arm_incremental_target = torch.clamp(
            self._right_arm_incremental_target + step_scale * right_arm_action,
            min=self.right_arm_default_joint_pos - target_limit,
            max=self.right_arm_default_joint_pos + target_limit,
        )
        left_target_error = self._left_arm_incremental_target - left_current_joint_pos
        right_target_error = self._right_arm_incremental_target - right_current_joint_pos
        debug.update(
            {
                "left_action_abs_max": left_arm_action.abs().amax().item(),
                "right_action_abs_max": right_arm_action.abs().amax().item(),
                "left_action_mean_abs": left_arm_action.abs().mean().item(),
                "right_action_mean_abs": right_arm_action.abs().mean().item(),
                "left_action_saturation_ratio": (left_arm_action.abs() > 0.95).float().mean().item(),
                "right_action_saturation_ratio": (right_arm_action.abs() > 0.95).float().mean().item(),
                "left_gripper_action_mean": 0.0,
                "right_gripper_action_mean": 0.0,
                "left_gripper_action_abs_mean": 0.0,
                "right_gripper_action_abs_mean": 0.0,
                "left_joint_target_error_norm": torch.linalg.vector_norm(left_target_error, dim=-1).mean().item(),
                "right_joint_target_error_norm": torch.linalg.vector_norm(right_target_error, dim=-1).mean().item(),
                "left_joint_target_error_abs_max": left_target_error.abs().amax().item(),
                "right_joint_target_error_abs_max": right_target_error.abs().amax().item(),
            }
        )

        self._left_gripper_closed[:] = True
        self._right_gripper_closed[:] = True
        left_gripper_target, right_gripper_target = self._fixed_gripper_targets()
        self.robot.set_joint_position_target(self._left_arm_incremental_target, joint_ids=self.left_arm_joint_ids)
        self.robot.set_joint_position_target(self._right_arm_incremental_target, joint_ids=self.right_arm_joint_ids)
        self.robot.set_joint_position_target(left_gripper_target, joint_ids=self.left_gripper_joint_ids)
        self.robot.set_joint_position_target(right_gripper_target, joint_ids=self.right_gripper_joint_ids)
        if not bool(self.cfg.use_runtime_fixed_joints):
            self._attach_objects_to_grippers()

    def _get_observations(self) -> dict[str, torch.Tensor]:
        if not bool(self.cfg.use_runtime_fixed_joints):
            self._attach_objects_to_grippers()
        return super()._get_observations()

    def _get_states(self) -> torch.Tensor:
        if not bool(self.cfg.use_runtime_fixed_joints):
            self._attach_objects_to_grippers()
        return super()._get_states()

    def _get_rewards(self) -> dict[str, torch.Tensor]:
        if not bool(self.cfg.use_runtime_fixed_joints):
            self._attach_objects_to_grippers()
        return super()._get_rewards()

    def _get_dones(self) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        if not bool(self.cfg.use_runtime_fixed_joints):
            self._attach_objects_to_grippers()
        return super()._get_dones()

    def _reset_idx(self, env_ids: Sequence[int] | torch.Tensor | None):
        super()._reset_idx(env_ids)
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        if not isinstance(env_ids, torch.Tensor):
            env_ids = torch.tensor(env_ids, device=self.device, dtype=torch.long)
        self._left_gripper_closed[env_ids] = True
        self._right_gripper_closed[env_ids] = True
        self._reset_grace_left[env_ids] = 0

        joint_pos = self.robot.data.joint_pos[env_ids].clone()
        joint_vel = self.robot.data.joint_vel[env_ids].clone()
        left_gripper_target, right_gripper_target = self._fixed_gripper_targets(env_ids)
        joint_pos[:, self.left_gripper_joint_ids] = left_gripper_target
        joint_pos[:, self.right_gripper_joint_ids] = right_gripper_target
        joint_vel[:, self.left_gripper_joint_ids] = 0.0
        joint_vel[:, self.right_gripper_joint_ids] = 0.0
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

        self.robot.set_joint_position_target(
            left_gripper_target,
            joint_ids=self.left_gripper_joint_ids,
            env_ids=env_ids,
        )
        self.robot.set_joint_position_target(
            right_gripper_target,
            joint_ids=self.right_gripper_joint_ids,
            env_ids=env_ids,
        )
        self._attach_objects_to_grippers(env_ids)
