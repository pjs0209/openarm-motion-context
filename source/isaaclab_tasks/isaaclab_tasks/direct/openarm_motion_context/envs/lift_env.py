# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""OpenArmRE: a simple contact-free bimanual KLT lift DirectMARL baseline."""

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
from isaaclab.sim.spawners.materials.physics_materials_cfg import RigidBodyMaterialCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_from_euler_xyz, quat_mul

from .lift_task_logic import (
    collect_openarm_re_trace_signals,
    compute_openarm_re_motion_prediction,
    compute_openarm_re_observations,
    compute_openarm_re_rewards,
    compute_openarm_re_state,
    compute_openarm_re_terminations,
    ensure_openarm_re_context,
    reset_openarm_re_context,
)
from .robot_cfg import OPENARM_ASSET_DIR, OPEN_ARM_HIGH_PD_CFG


@configclass
class OpenArmReBimanualLiftSceneCfg(InteractiveSceneCfg):
    """Scene without contact sensors for the OpenArmRE simple baseline."""

    replicate_physics = True

    robot: ArticulationCfg = OPEN_ARM_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    robot.spawn.activate_contact_sensors = False

    environment = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Scene",
        spawn=UsdFileCfg(usd_path=os.path.join(OPENARM_ASSET_DIR, "openarm_env_box.usd")),
    )

    object = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Scene/Environment/small_KLT",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.24), rot=(1.0, 0.0, 0.0, 0.0)),
        spawn=None,
    )


@configclass
class OpenArmReBimanualLiftEnvCfg(DirectMARLEnvCfg):
    """Config for the contact-force-free OpenArm bimanual KLT lift baseline."""

    decimation = 2
    episode_length_s = 10.0

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
    # Backward-compatible Hydra aliases used by the shared train/play scripts.
    intent_variant = "share_intent"
    intent_arch = "shared_intent_encoder"
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
    intent_task_label = "openarm_re_simple_reward"
    observation_spaces = {"left_arm": 60, "right_arm": 60}
    state_space = 73

    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120,
        render_interval=decimation,
        physics_material=RigidBodyMaterialCfg(
            static_friction=8.0,
            dynamic_friction=6.0,
            restitution=0.0,
            friction_combine_mode="max",
        ),
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
    scene: OpenArmReBimanualLiftSceneCfg = OpenArmReBimanualLiftSceneCfg(num_envs=64, env_spacing=3.0)

    # Action. Arm actions are accumulated into joint-position targets.
    action_scale = 0.7
    action_step_scale = 0.04
    gripper_open_target = 0.04
    gripper_close_target = 0.0
    gripper_switch_epsilon = 0.02
    ee_vel_scale = 1.0
    use_gripper_close_prior = False
    reset_grace_steps = 20

    # Reset randomization for paper training/evaluation.
    enable_reset_randomization = True
    reset_randomize_xy_range = 0.03
    reset_randomize_yaw_deg = 10.0

    # USD-authored collision boxes used as grasp targets. These must exist at runtime;
    # no hard-coded fallback is used.
    actor_object_pose_source = "apriltag"
    apriltag_names = ("apriltag_00", "apriltag_01", "apriltag_02")
    apriltag_measurement_source = "camera_transform"
    apriltag_camera_prims = (
        "realsense_d435i_left/camera_link/camera_color_frame/camera_color_optical_frame",
        "realsense_d435i_right/camera_link/camera_color_frame/camera_color_optical_frame",
        "realsense_d435i/camera_link/camera_color_frame/camera_color_optical_frame",
    )
    # Camera optical poses are reconstructed as T_world_mount * T_mount_camera.
    # All three mount poses come from articulation FK.
    apriltag_camera_mount_prims = (
        "openarm_left_link7",
        "openarm_right_link7",
        "openarm_body_link",
    )
    # Main paper training uses clean virtual AprilTag measurements. Keep the
    # perturbation hooks available for later measured-perception robustness runs.
    apriltag_sim_position_noise_std = 0.0
    apriltag_sim_rotation_noise_deg = 0.0
    apriltag_sim_dropout_prob = 0.0
    apriltag_sim_latency_steps = 0
    left_collision_target_prim = "{ENV_REGEX_NS}/Scene/Environment/small_KLT/Collision/Cube_01"
    right_collision_target_prim = "{ENV_REGEX_NS}/Scene/Environment/small_KLT/Collision/Cube"
    collision_target_z_offset = 0.025
    collision_target_x_rot_offset_deg = 180.0

    # Reward.
    reward_reach_scale = 2.5
    reward_orientation_scale = 2.0
    reward_grasp_scale = 10.0
    reward_lift_scale = 48.0
    lift_target_height = 0.15
    reward_goal_scale = 5.0
    reward_stability_scale = 10.0

    # Tilt is folded into lift reward as lift * (1 - normalized_tilt_penalty)^2.
    tilt_free_deg = 5.0
    tilt_bad_deg = 20.0
    reward_action_rate_scale = 1.0e-3
    reward_joint_vel_scale = 3.0e-4

    #std parameters
    reach_std = 0.25
    orientation_std = 0.8
    goal_std = 0.35
    # Termination / success.
    success_height_margin = 0.08
    success_roll_pitch_threshold = 10.0
    success_yaw_threshold = 10.0
    hold_required_steps = 90
    fall_height_margin = 0.10
    max_target_distance = 1.0
    max_tilt_deg = 30.0

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
        if self.intent_variant not in ("no_intent", "share_intent"):
            raise ValueError(
                "OpenArmRE supports intent_variant='no_intent' or 'share_intent' as a CLI alias. "
                f"Got {self.intent_variant!r}."
            )
        if self.intent_variant == "no_intent":
            self.communication_mode = "none"
        if self.communication_mode not in valid_modes:
            raise ValueError(f"Unsupported communication_mode={self.communication_mode!r}. Valid modes: {sorted(valid_modes)}")
        self.intent_variant = "no_intent" if self.communication_mode == "none" else "share_intent"
        self.intent_arch = "none" if self.communication_mode == "none" else "shared_intent_encoder"
        if int(self.motion_intent_dim) != 3 or int(self.motion_context_dim) != 3:
            raise ValueError("OpenArmRE paper communication requires motion 3D + context 3D.")

        self.own_observation_dim = self.base_own_obs_dim + (
            self.gripper_extra_obs_dim if self.use_gripper_proprio_obs else 0
        )
        if self.own_observation_dim != 30:
            raise ValueError(f"OpenArmRE own_observation_dim must be 30, got {self.own_observation_dim}.")
        self.communication_feature_dim = int(self.own_observation_dim)
        # Compatibility aliases for shared scripts/checkpoint metadata.
        self.intent_feature_dim = self.communication_feature_dim
        self.actor_partner_intent_dim = self.communication_feature_dim
        self.actor_input_dim = self.own_observation_dim + self.intent_horizon * self.communication_feature_dim
        self.observation_spaces = {
            "left_arm": self.actor_input_dim,
            "right_arm": self.actor_input_dim,
        }
        self.state_space = 2 * self.own_observation_dim + 13


class OpenArmReBimanualLiftEnv(DirectMARLEnv):
    """A minimal OpenArm bimanual lift task with no contact-force reward shaping."""

    cfg: OpenArmReBimanualLiftEnvCfg

    def __init__(self, cfg: OpenArmReBimanualLiftEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.robot = self.scene.articulations["robot"]
        self.object = self.scene.rigid_objects["object"]

        self.intent_variant = str(self.cfg.intent_variant)
        self.intent_arch = str(self.cfg.intent_arch)
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
                "OpenArmRE paper communication dimensions are inconsistent: "
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
        self._left_arm_incremental_target = self.robot.data.joint_pos[:, self.left_arm_joint_ids].clone()
        self._right_arm_incremental_target = self.robot.data.joint_pos[:, self.right_arm_joint_ids].clone()

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

        self.communication_enabled = True
        self.intent_share_enabled = True
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
        self.object_authored_local_pos = (self.object.data.root_pos_w[0, 0:3] - env_origin0).clone()
        self.object_authored_local_rot = self.object.data.root_quat_w[0].clone()

        print("[INFO] OpenArmRE paper motion-context task initialized")
        print(f"[INFO] communication_mode={self.communication_mode}")
        print(f"[INFO] communication_sharing={self.sharing_mode}")
        print(f"[INFO] own_observation_dim={self.own_observation_dim}")
        print(f"[INFO] communication_feature_dim={self.communication_feature_dim}")
        print(f"[INFO] actor_input_dim={self.actor_input_dim}")
        print("[INFO] 6D motion-context message = base-frame signed EE motion 3D + proprioceptive context 3D")
        print("[INFO] motion context = linear_activity, angular_activity, action_smoothness")
        print("[INFO] reward=reach + orientation + grasp_hint + dual-gated tilt-aware_lift + center - action_rate - joint_vel")
        print("[INFO] contact/force/learned-mode/auxiliary-intent rewards are not used")

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

        from .lift_task_logic import compute_openarm_re_motion_prediction

        return compute_openarm_re_motion_prediction(self, self._ctx)

    def get_paper_motion_context_trace_signals(self, env_index: int = 0) -> dict:
        """Return per-step scalar/vector signals for Paper motion-context trace analysis."""

        return collect_openarm_re_trace_signals(self, self._ctx, env_index)

    def _current_intent_feature_dict(self) -> dict[str, torch.Tensor]:
        """Backward-compatible alias for the current communication message."""

        return self._current_communication_message_dict()

    def _current_communication_message_dict(self) -> dict[str, torch.Tensor]:
        """Expose z_i according to communication_mode."""

        from .lift_task_logic import compute_openarm_re_coordination_messages

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
        """Accumulate normalized arm actions into joint-position targets."""

        left_arm_action = self.actions["left_arm"][:, :7]
        right_arm_action = self.actions["right_arm"][:, :7]
        target_limit = float(self.cfg.action_scale)
        self._left_arm_incremental_target = torch.clamp(
            self._left_arm_incremental_target + float(self.cfg.action_step_scale) * left_arm_action,
            min=self.left_arm_default_joint_pos - target_limit,
            max=self.left_arm_default_joint_pos + target_limit,
        )
        self._right_arm_incremental_target = torch.clamp(
            self._right_arm_incremental_target + float(self.cfg.action_step_scale) * right_arm_action,
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
        debug = self._ctx.setdefault("debug_stats", {})
        debug.update(
            {
                "action_scale_mean": float(self.cfg.action_step_scale),
                "fine_control_ratio": 1.0,
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

        object_state = self.object.data.default_root_state[env_ids].clone()
        object_state[:, 0:3] = self.scene.env_origins[env_ids] + self.object_authored_local_pos.unsqueeze(0)
        object_state[:, 3:7] = self.object_authored_local_rot.unsqueeze(0)
        if bool(self.cfg.enable_reset_randomization):
            xy_range = float(self.cfg.reset_randomize_xy_range)
            yaw_range = torch.deg2rad(torch.tensor(float(self.cfg.reset_randomize_yaw_deg), device=self.device))
            object_state[:, 0:2] += torch.empty((env_ids.numel(), 2), device=self.device).uniform_(-xy_range, xy_range)
            yaw = torch.empty(env_ids.numel(), device=self.device).uniform_(-yaw_range.item(), yaw_range.item())
            zeros = torch.zeros_like(yaw)
            yaw_quat = quat_from_euler_xyz(zeros, zeros, yaw)
            object_state[:, 3:7] = quat_mul(yaw_quat, object_state[:, 3:7])
        object_state[:, 7:] = 0.0
        self.object.write_root_pose_to_sim(object_state[:, :7], env_ids=env_ids)
        self.object.write_root_velocity_to_sim(object_state[:, 7:], env_ids=env_ids)

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
class OpenArmReIncrementalBimanualLiftEnvCfg(OpenArmReBimanualLiftEnvCfg):
    """Config using accumulated joint-position targets instead of default-pose offsets."""

    intent_task_label = "openarm_re_simple_reward_incremental"
    action_scale = 0.7
    action_step_scale = 0.04


class OpenArmReIncrementalBimanualLiftEnv(OpenArmReBimanualLiftEnv):
    """OpenArmRE lift task with incremental arm joint targets."""

    cfg: OpenArmReIncrementalBimanualLiftEnvCfg

    def __init__(self, cfg: OpenArmReIncrementalBimanualLiftEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        print(
            "[INFO] incremental arm targets enabled "
            f"step_scale={self.cfg.action_step_scale} action_scale={self.cfg.action_scale}"
        )
