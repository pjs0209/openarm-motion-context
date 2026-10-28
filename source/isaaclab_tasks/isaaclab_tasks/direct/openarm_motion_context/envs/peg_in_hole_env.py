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
from isaaclab.sim import SimulationCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass
from isaaclab_newton.physics import NewtonCfg

from isaaclab_tasks.direct.openarm_motion_context.legacy_math import quat_apply, quat_inv, quat_mul

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
from .gripper_control import update_gripper_command
from .backend_cfg import OpenArmPhysicsCfg, configure_openarm_contact_capacity
from .newton_assets import spawn_openarm_scene
from isaaclab_tasks.direct.openarm_motion_context.legacy_api import (
    read_data,
    write_root_pose,
    set_joint_position,
    tensor,
)


def _peg_in_hole_environment_cfg(filename: str) -> AssetBaseCfg:
    """Return the shared scene asset configured with one authored geometry."""

    return AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Scene",
        spawn=UsdFileCfg(
            func=spawn_openarm_scene, usd_path=os.path.join(OPENARM_ENVIRONMENT_ASSET_DIR, "peg_in_hole", filename)
        ),
    )


def _peg_in_hole_rigid_object_cfg(
    prim_name: str,
    pos: tuple[float, float, float],
    rot: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
) -> RigidObjectCfg:
    """Bind an Isaac Lab rigid object to a rigid root authored in the scene USD."""

    return RigidObjectCfg(
        prim_path=f"{{ENV_REGEX_NS}}/Scene/Environment/{prim_name}",
        init_state=RigidObjectCfg.InitialStateCfg(pos=pos, rot=(rot[1], rot[2], rot[3], rot[0])),
        spawn=None,
    )


@configclass
class OpenArmPegInHoleSceneCfg(InteractiveSceneCfg):
    """Arrow peg-in-hole scene using USD-authored rigid bodies."""

    replicate_physics = True

    robot: ArticulationCfg = OPEN_ARM_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    robot.spawn.activate_contact_sensors = False

    environment = _peg_in_hole_environment_cfg("openarm_env_peg_in_hole.usd")
    peg = _peg_in_hole_rigid_object_cfg("arrow_peg", (0.3, -0.2, 0.275))
    hole = _peg_in_hole_rigid_object_cfg("arrow_hole", (0.3, 0.2, 0.275))


@configclass
class OpenArmCirclePegInHoleSceneCfg(OpenArmPegInHoleSceneCfg):
    """Circular peg-in-hole scene.

    The object USDs author the peg axis along local +Z and the hole insertion
    axis along local -Z. The fixed-task attachment code reads those semantic
    frames and rotates both axes toward robot-base +X at initialization.
    """

    environment = _peg_in_hole_environment_cfg("openarm_env_peg_in_hole_circle.usd")

    # circle_peg is authored with grap_point -> peg_tip along local +Z.
    peg = _peg_in_hole_rigid_object_cfg(
        "circle_peg",
        (0.3, -0.2, 0.275),
        (1.0, 0.0, 0.0, 0.0),
    )

    # circle_hole is authored with hole_entrance -> hole_bottom along local -Z.
    # +90 deg about X maps local -Z to base/world +Y at the authored reset pose.
    hole = _peg_in_hole_rigid_object_cfg(
        "circle_hole",
        (0.3, 0.2, 0.275),
        (0.70710678, 0.70710678, 0.0, 0.0),
    )


@configclass
class OpenArmHexagonPegInHoleSceneCfg(OpenArmPegInHoleSceneCfg):
    """Hexagonal peg-in-hole scene."""

    environment = _peg_in_hole_environment_cfg("openarm_env_peg_in_hole_hexagon.usd")
    peg = _peg_in_hole_rigid_object_cfg("hexagon_peg", (0.3, -0.2, 0.275))
    hole = _peg_in_hole_rigid_object_cfg("hexagon_hole", (0.3, 0.2, 0.275))


@configclass
class OpenArmPegInHoleEnvCfg(DirectMARLEnvCfg):
    """Config for the OpenArm bimanual peg-in-hole task."""

    decimation = 2
    # 1000 policy steps at dt=1/120 and decimation=2.
    episode_length_s = 1000.0 / 60.0

    possible_agents = ["left_arm", "right_arm"]
    action_spaces = {"left_arm": 8, "right_arm": 8}
    arm_only_actions = False

    # Proprioceptive Motion Context Sharing policy interface.
    # communication_mode:
    #   none                     -> own_obs 30D + zero slot 30D
    #   motion_only              -> own_obs 30D + [signed EE motion 3D, zeros 27D]
    #   context_only             -> own_obs 30D + [motion context 3D, zeros 27D]
    #   motion_context           -> own_obs 30D + [motion 3D, context 3D, zeros 24D]
    #   motion_context_la        -> own_obs 30D + [motion 3D, T_v, T_omega, 0, zeros 24D]
    #   motion_context_s         -> own_obs 30D + [motion 3D, 0, 0, T_S, zeros 24D]
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
    # These 3 slots are the directly computed proprioceptive motion context.
    motion_context_dim = 3
    intent_feature_dim = 30
    actor_partner_intent_dim = 30
    # Context channels are [T_v, T_omega, T_S]. They compare signal EMAs
    # against the same EMAs H control steps ago and require no running scale.
    motion_context_representation = "trend"
    motion_context_trend_horizon = 5
    motion_context_trend_beta = 0.80
    motion_context_trend_linear_epsilon = 0.01
    motion_context_trend_angular_epsilon = 0.05
    motion_context_trend_action_epsilon = 0.01
    motion_context_twist_ang_scale = 3.0  # rad/s; fixed before the comparative study
    motion_context_update_scale = False
    motion_context_calibration_enabled = False
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
        physics=OpenArmPhysicsCfg(),
    )
    scene: OpenArmPegInHoleSceneCfg = OpenArmPegInHoleSceneCfg(num_envs=64, env_spacing=3.0)

    # Action.
    # Arm actions are accumulated into joint-position targets.
    action_scale = 0.8
    action_step_scale = 0.005
    gripper_open_target = 0.04
    gripper_close_target = 0.0
    gripper_switch_epsilon = 0.05
    gripper_target_speed_limit = 0.0
    ee_vel_scale = 1.0
    use_gripper_close_prior = False
    reset_grace_steps = 20

    # USD-authored peg/hole frames. These must exist at runtime.
    peg_geometry = "arrow"
    rotational_symmetry_order = 1
    peg_tip_prim = "{ENV_REGEX_NS}/Scene/Environment/arrow_peg/peg_tip"
    peg_grip_prim = "{ENV_REGEX_NS}/Scene/Environment/arrow_peg/grap_point"
    hole_grip_prim = "{ENV_REGEX_NS}/Scene/Environment/arrow_hole/grap_point"
    hole_entrance_prim = "{ENV_REGEX_NS}/Scene/Environment/arrow_hole/hole_entrance"
    hole_bottom_prim = "{ENV_REGEX_NS}/Scene/Environment/arrow_hole/hole_bottom"

    # Staged peg-in-hole reward.
    num_keypoints = 11
    keypoint_spacing = 0.005
    preinsert_clearance = 0.03
    # The initial peg-tip distance is much larger than the insertion clearance.
    # A broad approach scale keeps the staged reward informative before contact.
    preinsert_std = 0.10
    insert_pose_std = 0.02
    reward_preinsert_scale = 2.0
    # Optional reward variant: subtract each gated preparation term's maximum
    # so entering the insertion gate does not discard an earned positive bonus.
    # False preserves every historical staged-reward checkpoint.
    use_preinsert_deficit_reward: bool = False
    # Optional, versioned Circle experiment. Zero preserves the v4 reward.
    reward_axis_alignment_scale = 0.0
    # Power 1 preserves v7; a higher power discourages tilted near-mouth equilibria.
    alignment_shaping_power = 1.0
    # Optional Circle refinement. Zero keeps earlier reward versions identical.
    reward_lateral_alignment_scale = 0.0
    lateral_alignment_std = 0.04
    reward_insert_pose_scale = 6.0
    reward_depth_scale = 10.0
    reward_success_scale = 30.0
    # Compatibility values for config serialization. Runtime reward and
    # success depths are read from hole_entrance -> hole_bottom in the USD.
    target_insertion_depth = 0.020
    reward_action_rate_scale = 1.0e-3

    # Reset.
    # Optional future Circle study: vary the reset pose to avoid repeated
    # deterministic scenarios across evaluation seeds. Zero preserves v4/v5.
    reset_arm_joint_noise_rad = 0.0
    fixed_hole = False
    fixed_peg_and_hole = False

    # Termination / success.
    success_lateral_threshold = 0.012
    success_axis_threshold = 0.97
    require_contact_grasp: bool = False
    success_depth_threshold = 0.018
    success_depth_ratio = 0.90
    # Zero preserves the historical hole-bottom objective. CAD-calibrated
    # tasks can limit travel by the protruding shaft and bound its envelope.
    insertion_depth_limit = 0.0
    success_radial_clearance = 0.0
    peg_collision_radius = 0.0
    success_depth_overrun_tolerance = 0.0
    success_protocol = "legacy_marker_depth_v1"
    wall_penetration_depth_threshold = 0.010
    wall_penetration_lateral_threshold = 0.025
    hold_required_steps = 45
    max_tip_distance = 0.60

    def __post_init__(self):
        """Synchronize derived dimensions and simulation render interval."""

        self.scene.replicate_physics = True
        self.sim.render_interval = self.decimation
        configure_openarm_contact_capacity(self.sim.physics, self.scene.num_envs)
        if not 0.0 <= self.gripper_target_speed_limit < float("inf"):
            raise ValueError("gripper_target_speed_limit must be finite and nonnegative")
        for name in (
            "insertion_depth_limit",
            "success_radial_clearance",
            "peg_collision_radius",
            "success_depth_overrun_tolerance",
        ):
            if not 0.0 <= getattr(self, name) < float("inf"):
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.success_radial_clearance > 0 and self.peg_collision_radius <= 0:
            raise ValueError("A shaft-clearance check requires a positive peg_collision_radius")

        valid_modes = {
            "none",
            "motion_only",
            "context_only",
            "motion_context",
            "motion_context_la",
            "motion_context_s",
            "previous_action",
            "full_partner_observation",
        }
        if self.communication_mode not in valid_modes:
            raise ValueError(
                f"Unsupported communication_mode={self.communication_mode!r}. Valid modes: {sorted(valid_modes)}"
            )
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
        # Finalize native clone capacities/reset invariants for direct gym
        # callers as well as the train/play entrypoints.
        cfg.__post_init__()
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

        self.left_arm_default_joint_pos = read_data(self.robot.data, "default_joint_pos")[
            :, self.left_arm_joint_ids
        ].clone()
        self.right_arm_default_joint_pos = read_data(self.robot.data, "default_joint_pos")[
            :, self.right_arm_joint_ids
        ].clone()

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

        self._left_gripper_position_command = self.left_gripper_open_command.clone()
        self._right_gripper_position_command = self.right_gripper_open_command.clone()
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
        self._left_arm_incremental_target = read_data(self.robot.data, "joint_pos")[:, self.left_arm_joint_ids].clone()
        self._right_arm_incremental_target = read_data(self.robot.data, "joint_pos")[
            :, self.right_arm_joint_ids
        ].clone()

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
        self.peg_authored_local_pos = (read_data(self.peg.data, "root_pos_w")[0, 0:3] - env_origin0).clone()
        self.peg_authored_local_rot = read_data(self.peg.data, "root_quat_w")[0].clone()
        self.hole_authored_local_pos = (read_data(self.hole.data, "root_pos_w")[0, 0:3] - env_origin0).clone()
        self.hole_authored_local_rot = read_data(self.hole.data, "root_quat_w")[0].clone()

        print("[INFO] OpenArm PegInHole motion-context task initialized")
        self.target_insertion_depth = float(self._ctx["hole_depth"][0].item())
        if self.cfg.insertion_depth_limit > 0:
            self.target_insertion_depth = min(self.target_insertion_depth, self.cfg.insertion_depth_limit)
        self.success_depth_threshold = self.target_insertion_depth * float(self.cfg.success_depth_ratio)
        print(
            f"[INFO] geometry={self.cfg.peg_geometry} "
            f"usd_target_depth={self.target_insertion_depth:.4f}m "
            f"success_depth={self.success_depth_threshold:.4f}m "
            f"rotational_symmetry={self.cfg.rotational_symmetry_order}"
        )
        print(f"[INFO] communication_mode={self.communication_mode}")
        print(f"[INFO] communication_sharing={self.sharing_mode}")
        print(f"[INFO] own_observation_dim={self.own_observation_dim}")
        print(f"[INFO] communication_feature_dim={self.communication_feature_dim}")
        print(f"[INFO] actor_input_dim={self.actor_input_dim}")
        print("[INFO] 6D motion-context message = base-frame signed EE motion 3D + proprioceptive context 3D")
        print(f"[INFO] motion_context_representation={self.cfg.motion_context_representation}")
        print("[INFO] reward=preinsert + gated_insert_pose + gated_depth + success - action_rate")
        print(
            "[INFO] keypoints: pre target=hole_entrance-front pose, insert target=hole_bottom pose "
            f"num={self.cfg.num_keypoints} spacing={self.cfg.keypoint_spacing:.4f}m "
            f"pre_clearance={self.cfg.preinsert_clearance:.4f}m target_depth={self.target_insertion_depth:.4f}m"
        )
        print("[INFO] contact/force/auxiliary-intent rewards are not used")
        print(
            "[INFO] incremental arm targets enabled "
            f"step_scale={self.cfg.action_step_scale} action_scale={self.cfg.action_scale}"
        )

    def _pin_authored_peg_hole_roots(
        self, env_ids: torch.Tensor | None = None, peg: bool = False, hole: bool = False
    ) -> None:
        """Keep selected peg/hole rigid roots at their authored USD poses."""

        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        origins = self.scene.env_origins[env_ids]

        if hole:
            hole_state = read_data(self.hole.data, "default_root_state")[env_ids].clone()
            hole_state[:, 0:3] = origins + self.hole_authored_local_pos.unsqueeze(0)
            hole_state[:, 3:7] = self.hole_authored_local_rot.unsqueeze(0)
            hole_state[:, 7:] = 0.0
            write_root_pose(self.hole, hole_state[:, :7], env_ids=env_ids)
            self.hole.write_root_velocity_to_sim_index(root_velocity=hole_state[:, 7:], env_ids=env_ids)

        if peg:
            peg_state = read_data(self.peg.data, "default_root_state")[env_ids].clone()
            peg_state[:, 0:3] = origins + self.peg_authored_local_pos.unsqueeze(0)
            peg_state[:, 3:7] = self.peg_authored_local_rot.unsqueeze(0)
            peg_state[:, 7:] = 0.0
            write_root_pose(self.peg, peg_state[:, :7], env_ids=env_ids)
            self.peg.write_root_velocity_to_sim_index(root_velocity=peg_state[:, 7:], env_ids=env_ids)

    def set_pending_intents(
        self,
        predicted_intents: dict[str, torch.Tensor],
        policy_states: dict[str, torch.Tensor],
        current_actions: dict[str, torch.Tensor] | None = None,
    ) -> None:
        """Compatibility hook: store current communication messages for the next step."""

        del predicted_intents, policy_states
        if self.communication_feature_dim == 0 or not getattr(self, "communication_enabled", True):
            self.communication_buffer["left_intent"].zero_()
            self.communication_buffer["right_intent"].zero_()
            return

        if current_actions is not None:
            # Match the action that _pre_physics_step will actually apply.
            self._communication_action_override = {
                agent: torch.clamp(action, -1.0, 1.0) for agent, action in current_actions.items()
            }
            self._communication_previous_action_override = self.actions
            self._motion_context_cache_step = -1
            self._motion_context_cache = None
        try:
            current_messages = self._current_communication_message_dict()
        finally:
            self._communication_action_override = None
            self._communication_previous_action_override = None
        cached_context = getattr(self, "_motion_context_cache", None)
        if isinstance(cached_context, dict):
            self._motion_context_last_generated = {agent: value.detach() for agent, value in cached_context.items()}
            self._motion_context_last_generated_debug = {
                key: value.detach() for key, value in getattr(self, "_motion_context_debug", {}).items()
            }
        for agent, key in (("left_arm", "left_intent"), ("right_arm", "right_intent")):
            message = (
                current_messages[agent]
                .detach()
                .reshape(
                    self.num_envs,
                    self.intent_horizon,
                    self.communication_feature_dim,
                )
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
        """Update incremental targets and gripper state once per control step."""

        self._prev_action.copy_(self.action_manager.action)
        raw_actions = {agent: torch.clamp(action, -1.0, 1.0) for agent, action in actions.items()}
        action_tensor = torch.cat([raw_actions["left_arm"], raw_actions["right_arm"]], dim=-1)
        self._smoothed_action.copy_(action_tensor)
        self.action_manager.action = action_tensor
        self.actions = raw_actions

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
        left_current_joint_pos = read_data(self.robot.data, "joint_pos")[:, self.left_arm_joint_ids]
        right_current_joint_pos = read_data(self.robot.data, "joint_pos")[:, self.right_arm_joint_ids]
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

        self._left_gripper_closed = torch.where(
            left_set_close, torch.ones_like(self._left_gripper_closed), self._left_gripper_closed
        )
        self._right_gripper_closed = torch.where(
            right_set_close, torch.ones_like(self._right_gripper_closed), self._right_gripper_closed
        )
        self._left_gripper_closed = torch.where(
            left_set_open, torch.zeros_like(self._left_gripper_closed), self._left_gripper_closed
        )
        self._right_gripper_closed = torch.where(
            right_set_open, torch.zeros_like(self._right_gripper_closed), self._right_gripper_closed
        )

        in_reset_grace = self._reset_grace_left > 0
        if in_reset_grace.any():
            self._left_gripper_closed[in_reset_grace] = False
            self._right_gripper_closed[in_reset_grace] = False
            self._reset_grace_left[in_reset_grace] -= 1

        max_delta = float(self.cfg.gripper_target_speed_limit) * self.step_dt
        update_gripper_command(
            self._left_gripper_position_command,
            self._left_gripper_closed,
            self.left_gripper_open_command,
            self.left_gripper_close_command,
            max_delta,
        )
        update_gripper_command(
            self._right_gripper_position_command,
            self._right_gripper_closed,
            self.right_gripper_open_command,
            self.right_gripper_close_command,
            max_delta,
        )

    def _apply_action(self) -> None:
        """Hold the control-step targets across every physics substep."""

        set_joint_position(self.robot, self._left_arm_incremental_target, joint_ids=self.left_arm_joint_ids)
        set_joint_position(self.robot, self._right_arm_incremental_target, joint_ids=self.right_arm_joint_ids)
        set_joint_position(self.robot, self._left_gripper_position_command, joint_ids=self.left_gripper_joint_ids)
        set_joint_position(self.robot, self._right_gripper_position_command, joint_ids=self.right_gripper_joint_ids)

        if bool(getattr(self.cfg, "fixed_peg_and_hole", False)):
            self._pin_authored_peg_hole_roots(peg=True, hole=True)
        elif bool(getattr(self.cfg, "fixed_hole", False)):
            self._pin_authored_peg_hole_roots(peg=False, hole=True)

    def _get_observations(self) -> dict[str, torch.Tensor]:
        """Return Paper motion-context-compatible actor observations."""

        # set_pending_intents generated the message before the physics step.
        # Recomputing it here would advance the EMA/history a second time.
        self.extras["communication_messages"] = {
            "left_arm": self.communication_buffer["left_intent"].reshape(self.num_envs, -1),
            "right_arm": self.communication_buffer["right_intent"].reshape(self.num_envs, -1),
        }
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

        default_root_state = read_data(self.robot.data, "default_root_state")[env_ids].clone()
        default_root_state[:, 0:3] += self.scene.env_origins[env_ids]
        write_root_pose(self.robot, default_root_state[:, :7], env_ids=env_ids)
        self.robot.write_root_velocity_to_sim_index(root_velocity=default_root_state[:, 7:], env_ids=env_ids)

        default_joint_pos = read_data(self.robot.data, "default_joint_pos")[env_ids].clone()
        default_joint_vel = read_data(self.robot.data, "default_joint_vel")[env_ids].clone()
        default_joint_pos[:, self.left_gripper_joint_ids] = float(self.cfg.gripper_open_target)
        reset_noise = float(getattr(self.cfg, "reset_arm_joint_noise_rad", 0.0))
        default_joint_pos[:, self.right_gripper_joint_ids] = float(self.cfg.gripper_open_target)
        if reset_noise > 0.0:
            arm_joint_ids = torch.as_tensor(
                [*self.left_arm_joint_ids, *self.right_arm_joint_ids], device=self.device, dtype=torch.long
            )
            limits = read_data(self.robot.data, "soft_joint_pos_limits")[env_ids][:, arm_joint_ids]
            offset = (2.0 * torch.rand_like(default_joint_pos[:, arm_joint_ids]) - 1.0) * reset_noise
            default_joint_pos[:, arm_joint_ids] = torch.clamp(
                default_joint_pos[:, arm_joint_ids] + offset, min=limits[..., 0], max=limits[..., 1]
            )
        default_joint_vel[:] = 0.0
        self.robot.write_joint_state_to_sim_index(
            position=default_joint_pos, velocity=default_joint_vel, env_ids=env_ids
        )

        set_joint_position(
            self.robot,
            default_joint_pos[:, self.left_arm_joint_ids],
            joint_ids=self.left_arm_joint_ids,
            env_ids=env_ids,
        )
        set_joint_position(
            self.robot,
            default_joint_pos[:, self.right_arm_joint_ids],
            joint_ids=self.right_arm_joint_ids,
            env_ids=env_ids,
        )
        set_joint_position(
            self.robot,
            default_joint_pos[:, self.left_gripper_joint_ids],
            joint_ids=self.left_gripper_joint_ids,
            env_ids=env_ids,
        )
        set_joint_position(
            self.robot,
            default_joint_pos[:, self.right_gripper_joint_ids],
            joint_ids=self.right_gripper_joint_ids,
            env_ids=env_ids,
        )

        if not bool(getattr(self.cfg, "use_runtime_fixed_joints", False)):
            self._pin_authored_peg_hole_roots(env_ids=env_ids, peg=True, hole=True)

        for agent in self.cfg.possible_agents:
            self.actions[agent][env_ids] = 0.0
        self.action_manager.action[env_ids] = 0.0
        self._prev_action[env_ids] = 0.0
        self._smoothed_action[env_ids] = 0.0
        self._left_arm_incremental_target[env_ids] = read_data(self.robot.data, "joint_pos")[env_ids][
            :, self.left_arm_joint_ids
        ]
        self._right_arm_incremental_target[env_ids] = read_data(self.robot.data, "joint_pos")[env_ids][
            :, self.right_arm_joint_ids
        ]
        self._left_gripper_closed[env_ids] = False
        self._right_gripper_closed[env_ids] = False
        self._left_gripper_position_command[env_ids] = float(self.cfg.gripper_open_target)
        self._right_gripper_position_command[env_ids] = float(self.cfg.gripper_open_target)
        self._reset_grace_left[env_ids] = int(self.cfg.reset_grace_steps)
        self.communication_buffer["left_intent"][env_ids] = 0.0
        self.communication_buffer["right_intent"][env_ids] = 0.0
        self.intent_buffer = self.communication_buffer
        self._motion_context_last_generated = None
        self._motion_context_last_generated_debug = None

        reset_openarm_re_context(self, self._ctx, env_ids)

    def reset(self, seed: int | None = None, options: dict | None = None):
        """Refresh extras log after reset."""

        observations, extras = super().reset(seed=seed, options=options)
        self.extras["log"] = dict(self._ctx.get("debug_stats", {}))
        return observations, extras


@configclass
class OpenArmFixedPegAndHoleEnvCfg(OpenArmPegInHoleEnvCfg):
    """Arrow peg-in-hole config with both objects attached to the grippers."""

    action_spaces = {"left_arm": 7, "right_arm": 7}
    arm_only_actions = True
    fixed_peg_and_hole = False
    fixed_hole = False
    fixed_objects_to_grippers = True
    reset_grace_steps = 0
    # All three authored geometries are 80 mm thick along the gripper opening
    # axis. Each OpenArm finger moves symmetrically, so q=40 mm gives an
    # 80 mm fingertip opening for both held objects.
    fixed_peg_gripper_open_target = 0.040
    fixed_hole_gripper_open_target = 0.040
    # Keep the authored grip frames on the TCPs with physical constraints.
    # Tensor pose synchronization is only used once, before joint creation,
    # and when restoring a reset pose; it must not override contact dynamics.
    use_runtime_fixed_joints = True
    # Optional anti-windup for policies that push into contact constraints.
    # Zero preserves the physical-joint v4/v5 action path.
    max_joint_target_error_rad = 0.0
    # Diagnostic only until the physical-joint reset is validated in replay.
    # Zero preserves all completed Circle studies.
    experimental_reset_fixed_objects_to_tcp = False
    # Diagnostic only: after the runtime joints exist, reset held-object poses
    # from their authored joint frames instead of recomputing new frame angles.
    # PhysX does not update articulation joint local rotations after startup.
    experimental_preserve_authored_joint_frames_on_reset = False

    # Each held part extends from its wrist toward its own EE TCP. Peg front is
    # the tip; hole front is the entrance face. This base-frame axis is only a
    # fallback if a wrist and its TCP are numerically coincident.
    fixed_insertion_axis_b = (1.0, 0.0, 0.0)
    intent_task_label = "openarm_peg_in_hole"

    def __post_init__(self):
        super().__post_init__()
        if isinstance(self.sim.physics, NewtonCfg):
            # Native welds retain immutable grasp frames. Reset the free object
            # roots consistently instead of requesting a different weld angle.
            self.experimental_reset_fixed_objects_to_tcp = True
            self.experimental_preserve_authored_joint_frames_on_reset = True


@configclass
class OpenArmFixedCirclePegAndHoleEnvCfg(OpenArmFixedPegAndHoleEnvCfg):
    """Circular peg-in-hole variant using its dedicated scene and semantic frames."""

    scene: OpenArmCirclePegInHoleSceneCfg = OpenArmCirclePegInHoleSceneCfg(num_envs=64, env_spacing=3.0)
    peg_geometry = "circle"

    use_runtime_fixed_joints = True

    # The Circle hole depth axis is aligned with the peg tip axis at reset.
    # Its mouth therefore faces the incoming peg, rather than the right wrist outward axis.
    rotational_symmetry_order = 0
    # A distant tip may project past the entrance plane without touching the
    # aperture wall. This Circle-only near-mouth gate prevents reset states
    # from being mislabeled as invalid on their first training step.
    wall_penetration_max_tip_distance = 0.06
    peg_tip_prim = "{ENV_REGEX_NS}/Scene/Environment/circle_peg/peg_tip"
    peg_grip_prim = "{ENV_REGEX_NS}/Scene/Environment/circle_peg/grap_point"
    hole_grip_prim = "{ENV_REGEX_NS}/Scene/Environment/circle_hole/grap_point"
    hole_entrance_prim = "{ENV_REGEX_NS}/Scene/Environment/circle_hole/hole_entrance"
    hole_bottom_prim = "{ENV_REGEX_NS}/Scene/Environment/circle_hole/hole_bottom"
    target_insertion_depth = 0.025
    success_depth_threshold = 0.0225
    intent_task_label = "openarm_peg_in_hole_circle"


@configclass
class OpenArmFixedCirclePegAndHoleV7EnvCfg(OpenArmFixedCirclePegAndHoleEnvCfg):
    """Replay-checked Circle v7 preset; keep the original Circle task unchanged.

    This fixes train/evaluation configuration drift, not learning-seed variance:
    seed 43 can still fail even though seed 42 succeeded. A separate task
    prevents accidental omission of the physical reset and reward overrides.
    """

    max_joint_target_error_rad = 0.2
    reward_axis_alignment_scale = 4.0
    reward_lateral_alignment_scale = 3.0
    lateral_alignment_std = 0.04
    reset_arm_joint_noise_rad = 0.03
    experimental_reset_fixed_objects_to_tcp = True


@configclass
class OpenArmFixedCirclePegAndHoleV8EnvCfg(OpenArmFixedCirclePegAndHoleV7EnvCfg):
    """Circle v8: change only axis-quality shaping in the preinsert/lateral reward."""

    alignment_shaping_power = 8.0


@configclass
class OpenArmFixedHexagonPegAndHoleEnvCfg(OpenArmFixedPegAndHoleEnvCfg):
    """Hexagonal peg-in-hole variant using its dedicated scene and semantic frames."""

    scene: OpenArmHexagonPegInHoleSceneCfg = OpenArmHexagonPegInHoleSceneCfg(num_envs=64, env_spacing=3.0)
    peg_geometry = "hexagon"
    rotational_symmetry_order = 6
    peg_tip_prim = "{ENV_REGEX_NS}/Scene/Environment/hexagon_peg/peg_tip"
    peg_grip_prim = "{ENV_REGEX_NS}/Scene/Environment/hexagon_peg/grap_point"
    hole_grip_prim = "{ENV_REGEX_NS}/Scene/Environment/hexagon_hole/grap_point"
    hole_entrance_prim = "{ENV_REGEX_NS}/Scene/Environment/hexagon_hole/hole_entrance"
    hole_bottom_prim = "{ENV_REGEX_NS}/Scene/Environment/hexagon_hole/hole_bottom"
    target_insertion_depth = 0.025
    success_depth_threshold = 0.0225
    intent_task_label = "openarm_peg_in_hole_hexagon"


class OpenArmFixedPegAndHoleEnv(OpenArmPegInHoleEnv):
    """OpenArm peg-in-hole task with peg/hole attached to left/right grippers."""

    cfg: OpenArmFixedPegAndHoleEnvCfg

    def __init__(self, cfg: OpenArmFixedPegAndHoleEnvCfg, render_mode: str | None = None, **kwargs):
        newton = isinstance(cfg.sim.physics, NewtonCfg)
        self._newton_grasp_frames = {}
        if newton and not cfg.use_runtime_fixed_joints:
            raise ValueError("Newton Peg requires physical grasp constraints; tensor-only attachment is not supported")
        v13_diagnostic = bool(getattr(cfg, "require_backside_hole_grasp", False))
        if v13_diagnostic:
            print("[PEG BACKSIDE INIT] before scene/physics initialization", flush=True)
        if newton:
            from functools import partial
            from isaaclab_newton.cloner import newton_builder_world_hook
            from .newton_peg import add_peg_welds

            with newton_builder_world_hook(partial(add_peg_welds, cfg=cfg, cache=self._newton_grasp_frames)):
                super().__init__(cfg, render_mode, **kwargs)
        else:
            super().__init__(cfg, render_mode, **kwargs)
        if v13_diagnostic:
            print("[PEG BACKSIDE INIT] scene/physics initialized", flush=True)
        self.actions = {
            agent: torch.zeros((self.num_envs, self.cfg.action_spaces[agent]), device=self.device)
            for agent in self.cfg.possible_agents
        }
        self.action_manager = SimpleNamespace(action=torch.zeros((self.num_envs, 14), device=self.device))
        self._prev_action = torch.zeros((self.num_envs, 14), device=self.device)
        self._smoothed_action = torch.zeros((self.num_envs, 14), device=self.device)
        expected_arm_actions = len(self.left_arm_joint_ids)
        if self.cfg.action_spaces["left_arm"] != expected_arm_actions:
            raise RuntimeError(
                "Fixed peg-in-hole must expose one policy action per arm joint: "
                f"actions={self.cfg.action_spaces['left_arm']} joints={expected_arm_actions}."
            )
        self._runtime_fixed_joints_created = newton
        if newton:
            if set(self._newton_grasp_frames) != {"left", "right"}:
                raise RuntimeError("Newton model was built without both physical grasp constraints")
            # Align the initial physical state to the same configured FK used
            # by the pre-finalization builder hook, before evaluating the grasp.
            env_ids = torch.arange(self.num_envs, device=self.device)
            OpenArmPegInHoleEnv._reset_idx(self, env_ids)
            self.sim.forward()
            self.scene.update(dt=self.physics_dt)
            for side in ("left", "right"):
                frame = self._newton_grasp_frames[side][0]
                quat = frame.ExtractRotation().GetQuat()
                rotation = torch.tensor([quat.GetReal(), *quat.GetImaginary()], device=self.device, dtype=torch.float32)
                position = torch.tensor(list(frame.ExtractTranslation()), device=self.device, dtype=torch.float32)
                setattr(self, f"_{side}_fixed_joint_local_quat", rotation.repeat(self.num_envs, 1))
                setattr(self, f"_{side}_fixed_joint_local_pos", position.repeat(self.num_envs, 1))
        if v13_diagnostic:
            print("[PEG BACKSIDE INIT] before object attachment", flush=True)
        self._attach_objects_to_grippers()
        if v13_diagnostic:
            print("[PEG BACKSIDE INIT] objects attached", flush=True)
        if bool(self.cfg.use_runtime_fixed_joints):
            if not newton:
                self._initialize_runtime_fixed_joints(v13_diagnostic)
            else:
                self.sim.forward()
            self.scene.update(dt=self.physics_dt)
            if v13_diagnostic:
                print("[PEG BACKSIDE INIT] post-joint physics reset complete", flush=True)

            self._validate_fixed_joint_initialization()

    def _initialize_runtime_fixed_joints(self, diagnostic: bool = False) -> None:
        """Author coincident attachment frames before the GUI rebuilds PhysX."""

        # Stopping invalidates the robot's tensor view, including body_names.
        body_names = tuple(self.robot.body_names)
        if self.sim.has_gui:
            # Stop before authoring so GUI transform restoration cannot undo
            # coincident frames. Native3 no longer has the 2.x app-exit guard.
            self.sim.stop()
        self._sync_attachment_poses_to_usd()
        self.sim.forward()
        self._create_runtime_fixed_joints(body_names=body_names)
        if diagnostic:
            print("[PEG BACKSIDE INIT] runtime joints authored", flush=True)
        self.sim.reset()

    def _validate_fixed_joint_initialization(self) -> None:
        """Check wrist-to-TCP mounting directions and physical grip frames."""

        metrics = compute_peg_in_hole_metrics(self, self._ctx)
        env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        left_outward_w = self._desired_outward_axis_w(env_ids, "left")
        right_outward_w = self._desired_outward_axis_w(env_ids, "right")

        # Peg front is +peg_axis. The historical Circle variants only checked
        # peg/hole axis agreement. Opt-in corrected grasps also require the
        # right wrist to remain behind the hole's closed side.
        peg_axis_to_target = (metrics["peg_axis_w"] * left_outward_w).sum(dim=-1)
        if self.cfg.peg_geometry == "circle" and bool(getattr(self.cfg, "require_backside_hole_grasp", False)):
            hole_axis_to_target = ((-metrics["hole_axis_w"]) * right_outward_w).sum(dim=-1)
        elif self.cfg.peg_geometry == "circle":
            hole_axis_to_target = metrics["axis_alignment"]
        else:
            hole_axis_to_target = ((-metrics["hole_axis_w"]) * right_outward_w).sum(dim=-1)

        left_center = self._gripper_center_w(env_ids, "left")
        right_center = self._gripper_center_w(env_ids, "right")
        peg_grip_error = torch.linalg.vector_norm(metrics["peg_grip_w"] - left_center, dim=-1)
        hole_grip_w = read_data(self.hole.data, "root_pos_w") + quat_apply(
            read_data(self.hole.data, "root_quat_w"), self._ctx["hole_grip_local_pos"]
        )
        hole_grip_error = torch.linalg.vector_norm(hole_grip_w - right_center, dim=-1)

        alignment_min = float(metrics["axis_alignment"].min().item())
        target_axis_min = float(torch.minimum(peg_axis_to_target, hole_axis_to_target).min().item())
        grip_error_max = float(torch.maximum(peg_grip_error, hole_grip_error).max().item())

        if bool(getattr(self.cfg, "require_backside_hole_grasp", False)):
            # The Circle mouth lies opposite its entrance->bottom depth axis.
            # A right wrist pointing *along* that depth axis holds the hole
            # through its opening even when peg/hole axes appear aligned.
            peg_to_mouth = metrics["hole_entrance_w"] - metrics["peg_tip_w"]
            entrance_clearance = (peg_to_mouth * metrics["peg_axis_w"]).sum(dim=-1)
            lateral_offset = torch.linalg.vector_norm(
                peg_to_mouth - entrance_clearance[:, None] * metrics["peg_axis_w"], dim=-1
            )
            backside_min = float(hole_axis_to_target.min().item())
            clearance_min = float(entrance_clearance.min().item())
            lateral_max = float(lateral_offset.max().item())
            if backside_min < 0.90 or clearance_min < 0.02 or lateral_max > 0.05:
                print(
                    "[DEBUG] Circle grasp init: "
                    f"left_q={read_data(self.robot.data, 'joint_pos')[0, self.left_arm_joint_ids].tolist()}, "
                    f"right_q={read_data(self.robot.data, 'joint_pos')[0, self.right_arm_joint_ids].tolist()}, "
                    f"left_outward={left_outward_w[0].tolist()}, "
                    f"right_outward={right_outward_w[0].tolist()}, "
                    f"peg_axis={metrics['peg_axis_w'][0].tolist()}, "
                    f"hole_axis={metrics['hole_axis_w'][0].tolist()}"
                )
                raise RuntimeError(
                    "Circle backside-grasp initialization failed: "
                    f"backside_dot_min={backside_min:.4f}, "
                    f"entrance_clearance_min={clearance_min:.4f}m, "
                    f"lateral_offset_max={lateral_max:.4f}m."
                )
            print(
                f"[INFO] Circle backside grasp: backside_dot_min={backside_min:.4f}, "
                f"entrance_clearance_min={clearance_min:.4f}m, "
                f"lateral_offset_max={lateral_max:.4f}m"
            )

        # Circle's physical fixed joints can settle several degrees across
        # 128 environments. A 0.90 initialization floor still keeps the smooth
        # insertion gate active; the stricter 0.97 success test is unchanged.
        initialization_axis_floor = 0.90 if self.cfg.peg_geometry == "circle" else 0.95
        if target_axis_min < initialization_axis_floor or grip_error_max > 0.002:
            raise RuntimeError(
                f"{self.cfg.peg_geometry} FixedJoint initialization is inconsistent: "
                f"axis_alignment_min={alignment_min:.4f}, "
                f"outward_alignment_min={target_axis_min:.4f}, "
                f"grip_error_max={grip_error_max:.5f}m."
            )

        print(
            f"[INFO] {self.cfg.peg_geometry} FixedJoints: "
            f"axis_alignment_min={alignment_min:.4f}, "
            f"outward_alignment_min={target_axis_min:.4f}, "
            f"grip_error_max={grip_error_max:.5f}m"
        )

    def _fixed_gripper_targets(self, env_ids: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
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

    @staticmethod
    def _quat_between_vectors(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Return wxyz quaternions rotating ``source`` onto ``target``.

        Both tensors are batched ``(N, 3)`` vectors.  The opposite-vector case
        is handled explicitly so the result stays finite during reset.
        """

        source = source / torch.linalg.vector_norm(source, dim=-1, keepdim=True).clamp_min(1.0e-8)
        target = target / torch.linalg.vector_norm(target, dim=-1, keepdim=True).clamp_min(1.0e-8)

        dot = torch.clamp((source * target).sum(dim=-1, keepdim=True), -1.0, 1.0)
        cross = torch.linalg.cross(source, target, dim=-1)
        quat = torch.cat([1.0 + dot, cross], dim=-1)

        opposite = dot.squeeze(-1) < -0.999999
        if opposite.any():
            src = source[opposite]
            x_axis = torch.tensor([1.0, 0.0, 0.0], device=source.device, dtype=source.dtype).expand_as(src)
            y_axis = torch.tensor([0.0, 1.0, 0.0], device=source.device, dtype=source.dtype).expand_as(src)
            axis = torch.linalg.cross(src, x_axis, dim=-1)
            use_y = torch.linalg.vector_norm(axis, dim=-1, keepdim=True) < 1.0e-6
            axis = torch.where(use_y, torch.linalg.cross(src, y_axis, dim=-1), axis)
            axis = axis / torch.linalg.vector_norm(axis, dim=-1, keepdim=True).clamp_min(1.0e-8)
            opposite_quat = torch.cat(
                [torch.zeros((axis.shape[0], 1), device=axis.device, dtype=axis.dtype), axis], dim=-1
            )
            quat[opposite] = opposite_quat

        return quat / torch.linalg.vector_norm(quat, dim=-1, keepdim=True).clamp_min(1.0e-8)

    def _desired_outward_axis_w(self, env_ids: torch.Tensor, side: str) -> torch.Tensor:
        """Return one arm's physical wrist-to-EE-TCP direction in world coordinates."""

        if side not in ("left", "right"):
            raise ValueError(f"Unsupported side: {side!r}")
        body_state = read_data(self.robot.data, "body_state_w")
        wrist_w = body_state[env_ids, self._ctx[f"{side}_wrist_body_id"], 0:3]
        tcp_w = body_state[env_ids, self._ctx[f"{side}_ee_body_id"], 0:3]
        wrist_to_tcp_w = tcp_w - wrist_w
        axis_norm = torch.linalg.vector_norm(wrist_to_tcp_w, dim=-1, keepdim=True)

        axis_b = (
            torch.as_tensor(
                getattr(self.cfg, "fixed_insertion_axis_b", (1.0, 0.0, 0.0)),
                device=self.device,
                dtype=torch.float32,
            )
            .reshape(1, 3)
            .expand(len(env_ids), -1)
        )
        axis_b = axis_b / torch.linalg.vector_norm(axis_b, dim=-1, keepdim=True).clamp_min(1.0e-8)
        robot_root_quat = read_data(self.robot.data, "root_quat_w")[env_ids]
        fallback_axis_w = quat_apply(robot_root_quat, axis_b)
        return torch.where(
            axis_norm > 1.0e-5,
            wrist_to_tcp_w / axis_norm.clamp_min(1.0e-8),
            fallback_axis_w,
        )

    def _gripper_center_w(self, env_ids: torch.Tensor, side: str) -> torch.Tensor:
        """Return the authored grasp center at the gripper TCP.

        Finger rigid-body origins sit near their joints, so averaging them
        moves a held object into the wrist. The robot asset's EE TCP is the
        intended center of the fingertip opening.
        """

        return read_data(self.robot.data, "body_state_w")[env_ids, self._ctx[f"{side}_ee_body_id"], :3]

    def _attach_objects_to_grippers(self, env_ids: torch.Tensor | None = None) -> None:
        """Synchronize the authored peg/hole grip frames with the OpenArm TCPs."""

        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)

        left_ee_pos = read_data(self.robot.data, "body_state_w")[env_ids, self._ctx["left_ee_body_id"], 0:3]
        left_ee_quat = read_data(self.robot.data, "body_state_w")[env_ids, self._ctx["left_ee_body_id"], 3:7]
        right_ee_pos = read_data(self.robot.data, "body_state_w")[env_ids, self._ctx["right_ee_body_id"], 0:3]
        right_ee_quat = read_data(self.robot.data, "body_state_w")[env_ids, self._ctx["right_ee_body_id"], 3:7]
        left_grip_center = self._gripper_center_w(env_ids, "left")
        right_grip_center = self._gripper_center_w(env_ids, "right")

        peg_grip_local_pos = self._ctx["peg_grip_local_pos"][env_ids]
        peg_grip_local_quat = self._ctx["peg_grip_local_quat"][env_ids]
        hole_grip_local_pos = self._ctx["hole_grip_local_pos"][env_ids]
        hole_grip_local_quat = self._ctx["hole_grip_local_quat"][env_ids]

        self._record_attachment_errors(env_ids)

        # Start from the natural "grip frame == TCP frame" attachment.
        peg_root_quat_nominal = quat_mul(left_ee_quat, quat_inv(peg_grip_local_quat))
        hole_root_quat_nominal = quat_mul(right_ee_quat, quat_inv(hole_grip_local_quat))

        # Rotate each object about its grip point only. The peg tip follows the
        # left wrist -> TCP direction. Since hole_axis is entrance -> bottom,
        # its negative follows the right wrist -> TCP direction.
        peg_axis_local = self._ctx["peg_tip_local_pos"][env_ids] - self._ctx["peg_grip_local_pos"][env_ids]
        peg_axis_local = peg_axis_local / torch.linalg.vector_norm(peg_axis_local, dim=-1, keepdim=True).clamp_min(
            1.0e-8
        )
        hole_axis_local = self._ctx["hole_bottom_local_pos"][env_ids] - self._ctx["hole_entrance_local_pos"][env_ids]
        hole_axis_local = hole_axis_local / torch.linalg.vector_norm(hole_axis_local, dim=-1, keepdim=True).clamp_min(
            1.0e-8
        )

        desired_peg_axis_w = self._desired_outward_axis_w(env_ids, "left")
        # The Circle wrists point in nearly the same world direction at reset.
        # Negating the right wrist made the authored insertion axis oppose the
        # peg (measured dot ~-0.997). Aim the hole depth along the peg instead;
        # this rotates the hole about its grip without moving the fixed joint.
        if self.cfg.peg_geometry == "circle":
            desired_hole_axis_w = desired_peg_axis_w
        else:
            desired_hole_axis_w = -self._desired_outward_axis_w(env_ids, "right")
        peg_axis_nominal_w = quat_apply(peg_root_quat_nominal, peg_axis_local)
        hole_axis_nominal_w = quat_apply(hole_root_quat_nominal, hole_axis_local)

        peg_delta_quat = self._quat_between_vectors(peg_axis_nominal_w, desired_peg_axis_w)
        hole_delta_quat = self._quat_between_vectors(hole_axis_nominal_w, desired_hole_axis_w)
        peg_root_quat = quat_mul(peg_delta_quat, peg_root_quat_nominal)
        hole_root_quat = quat_mul(hole_delta_quat, hole_root_quat_nominal)
        if self._runtime_fixed_joints_created and bool(self.cfg.experimental_preserve_authored_joint_frames_on_reset):
            # Keep the object-side authored grip frame coincident with the
            # immutable EE-side PhysX frame. Recomputing the desired axis on
            # reset can demand a different joint angle and make the hole snap
            # at the first physics step even if its grip position is correct.
            peg_root_quat = quat_mul(
                quat_mul(left_ee_quat, self._left_fixed_joint_local_quat[env_ids]),
                quat_inv(peg_grip_local_quat),
            )
            hole_root_quat = quat_mul(
                quat_mul(right_ee_quat, self._right_fixed_joint_local_quat[env_ids]),
                quat_inv(hole_grip_local_quat),
            )

        # Place each authored grasp point at the robot asset's grasp TCP.
        peg_root_pos = left_grip_center - quat_apply(peg_root_quat, peg_grip_local_pos)
        hole_root_pos = right_grip_center - quat_apply(hole_root_quat, hole_grip_local_pos)

        if not hasattr(self, "_peg_attachment_pose_w"):
            self._peg_attachment_pose_w = torch.zeros((self.num_envs, 7), device=self.device)
            self._hole_attachment_pose_w = torch.zeros((self.num_envs, 7), device=self.device)
        self._peg_attachment_pose_w[env_ids, 0:3] = peg_root_pos
        self._peg_attachment_pose_w[env_ids, 3:7] = peg_root_quat
        self._hole_attachment_pose_w[env_ids, 0:3] = hole_root_pos
        self._hole_attachment_pose_w[env_ids, 3:7] = hole_root_quat

        # The object-side joint frames are the authored grasp frames.  Since
        # aligning the insertion axis may rotate that frame relative to the TCP,
        # store the matching EE-side local orientation for FixedJoint creation.
        peg_grip_world_quat = quat_mul(peg_root_quat, peg_grip_local_quat)
        hole_grip_world_quat = quat_mul(hole_root_quat, hole_grip_local_quat)
        left_joint_local_quat = quat_mul(quat_inv(left_ee_quat), peg_grip_world_quat)
        right_joint_local_quat = quat_mul(quat_inv(right_ee_quat), hole_grip_world_quat)
        # FixedJoint frame 0 is exactly the EE TCP origin. Frame 1 is the
        # authored object grap_point (set in _create_runtime_fixed_joints).
        left_joint_local_pos = torch.zeros_like(left_ee_pos)
        right_joint_local_pos = torch.zeros_like(right_ee_pos)

        if not hasattr(self, "_left_fixed_joint_local_quat"):
            self._left_fixed_joint_local_quat = torch.zeros((self.num_envs, 4), device=self.device)
            self._right_fixed_joint_local_quat = torch.zeros((self.num_envs, 4), device=self.device)
        self._left_fixed_joint_local_quat[env_ids] = left_joint_local_quat
        self._right_fixed_joint_local_quat[env_ids] = right_joint_local_quat
        if not hasattr(self, "_left_fixed_joint_local_pos"):
            self._left_fixed_joint_local_pos = torch.zeros((self.num_envs, 3), device=self.device)
            self._right_fixed_joint_local_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self._left_fixed_joint_local_pos[env_ids] = left_joint_local_pos
        self._right_fixed_joint_local_pos[env_ids] = right_joint_local_pos

        left_ee_lin_vel = read_data(self.robot.data, "body_state_w")[env_ids, self._ctx["left_ee_body_id"], 7:10]
        left_ee_ang_vel = read_data(self.robot.data, "body_state_w")[env_ids, self._ctx["left_ee_body_id"], 10:13]
        peg_grip_offset_w = quat_apply(peg_root_quat, peg_grip_local_pos)
        peg_root_lin_vel = left_ee_lin_vel - torch.cross(left_ee_ang_vel, peg_grip_offset_w, dim=-1)

        peg_state = read_data(self.peg.data, "default_root_state")[env_ids].clone()
        peg_state[:, 0:3] = peg_root_pos
        peg_state[:, 3:7] = peg_root_quat
        peg_state[:, 7:10] = peg_root_lin_vel
        peg_state[:, 10:13] = left_ee_ang_vel
        write_root_pose(self.peg, peg_state[:, :7], env_ids=env_ids)
        self.peg.write_root_velocity_to_sim_index(root_velocity=peg_state[:, 7:], env_ids=env_ids)

        right_ee_lin_vel = read_data(self.robot.data, "body_state_w")[env_ids, self._ctx["right_ee_body_id"], 7:10]
        right_ee_ang_vel = read_data(self.robot.data, "body_state_w")[env_ids, self._ctx["right_ee_body_id"], 10:13]
        hole_grip_offset_w = quat_apply(hole_root_quat, hole_grip_local_pos)
        hole_root_lin_vel = right_ee_lin_vel - torch.cross(right_ee_ang_vel, hole_grip_offset_w, dim=-1)

        hole_state = read_data(self.hole.data, "default_root_state")[env_ids].clone()
        hole_state[:, 0:3] = hole_root_pos
        hole_state[:, 3:7] = hole_root_quat
        hole_state[:, 7:10] = hole_root_lin_vel
        hole_state[:, 10:13] = right_ee_ang_vel
        write_root_pose(self.hole, hole_state[:, :7], env_ids=env_ids)
        self.hole.write_root_velocity_to_sim_index(root_velocity=hole_state[:, 7:], env_ids=env_ids)

    def _record_attachment_errors(self, env_ids: torch.Tensor | None = None) -> None:
        """Record authored grip-frame displacement from the corresponding TCP."""

        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        left_ee_pos = self._gripper_center_w(env_ids, "left")
        right_ee_pos = self._gripper_center_w(env_ids, "right")
        peg_grip_pos = read_data(self.peg.data, "root_pos_w")[env_ids] + quat_apply(
            read_data(self.peg.data, "root_quat_w")[env_ids], self._ctx["peg_grip_local_pos"][env_ids]
        )
        hole_grip_pos = read_data(self.hole.data, "root_pos_w")[env_ids] + quat_apply(
            read_data(self.hole.data, "root_quat_w")[env_ids], self._ctx["hole_grip_local_pos"][env_ids]
        )
        peg_attach_error = torch.linalg.vector_norm(peg_grip_pos - left_ee_pos, dim=-1)
        hole_attach_error = torch.linalg.vector_norm(hole_grip_pos - right_ee_pos, dim=-1)
        self._ctx.setdefault("debug_stats", {}).update(
            {
                "peg_hole/peg_attach_error": peg_attach_error.mean().item(),
                "peg_hole/hole_attach_error": hole_attach_error.mean().item(),
                "peg_hole/peg_attach_error_max": peg_attach_error.max().item(),
                "peg_hole/hole_attach_error_max": hole_attach_error.max().item(),
            }
        )

    def _sync_attachment_poses_to_usd(self) -> None:
        """Mirror tensor attachment poses into USD before creating fixed joints."""

        from isaaclab.sim.utils import get_current_stage
        from pxr import Gf, UsdGeom  # type: ignore

        stage = get_current_stage()
        for env_id in range(self.num_envs):
            for root_template, pose_w in (
                (self.peg.cfg.prim_path, self._peg_attachment_pose_w[env_id]),
                (self.hole.cfg.prim_path, self._hole_attachment_pose_w[env_id]),
            ):
                prim_path = self._resolve_env_prim_path(root_template, env_id)
                prim = stage.GetPrimAtPath(prim_path)
                if not prim.IsValid():
                    raise RuntimeError(f"Attachment root prim not found while synchronizing USD: {prim_path}")

                pose_values = pose_w.detach().cpu().tolist()
                local_pos, local_quat = sim_utils.convert_world_pose_to_local(
                    tuple(float(value) for value in pose_values[0:3]),
                    tuple(float(value) for value in pose_values[3:7]),
                    prim.GetParent(),
                )
                if local_quat is None:
                    raise RuntimeError(f"Missing attachment orientation for {prim_path}.")
                xformable = UsdGeom.Xformable(prim)
                original_scale = Gf.Transform(xformable.GetLocalTransformation()).GetScale()
                transform = Gf.Transform()
                transform.SetTranslation(Gf.Vec3d(*local_pos))
                transform.SetRotation(Gf.Rotation(Gf.Quatd(*local_quat)))
                transform.SetScale(original_scale)
                # Some referenced assets already author xformOp:orient with a
                # different precision. A dedicated matrix op avoids trying to
                # redefine that attribute while preserving the root scale.
                matrix_op = UsdGeom.XformOp(prim.GetAttribute("xformOp:transform:fixedJointInit"))
                if not matrix_op:
                    matrix_op = xformable.AddTransformOp(UsdGeom.XformOp.PrecisionDouble, "fixedJointInit")
                matrix_op.Set(transform.GetMatrix())
                xformable.SetXformOpOrder([matrix_op])

    def _rigid_body_prim_path(self, env_id: int, root_path_template: str, body_name: str) -> str:
        """Resolve a rigid body prim path under an env root by body name."""

        from isaaclab.sim.utils import get_current_stage
        from pxr import UsdPhysics  # type: ignore

        env_ns = f"/World/envs/env_{env_id}"
        root_path = (
            str(root_path_template)
            .replace("{ENV_REGEX_NS}", env_ns)
            .replace("/World/envs/env_.*/", f"{env_ns}/")
            .replace("/World/envs/env_[^/]+/", f"{env_ns}/")
        )
        candidate = f"{root_path}/{body_name}"
        stage = get_current_stage()
        prim = stage.GetPrimAtPath(candidate)
        if prim.IsValid() and prim.HasAPI(UsdPhysics.RigidBodyAPI):
            return candidate

        root = stage.GetPrimAtPath(root_path)
        if not root.IsValid():
            raise RuntimeError(f"Runtime fixed joint root prim not found: {root_path}")
        for prim in stage.Traverse():
            path = prim.GetPath().pathString
            if (
                path.startswith(root_path + "/")
                and path.split("/")[-1] == body_name
                and prim.HasAPI(UsdPhysics.RigidBodyAPI)
            ):
                return path
        raise RuntimeError(f"Runtime fixed joint rigid body prim not found: root={root_path}, body={body_name}")

    @staticmethod
    def _resolve_env_prim_path(path_template: str, env_id: int) -> str:
        env_ns = f"/World/envs/env_{env_id}"
        return (
            str(path_template)
            .replace("{ENV_REGEX_NS}", env_ns)
            .replace("/World/envs/env_.*/", f"{env_ns}/")
            .replace("/World/envs/env_[^/]+/", f"{env_ns}/")
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

        from isaaclab.sim.utils import get_current_stage
        from pxr import Sdf, UsdPhysics  # type: ignore

        stage = get_current_stage()
        parent_path = "/".join(str(joint_path).split("/")[:-1])
        if parent_path:
            stage.DefinePrim(parent_path, "Xform")
        joint = UsdPhysics.FixedJoint.Define(stage, joint_path)
        # Author constraint frames before connecting bodies. PhysX processes
        # USD edits incrementally; connecting the bodies first briefly creates
        # an identity-frame joint and reports a false disjoint-transform snap.
        joint.CreateLocalPos0Attr(local_pos0)
        joint.CreateLocalRot0Attr(local_rot0)
        joint.CreateLocalPos1Attr(local_pos1)
        joint.CreateLocalRot1Attr(local_rot1)
        if hasattr(joint, "CreateCollisionEnabledAttr"):
            joint.CreateCollisionEnabledAttr(False)
        joint.CreateBody0Rel().SetTargets([Sdf.Path(body0_path)])
        joint.CreateBody1Rel().SetTargets([Sdf.Path(body1_path)])

    def _create_runtime_fixed_joints(self, body_names: Sequence[str] | None = None) -> None:
        """Create fixed joints from gripper TCP bodies to peg/hole rigid bodies."""

        if self._runtime_fixed_joints_created:
            return

        from pxr import Gf  # type: ignore

        if body_names is None:
            body_names = self.robot.body_names
        left_body_name = body_names[self._ctx["left_ee_body_id"]]
        right_body_name = body_names[self._ctx["right_ee_body_id"]]
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
            left_local_rot = self._gf_quat_from_tensor(self._left_fixed_joint_local_quat[env_id])
            right_local_rot = self._gf_quat_from_tensor(self._right_fixed_joint_local_quat[env_id])

            self._define_fixed_joint(
                f"{env_ns}/RuntimeFixedJoints/LeftPegFixedJoint",
                left_body_path,
                peg_body_path,
                self._gf_vec3_from_tensor(self._left_fixed_joint_local_pos[env_id]),
                left_local_rot,
                self._gf_vec3_from_tensor(peg_grip_local_pos),
                self._gf_quat_from_tensor(peg_grip_local_quat),
            )
            self._define_fixed_joint(
                f"{env_ns}/RuntimeFixedJoints/RightHoleFixedJoint",
                right_body_path,
                hole_body_path,
                self._gf_vec3_from_tensor(self._right_fixed_joint_local_pos[env_id]),
                right_local_rot,
                self._gf_vec3_from_tensor(hole_grip_local_pos),
                self._gf_quat_from_tensor(hole_grip_local_quat),
            )

        self._runtime_fixed_joints_created = True
        self.sim.forward()
        print("[INFO] Runtime FixedJoints created: left_ee_tcp <-> peg/grap_point, right_ee_tcp <-> hole/grap_point.")

    def _pre_physics_step(self, actions: dict[str, torch.Tensor]) -> None:
        """Update fixed-task arm targets once per control step."""

        self._prev_action.copy_(self.action_manager.action)
        raw_actions = {agent: torch.clamp(action, -1.0, 1.0) for agent, action in actions.items()}
        action_tensor = torch.cat([raw_actions["left_arm"], raw_actions["right_arm"]], dim=-1)
        self._smoothed_action.copy_(action_tensor)
        self.action_manager.action = action_tensor
        self.actions = raw_actions

        base_step_scale = float(self.cfg.action_step_scale)
        step_scale = torch.full((self.num_envs, 1), base_step_scale, device=self.device)
        debug = self._ctx.setdefault("debug_stats", {})
        try:
            metrics = compute_peg_in_hole_metrics(self, self._ctx)
            near_entrance = (
                (metrics["tip_dist"] < 0.03)
                & (metrics["lateral_error"] < 0.05)
                & (metrics["axis_alignment"] > 0.7)
                & (metrics["insertion_depth"] > -metrics["hole_depth"])
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
        left_current_joint_pos = read_data(self.robot.data, "joint_pos")[:, self.left_arm_joint_ids]
        right_current_joint_pos = read_data(self.robot.data, "joint_pos")[:, self.right_arm_joint_ids]
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
        max_target_error = float(self.cfg.max_joint_target_error_rad)
        if max_target_error > 0.0:
            self._left_arm_incremental_target = torch.clamp(
                self._left_arm_incremental_target,
                min=left_current_joint_pos - max_target_error,
                max=left_current_joint_pos + max_target_error,
            )
            self._right_arm_incremental_target = torch.clamp(
                self._right_arm_incremental_target,
                min=right_current_joint_pos - max_target_error,
                max=right_current_joint_pos + max_target_error,
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

    def _apply_action(self) -> None:
        """Hold the arm targets and fixed gripper openings during substeps."""

        self._left_gripper_closed[:] = True
        self._right_gripper_closed[:] = True
        left_gripper_target, right_gripper_target = self._fixed_gripper_targets()
        set_joint_position(self.robot, self._left_arm_incremental_target, joint_ids=self.left_arm_joint_ids)
        set_joint_position(self.robot, self._right_arm_incremental_target, joint_ids=self.right_arm_joint_ids)
        set_joint_position(self.robot, left_gripper_target, joint_ids=self.left_gripper_joint_ids)
        set_joint_position(self.robot, right_gripper_target, joint_ids=self.right_gripper_joint_ids)
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
        else:
            self._record_attachment_errors()
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

        joint_pos = read_data(self.robot.data, "joint_pos")[env_ids].clone()
        joint_vel = read_data(self.robot.data, "joint_vel")[env_ids].clone()
        left_gripper_target, right_gripper_target = self._fixed_gripper_targets(env_ids)
        joint_pos[:, self.left_gripper_joint_ids] = left_gripper_target
        joint_pos[:, self.right_gripper_joint_ids] = right_gripper_target
        joint_vel[:, self.left_gripper_joint_ids] = 0.0
        joint_vel[:, self.right_gripper_joint_ids] = 0.0
        self.robot.write_joint_state_to_sim_index(position=joint_pos, velocity=joint_vel, env_ids=env_ids)

        set_joint_position(self.robot, left_gripper_target, joint_ids=self.left_gripper_joint_ids, env_ids=env_ids)
        set_joint_position(self.robot, right_gripper_target, joint_ids=self.right_gripper_joint_ids, env_ids=env_ids)
        # After joint activation the objects are non-root articulation links;
        # direct root-pose writes are ignored by PhysX and can break reset.
        # Resetting the robot's joint state moves the attached bodies instead.
        if not bool(self.cfg.use_runtime_fixed_joints):
            self._attach_objects_to_grippers(env_ids)
        elif bool(self.cfg.experimental_reset_fixed_objects_to_tcp):
            # Refresh the robot TCP frames after the joint-state write, then
            # place both held bodies at their matching authored grip frames.
            self.sim.forward()
            self.scene.update(dt=self.physics_dt)
            self._attach_objects_to_grippers(env_ids)
            self.sim.forward()
