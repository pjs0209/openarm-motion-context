"""Matched Lift plus the authored 2.102 kg ball and reward-only centering.

Existing 30D Lift observations, controls, base rewards, and success/dones are
retained. Four rendered-camera channels add ball position and visibility. Ball
simulator pose is used only for reward/diagnostics, never as a camera fallback.
"""

import math
import os

import torch

from isaaclab.assets import RigidObjectCfg
from isaaclab.sensors import TiledCameraCfg
from isaaclab_physx.renderers import IsaacRtxRendererCfg
import isaaclab.sim as sim_utils
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass
from isaaclab_tasks.direct.openarm_motion_context.legacy_math import quat_apply, quat_inv, quat_mul

from .ball_center import ball_center_metrics, ball_offcenter_penalty
from .ball_lift_reward import ball_lift_bootstrap, ball_height_upright_bridge, bilateral_closure_gate
from .ball_camera import estimate_ball_camera_position, ball_semantic_mask
from .lift_task_logic import get_actor_object_pose_w, _closure_from_opening
from .lift_env import (
    OpenArmReBimanualLiftSceneCfg,
    OpenArmReIncrementalBimanualLiftEnv,
    OpenArmReIncrementalBimanualLiftEnvCfg,
)
from .robot_cfg import OPENARM_ENVIRONMENT_ASSET_DIR
from .newton_assets import spawn_openarm_scene
from isaaclab_tasks.direct.openarm_motion_context.legacy_api import (
    read_data,
    write_root_pose,
    set_joint_position,
    tensor,
)


@configclass
class OpenArmLiftWithBallSceneCfg(OpenArmReBimanualLiftSceneCfg):
    environment = OpenArmReBimanualLiftSceneCfg().environment.replace(
        spawn=UsdFileCfg(
            func=spawn_openarm_scene,
            usd_path=os.path.join(OPENARM_ENVIRONMENT_ASSET_DIR, "lift", "openarm_env_box_ball_matched.usda"),
        )
    )
    ball = RigidObjectCfg(prim_path="{ENV_REGEX_NS}/Scene/Environment/Ball", spawn=None)
    ball_camera = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/BallCamera",
        update_period=1 / 30,
        height=96,
        width=96,
        data_types=["distance_to_image_plane", "semantic_segmentation"],
        renderer_cfg=IsaacRtxRendererCfg(colorize_semantic_segmentation=False),
        spawn=sim_utils.PinholeCameraCfg(focal_length=24.0, horizontal_aperture=30.0, clipping_range=(0.05, 3.0)),
        offset=TiledCameraCfg.OffsetCfg(pos=(0.35, 0.0, 0.9), rot=(1.0, 0.0, 0.0, 0.0), convention="ros"),
    )


@configclass
class OpenArmLiftWithBallEnvCfg(OpenArmReIncrementalBimanualLiftEnvCfg):
    scene: OpenArmLiftWithBallSceneCfg = OpenArmLiftWithBallSceneCfg(num_envs=64, env_spacing=3.0)
    intent_task_label = "openarm_lift_with_ball"
    camera_extra_observation_dim = 4
    ball_camera_position_scale = 0.20
    ball_camera_visual_radius = 0.04
    reward_ball_center_scale = 2.0
    ball_center_penalty_scale = 0.0
    ball_center_penalty_full_error = 0.10
    # Opt-in only: keep the frozen camera-v1 benchmark unchanged.
    ball_reward_profile = "original"
    ball_lift_bootstrap_scale = 0.0
    ball_height_bridge_scale = 0.0
    ball_upright_bridge_scale = 0.0
    ball_upright_decay_deg = 20.0
    ball_bridge_open_floor = 0.0
    ball_center_std = 0.08
    ball_center_xy = (0.0, 0.0)
    ball_center_half_extents_xy = (0.18, 0.10)
    ball_center_height_range = (0.025, 0.20)
    ball_center_success_threshold = 0.05
    expected_ball_mass_kg = 2.102

    def __post_init__(self):
        if self.ball_reward_profile not in (
            "original",
            "lift_bootstrap",
            "lift_retention",
            "lift_bridge",
            "center_penalty",
            "re_gated_center",
            "re_gated_center_lift40",
            "height_upright_bridge",
        ):
            raise ValueError("Unknown ball reward profile")
        if self.ball_reward_profile == "lift_bootstrap":
            self.reward_grasp_scale = 3.0
            self.reward_lift_scale = 15.0
            self.ball_lift_bootstrap_scale = 5.0
        elif self.ball_reward_profile == "lift_retention":
            # Separate v3 experiment: restore grasp shaping and gate only
            # the additional height credit by measured bilateral closure.
            self.reward_grasp_scale = 5.0
            self.reward_lift_scale = 15.0
            self.ball_lift_bootstrap_scale = 5.0
            if not math.isfinite(float(self.success_min_closure)) or self.success_min_closure <= 0:
                raise ValueError("Retention profile requires a positive closure threshold")
        elif self.ball_reward_profile == "lift_bridge":
            # v4 soft bridge: retain some early elevation credit while
            # preserving the full closed-gripper incentive of v3.
            self.reward_grasp_scale = 5.0
            self.reward_lift_scale = 15.0
            self.ball_lift_bootstrap_scale = 5.0
            self.ball_bridge_open_floor = 0.25
            if not math.isfinite(float(self.success_min_closure)) or self.success_min_closure <= 0:
                raise ValueError("Bridge profile requires a positive closure threshold")
        elif self.ball_reward_profile in ("re_gated_center", "re_gated_center_lift40"):
            # Center only after bilateral lifting. The lift40 pilot changes
            # exactly the parent Lift coefficient, not the grasp/center terms.
            self.reward_ball_center_scale = 8.0
            self.ball_center_std = 0.12
            if self.ball_reward_profile == "re_gated_center_lift40":
                self.reward_lift_scale = 40.0
        elif self.ball_reward_profile == "center_penalty":
            # Exactly the parent Lift reward plus bounded off-center cost.
            self.reward_grasp_scale = 5.0
            self.reward_lift_scale = 20.0
            self.reward_ball_center_scale = 0.0
            self.ball_lift_bootstrap_scale = 0.0
            self.ball_center_penalty_scale = 2.0
        elif self.ball_reward_profile == "height_upright_bridge":
            # Preserve Lift reward/success, then add grasp-gated elevation
            # credit that remains informative when the heavy ball tilts.
            self.reward_grasp_scale = 5.0
            self.reward_lift_scale = 20.0
            self.reward_ball_center_scale = 0.0
            self.ball_lift_bootstrap_scale = 0.0
            self.ball_center_penalty_scale = 2.0
            self.ball_height_bridge_scale = 15.0
            self.ball_upright_bridge_scale = 10.0
            self.ball_upright_decay_deg = 20.0
        elif self.ball_lift_bootstrap_scale != 0:
            raise ValueError("Original profile must have zero lift bootstrap")
        if (
            self.ball_reward_profile not in ("center_penalty", "height_upright_bridge")
            and self.ball_center_penalty_scale != 0
        ):
            raise ValueError("Only center_penalty or height_upright_bridge may activate off-center cost")
        if self.ball_reward_profile != "lift_bridge" and self.ball_bridge_open_floor != 0:
            raise ValueError("Non-bridge reward profile must have zero open floor")
        if not math.isfinite(float(self.ball_center_penalty_full_error)) or (
            self.ball_center_penalty_full_error <= self.ball_center_success_threshold
        ):
            raise ValueError("Off-center penalty must saturate beyond the central threshold")
        super().__post_init__()
        if self.actor_object_pose_source != "apriltag" or self.apriltag_measurement_source != "camera_transform":
            raise ValueError("Matched ball task requires the same camera-transform AprilTag path as Lift")
        expected_center_scale = (
            8.0 if self.ball_reward_profile in ("re_gated_center", "re_gated_center_lift40") else 2.0
        )
        if (
            not math.isfinite(float(self.reward_ball_center_scale))
            or not 0 <= self.reward_ball_center_scale <= expected_center_scale
        ):
            raise ValueError("Ball reward scale exceeds the selected profile bound")
        if not math.isfinite(self.ball_camera_position_scale) or self.ball_camera_position_scale <= 0:
            raise ValueError("Camera ball position scale must be positive and finite")
        ball_center_metrics(
            torch.zeros(1, 3),
            self.ball_center_xy,
            self.ball_center_half_extents_xy,
            self.ball_center_height_range,
            self.ball_center_std,
            self.ball_center_success_threshold,
        )


class OpenArmLiftWithBallEnv(OpenArmReIncrementalBimanualLiftEnv):
    cfg: OpenArmLiftWithBallEnvCfg

    def __init__(self, cfg, render_mode=None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self.ball = self.scene.rigid_objects["ball"]
        box_q = read_data(self.object.data, "root_quat_w")[0:1]
        self._ball_reset_local_position = quat_apply(
            quat_inv(box_q),
            read_data(self.ball.data, "root_pos_w")[0:1] - read_data(self.object.data, "root_pos_w")[0:1],
        )
        self._ball_reset_local_rotation = quat_mul(quat_inv(box_q), read_data(self.ball.data, "root_quat_w")[0:1])
        masses = read_data(self.ball.data, "body_mass")
        if not torch.allclose(masses, torch.full_like(masses, cfg.expected_ball_mass_kg), atol=0.01, rtol=0):
            raise ValueError(f"Unexpected ball masses: {masses}")
        print(
            f"[INFO] matched ball mass={cfg.expected_ball_mass_kg}kg; actor={cfg.actor_input_dim}D critic={cfg.state_space}D; rendered depth/semantic ball observation"
        )

    def _setup_scene(self):
        super()._setup_scene()
        for prim in sim_utils.find_matching_prims(self.cfg.scene.ball.prim_path):
            sim_utils.add_labels(prim, ["ball"], instance_name="class")

    def get_additional_actor_camera_observation(self, side):
        del side  # One shared calibrated overhead view, equally available to both arms.
        step = int(self.common_step_counter)
        if getattr(self, "_ball_camera_observation_cache_step", -1) == step:
            return self._ball_camera_observation_cache
        camera = self.scene.sensors["ball_camera"]
        data = camera.data
        depth = tensor(data.output["distance_to_image_plane"])
        segmentation = tensor(data.output["semantic_segmentation"])
        info = data.info.get("semantic_segmentation", {})
        mask = ball_semantic_mask(segmentation, info)
        camera_position, visible = estimate_ball_camera_position(
            depth, mask, read_data(data, "intrinsic_matrices"), radius=self.cfg.ball_camera_visual_radius
        )
        # DirectMARLEnv does not rerender immediately on reset. At 30 Hz,
        # suppress the first two 60 Hz control observations to prevent stale
        # previous-episode images from being passed as current measurements.
        visible &= self.episode_length_buf >= 2
        measured_world = read_data(data, "pos_w") + quat_apply(read_data(data, "quat_w_ros"), camera_position)
        box_position, box_rotation = get_actor_object_pose_w(self, self._ctx)
        measured_box = quat_apply(quat_inv(box_rotation), measured_world - box_position)
        normalized = (measured_box / self.cfg.ball_camera_position_scale).clamp(-5, 5)
        normalized = torch.where(visible[:, None], normalized, torch.zeros_like(normalized))
        observation = torch.cat([normalized, visible.float()[:, None]], dim=-1)
        self._ball_camera_observation_cache = observation
        self._ball_camera_observation_cache_step = step
        self._ball_camera_detected = visible
        return observation

    def _ball_metrics(self):
        local = quat_apply(
            quat_inv(read_data(self.object.data, "root_quat_w")),
            read_data(self.ball.data, "root_pos_w") - read_data(self.object.data, "root_pos_w"),
        )
        metrics = ball_center_metrics(
            local,
            self.cfg.ball_center_xy,
            self.cfg.ball_center_half_extents_xy,
            self.cfg.ball_center_height_range,
            self.cfg.ball_center_std,
            self.cfg.ball_center_success_threshold,
        )
        return local, metrics

    def _get_rewards(self):
        rewards = super()._get_rewards()
        _, metrics = self._ball_metrics()
        # No bonus for camping at the initial box height or a fallen ball.
        # Base Lift reward and success remain untouched.
        lift = self._ctx["trace_signals"]["height_progress"].clamp(0, 1)
        if self.cfg.ball_reward_profile in ("re_gated_center", "re_gated_center_lift40"):
            # Box-local version of the old RE term: zero at floor or without bilateral grasp.
            grasp_gate = self._ctx["trace_signals"]["lift_gate"].clamp(0, 1)
            centeredness = torch.where(
                metrics["inside"], 1.0 - torch.tanh(metrics["error"] / self.cfg.ball_center_std), torch.zeros_like(lift)
            )
            center_bonus = self.cfg.reward_ball_center_scale * grasp_gate * lift.square() * centeredness
            self.extras["log"]["ball_center_grasp_gate"] = grasp_gate.mean().item()
        else:
            center_bonus = self.cfg.reward_ball_center_scale * metrics["score"] * lift
        bootstrap = ball_lift_bootstrap(
            lift,
            self._ctx["trace_signals"]["object_tilt_deg"],
            metrics["inside"],
            self.cfg.ball_lift_bootstrap_scale,
            self.cfg.tilt_free_deg,
            self.cfg.tilt_bad_deg,
        )
        if self.cfg.ball_reward_profile in ("lift_retention", "lift_bridge"):
            # Reuse the exact measured joint-position closure definition.
            # Commanded gripper action and contact-force stubs are not evidence
            # of retention. This soft gate still does not certify actual contact.
            left_opening = read_data(self.robot.data, "joint_pos")[:, self._ctx["left_gripper_joint_ids"]].mean(dim=-1)
            right_opening = read_data(self.robot.data, "joint_pos")[:, self._ctx["right_gripper_joint_ids"]].mean(
                dim=-1
            )
            gate = bilateral_closure_gate(
                _closure_from_opening(self, left_opening),
                _closure_from_opening(self, right_opening),
                self.cfg.success_min_closure,
            )
            if self.cfg.ball_reward_profile == "lift_bridge":
                gate = self.cfg.ball_bridge_open_floor + (1.0 - self.cfg.ball_bridge_open_floor) * gate
            bootstrap = bootstrap * gate
            self.extras["log"]["ball_bootstrap_closure_gate"] = gate.mean().item()
        center_penalty = ball_offcenter_penalty(
            metrics["error"],
            metrics["inside"],
            self.cfg.ball_center_success_threshold,
            self.cfg.ball_center_penalty_full_error,
            self.cfg.ball_center_penalty_scale,
        )
        bridge_height = lift * 0.0
        bridge_upright = lift * 0.0
        if self.cfg.ball_reward_profile == "height_upright_bridge":
            bridge_height, bridge_upright = ball_height_upright_bridge(
                lift,
                self._ctx["trace_signals"]["object_tilt_deg"],
                metrics["inside"],
                self._ctx["trace_signals"]["lift_gate"],
                self.cfg.ball_height_bridge_scale,
                self.cfg.ball_upright_bridge_scale,
                self.cfg.ball_upright_decay_deg,
            )
            center_penalty = center_penalty * lift
        bonus = center_bonus + bootstrap + center_penalty + bridge_height + bridge_upright
        for agent in rewards:
            rewards[agent] = rewards[agent] + bonus
        self.extras["log"].update(
            {
                "ball_center_error": metrics["error"].mean().item(),
                "ball_inside_ratio": metrics["inside"].float().mean().item(),
                "ball_center_ok_ratio": metrics["center_ok"].float().mean().item(),
                "reward_ball_center": center_bonus.mean().item(),
                "reward_ball_center_penalty": center_penalty.mean().item(),
                "reward_ball_lift_bootstrap": bootstrap.mean().item(),
                "reward_ball_height_bridge": bridge_height.mean().item(),
                "reward_ball_upright_bridge": bridge_upright.mean().item(),
            }
        )
        return rewards

    def _reset_idx(self, env_ids):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        if not isinstance(env_ids, torch.Tensor):
            env_ids = torch.tensor(env_ids, device=self.device, dtype=torch.long)
        super()._reset_idx(env_ids)
        self._ball_camera_observation_cache_step = -1
        # Use the just-written reset target, not stale pre-reset asset buffers.
        position = self._ctx["initial_object_pos_w"][env_ids]
        rotation = self._ctx["initial_object_quat_w"][env_ids]
        state = read_data(self.ball.data, "default_root_state")[env_ids].clone()
        state[:, :3] = position + quat_apply(rotation, self._ball_reset_local_position.expand(env_ids.numel(), -1))
        state[:, 3:7] = quat_mul(rotation, self._ball_reset_local_rotation.expand(env_ids.numel(), -1))
        state[:, 7:] = 0
        write_root_pose(self.ball, state[:, :7], env_ids=env_ids)
        self.ball.write_root_velocity_to_sim_index(root_velocity=state[:, 7:], env_ids=env_ids)

    def get_paper_motion_context_trace_signals(self, env_index=0):
        signals = super().get_paper_motion_context_trace_signals(env_index)
        local, metrics = self._ball_metrics()
        index = int(env_index)
        camera_observation = self.get_additional_actor_camera_observation("left")
        signals.update(
            {
                "ball_camera_observation": camera_observation[index].detach().cpu().tolist(),
                "ball_position_box": local[index].detach().cpu().tolist(),
                "ball_position_w": read_data(self.ball.data, "root_pos_w")[index].detach().cpu().tolist(),
                "ball_center_error": float(metrics["error"][index]),
                "ball_inside": bool(metrics["inside"][index]),
                "ball_center_ok": bool(metrics["center_ok"][index]),
            }
        )
        return signals
