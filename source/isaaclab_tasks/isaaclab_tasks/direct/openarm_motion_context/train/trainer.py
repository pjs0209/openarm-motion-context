"""Proprioceptive Motion Context Sharing trainer.

이 trainer는 skrl agent와 Isaac Lab env 사이에서 데이터 흐름을 조율한다.
주요 역할은 다음과 같다.

1. policy inference 전에 env에서 z=[signed EE motion, motion context]를 읽는다.
2. 각 팔의 z_t를 env buffer를 통해 상대 팔 observation에 전달한다.
3. 사후 context behavior 분석을 위해 episode별 JSON trace를 저장한다.
4. 긴 Isaac Sim 학습 중 reward, grip, motion context 상태를 읽기 쉬운 로그로 출력한다.

reward tensor와 observation tensor는 여기서 만들지 않는다. 그 경계는 env와
task logic에 남겨두고, 이 파일은 handoff와 logging만 담당한다.

파일 구조:
    1. Logging helper
       - context/vector formatting helpers.
    2. MotionContextTrainer trace helpers
       - episode별 JSON trace 생성/저장.
    3. Debug summary
       - 긴 학습 로그를 사람이 읽을 수 있는 블록으로 출력.
    4. Communication handoff
       - communication_mode에 맞는 상대 팔 message를 즉시 전달한다.
    5. train/eval loop
       - skrl agent call, env.step, trace/debug emission.
"""

from __future__ import annotations

from collections import deque
import json
import math
from numbers import Number
import os
import sys

import torch

from ..communication.motion_context import motion_context_layout
from .action_trace import record_policy_mean_actions
from .evaluation_state import capture_physical_start
import tqdm

from skrl.trainers.torch import SequentialTrainer


def _to_float_list(values, width: int | None = None) -> list[float]:
    """tensor/list를 logging 문자열에 넣을 CPU float list로 바꾼다."""

    if isinstance(values, torch.Tensor):
        data = values.detach().float().flatten().cpu().tolist()
    elif isinstance(values, (list, tuple)):
        try:
            data = torch.as_tensor(values, dtype=torch.float32).flatten().tolist()
        except (TypeError, ValueError):
            data = []
    else:
        data = []
    if width is not None:
        if len(data) < width:
            data = data + [0.0] * (width - len(data))
        data = data[:width]
    return data


def format_motion(motion) -> str:
    """deterministic 3D motion intent를 문자열로 만든다."""

    vals = _to_float_list(motion, 3)
    norm = math.sqrt(sum(value * value for value in vals))
    return f"[mx={vals[0]:+.2f}, my={vals[1]:+.2f}, mz={vals[2]:+.2f}; norm={norm:.2f}]"


_COMMUNICATION_PAYLOAD_DIMS = {
    "none": 0,
    "motion_only": 3,
    "context_only": 3,
    "motion_context": 6,
    "motion_context_la": 6,
    "motion_context_s": 6,
    "previous_action": 8,
    "full_partner_observation": 30,
}


def _communication_layout(mode: str, representation: str = "trend") -> list[str]:
    """Return the semantic layout of the active (unpadded) payload."""

    if mode == "motion_only":
        return ["signed_motion_x", "signed_motion_y", "signed_motion_z"]
    if mode == "context_only":
        return list(motion_context_layout(representation))
    if mode in ("motion_context", "motion_context_la", "motion_context_s"):
        return [
            "signed_motion_x",
            "signed_motion_y",
            "signed_motion_z",
            *motion_context_layout(representation),
        ]
    if mode == "previous_action":
        return [f"previous_action_{index}" for index in range(8)]
    if mode == "full_partner_observation":
        return [f"partner_observation_{index}" for index in range(30)]
    return []


class MotionContextTrainer(SequentialTrainer):
    """no_intent/share_intent paper motion-context 실험을 위한 custom skrl trainer."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.debug_interval = int(self.cfg.get("debug_interval", 0))
        self.intent_share_warmup_timesteps = 0
        self.apply_intent_warmup_in_eval = False
        self.save_mode_trace = bool(self.cfg.get("save_mode_trace", False))
        self.mode_trace_dir = str(self.cfg.get("mode_trace_dir", "logs/motion_context_traces"))
        self.mode_trace_env_index = int(self.cfg.get("mode_trace_env_index", 0))
        self.mode_trace_format = str(self.cfg.get("mode_trace_format", "json")).lower()
        self.mode_trace_task = str(self.cfg.get("mode_trace_task", self.cfg.get("task_name", "")))
        self.mode_trace_checkpoint = str(self.cfg.get("mode_trace_checkpoint", self.cfg.get("checkpoint", "")))
        self.policy_warmstart_manifest = None
        if self.mode_trace_checkpoint:
            transfer_manifest = os.path.join(
                os.path.dirname(os.path.dirname(self.mode_trace_checkpoint)), "params", "policy_warmstart.json"
            )
            if os.path.isfile(transfer_manifest):
                with open(transfer_manifest) as stream:
                    self.policy_warmstart_manifest = json.load(stream)
        self.num_trace_episodes = int(self.cfg.get("num_trace_episodes", 0))
        self.diagnostic_mask = str(self.cfg.get("diagnostic_mask", "none"))
        if bool(self.cfg.get("deterministic_eval", False)):
            self.stochastic_evaluation = False
        self._mode_trace_episode_index = 0
        self._mode_trace = self._new_mode_trace()
        self._in_eval = False
        self._missing_mode_trace_warned = False
        self._mode_trace_last_done = {"terminated": False, "truncated": False}
        self._episode_returns = None
        self._episode_lengths = None
        self._episode_history = deque(maxlen=100)
        if self.save_mode_trace:
            print(
                f"[INFO] Motion-context trace saving enabled: dir={self.mode_trace_dir} "
                f"env_index={self.mode_trace_env_index} episodes={self.num_trace_episodes}"
            )
        if hasattr(self.agents, "set_motion_context_state_provider"):
            self.agents.set_motion_context_state_provider(self._motion_context_checkpoint_state)
        if hasattr(self.agents, "set_motion_context_state_loader"):
            self.agents.set_motion_context_state_loader(self._load_motion_context_checkpoint_state)

    @staticmethod
    def _new_mode_trace() -> dict:
        """분석 코드가 기대하는 모든 field를 가진 빈 trace dict를 만든다."""

        return {
            "success_signal": [],
            "left_motion": [],
            "right_motion": [],
            "left_message": [],
            "right_message": [],
            "left_policy_message": [],
            "left_policy_mean_action": [],
            "right_policy_mean_action": [],
            "right_policy_message": [],
            "object_z": [],
            "object_dz": [],
            "tip_dist": [],
            "lateral_error": [],
            "axis_alignment": [],
            "insertion_depth": [],
            "target_insertion_depth": [],
            "shaft_radial_envelope": [],
            "keypoint_dist": [],
            "preinsert_dist": [],
            "insert_pose_dist": [],
            "keypoint_reward_baseline": [],
            "keypoint_reward_coarse": [],
            "keypoint_reward_fine": [],
            "preinsert_reward": [],
            "insert_pose_reward": [],
            "depth_reward": [],
            "axis_gate": [],
            "lateral_gate": [],
            "insertion_gate": [],
            "depth_progress": [],
            "reward_left": [],
            "reward_right": [],
            "reward_mean": [],
            "left_ee_vel": [],
            "right_ee_vel": [],
            "left_motion_context": [],
            "right_motion_context": [],
            "left_translation_trend": [],
            "left_rotation_trend": [],
            "left_stability_trend": [],
            "left_action_smoothness": [],
            "right_translation_trend": [],
            "right_rotation_trend": [],
            "right_stability_trend": [],
            "right_action_smoothness": [],
            "left_linear_speed_raw": [],
            "left_angular_speed_raw": [],
            "left_action_variation_raw": [],
            "left_action_variation_ema": [],
            "right_linear_speed_raw": [],
            "right_angular_speed_raw": [],
            "right_action_variation_raw": [],
            "right_action_variation_ema": [],
            "context_linear_scale": [],
            "context_angular_scale": [],
            "context_action_variation_scale": [],
            "motion_context_calibrated": [],
            "motion_sharing_active": [],
            "context_sharing_active": [],
            "calibration_cv_linear": [],
            "calibration_cv_angular": [],
            "calibration_cv_action_variation": [],
            "calibration_cv_max": [],
            "calibration_stable_count": [],
            "calibration_history_count": [],
            "calibration_window_filled": [],
            "motion_context_scale_frozen": [],
            "left_ee_pos_b": [],
            "right_ee_pos_b": [],
            "left_ee_quat_b": [],
            "right_ee_quat_b": [],
            "left_target_dist": [],
            "right_target_dist": [],
            "left_target_quat_error": [],
            "right_target_quat_error": [],
            "left_gripper_opening": [],
            "right_gripper_opening": [],
            "left_prev_gripper_action": [],
            "right_prev_gripper_action": [],
            "left_closure": [],
            "right_closure": [],
            "left_ee_lin_vel": [],
            "right_ee_lin_vel": [],
            "left_contact_min": [],
            "right_contact_min": [],
            "left_force_min": [],
            "right_force_min": [],
            "left_grip_score": [],
            "right_grip_score": [],
            "left_lift_reward": [],
            "right_lift_reward": [],
            "object_tilt_deg": [],
            "success_roll_deg": [],
            "success_pitch_deg": [],
            "success_roll_pitch_deg": [],
            "success_yaw_deg": [],
            "goal_error": [],
            "xy_error": [],
            "hprog": [],
            "left_arm_action_magnitude": [],
            "right_arm_action_magnitude": [],
            "left_arm_action": [],
            "right_arm_action": [],
            "left_action_grip": [],
            "right_action_grip": [],
            "height_ok": [],
            "tilt_ok": [],
            "left_distance_ok": [],
            "right_distance_ok": [],
            "left_orientation_ok": [],
            "right_orientation_ok": [],
            "left_closure_ok": [],
            "right_closure_ok": [],
            "left_finger_span_ok": [],
            "right_finger_span_ok": [],
            "left_span_ok": [],
            "right_span_ok": [],
            "success_hold_count": [],
            "left_grasp_ok": [],
            "right_grasp_ok": [],
            "dual_grasp_ok": [],
            "hold_ok": [],
            "strict_success": [],
            "drop": [],
            "far": [],
            "tilt_fail": [],
            "invalid": [],
            "secure_bi_min": [],
            "secure_bi_sqrt": [],
            "grasp_imbalance": [],
            "reward_gap": [],
            "left_right_closure_gap": [],
            "left_right_contact_gap": [],
        }

    def _resolve_env_attr(self, attr_name: str, default=None):
        """skrl/gym wrapper 층을 따라가며 task env attribute를 읽는다."""

        queue = [self.env]
        visited = set()
        while queue:
            env = queue.pop(0)
            if env is None or id(env) in visited:
                continue
            visited.add(id(env))
            try:
                if hasattr(env, attr_name):
                    return getattr(env, attr_name)
            except Exception:
                pass
            try:
                cfg = getattr(env, "cfg", None)
                if cfg is not None and hasattr(cfg, attr_name):
                    return getattr(cfg, attr_name)
            except Exception:
                pass
            for child_name in ("_env", "_unwrapped", "env", "unwrapped"):
                try:
                    child = getattr(env, child_name, None)
                except Exception:
                    child = None
                if child is not None and child is not env:
                    queue.append(child)
        return default

    def _set_env_attr(self, attr_name: str, value) -> None:
        """해당 attribute를 노출하는 wrapper/task layer 전체에 값을 쓴다."""

        queue = [self.env]
        visited = set()
        updated = False
        force_set = attr_name.startswith("_motion_context_")
        while queue:
            env = queue.pop(0)
            if env is None or id(env) in visited:
                continue
            visited.add(id(env))
            try:
                cfg = getattr(env, "cfg", None)
                if force_set or hasattr(env, attr_name) or (cfg is not None and hasattr(cfg, attr_name)):
                    setattr(env, attr_name, value)
                    updated = True
            except Exception:
                pass
            for child_name in ("_env", "_unwrapped", "env", "unwrapped"):
                try:
                    child = getattr(env, child_name, None)
                except Exception:
                    child = None
                if child is not None and child is not env:
                    queue.append(child)
        if not updated:
            try:
                setattr(self.env, attr_name, value)
            except Exception:
                pass

    def _motion_context_checkpoint_state(self) -> dict:
        """Return representation metadata saved beside policy checkpoints."""

        raw_scale = self._resolve_env_attr("_motion_context_running_scale", None)
        if isinstance(raw_scale, torch.Tensor) and raw_scale.numel() >= 3:
            running_scale = raw_scale.detach().float().cpu().clone()
        else:
            running_scale = torch.tensor(
                [
                    float(self._resolve_env_attr("motion_context_lin_scale_init", 0.10) or 0.10),
                    float(self._resolve_env_attr("motion_context_ang_scale_init", 0.50) or 0.50),
                    float(self._resolve_env_attr("motion_context_action_scale_init", 0.50) or 0.50),
                ],
                dtype=torch.float32,
            )

        raw_step = self._resolve_env_attr("_motion_context_scale_update_step", -1)
        step = int(-1 if raw_step is None else raw_step)
        common_step = int(self._resolve_env_attr("common_step_counter", 0) or 0)
        scale_frozen = bool(self._resolve_env_attr("_motion_context_scale_frozen", False))
        calibrated = bool(self._resolve_env_attr("_motion_context_calibrated", False))
        history = self._resolve_env_attr("_motion_context_calibration_history", None)
        if isinstance(history, torch.Tensor):
            history = history.detach().float().cpu().clone()
        else:
            history = torch.empty((0, 3), dtype=torch.float32)
        cv = self._resolve_env_attr("_motion_context_calibration_cv", None)
        if isinstance(cv, torch.Tensor) and cv.numel() >= 3:
            cv = cv.detach().float().cpu().flatten()[:3].clone()
        else:
            cv = torch.zeros(3, dtype=torch.float32)
        return {
            "running_scale": running_scale,
            "scale_update_step": step,
            "common_step_counter": common_step,
            "scale_frozen": scale_frozen,
            "context_calibrated": calibrated,
            "calibration_enabled": bool(self._resolve_env_attr("motion_context_calibration_enabled", True)),
            "calibration_history": history,
            "calibration_history_index": int(
                self._resolve_env_attr("_motion_context_calibration_history_index", 0) or 0
            ),
            "calibration_history_count": int(
                self._resolve_env_attr("_motion_context_calibration_history_count", 0) or 0
            ),
            "calibration_stable_count": int(self._resolve_env_attr("_motion_context_calibration_stable_count", 0) or 0),
            "calibration_cv": cv,
            "calibration_window": int(self._resolve_env_attr("motion_context_calibration_window", 256) or 256),
            "calibration_cv_threshold": float(self._resolve_env_attr("motion_context_calibration_cv_threshold", 0.02)),
            "calibration_confirmations": int(
                self._resolve_env_attr("motion_context_calibration_confirmations", 64) or 64
            ),
            "calibration_epsilon": float(
                self._resolve_env_attr("motion_context_calibration_epsilon", 1.0e-6) or 1.0e-6
            ),
            "representation": str(self._resolve_env_attr("motion_context_representation", "trend")),
            "normalization": {
                "trend": "relative_ema_trend",
                "absolute_fixed": "fixed_physical_scale",
                "signed_twist": "fixed_angular_velocity_scale",
            }[str(self._resolve_env_attr("motion_context_representation", "trend"))],
            "trend_horizon": int(self._resolve_env_attr("motion_context_trend_horizon", 5) or 5),
            "trend_beta": float(self._resolve_env_attr("motion_context_trend_beta", 0.80) or 0.80),
            "twist_angular_scale": float(self._resolve_env_attr("motion_context_twist_ang_scale", 3.0)),
            "trend_epsilon": [
                float(self._resolve_env_attr("motion_context_trend_linear_epsilon", 0.01) or 0.01),
                float(self._resolve_env_attr("motion_context_trend_angular_epsilon", 0.05) or 0.05),
                float(self._resolve_env_attr("motion_context_trend_action_epsilon", 0.01) or 0.01),
            ],
        }

    def _load_motion_context_checkpoint_state(self, state) -> None:
        """Restore env-side normalization state from a checkpoint sidecar."""

        if not isinstance(state, dict):
            return
        running_scale = state.get("running_scale", None)
        if running_scale is None:
            return
        device = self._resolve_env_attr("device", self.env.device if hasattr(self.env, "device") else "cpu")
        scale_tensor = torch.as_tensor(running_scale, dtype=torch.float32, device=device).flatten()
        if scale_tensor.numel() < 3:
            return
        self._set_env_attr("_motion_context_running_scale", scale_tensor[:3].clone())
        self._set_env_attr("_motion_context_scale_update_step", int(state.get("scale_update_step", -1)))
        for attr_name, state_name, default in (
            ("motion_context_calibration_enabled", "calibration_enabled", True),
            ("motion_context_calibration_window", "calibration_window", 256),
            ("motion_context_calibration_cv_threshold", "calibration_cv_threshold", 0.02),
            ("motion_context_calibration_confirmations", "calibration_confirmations", 64),
            ("motion_context_calibration_epsilon", "calibration_epsilon", 1.0e-6),
        ):
            self._set_env_attr(attr_name, state.get(state_name, default))
        has_adaptive_state = "context_calibrated" in state
        calibrated = bool(state.get("context_calibrated", False))
        self._set_env_attr("_motion_context_calibrated", calibrated)
        self._set_env_attr(
            "_motion_context_scale_frozen",
            bool(state.get("scale_frozen", calibrated)) if has_adaptive_state else False,
        )
        history = state.get("calibration_history", None)
        if history is not None:
            history_tensor = torch.as_tensor(history, dtype=torch.float32, device=device)
            if history_tensor.ndim == 2 and history_tensor.shape[-1] == 3:
                self._set_env_attr("_motion_context_calibration_history", history_tensor.clone())
        cv = torch.as_tensor(state.get("calibration_cv", [0.0, 0.0, 0.0]), dtype=torch.float32, device=device)
        if cv.numel() >= 3:
            self._set_env_attr("_motion_context_calibration_cv", cv.flatten()[:3].clone())
        self._set_env_attr("_motion_context_calibration_history_index", int(state.get("calibration_history_index", 0)))
        self._set_env_attr("_motion_context_calibration_history_count", int(state.get("calibration_history_count", 0)))
        self._set_env_attr("_motion_context_calibration_stable_count", int(state.get("calibration_stable_count", 0)))
        self._set_env_attr("_motion_context_calibration_event_emitted", calibrated)

    def _intent_share_warmup_timesteps(self) -> int:
        """intent-share warmup 길이를 반환한다. eval에서는 요청하지 않으면 warmup을 끈다."""

        if self._in_eval and not self.apply_intent_warmup_in_eval:
            return 0
        return max(self.intent_share_warmup_timesteps, 0)

    def _current_motion_intents(self) -> dict[str, torch.Tensor]:
        """task env에서 deterministic motion intent를 가져온다."""

        getter = self._resolve_env_attr("get_current_motion_intents", None)
        if callable(getter):
            return getter()
        infos = getattr(self.env, "extras", {})
        motion = infos.get("motion_intents", {}) if isinstance(infos, dict) else {}
        return motion if isinstance(motion, dict) else {}

    @staticmethod
    def _scalar(value, default=0.0):
        """scalar처럼 쓰는 tensor/value를 로그용 float으로 변환한다."""

        if value is None:
            return default
        if isinstance(value, torch.Tensor):
            return value.item() if value.numel() == 1 else value.float().mean().item()
        return float(value)

    @staticmethod
    def _env_scalar(value, env_index: int, default=0.0):
        """batched tensor에서 env 하나의 scalar 값을 고른다."""

        if value is None:
            return default
        if isinstance(value, torch.Tensor):
            if value.numel() == 0:
                return default
            data = value.detach().float()
            if data.dim() == 0:
                return float(data.cpu())
            env_index = int(max(0, min(env_index, data.shape[0] - 1)))
            return float(data.reshape(data.shape[0], -1)[env_index, 0].cpu())
        return float(value)

    def _track_environment_info(self, infos: dict) -> None:
        """Send finite scalar task diagnostics to the skrl event logger."""

        values = infos.get(self.environment_info, {}) if isinstance(infos, dict) else {}
        if not isinstance(values, dict):
            return
        for key, value in values.items():
            scalar = None
            if isinstance(value, torch.Tensor) and value.numel() == 1:
                scalar = float(value.detach().cpu())
            elif isinstance(value, Number) and not isinstance(value, bool):
                scalar = float(value)
            if scalar is not None and math.isfinite(scalar):
                self.agents.track_data(f"Info / {key}", scalar)

    @staticmethod
    def _env_vector(value, env_index: int, width: int) -> list[float]:
        """batched tensor에서 env 하나의 고정 폭 vector 값을 고른다."""

        if not isinstance(value, torch.Tensor) or value.numel() == 0:
            return [0.0] * width
        data = value.detach().float()
        if data.dim() == 1:
            vec = data[:width]
        else:
            env_index = int(max(0, min(env_index, data.shape[0] - 1)))
            vec = data.reshape(data.shape[0], -1)[env_index, :width]
        values = [float(v) for v in vec.cpu().tolist()]
        if len(values) < width:
            values.extend([0.0] * (width - len(values)))
        return values[:width]

    @staticmethod
    def _fmt_motion(value: torch.Tensor | None):
        """batch가 있을 수 있는 motion tensor를 로그 문자열로 만든다."""

        if isinstance(value, torch.Tensor) and value.numel() > 0:
            value = value.detach().float().reshape(value.shape[0], -1)[:, :3].mean(dim=0)
        return format_motion(value)

    def _append_mode_trace(
        self,
        outputs,
        rewards,
        infos,
        trace_signals: dict | None = None,
        policy_states: dict[str, torch.Tensor] | None = None,
    ) -> None:
        """Append one coherent evaluation transition to the JSON trace.

        ``trace_signals`` is sampled before ``env.step`` so state, outgoing
        message, and action all describe time t. Rewards describe t -> t+1.
        This avoids recording the auto-reset state as the terminal task state.
        """

        if not self.save_mode_trace or not self._in_eval:
            return
        if self.num_trace_episodes > 0 and self._mode_trace_episode_index >= self.num_trace_episodes:
            return
        if self.mode_trace_format != "json":
            raise ValueError(f"Unsupported mode_trace_format: {self.mode_trace_format}. Only 'json' is supported.")
        env_index = int(max(0, min(self.mode_trace_env_index, self.env.num_envs - 1)))
        step_index = len(self._mode_trace["left_motion_context"])
        record_policy_mean_actions(self._mode_trace, outputs, env_index)
        self._mode_trace["left_motion"].append(
            self._env_vector(outputs.get("left_arm", {}).get("motion_intent"), env_index, 3)
        )
        self._mode_trace["right_motion"].append(
            self._env_vector(outputs.get("right_arm", {}).get("motion_intent"), env_index, 3)
        )
        slot_dim = int(self._resolve_env_attr("communication_feature_dim", 0) or 0)
        self._mode_trace["left_message"].append(
            self._env_vector(outputs.get("left_arm", {}).get("intent"), env_index, slot_dim)
        )
        self._mode_trace["right_message"].append(
            self._env_vector(outputs.get("right_arm", {}).get("intent"), env_index, slot_dim)
        )
        own_dim = int(self._resolve_env_attr("own_observation_dim", 30))
        for side, agent in (("left", "left_arm"), ("right", "right_arm")):
            state = policy_states.get(agent) if policy_states is not None else None
            partner_slot = state[..., own_dim : own_dim + slot_dim] if state is not None else None
            self._mode_trace[f"{side}_policy_message"].append(self._env_vector(partner_slot, env_index, slot_dim))

        reward_left = 0.0
        reward_right = 0.0
        if isinstance(rewards, dict):
            reward_left = self._env_scalar(rewards.get("left_arm"), env_index)
            reward_right = self._env_scalar(rewards.get("right_arm"), env_index)
        self._mode_trace["reward_left"].append(float(reward_left))
        self._mode_trace["reward_right"].append(float(reward_right))
        self._mode_trace["reward_mean"].append(float(0.5 * (reward_left + reward_right)))

        if trace_signals is None:
            trace_getter = self._resolve_env_attr("get_paper_motion_context_trace_signals", None)
            signals = trace_getter(env_index) if callable(trace_getter) else {}
        else:
            signals = trace_signals
        self._mode_trace["success_signal"].append(bool(signals.get("success", False)))
        if "ball_position_box" in signals:
            for key in (
                "ball_position_box",
                "ball_position_w",
                "ball_center_error",
                "ball_inside",
                "ball_center_ok",
                "ball_camera_observation",
            ):
                self._mode_trace.setdefault(key, []).append(signals[key])
        if "peg_tip_w" in signals:
            for key in ("peg_tip_w", "hole_entrance_w", "hole_bottom_w"):
                self._mode_trace.setdefault(key, []).append(signals[key])
        if "left_grasp_error_m" in signals:
            for key in (
                "left_grasp_error_m",
                "right_grasp_error_m",
                "left_gripper_command_closed",
                "right_gripper_command_closed",
            ):
                self._mode_trace.setdefault(key, []).append(signals[key])
        for key in (
            "object_z",
            "object_dz",
            "tip_dist",
            "lateral_error",
            "axis_alignment",
            "insertion_depth",
            "target_insertion_depth",
            "shaft_radial_envelope",
            "keypoint_dist",
            "preinsert_dist",
            "insert_pose_dist",
            "keypoint_reward_baseline",
            "keypoint_reward_coarse",
            "keypoint_reward_fine",
            "preinsert_reward",
            "insert_pose_reward",
            "depth_reward",
            "axis_gate",
            "lateral_gate",
            "insertion_gate",
            "depth_progress",
            "left_ee_pos_b",
            "right_ee_pos_b",
            "left_ee_quat_b",
            "right_ee_quat_b",
            "left_ee_vel",
            "right_ee_vel",
            "left_motion_context",
            "right_motion_context",
            "left_translation_trend",
            "left_rotation_trend",
            "left_stability_trend",
            "left_action_smoothness",
            "right_translation_trend",
            "right_rotation_trend",
            "right_stability_trend",
            "right_action_smoothness",
            "left_linear_speed_raw",
            "left_angular_speed_raw",
            "left_action_variation_raw",
            "left_action_variation_ema",
            "right_linear_speed_raw",
            "right_angular_speed_raw",
            "right_action_variation_raw",
            "right_action_variation_ema",
            "context_linear_scale",
            "context_angular_scale",
            "context_action_variation_scale",
            "motion_context_calibrated",
            "motion_sharing_active",
            "context_sharing_active",
            "calibration_cv_linear",
            "calibration_cv_angular",
            "calibration_cv_action_variation",
            "calibration_cv_max",
            "calibration_stable_count",
            "calibration_history_count",
            "calibration_window_filled",
            "motion_context_scale_frozen",
            "left_target_dist",
            "right_target_dist",
            "left_target_quat_error",
            "right_target_quat_error",
            "left_gripper_opening",
            "right_gripper_opening",
            "left_prev_gripper_action",
            "right_prev_gripper_action",
            "left_closure",
            "right_closure",
            "left_ee_lin_vel",
            "right_ee_lin_vel",
            "left_contact_min",
            "right_contact_min",
            "left_force_min",
            "right_force_min",
            "left_grip_score",
            "right_grip_score",
            "left_lift_reward",
            "right_lift_reward",
            "object_tilt_deg",
            "success_roll_deg",
            "success_pitch_deg",
            "success_roll_pitch_deg",
            "success_yaw_deg",
            "goal_error",
            "xy_error",
            "hprog",
            "left_arm_action_magnitude",
            "right_arm_action_magnitude",
            "left_arm_action",
            "right_arm_action",
            "left_action_grip",
            "right_action_grip",
            "height_ok",
            "tilt_ok",
            "left_distance_ok",
            "right_distance_ok",
            "left_orientation_ok",
            "right_orientation_ok",
            "left_closure_ok",
            "right_closure_ok",
            "left_finger_span_ok",
            "right_finger_span_ok",
            "left_span_ok",
            "right_span_ok",
            "success_hold_count",
            "left_grasp_ok",
            "right_grasp_ok",
            "dual_grasp_ok",
            "hold_ok",
            "strict_success",
            "drop",
            "far",
            "tilt_fail",
            "invalid",
            "secure_bi_min",
            "secure_bi_sqrt",
            "grasp_imbalance",
            "reward_gap",
            "left_right_closure_gap",
            "left_right_contact_gap",
        ):
            vector_default = (
                [0.0, 0.0, 0.0, 1.0]
                if key.endswith("_ee_quat_b")
                else [0.0, 0.0, 0.0]
                if key.endswith("_ee_vel")
                or key.endswith("_ee_lin_vel")
                or key.endswith("_ee_pos_b")
                or key.endswith("_motion_context")
                else [0.0] * 7
                if key.endswith("_arm_action")
                else 0.0
            )
            self._mode_trace[key].append(signals.get(key, vector_default))

    def _agent_done_value(self, done_dict, agent: str) -> bool:
        """terminated/truncated dict에서 trace 대상 env 하나의 done 값을 안전하게 읽는다."""

        if not isinstance(done_dict, dict):
            return False
        value = done_dict.get(agent)
        if value is None:
            value = done_dict.get("__all__")
        if value is None:
            return False
        env_index = max(int(self.mode_trace_env_index), 0)
        try:
            if isinstance(value, torch.Tensor):
                if value.numel() == 0:
                    return False
                flat = value.reshape(-1)
                env_index = min(env_index, flat.shape[0] - 1)
                return bool(flat[env_index].detach().cpu().item())
            if isinstance(value, (list, tuple)):
                if not value:
                    return False
                env_index = min(env_index, len(value) - 1)
                return bool(value[env_index])
            return bool(value)
        except Exception:
            return False

    def _trace_env_done(self, terminated, truncated) -> tuple[bool, bool, bool]:
        """trace 대상 env가 이번 step에서 scenario 종료됐는지 판단한다.

        Isaac Lab multi-agent wrapper에서는 episode가 끝나도 self.env.agents가
        비지 않을 수 있다. trace 파일 경계는 terminated/truncated의 trace env
        index를 직접 읽어 결정한다.
        """

        agents = ("left_arm", "right_arm")
        terminated_done = any(self._agent_done_value(terminated, agent) for agent in agents)
        truncated_done = any(self._agent_done_value(truncated, agent) for agent in agents)
        return terminated_done or truncated_done, terminated_done, truncated_done

    def _trace_terminal_info(self, terminated, truncated) -> dict[str, bool]:
        """Return terminal reason flags for the trace env from env termination masks."""

        trace_done, trace_terminated, trace_truncated = self._trace_env_done(terminated, truncated)
        device = None
        for value in (terminated, truncated):
            if isinstance(value, dict):
                tensors = [v for v in value.values() if isinstance(v, torch.Tensor)]
                if tensors:
                    device = tensors[0].device
                    break
        if device is None:
            device = torch.device("cpu")
        env_index = int(max(0, min(int(self.mode_trace_env_index), int(self.env.num_envs) - 1)))
        num_envs = int(self.env.num_envs)

        def _flag(name: str) -> bool:
            mask = self._ctx_mask(name, num_envs, device)
            if mask.numel() <= env_index:
                return False
            return bool(mask[env_index].detach().cpu())

        drop = _flag("termination_drop")
        far = _flag("termination_far")
        tilt_fail = _flag("termination_tilt_fail")
        invalid = _flag("termination_invalid") or _flag("termination_side_wall")
        success = _flag("termination_success")
        # Some Isaac Lab wrappers reset task buffers before trace collection.
        # If an env reports terminal=True without any failure/truncation flag,
        # the terminal reason is the success flag returned by _get_dones().
        if trace_terminated and not trace_truncated and not (success or drop or far or tilt_fail or invalid):
            success = True
        return {
            "done": bool(trace_done),
            "terminated": bool(trace_terminated),
            "truncated": bool(trace_truncated),
            "success": bool(success),
            "drop": bool(drop),
            "far": bool(far),
            "tilt_fail": bool(tilt_fail),
            "invalid": bool(invalid),
        }

    def _save_mode_trace(self, *, terminated_at_end: bool = False, truncated_at_end: bool = False) -> bool:
        """현재 episode trace와 metadata를 디스크에 저장한다."""

        if not self.save_mode_trace or not self._in_eval or not self._mode_trace["left_motion_context"]:
            if self.save_mode_trace and self._in_eval and not self._mode_trace["left_motion_context"]:
                print(
                    "[WARN] save_mode_trace=True but no trace samples were collected. "
                    "Check that eval is running and _append_mode_trace() is called."
                )
            return False

        os.makedirs(self.mode_trace_dir, exist_ok=True)
        path = os.path.join(
            self.mode_trace_dir,
            f"motion_context_trace_episode_{self._mode_trace_episode_index:03d}.json",
        )

        trace = dict(self._mode_trace)
        trace["episode_index"] = self._mode_trace_episode_index
        trace["trace_episode_index"] = self._mode_trace_episode_index
        trace["scenario_index"] = self._mode_trace_episode_index
        trace["episode_length"] = len(trace["left_motion_context"])
        trace["num_steps"] = len(trace["left_motion_context"])
        trace["terminated_at_end"] = bool(terminated_at_end)
        trace["truncated_at_end"] = bool(truncated_at_end)
        trace["trace_env_index"] = int(self.mode_trace_env_index)

        success_signal = [bool(value) for value in trace.get("success_signal", [])]
        terminal_info = dict(getattr(self, "_mode_trace_last_done", {}) or {})
        terminal_success = bool(terminal_info.get("success", False))
        trace["success"] = bool(terminal_success or any(success_signal))
        trace["success_step"] = next(
            (index for index, value in enumerate(success_signal) if value),
            trace["num_steps"] - 1 if terminal_success and trace["num_steps"] > 0 else None,
        )
        terminal_drop = bool(terminal_info.get("drop", False))
        terminal_far = bool(terminal_info.get("far", False))
        terminal_tilt_fail = bool(terminal_info.get("tilt_fail", False))
        terminal_invalid = bool(terminal_info.get("invalid", False))
        if trace["success"]:
            termination_reason = "success"
        elif terminal_drop or bool(any(bool(value) for value in trace.get("drop", []))):
            termination_reason = "drop"
        elif terminal_far or bool(any(bool(value) for value in trace.get("far", []))):
            termination_reason = "far"
        elif terminal_tilt_fail or bool(any(bool(value) for value in trace.get("tilt_fail", []))):
            termination_reason = "tilt_fail"
        elif terminal_invalid or bool(any(bool(value) for value in trace.get("invalid", []))):
            termination_reason = "invalid"
        elif bool(truncated_at_end):
            termination_reason = "timeout"
        else:
            termination_reason = "unknown"
        trace["termination_reason"] = termination_reason
        trace["terminal_success"] = terminal_success
        trace["diagnostic_mask"] = self.diagnostic_mask
        trace["terminal_flags"] = {
            "success": terminal_success,
            "drop": terminal_drop,
            "far": terminal_far,
            "tilt_fail": terminal_tilt_fail,
            "invalid": terminal_invalid,
        }

        # Agent/MAPPO owns the architecture debug values. Read through the public
        # getter so train and eval trace metadata stay identical.
        agent_debug = self.agents.get_motion_context_debug() if hasattr(self.agents, "get_motion_context_debug") else {}
        left_debug = agent_debug.get("left_arm", {})
        right_debug = agent_debug.get("right_arm", {})
        shared_intent_encoder = bool(
            left_debug.get("shared_intent_encoder", False) or right_debug.get("shared_intent_encoder", False)
        )
        context_scale = [0.0, 0.0, 0.0]
        try:
            raw_scale = self._resolve_env_attr("_motion_context_running_scale", None)
            if isinstance(raw_scale, torch.Tensor) and raw_scale.numel() >= 3:
                context_scale = raw_scale[:3].detach().float().cpu().tolist()
        except Exception:
            context_scale = [0.0, 0.0, 0.0]

        def _debug_or_env_float(key: str, env_attr: str, default: float) -> float:
            """left/right agent debug에서 값을 읽고, 없으면 env cfg로 fallback한다."""
            if key in left_debug:
                try:
                    return float(left_debug.get(key, default))
                except Exception:
                    pass
            if key in right_debug:
                try:
                    return float(right_debug.get(key, default))
                except Exception:
                    pass
            try:
                value = self._resolve_env_attr(env_attr, default)
                return float(default if value is None else value)
            except Exception:
                return float(default)

        def _env_int(attr: str, default: int = 0) -> int:
            try:
                value = self._resolve_env_attr(attr, default)
                return int(default if value is None else value)
            except Exception:
                return int(default)

        def _env_float(attr: str, default: float = 0.0) -> float:
            try:
                value = self._resolve_env_attr(attr, default)
                return float(default if value is None else value)
            except Exception:
                return float(default)

        def _env_str(attr: str, default: str = "") -> str:
            try:
                value = self._resolve_env_attr(attr, default)
                return str(value if value is not None else default)
            except Exception:
                return str(default)

        env_cfg = self._resolve_env_attr("cfg", None)
        trace_seed = int(getattr(env_cfg, "seed", 0) or 0) if env_cfg is not None else 0

        communication_mode = _env_str("communication_mode", "")
        context_representation = _env_str("motion_context_representation", "trend")
        context_layout = list(motion_context_layout(context_representation))
        communication_slot_dim = _env_int("communication_feature_dim", _env_int("intent_feature_dim", 0))
        communication_payload_capacity_dim = min(
            _env_int("own_observation_dim", 30)
            if communication_mode == "full_partner_observation"
            else int(_COMMUNICATION_PAYLOAD_DIMS.get(communication_mode, communication_slot_dim)),
            communication_slot_dim,
        )
        communication_payload_dim = (
            0 if _env_str("sharing_mode", "") == "no_share" else communication_payload_capacity_dim
        )
        trace["metadata"] = {
            "trace_schema": "openarm_motion_context/v3",
            "timeline_convention": "state_message_action_t_reward_t_to_t_plus_1",
            "communication_delay_steps": 1,
            "seed": trace_seed,
            "training_seed": self.cfg.get("mode_trace_training_seed"),
            "evaluation_seed": trace_seed,
            "context_dim": _env_int("motion_context_dim", 0),
            "motion_intent_dim": _env_int("motion_intent_dim", 0),
            "communication_mode": communication_mode,
            "diagnostic_mask": self.diagnostic_mask,
            "communication_feature_dim": communication_slot_dim,
            "communication_slot_dim": communication_slot_dim,
            "communication_payload_dim": communication_payload_dim,
            "communication_payload_capacity_dim": communication_payload_capacity_dim,
            "communication_payload_layout": (
                _communication_layout(communication_mode, context_representation) if communication_payload_dim else []
            ),
            "intent_feature_dim": _env_int("intent_feature_dim", 0),
            "intent_dim": _env_int("intent_feature_dim", 0),
            "own_obs_dim": _env_int("own_observation_dim", 0),
            "actor_input_dim": _env_int("actor_input_dim", 0),
            "motion_intent_horizon": _env_int("motion_intent_horizon", 0),
            "motion_context_representation": context_representation,
            "legacy_trend_trace_fields_valid": context_representation == "trend",
            "motion_context_layout": ["signed_motion_x", "signed_motion_y", "signed_motion_z", *context_layout],
            "env_index": int(self.mode_trace_env_index),
            "trace_env_index": int(self.mode_trace_env_index),
            "trace_episode_index": int(self._mode_trace_episode_index),
            "scenario_index": int(self._mode_trace_episode_index),
            "num_steps": int(trace["num_steps"]),
            "terminated_at_end": bool(terminated_at_end),
            "truncated_at_end": bool(truncated_at_end),
            "termination_reason": termination_reason,
            "task": self.mode_trace_task,
            "checkpoint": self.mode_trace_checkpoint,
            "policy_warmstart": getattr(self, "policy_warmstart_manifest", None),
            "sharing_mode": _env_str("sharing_mode", ""),
            "method": communication_mode,
            "method_name": "Proprioceptive Motion Context Sharing",
            "reward": _env_str("intent_task_label", "openarm_task"),
            "context_normalization": {
                "trend": "relative_ema_trend",
                "absolute_fixed": "fixed_physical_scale",
                "signed_twist": "fixed_angular_velocity_scale",
            }[context_representation],
            "context_dt": _env_float("step_dt", 1.0),
            "context_layout": context_layout,
            "legacy_context_aliases": {
                "linear_activity": "translation_trend",
                "angular_activity": "rotation_trend",
                "motion_smoothness": "stability_trend",
            }
            if context_representation == "trend"
            else {},
            "context_trend_horizon": _env_int("motion_context_trend_horizon", 5),
            "context_trend_beta": _env_float("motion_context_trend_beta", 0.80),
            "context_trend_linear_epsilon": _env_float("motion_context_trend_linear_epsilon", 0.01),
            "context_trend_angular_epsilon": _env_float("motion_context_trend_angular_epsilon", 0.05),
            "context_trend_action_epsilon": _env_float("motion_context_trend_action_epsilon", 0.01),
            "context_twist_angular_scale": _env_float("motion_context_twist_ang_scale", 3.0),
            "context_update_scale": False,
            "context_calibration_enabled": False,
            "context_calibration_window": _env_int("motion_context_calibration_window", 256),
            "context_calibration_cv_threshold": _env_float("motion_context_calibration_cv_threshold", 0.02),
            "context_calibration_confirmations": _env_int("motion_context_calibration_confirmations", 64),
            "context_calibration_epsilon": _env_float("motion_context_calibration_epsilon", 1.0e-6),
            "context_calibrated": bool(self._resolve_env_attr("_motion_context_calibrated", False)),
            "context_running_scale": context_scale,
            "success_lateral_threshold": _env_float("success_lateral_threshold", 0.012),
            "success_axis_threshold": _env_float("success_axis_threshold", 0.85),
            "success_depth_threshold": _env_float("success_depth_threshold", 0.045),
            "target_insertion_depth": _env_float("target_insertion_depth", 0.05),
            "insertion_depth_limit": _env_float("insertion_depth_limit", 0.0),
            "success_radial_clearance": _env_float("success_radial_clearance", 0.0),
            "peg_collision_radius": _env_float("peg_collision_radius", 0.0),
            "success_depth_overrun_tolerance": _env_float("success_depth_overrun_tolerance", 0.0),
            "success_protocol": str(self._resolve_env_attr("success_protocol", "legacy")),
            "success_depth_ratio": _env_float("success_depth_ratio", 0.90),
            "rotational_symmetry_order": _env_int("rotational_symmetry_order", 0),
            "wall_penetration_lateral_threshold": _env_float("wall_penetration_lateral_threshold", 0.015),
            "wall_penetration_depth_threshold": _env_float("wall_penetration_depth_threshold", 0.0),
            "max_joint_target_error_rad": _env_float("max_joint_target_error_rad", 0.0),
            "reward_axis_alignment_scale": _env_float("reward_axis_alignment_scale", 0.0),
            "alignment_shaping_power": _env_float("alignment_shaping_power", 1.0),
            "depth_overrun_std": _env_float("depth_overrun_std", 0.0),
            "reward_insert_settle_scale": _env_float("reward_insert_settle_scale", 0.0),
            "insert_settle_axis_width": _env_float("insert_settle_axis_width", 0.0),
            "insert_settle_speed_std": _env_float("insert_settle_speed_std", 0.0),
            "reward_lateral_alignment_scale": _env_float("reward_lateral_alignment_scale", 0.0),
            "lateral_alignment_std": _env_float("lateral_alignment_std", 0.04),
            "reset_arm_joint_noise_rad": _env_float("reset_arm_joint_noise_rad", 0.0),
            "experimental_reset_fixed_objects_to_tcp": bool(
                self._resolve_env_attr("experimental_reset_fixed_objects_to_tcp", False)
            ),
            "shared_intent_encoder": shared_intent_encoder,
            "context_classifier": False,
            "auxiliary_loss": False,
            "actor_backbone_shared": False,
            "action_head_shared": False,
            "partner_message_embed_dim": int(
                left_debug.get("partner_message_embed_dim", right_debug.get("partner_message_embed_dim", 0))
            ),
            "partner_intent_embed_dim": int(
                left_debug.get("partner_intent_embed_dim", right_debug.get("partner_intent_embed_dim", 0))
            ),
            "actor_action_input_dim": int(
                left_debug.get("actor_action_input_dim", right_debug.get("actor_action_input_dim", 0))
            ),
            "gripper_head_input_dim": int(
                left_debug.get("gripper_head_input_dim", right_debug.get("gripper_head_input_dim", 0))
            ),
            "architecture": ("agent_specific_actor_partner_context_encoder" if shared_intent_encoder else "no_intent"),
            "actor_input_excludes_contact_reward_success_object_z": True,
            "policy_mean_action_semantics": "pre-clamp latent Gaussian mean; not sampled executed actions",
            "deterministic_eval": not bool(getattr(self, "stochastic_evaluation", True)),
        }

        if "lift" in str(self.mode_trace_task).lower():
            trace["metadata"].update(
                {
                    "lift_target_height": _env_float("lift_target_height", 0.10),
                    "success_height_margin": _env_float("success_height_margin", 0.10),
                    "success_grasp_distance": _env_float("success_grasp_distance", 0.08),
                    "success_grasp_orientation_deg": _env_float("success_grasp_orientation_deg", 30.0),
                    "success_min_closure": _env_float("success_min_closure", 0.30),
                    "success_tilt_threshold_deg": _env_float("success_tilt_threshold_deg", 10.0),
                    "hold_required_steps": _env_int("hold_required_steps", 60),
                    "actor_object_pose_source": _env_str("actor_object_pose_source", ""),
                    "apriltag_measurement_source": _env_str("apriltag_measurement_source", ""),
                    "camera_extra_observation_dim": _env_int("camera_extra_observation_dim", 0),
                }
            )

        if bool(self._resolve_env_attr("require_contact_grasp", False)):
            trace["metadata"].update(
                {
                    "grasp_attachment_mode": "contact_friction",
                    "initialization": "pregrasp_on_reset",
                    "grasp_success_distance": _env_float("grasp_success_distance", 0.020),
                    "grasp_drop_distance": _env_float("grasp_drop_distance", 0.040),
                    "grasp_validity_source": "grasp_point_distance_and_closed_command_not_contact_force",
                }
            )

        if "ball_position_box" in trace:
            trace["metadata"].update(
                {
                    "ball_observation_source": "rendered_axial_depth_and_oracle_semantic_image_mask",
                    "ball_pose_ground_truth_in_actor": False,
                    "ball_camera_position_scale": _env_float("ball_camera_position_scale", 0.20),
                    "reward_ball_center_scale": _env_float("reward_ball_center_scale", 2.0),
                    "ball_reward_profile": _env_str("ball_reward_profile", "original"),
                    "reward_grasp_scale": _env_float("reward_grasp_scale", 5.0),
                    "reward_lift_scale": _env_float("reward_lift_scale", 20.0),
                    "ball_height_bridge_scale": _env_float("ball_height_bridge_scale", 0.0),
                    "ball_upright_bridge_scale": _env_float("ball_upright_bridge_scale", 0.0),
                    "ball_upright_decay_deg": _env_float("ball_upright_decay_deg", 20.0),
                    "ball_center_penalty_scale": _env_float("ball_center_penalty_scale", 0.0),
                    "ball_center_success_threshold": _env_float("ball_center_success_threshold", 0.05),
                    "ball_mass_kg": _env_float("expected_ball_mass_kg", 2.102),
                    "lift_success_definition_unchanged": True,
                    "ball_center_success_is_separate_metric": True,
                }
            )

        with open(path, "w", encoding="utf-8") as f:
            json.dump(trace, f, indent=2)

        print(
            f"[TRACE] Saved scenario trace {self._mode_trace_episode_index:03d}: "
            f"steps={trace['num_steps']} terminated={bool(terminated_at_end)} "
            f"truncated={bool(truncated_at_end)} path={path}"
        )

        self._mode_trace_episode_index += 1
        self._mode_trace = self._new_mode_trace()
        self._mode_trace_last_done = {"terminated": False, "truncated": False}
        return True

    @staticmethod
    def _stack_agent_tensors(values, dtype=None) -> torch.Tensor | None:
        if isinstance(values, dict):
            tensors = [v for v in values.values() if isinstance(v, torch.Tensor)]
            if not tensors:
                return None
            if dtype is torch.bool:
                flat = [v.detach().reshape(v.shape[0], -1).bool().any(dim=-1) for v in tensors]
                out = torch.stack(flat, dim=0).any(dim=0)
            else:
                flat = [v.detach().float().reshape(v.shape[0], -1).mean(dim=-1) for v in tensors]
                out = torch.stack(flat, dim=0).mean(dim=0)
        elif isinstance(values, torch.Tensor):
            if dtype is torch.bool:
                out = values.detach().reshape(values.shape[0], -1).bool().any(dim=-1)
            else:
                out = values.detach().float().reshape(values.shape[0], -1).mean(dim=-1)
        else:
            return None
        return out.to(dtype=dtype) if dtype is not None else out

    def _ctx_mask(self, name: str, width: int, device) -> torch.Tensor:
        ctx = self._resolve_env_attr("_ctx", {})
        value = ctx.get(name) if isinstance(ctx, dict) else None
        if isinstance(value, torch.Tensor) and value.numel() >= width:
            return value.detach().reshape(-1)[:width].to(device=device, dtype=torch.bool)
        return torch.zeros(width, device=device, dtype=torch.bool)

    def _update_episode_history(self, rewards, terminated, truncated) -> None:
        reward = self._stack_agent_tensors(rewards, dtype=torch.float32)
        done = self._stack_agent_tensors(terminated, dtype=torch.bool)
        trunc = self._stack_agent_tensors(truncated, dtype=torch.bool)
        if reward is None:
            return
        num_envs = int(reward.shape[0])
        device = reward.device
        if done is None:
            done = torch.zeros(num_envs, device=device, dtype=torch.bool)
        if trunc is None:
            trunc = torch.zeros(num_envs, device=device, dtype=torch.bool)
        done = done.to(device=device, dtype=torch.bool)
        trunc = trunc.to(device=device, dtype=torch.bool)

        if self._episode_returns is None or self._episode_returns.numel() != num_envs:
            self._episode_returns = torch.zeros(num_envs, device=device)
            self._episode_lengths = torch.zeros(num_envs, device=device, dtype=torch.long)

        self._episode_returns += reward
        self._episode_lengths += 1
        finished = done | trunc
        if not bool(finished.any()):
            return

        success = self._ctx_mask("termination_success", num_envs, device)
        drop = self._ctx_mask("termination_drop", num_envs, device)
        far = self._ctx_mask("termination_far", num_envs, device)
        tilt = self._ctx_mask("termination_tilt_fail", num_envs, device)
        invalid = self._ctx_mask("termination_invalid", num_envs, device)
        side_wall = self._ctx_mask("termination_side_wall", num_envs, device)
        idxs = torch.nonzero(finished, as_tuple=False).flatten()
        for idx in idxs.detach().cpu().tolist():
            timeout = bool(trunc[idx].detach().cpu()) and not bool(done[idx].detach().cpu())
            self._episode_history.append(
                {
                    "return": float(self._episode_returns[idx].detach().cpu()),
                    "length": int(self._episode_lengths[idx].detach().cpu()),
                    "success": float(success[idx].detach().cpu()),
                    "timeout": float(timeout),
                    "drop": float(drop[idx].detach().cpu()),
                    "far": float(far[idx].detach().cpu()),
                    "tilt_fail": float(tilt[idx].detach().cpu()),
                    "invalid": float((invalid[idx] | side_wall[idx]).detach().cpu()),
                }
            )
        self._episode_returns[finished] = 0.0
        self._episode_lengths[finished] = 0

    def _episode_summary(self) -> dict[str, float]:
        if not self._episode_history:
            return {}
        rows = list(self._episode_history)
        count = float(len(rows))

        def _mean(key: str) -> float:
            return sum(float(row.get(key, 0.0)) for row in rows) / max(count, 1.0)

        return {
            "count": count,
            "success_rate": _mean("success"),
            "mean_return": _mean("return"),
            "mean_length": _mean("length"),
            "timeout": _mean("timeout"),
            "drop": _mean("drop"),
            "far": _mean("far"),
            "tilt_fail": _mean("tilt_fail"),
            "invalid": _mean("invalid"),
        }

    def _emit_debug_summary(self, timestep: int, actions, outputs, infos) -> None:
        """task/reward/grip/context 상태를 사람이 읽기 쉬운 형태로 출력한다."""

        if not self.debug_interval or timestep % self.debug_interval != 0:
            return

        warmup_steps = self._intent_share_warmup_timesteps()
        warmup_active = warmup_steps > 0 and timestep < warmup_steps
        sharing_mode = self._resolve_env_attr("sharing_mode", "unknown")
        communication_mode = str(self._resolve_env_attr("communication_mode", "motion_context") or "motion_context")
        communication_enabled = bool(
            self._resolve_env_attr("communication_enabled", communication_mode != "none" and sharing_mode != "no_share")
        )
        own_dim = int(self._resolve_env_attr("own_observation_dim", 0) or 0)
        slot_dim = int(
            self._resolve_env_attr("communication_feature_dim", self._resolve_env_attr("intent_feature_dim", 0)) or 0
        )
        motion_dim = int(self._resolve_env_attr("motion_intent_dim", 3) or 3)
        context_dim = int(self._resolve_env_attr("motion_context_dim", 3) or 0)
        if communication_mode == "none":
            context_dim = 0
        payload_dim = min(
            own_dim
            if communication_mode == "full_partner_observation"
            else int(_COMMUNICATION_PAYLOAD_DIMS.get(communication_mode, slot_dim)),
            slot_dim,
        )
        actor_in_dim = own_dim + slot_dim
        phase = "eval" if self._in_eval else "train"
        log_info = infos.get("log", {})
        context_debug = (
            self.agents.get_motion_context_debug() if hasattr(self.agents, "get_motion_context_debug") else {}
        )
        left_debug = context_debug.get("left_arm", {})
        right_debug = context_debug.get("right_arm", {})
        if communication_mode == "none":
            arch_line = "actor=agent_specific partner_context=disabled action_head=agent_specific"
        else:
            arch_line = "actor=agent_specific partner_context_encoder=agent_specific context_classifier=disabled action_head=agent_specific"

        task_label = str(self._resolve_env_attr("intent_task_label", "openarm_task"))
        print(f"[{phase} {timestep:05d}] task={task_label} communication={communication_mode} sharing={sharing_mode}")
        if not getattr(self, "_paper_static_debug_printed", False):
            print(f"  arch: {arch_line}")
            print(
                "  sharing: "
                f"warmup={'on' if warmup_active else 'off'}({warmup_steps}) "
                f"communication={'on' if communication_enabled else 'off'} "
                "strategy=direct_context"
            )
            print(
                f"  dims: own={own_dim} payload={payload_dim} slot={slot_dim} "
                f"motion={motion_dim if communication_mode in ('motion_only', 'motion_context') else 0} "
                f"context={context_dim if communication_mode in ('context_only', 'motion_context') else 0} "
                f"actor_in={actor_in_dim}"
            )
            self._paper_static_debug_printed = True

        debug_verbose = bool(self.cfg.get("debug_verbose", False)) or bool(
            self._resolve_env_attr("debug_verbose", False)
        )
        if not debug_verbose:

            def _log(name: str, default: float = 0.0) -> float:
                return self._scalar(log_info.get(name), default=default)

            def _has_log(name: str) -> bool:
                return name in log_info

            def _print_action_debug() -> None:
                print(
                    "  action_debug:\n"
                    f"    L(abs_max={_log('left_action_abs_max'):.2f}, mean_abs={_log('left_action_mean_abs'):.2f}, "
                    f"sat={_log('left_action_saturation_ratio'):.2f}, "
                    f"target_err={_log('left_joint_target_error_norm'):.2f}, "
                    f"target_err_max={_log('left_joint_target_error_abs_max'):.2f})\n"
                    f"    R(abs_max={_log('right_action_abs_max'):.2f}, mean_abs={_log('right_action_mean_abs'):.2f}, "
                    f"sat={_log('right_action_saturation_ratio'):.2f}, "
                    f"target_err={_log('right_joint_target_error_norm'):.2f}, "
                    f"target_err_max={_log('right_joint_target_error_abs_max'):.2f})"
                )

            episode_summary = self._episode_summary()
            if episode_summary:
                print(
                    "  episode(last100)\n"
                    f"    result : n={episode_summary['count']:.0f} "
                    f"success={episode_summary['success_rate']:.2f} "
                    f"return={episode_summary['mean_return']:.1f} "
                    f"length={episode_summary['mean_length']:.1f}\n"
                    f"    end    : timeout={episode_summary['timeout']:.2f} "
                    f"drop={episode_summary['drop']:.2f} "
                    f"far={episode_summary['far']:.2f} "
                    f"tilt={episode_summary['tilt_fail']:.2f} "
                    f"invalid={episode_summary['invalid']:.2f}"
                )

            def _message_slice(agent: str, start: int, width: int) -> list[float]:
                buffer = self._resolve_env_attr("communication_buffer", {})
                if not isinstance(buffer, dict):
                    return [0.0] * width
                value = buffer.get(f"{agent}_intent")
                if not isinstance(value, torch.Tensor) or value.ndim < 2:
                    return [0.0] * width
                row_tensor = value.detach().float().reshape(value.shape[0], -1)[0, start : start + width]
                row = row_tensor.cpu().tolist()
                if len(row) < width:
                    row.extend([0.0] * (width - len(row)))
                return row

            if communication_mode == "none":
                print("  communication: disabled")
            else:
                left_payload = _message_slice("left", 0, payload_dim)
                right_payload = _message_slice("right", 0, payload_dim)
                left_payload_norm = float(torch.linalg.vector_norm(torch.tensor(left_payload))) if left_payload else 0.0
                right_payload_norm = (
                    float(torch.linalg.vector_norm(torch.tensor(right_payload))) if right_payload else 0.0
                )
                if communication_mode in ("motion_only", "motion_context", "motion_context_la", "motion_context_s"):
                    left_received_motion = f" motion={format_motion(right_payload[:motion_dim])}"
                    right_received_motion = f" motion={format_motion(left_payload[:motion_dim])}"
                else:
                    left_received_motion = ""
                    right_received_motion = ""
                if communication_mode in ("context_only", "motion_context", "motion_context_la", "motion_context_s"):
                    context_start = motion_dim if communication_mode != "context_only" else 0
                    left_received_context = right_payload[context_start : context_start + context_dim]
                    right_received_context = left_payload[context_start : context_start + context_dim]
                    context_names = motion_context_layout(
                        str(self._resolve_env_attr("motion_context_representation", "trend"))
                    )
                    left_received_context_text = (
                        " context=["
                        + ", ".join(
                            f"{name}={float(value):+.2f}" for name, value in zip(context_names, left_received_context)
                        )
                        + "]"
                    )
                    right_received_context_text = (
                        " context=["
                        + ", ".join(
                            f"{name}={float(value):+.2f}" for name, value in zip(context_names, right_received_context)
                        )
                        + "]"
                    )
                else:
                    left_received_context_text = ""
                    right_received_context_text = ""
                print(
                    f"  communication (payload={payload_dim}/{slot_dim})\n"
                    f"    L <- R : norm={right_payload_norm:.2f}"
                    f"{left_received_motion}{left_received_context_text}\n"
                    f"    R <- L : norm={left_payload_norm:.2f}"
                    f"{right_received_motion}{right_received_context_text}"
                )
                signals = {}
                if hasattr(self.env, "get_paper_motion_context_trace_signals"):
                    try:
                        signals = self.env.get_paper_motion_context_trace_signals(0)
                    except Exception:
                        signals = {}
                if (
                    communication_mode in ("context_only", "motion_context", "motion_context_la", "motion_context_s")
                    and signals
                ):
                    left_generated = signals.get("left_motion_context", [0.0, 0.0, 0.0])
                    right_generated = signals.get("right_motion_context", [0.0, 0.0, 0.0])
                    print(
                        "    generated: L=["
                        + ", ".join(f"{name}={float(value):+.2f}" for name, value in zip(context_names, left_generated))
                        + "] R=["
                        + ", ".join(
                            f"{name}={float(value):+.2f}" for name, value in zip(context_names, right_generated)
                        )
                        + "]"
                    )
                    print(
                        "    raw/ema : "
                        f"L=[v={signals.get('left_linear_speed_raw', 0.0):.3f},"
                        f"w={signals.get('left_angular_speed_raw', 0.0):.3f},"
                        f"da={signals.get('left_action_variation_raw', 0.0):.3f}/"
                        f"{signals.get('left_action_variation_ema', 0.0):.3f}] "
                        f"R=[v={signals.get('right_linear_speed_raw', 0.0):.3f},"
                        f"w={signals.get('right_angular_speed_raw', 0.0):.3f},"
                        f"da={signals.get('right_action_variation_raw', 0.0):.3f}/"
                        f"{signals.get('right_action_variation_ema', 0.0):.3f}]"
                    )
                raw_scale = self._resolve_env_attr("_motion_context_running_scale", None)
                if isinstance(raw_scale, torch.Tensor) and raw_scale.numel() >= 3:
                    scale = raw_scale[:3].detach().float().cpu().tolist()
                    calibrated = bool(self._resolve_env_attr("_motion_context_calibrated", False))
                    calibration_enabled = bool(self._resolve_env_attr("motion_context_calibration_enabled", True))
                    cv = self._resolve_env_attr("_motion_context_calibration_cv", None)
                    if isinstance(cv, torch.Tensor) and cv.numel() >= 3:
                        cv_values = cv[:3].detach().float().cpu().tolist()
                    else:
                        cv_values = [0.0, 0.0, 0.0]
                    history_count = int(self._resolve_env_attr("_motion_context_calibration_history_count", 0) or 0)
                    window = int(self._resolve_env_attr("motion_context_calibration_window", 256) or 256)
                    stable_count = int(self._resolve_env_attr("_motion_context_calibration_stable_count", 0) or 0)
                    confirmations = int(self._resolve_env_attr("motion_context_calibration_confirmations", 64) or 64)
                    motion_active = communication_mode in (
                        "motion_only",
                        "motion_context",
                        "motion_context_la",
                        "motion_context_s",
                    )
                    context_active = communication_mode in (
                        "context_only",
                        "motion_context",
                        "motion_context_la",
                        "motion_context_s",
                    ) and ((not calibration_enabled) or calibrated)
                    print(
                        "    scale  : "
                        f"lin={scale[0]:.3f} ang={scale[1]:.3f} action={scale[2]:.3f} "
                        f"frozen={bool(self._resolve_env_attr('_motion_context_scale_frozen', False))}"
                    )
                    print(
                        "    calibration: "
                        f"enabled={calibration_enabled} calibrated={calibrated} "
                        f"motion_active={motion_active} context_active={context_active}\n"
                        "      cv=["
                        f"lin={cv_values[0]:.5f},ang={cv_values[1]:.5f},action={cv_values[2]:.5f}] "
                        f"history={history_count}/{window} stable={stable_count}/{confirmations}"
                    )

            height_ok = _log("height_ok_ratio")
            tilt_ok = _log("tilt_ok_ratio")
            left_grasp_ok = _log("left_grasp_ok_ratio")
            right_grasp_ok = _log("right_grasp_ok_ratio")
            hold_ok = _log("hold_ok_ratio")
            if min(left_grasp_ok, right_grasp_ok) < 0.5:
                bottleneck = "grasp"
            elif height_ok < 0.5:
                bottleneck = "height"
            elif tilt_ok < 0.5:
                bottleneck = "tilt"
            elif hold_ok < 0.5:
                bottleneck = "hold"
            else:
                bottleneck = "none"

            if task_label.startswith("openarm_peg_in_hole"):
                success_rate = _log("peg_hole/success_rate")
                axis_align = _log("peg_hole/axis_alignment")
                axis_error_deg = _log("peg_hole/axis_error_deg")
                twist_align = _log("peg_hole/twist_alignment", 1.0)
                twist_error_deg = _log("peg_hole/twist_error_deg")
                lateral_error = _log("peg_hole/lateral_error")
                insertion_depth = _log("peg_hole/insertion_depth")
                target_depth = _log("peg_hole/target_insertion_depth", 0.025)
                axis_gate = _log("peg_hole/axis_gate", max(0.0, min(axis_align, 1.0)))
                lateral_gate = _log("peg_hole/lateral_gate", 1.0)
                twist_gate = _log("peg_hole/twist_gate", 1.0)
                insertion_gate = _log("peg_hole/insertion_gate", axis_gate * lateral_gate)
                depth_progress = _log(
                    "peg_hole/depth_progress",
                    max(0.0, min(insertion_depth / max(target_depth, 1.0e-6), 1.0)),
                )
                keypoint_dist = _log("peg_hole/keypoint_dist")
                lateral_threshold = _log("peg_hole/success_lateral_threshold", 0.012)
                axis_threshold = _log("peg_hole/success_axis_threshold", 0.85)
                depth_threshold = _log("peg_hole/success_depth_threshold", target_depth)
                preinsert_dist = _log("peg_hole/preinsert_dist")
                insert_pose_dist = _log("peg_hole/insert_pose_dist")
                strict_success = _log("strict_success_ratio", _log("success_ratio"))
                if axis_align < 0.0:
                    bottleneck = "axis_direction"
                elif preinsert_dist > 0.05:
                    bottleneck = "preinsert"
                elif axis_gate < 0.2:
                    bottleneck = "axis_alignment"
                elif twist_gate < 0.2:
                    bottleneck = "twist_alignment"
                elif lateral_gate < 0.2:
                    bottleneck = "lateral"
                elif depth_progress < 0.8:
                    bottleneck = "insertion_depth"
                elif strict_success < 0.3:
                    bottleneck = "hold"
                else:
                    bottleneck = "solved"
                print(
                    "  stage_reward: "
                    f"team={_log('team_reward', _log('left_reward')):+.2f} "
                    f"preinsert={_log('peg_hole/r_preinsert'):.2f} "
                    f"insert_pose={_log('peg_hole/r_insert_pose'):.2f} "
                    f"depth={_log('peg_hole/r_depth'):.2f} "
                    f"success_bonus={success_rate:.2f} "
                    f"bottleneck={bottleneck}"
                )
                print(
                    "  peg_hole: "
                    f"preinsert_dist={preinsert_dist:.3f} "
                    f"insert_pose_dist={insert_pose_dist:.3f} "
                    f"kp_dist={keypoint_dist:.3f} "
                    f"kp_d0={_log('peg_hole/keypoint_dist_0'):.3f} "
                    f"kp_dlast={_log('peg_hole/keypoint_dist_last'):.3f} "
                    f"axis_align={axis_align:.3f}/{axis_threshold:.3f} "
                    f"axis_error={axis_error_deg:.1f}deg "
                    f"twist={twist_align:.3f} ({twist_error_deg:.1f}deg) "
                    f"kp_num={_log('peg_hole/keypoint_num'):.0f} "
                    f"kp_spacing={_log('peg_hole/keypoint_spacing'):.3f} "
                    f"target_depth={target_depth:.3f}"
                )
                print(
                    "  keypoint_attach: "
                    f"held_k0_err={_log('peg_hole/held_k0_error'):.6f} "
                    f"target_k0_err={_log('peg_hole/fixed_k0_error'):.6f} "
                    f"held_spacing={_log('peg_hole/held_keypoint_spacing'):.4f} "
                    f"target_spacing={_log('peg_hole/fixed_keypoint_spacing'):.4f}"
                )
                print(
                    "  object_attach: "
                    f"pre_sync_mean=peg:{_log('peg_hole/peg_attach_error'):.6f}/"
                    f"hole:{_log('peg_hole/hole_attach_error'):.6f}m "
                    f"max=peg:{_log('peg_hole/peg_attach_error_max'):.6f}/"
                    f"hole:{_log('peg_hole/hole_attach_error_max'):.6f}m"
                )
                print(
                    "  geometry_gate: "
                    f"tip_dist={_log('peg_hole/tip_dist'):.3f} "
                    f"lateral={lateral_error:.3f}/{lateral_threshold:.3f} "
                    f"depth={insertion_depth:.3f}/{depth_threshold:.3f} "
                    f"depth_progress={depth_progress:.2f} "
                    f"axis_gate={axis_gate:.2f} "
                    f"lateral_gate={lateral_gate:.2f} "
                    f"twist_gate={twist_gate:.2f} "
                    f"insertion_gate={insertion_gate:.2f} "
                    f"hold={_log('hold_count_mean'):.1f}/{_log('hold_required_steps'):.0f} "
                    f"step_scale={_log('action_scale_mean'):.3f} "
                    f"fine_ratio={_log('fine_control_ratio'):.2f}"
                )
                _print_action_debug()
            elif task_label.startswith("openarm_peg_in_hole"):
                success_rate = _log("peg_hole/success_rate")
                axis_align = _log("peg_hole/axis_alignment")
                lateral_error = _log("peg_hole/lateral_error")
                insertion_depth = _log("peg_hole/insertion_depth")
                target_depth = _log("peg_hole/target_insertion_depth", 0.025)
                dual_grasp_gate = _log("peg_hole/dual_grasp_gate")
                if dual_grasp_gate < _log("grasp_gate_threshold", 0.5):
                    bottleneck = "grasp"
                elif lateral_error > _log("lateral_gate_threshold", 0.015):
                    bottleneck = "lateral"
                elif axis_align < _log("axis_gate_threshold", 0.85):
                    bottleneck = "alignment"
                elif insertion_depth < target_depth:
                    bottleneck = "insert"
                elif success_rate < 0.5:
                    bottleneck = "hold/success"
                else:
                    bottleneck = "none"
                print(
                    "  reward: "
                    f"team={_log('team_reward', _log('left_reward')):+.2f} "
                    f"approach={_log('peg_hole/r_approach'):.2f} "
                    f"lateral={_log('peg_hole/r_lateral'):.2f} "
                    f"axis={_log('peg_hole/r_axis'):.2f} "
                    f"reach={_log('peg_hole/r_reach'):.2f} "
                    f"ori={_log('peg_hole/r_orientation'):.2f} "
                    f"grasp={_log('peg_hole/r_grasp'):.2f} "
                    f"insert={_log('peg_hole/r_insert'):.2f} "
                    f"success={success_rate:.2f} bottleneck={bottleneck}"
                )
                print(
                    "  peg_hole: "
                    f"tip_dist={_log('peg_hole/tip_dist'):.3f} "
                    f"lateral={lateral_error:.3f} "
                    f"axis_align={axis_align:.3f} "
                    f"depth={insertion_depth:.3f}/{target_depth:.3f} "
                    f"insert_gate={_log('peg_hole/insert_gate_ratio'):.2f} "
                    f"grasp_gate={dual_grasp_gate:.2f} "
                    f"obs_insert_ratio={_log('peg_hole/grasp_ready_obs_ratio'):.2f}"
                )
                print(
                    "  grasp_targets:\n"
                    f"    L(dist={_log('left_grasp_dist'):.3f}, near={_log('left_near_target'):.2f}, "
                    f"ori={_log('left_grasp_orientation'):.2f}, inside={_log('left_target_inside_gripper'):.2f}, "
                    f"closure={_log('left_closure'):.2f}, grasp={_log('left_grasp'):.2f})\n"
                    f"    R(dist={_log('right_grasp_dist'):.3f}, near={_log('right_near_target'):.2f}, "
                    f"ori={_log('right_grasp_orientation'):.2f}, inside={_log('right_target_inside_gripper'):.2f}, "
                    f"closure={_log('right_closure'):.2f}, grasp={_log('right_grasp'):.2f})"
                )
                _print_action_debug()
            elif task_label.startswith("openarm_lift") or task_label.startswith("openarm_re_"):
                left_grasp = _log("left_grasp_hint")
                right_grasp = _log("right_grasp_hint")
                grasp_ok = _log("success_grasp_ok_ratio")
                strict_success = _log("strict_success_ratio", _log("success_ratio"))
                if grasp_ok < 0.5:
                    bottleneck = "grasp"
                elif height_ok < 0.5:
                    bottleneck = "lift"
                elif tilt_ok < 0.5:
                    bottleneck = "tilt"
                elif strict_success < 0.5:
                    bottleneck = "hold"
                else:
                    bottleneck = "solved"
                print(
                    "  reward\n"
                    f"    total  : {_log('team_reward', _log('left_reward')):+.2f}\n"
                    f"    terms  : approach={_log('reward_approach_term', _log('reward_reach_term')):+.2f} "
                    f"grasp={_log('reward_grasp_term'):+.2f} "
                    f"lift={_log('reward_lift_term'):+.2f} "
                    f"success={_log('reward_success_term'):+.2f} "
                    f"action=-{_log('reward_action_term'):.4f} "
                    f"joint_vel=-{_log('reward_joint_vel_term'):.4f}"
                )
                print(
                    "  task state\n"
                    f"    grasp  : dist=L:{_log('left_actor_dist'):.3f}/R:{_log('right_actor_dist'):.3f}m "
                    f"score=L:{left_grasp:.2f}/R:{right_grasp:.2f}\n"
                    f"             ori_err=L:{_log('left_ori_err'):.2f}/R:{_log('right_ori_err'):.2f}rad "
                    f"closure=L:{_log('left_closure'):.2f}/R:{_log('right_closure'):.2f}\n"
                    f"             inside=L:{_log('left_target_inside_gripper'):.2f}/"
                    f"R:{_log('right_target_inside_gripper'):.2f} "
                    f"midline=L:{_log('left_target_to_gripper_midline_dist'):.3f}/"
                    f"R:{_log('right_target_to_gripper_midline_dist'):.3f}m\n"
                    f"             ready=L:{_log('left_pose_readiness'):.2f}/R:{_log('right_pose_readiness'):.2f} "
                    f"feas=L:{_log('left_grasp_feasibility'):.2f}/"
                    f"R:{_log('right_grasp_feasibility'):.2f}\n"
                    f"             "
                    f"early_close=L:{_log('left_premature_closure'):.2f}/"
                    f"R:{_log('right_premature_closure'):.2f}\n"
                    f"             valid=L[d:{_log('left_distance_ok_ratio'):.2f},"
                    f"o:{_log('left_orientation_ok_ratio'):.2f},"
                    f"c:{_log('left_closure_ok_ratio'):.2f},"
                    f"s:{_log('left_finger_span_ok_ratio'):.2f}] "
                    f"R[d:{_log('right_distance_ok_ratio'):.2f},"
                    f"o:{_log('right_orientation_ok_ratio'):.2f},"
                    f"c:{_log('right_closure_ok_ratio'):.2f},"
                    f"s:{_log('right_finger_span_ok_ratio'):.2f}]\n"
                    f"    lift   : dz(mean/max)={_log('object_height_delta'):.3f}/"
                    f"{_log('object_height_delta_max'):.3f}m "
                    f"target={float(self._resolve_env_attr('success_height_margin', 0.10)):.3f}m "
                    f"reached={height_ok:.2f} "
                    f"tilt={_log('object_tilt_deg'):.1f}/"
                    f"{float(self._resolve_env_attr('success_tilt_threshold_deg', 10.0)):.1f}deg\n"
                    f"    success: hold={_log('hold_count_max'):.0f}/{_log('hold_required_steps'):.0f} "
                    f"rate={strict_success:.2f} bottleneck={bottleneck}\n"
                    f"    april  : grasp_target_tf_error="
                    f"{max(_log('left_actor_target_pos_error_max'), _log('right_actor_target_pos_error_max')):.6f}m "
                    f"height_error={_log('object_height_estimation_error_max'):.6f}m"
                )
                if _has_log("ball_xy_error") or _has_log("ball_center_tracking"):
                    print(f"  ball: xy_err={_log('ball_xy_error'):.3f} center={_log('ball_center_tracking'):.2f}")
            else:
                print(
                    "  reward: "
                    f"L={_log('left_reward_total'):+.2f} R={_log('right_reward_total'):+.2f} "
                    f"grip=L:{_log('left_grasp_term'):+.2f}/R:{_log('right_grasp_term'):+.2f} "
                    f"lift=L:{_log('left_lift_term'):+.2f}/R:{_log('right_lift_term'):+.2f} "
                    f"success h={height_ok:.2f} g=L:{left_grasp_ok:.2f}/R:{right_grasp_ok:.2f} "
                    f"hold={hold_ok:.2f} bottleneck={bottleneck}"
                )
                print(
                    "  grasp:\n"
                    f"    L(act={_log('left_grip_action'):+.2f}, close={_log('left_closure'):.2f}, "
                    f"cmin={_log('left_contact_min'):.2f}, pose_gate={_log('left_grasp_distance_gate'):.2f}, "
                    f"local={_log('left_local_grasp', _log('left_grasp_reward')):.3f})\n"
                    f"    R(act={_log('right_grip_action'):+.2f}, close={_log('right_closure'):.2f}, "
                    f"cmin={_log('right_contact_min'):.2f}, pose_gate={_log('right_grasp_distance_gate'):.2f}, "
                    f"local={_log('right_local_grasp', _log('right_grasp_reward')):.3f})"
                )
                print(
                    "  object: "
                    f"dz={_log('object_height_delta'):.3f} hprog={_log('lift_gate'):.2f} "
                    f"tilt={_log('object_tilt_deg'):.1f}deg pen={_log('object_tilt_penalty_term'):+.2f} "
                    f"drop={_log('hold_height_drop'):.4f} xerr={_log('x_error', _log('xy_error')):.3f}"
                )
                if _has_log("ball_xy_error") or _has_log("ball_center_tracking"):
                    print(
                        "  ball: "
                        f"xy_err={_log('ball_xy_error'):.3f} "
                        f"center={_log('ball_center_tracking'):.2f} "
                        f"reward={_log('ball_center_reward'):.2f}"
                    )
            return
        context_names = motion_context_layout(str(self._resolve_env_attr("motion_context_representation", "trend")))
        if communication_mode == "none" or context_dim <= 0:
            print("  communication: disabled")
            print("    share_to_partner: disabled")
            print("    alignment: disabled")
        else:
            signals = {}
            if hasattr(self.env, "get_paper_motion_context_trace_signals"):
                try:
                    signals = self.env.get_paper_motion_context_trace_signals(0)
                except Exception:
                    signals = {}
            left_context = signals.get("left_motion_context", [0.0, 0.0, 0.0])
            right_context = signals.get("right_motion_context", [0.0, 0.0, 0.0])
            print("  communication:")
            print(
                f"    L self: motion={self._fmt_motion(outputs['left_arm'].get('motion_intent'))} "
                "context=["
                + ", ".join(f"{name}={float(value):+.2f}" for name, value in zip(context_names, left_context))
                + "]"
            )
            print(
                f"    R self: motion={self._fmt_motion(outputs['right_arm'].get('motion_intent'))} "
                "context=["
                + ", ".join(f"{name}={float(value):+.2f}" for name, value in zip(context_names, right_context))
                + "]"
            )
            print(
                "    raw/ema: "
                f"L=[v={float(signals.get('left_linear_speed_raw', 0.0)):.3f}, "
                f"w={float(signals.get('left_angular_speed_raw', 0.0)):.3f}, "
                f"da={float(signals.get('left_action_variation_raw', 0.0)):.3f}/"
                f"{float(signals.get('left_action_variation_ema', 0.0)):.3f}] "
                f"R=[v={float(signals.get('right_linear_speed_raw', 0.0)):.3f}, "
                f"w={float(signals.get('right_angular_speed_raw', 0.0)):.3f}, "
                f"da={float(signals.get('right_action_variation_raw', 0.0)):.3f}/"
                f"{float(signals.get('right_action_variation_ema', 0.0)):.3f}]"
            )
            print("    share_to_partner:")
            if not communication_enabled:
                print("      L->R z=[motion masked, context masked]")
                print("      R->L z=[motion masked, context masked]")
            else:
                print("      L->R z=[motion kept, context kept]")
                print("      R->L z=[motion kept, context kept]")
            print("    context_classifier: disabled")
        print(
            "  context_config:\n"
            f"    z=[signed_EE_motion_3D, {', '.join(context_names)}]\n"
            "    context_classifier=disabled auxiliary_loss=disabled warmup=disabled"
        )

        left_reward_total = self._scalar(log_info.get("left_reward_total"))
        right_reward_total = self._scalar(log_info.get("right_reward_total"))
        left_reach_term = self._scalar(log_info.get("left_reach_term"))
        right_reach_term = self._scalar(log_info.get("right_reach_term"))
        left_position_penalty_term = self._scalar(log_info.get("left_position_penalty_term"))
        right_position_penalty_term = self._scalar(log_info.get("right_position_penalty_term"))
        left_orientation_penalty_term = self._scalar(log_info.get("left_orientation_penalty_term"))
        right_orientation_penalty_term = self._scalar(log_info.get("right_orientation_penalty_term"))
        left_grasp_term = self._scalar(log_info.get("left_grasp_term"))
        right_grasp_term = self._scalar(log_info.get("right_grasp_term"))
        left_lift_term = self._scalar(log_info.get("left_lift_term"))
        right_lift_term = self._scalar(log_info.get("right_lift_term"))
        left_goal_term = self._scalar(log_info.get("left_goal_term"))
        right_goal_term = self._scalar(log_info.get("right_goal_term"))

        left_dist = self._scalar(log_info.get("left_dist"))
        right_dist = self._scalar(log_info.get("right_dist"))
        left_ori_err = self._scalar(log_info.get("left_ori_err"))
        right_ori_err = self._scalar(log_info.get("right_ori_err"))
        left_grasp_distance_gate = self._scalar(log_info.get("left_grasp_distance_gate"))
        right_grasp_distance_gate = self._scalar(log_info.get("right_grasp_distance_gate"))
        left_force_inner = self._scalar(log_info.get("left_grasp_force_inner"))
        left_force_outer = self._scalar(log_info.get("left_grasp_force_outer"))
        right_force_inner = self._scalar(log_info.get("right_grasp_force_inner"))
        right_force_outer = self._scalar(log_info.get("right_grasp_force_outer"))
        left_force_min = self._scalar(
            log_info.get("left_grasp_force_min"), default=min(left_force_inner, left_force_outer)
        )
        right_force_min = self._scalar(
            log_info.get("right_grasp_force_min"), default=min(right_force_inner, right_force_outer)
        )
        left_force_avg = self._scalar(
            log_info.get("left_grasp_force_avg"), default=0.5 * (left_force_inner + left_force_outer)
        )
        right_force_avg = self._scalar(
            log_info.get("right_grasp_force_avg"), default=0.5 * (right_force_inner + right_force_outer)
        )
        left_contact_min = self._scalar(log_info.get("left_contact_min"))
        right_contact_min = self._scalar(log_info.get("right_contact_min"))
        left_contact_avg = self._scalar(log_info.get("left_contact_avg"))
        right_contact_avg = self._scalar(log_info.get("right_contact_avg"))
        left_closure = self._scalar(log_info.get("left_closure"))
        right_closure = self._scalar(log_info.get("right_closure"))
        left_grip_action = self._scalar(log_info.get("left_grip_action"))
        right_grip_action = self._scalar(log_info.get("right_grip_action"))
        left_gripper_opening = self._scalar(log_info.get("left_gripper_opening"))
        right_gripper_opening = self._scalar(log_info.get("right_gripper_opening"))
        left_local_grasp = self._scalar(
            log_info.get("left_local_grasp"), default=self._scalar(log_info.get("left_grasp_reward"))
        )
        right_local_grasp = self._scalar(
            log_info.get("right_local_grasp"), default=self._scalar(log_info.get("right_grasp_reward"))
        )
        left_lift_reward_raw = self._scalar(log_info.get("left_lift_reward"))
        right_lift_reward_raw = self._scalar(log_info.get("right_lift_reward"))
        secure_local = self._scalar(
            log_info.get("secure_local"), default=self._scalar(log_info.get("secure_bilateral"))
        )

        left_obs_prev_gripper_action = self._scalar(log_info.get("left_obs_prev_gripper_action"))
        right_obs_prev_gripper_action = self._scalar(log_info.get("right_obs_prev_gripper_action"))
        left_obs_closure = self._scalar(log_info.get("left_obs_closure"))
        right_obs_closure = self._scalar(log_info.get("right_obs_closure"))
        left_obs_ee_lin_vel_norm = self._scalar(log_info.get("left_obs_ee_lin_vel_norm"))
        right_obs_ee_lin_vel_norm = self._scalar(log_info.get("right_obs_ee_lin_vel_norm"))
        left_set_close = self._scalar(log_info.get("left_set_close"))
        right_set_close = self._scalar(log_info.get("right_set_close"))
        left_set_open = self._scalar(log_info.get("left_set_open"))
        right_set_open = self._scalar(log_info.get("right_set_open"))
        left_can_close = self._scalar(log_info.get("left_can_close"))
        right_can_close = self._scalar(log_info.get("right_can_close"))
        left_close_x_gate = self._scalar(log_info.get("left_close_x_gate"))
        left_close_y_gate = self._scalar(log_info.get("left_close_y_gate"))
        left_close_z_gate = self._scalar(log_info.get("left_close_z_gate"))
        right_close_x_gate = self._scalar(log_info.get("right_close_x_gate"))
        right_close_y_gate = self._scalar(log_info.get("right_close_y_gate"))
        right_close_z_gate = self._scalar(log_info.get("right_close_z_gate"))
        left_close_eps = self._scalar(log_info.get("left_close_eps"), default=0.0)
        right_close_eps = self._scalar(log_info.get("right_close_eps"), default=0.0)
        left_open_eps = self._scalar(log_info.get("left_open_eps"), default=0.0)
        right_open_eps = self._scalar(log_info.get("right_open_eps"), default=0.0)

        object_height = self._scalar(log_info.get("object_height"))
        object_height_delta = self._scalar(log_info.get("object_height_delta"))
        lift_gate = self._scalar(log_info.get("lift_gate"))
        object_tilt_deg = self._scalar(log_info.get("object_tilt_deg"))
        success_roll_deg = self._scalar(log_info.get("success_roll_deg"))
        success_pitch_deg = self._scalar(log_info.get("success_pitch_deg"))
        success_roll_pitch_deg = self._scalar(
            log_info.get("success_roll_pitch_deg"),
            default=max(success_roll_deg, success_pitch_deg),
        )
        success_yaw_deg = self._scalar(log_info.get("success_yaw_deg"))
        object_tilt_excess = self._scalar(log_info.get("object_tilt_excess"))
        object_tilt_penalty = self._scalar(log_info.get("object_tilt_penalty_term"))
        target_delta = self._scalar(log_info.get("target_delta"))
        x_error = self._scalar(log_info.get("x_error"), default=self._scalar(log_info.get("xy_error")))
        goal_tracking = self._scalar(log_info.get("goal_tracking"))
        hold_active_gate = self._scalar(log_info.get("hold_active_gate"))
        hold_height_drop = self._scalar(log_info.get("hold_height_drop"))
        hold_height_drop_penalty = self._scalar(log_info.get("hold_height_drop_penalty"))
        height_ok = self._scalar(log_info.get("height_ok_ratio"))
        tilt_ok = self._scalar(log_info.get("tilt_ok_ratio"))
        left_grasp_ok = self._scalar(log_info.get("left_grasp_ok_ratio"))
        right_grasp_ok = self._scalar(log_info.get("right_grasp_ok_ratio"))
        dual_grasp_ok = self._scalar(log_info.get("dual_grasp_ok_ratio"))
        hold_ok = self._scalar(log_info.get("hold_ok_ratio"))
        strict_success = self._scalar(log_info.get("strict_success_ratio"))
        secure_bi_min = self._scalar(log_info.get("secure_bi_min"), default=min(left_local_grasp, right_local_grasp))
        secure_bi_sqrt = self._scalar(log_info.get("secure_bi_sqrt"))
        grasp_imbalance = self._scalar(
            log_info.get("grasp_imbalance"), default=abs(left_local_grasp - right_local_grasp)
        )
        reward_gap = self._scalar(log_info.get("reward_gap"), default=abs(left_reward_total - right_reward_total))
        closure_gap = self._scalar(log_info.get("left_right_closure_gap"), default=abs(left_closure - right_closure))
        contact_gap = self._scalar(
            log_info.get("left_right_contact_gap"), default=abs(left_contact_min - right_contact_min)
        )

        if dual_grasp_ok < 0.5:
            bottleneck = "dual_grasp"
        elif height_ok < 0.5:
            bottleneck = "height"
        elif tilt_ok < 0.5:
            bottleneck = "tilt"
        elif hold_ok < 0.5:
            bottleneck = "hold"
        else:
            bottleneck = "none"

        print(
            "  task_success:\n"
            f"    height_ok={height_ok:.3f} tilt_ok={tilt_ok:.3f} "
            f"L_grasp_ok={left_grasp_ok:.3f} R_grasp_ok={right_grasp_ok:.3f} "
            f"dual_grasp_ok={dual_grasp_ok:.3f} hold_ok={hold_ok:.3f} strict_success={strict_success:.3f}\n"
            f"    bottleneck={bottleneck}"
        )
        print(
            "  bimanual_balance:\n"
            f"    secure_min={secure_bi_min:.3f} secure_sqrt={secure_bi_sqrt:.3f}\n"
            f"    grasp_imbalance={grasp_imbalance:.3f} reward_gap={reward_gap:.3f} "
            f"closure_gap={closure_gap:.3f} contact_gap={contact_gap:.3f}"
        )
        print(
            "  lift_stability:\n"
            f"    obj_z={object_height:.3f} dz={object_height_delta:.3f} hprog={lift_gate:.3f}\n"
            f"    tilt={object_tilt_deg:.1f}deg roll_pitch={success_roll_pitch_deg:.1f}deg "
            f"yaw={success_yaw_deg:.1f}deg tilt_excess={object_tilt_excess:.3f} tilt_pen={object_tilt_penalty:+.3f}\n"
            f"    hold_gate={hold_active_gate:.3f} height_drop={hold_height_drop:.4f} drop_pen={-hold_height_drop_penalty:+.3f}\n"
            f"    obj_x_err={x_error:.3f} x_center_track={goal_tracking:.3f} goal_err={target_delta:.3f}"
        )
        print(
            "  reward_terms:\n"
            f"    L(total={left_reward_total:+.3f}, reach={left_reach_term:+.3f}, pos_pen={left_position_penalty_term:+.3f}, "
            f"ori_pen={left_orientation_penalty_term:+.3f}, grip={left_grasp_term:+.3f}, "
            f"lift={left_lift_term:+.3f}, xy_center={left_goal_term:+.3f})\n"
            f"    R(total={right_reward_total:+.3f}, reach={right_reach_term:+.3f}, pos_pen={right_position_penalty_term:+.3f}, "
            f"ori_pen={right_orientation_penalty_term:+.3f}, grip={right_grasp_term:+.3f}, "
            f"lift={right_lift_term:+.3f}, xy_center={right_goal_term:+.3f})"
        )

        print(
            "  reward: "
            f"L(total={left_reward_total:+.3f}, reach={left_reach_term:+.3f}, pos={left_position_penalty_term:+.3f}, "
            f"ori={left_orientation_penalty_term:+.3f}, grip={left_grasp_term:+.3f}, "
            f"lift={left_lift_term:+.3f}, xy_center={left_goal_term:+.3f}) | "
            f"R(total={right_reward_total:+.3f}, reach={right_reach_term:+.3f}, pos={right_position_penalty_term:+.3f}, "
            f"ori={right_orientation_penalty_term:+.3f}, grip={right_grasp_term:+.3f}, "
            f"lift={right_lift_term:+.3f}, xy_center={right_goal_term:+.3f})"
        )
        print(
            "  position: "
            f"L(dist={left_dist:.3f}, ori_err={left_ori_err:.3f}, "
            f"grasp_pose_gate={left_grasp_distance_gate:.3f}) | "
            f"R(dist={right_dist:.3f}, ori_err={right_ori_err:.3f}, "
            f"grasp_pose_gate={right_grasp_distance_gate:.3f})"
        )
        print(
            "  obs_extra:\n"
            f"    L(closure={left_obs_closure:.3f}, ee_v={left_obs_ee_lin_vel_norm:.3f})\n"
            f"    R(closure={right_obs_closure:.3f}, ee_v={right_obs_ee_lin_vel_norm:.3f})"
        )
        print(
            "  grip_debug:\n"
            f"    L(action={left_grip_action:+.2f}, can_close={left_can_close:.3f}, "
            f"set_close={left_set_close:.3f}, set_open={left_set_open:.3f}, "
            f"close_eps={left_close_eps:.2f}, open_eps={left_open_eps:.2f}, "
            f"x_gate={left_close_x_gate:.3f}, y_gate={left_close_y_gate:.3f}, z_gate={left_close_z_gate:.3f}, "
            f"closure={left_closure:.3f}, prev_act={left_obs_prev_gripper_action:+.3f})\n"
            f"    R(action={right_grip_action:+.2f}, can_close={right_can_close:.3f}, "
            f"set_close={right_set_close:.3f}, set_open={right_set_open:.3f}, "
            f"close_eps={right_close_eps:.2f}, open_eps={right_open_eps:.2f}, "
            f"x_gate={right_close_x_gate:.3f}, y_gate={right_close_y_gate:.3f}, z_gate={right_close_z_gate:.3f}, "
            f"closure={right_closure:.3f}, prev_act={right_obs_prev_gripper_action:+.3f})"
        )
        print(
            "  grasp:\n"
            f"    L(close={left_set_close:.3f}, open={left_set_open:.3f}, action={left_grip_action:+.2f}, "
            f"contact_min={left_contact_min:.3f}, force_min={left_force_min:.1f}, "
            f"grip_score/local={left_local_grasp:.3f}, lift_rew={left_lift_reward_raw:.3f})\n"
            f"    R(close={right_set_close:.3f}, open={right_set_open:.3f}, action={right_grip_action:+.2f}, "
            f"contact_min={right_contact_min:.3f}, force_min={right_force_min:.1f}, "
            f"grip_score/local={right_local_grasp:.3f}, lift_rew={right_lift_reward_raw:.3f})\n"
            f"    dual_grasp_signal={secure_bi_sqrt:.3f} secure_local={secure_local:.3f}"
        )
        print("  grip:")
        print(
            f"    L(open={left_gripper_opening:.3f}, close={left_closure:.3f}, action={left_grip_action:+.2f}, "
            f"force=({left_force_inner:.1f},{left_force_outer:.1f}), f_min={left_force_min:.1f}, "
            f"f_avg={left_force_avg:.1f}, c_min={left_contact_min:.3f}, c_avg={left_contact_avg:.3f}, "
            f"local={left_local_grasp:.3f}, grip_rew={left_grasp_term:.3f}, lift_rew={left_lift_reward_raw:.3f})"
        )
        print(
            f"    R(open={right_gripper_opening:.3f}, close={right_closure:.3f}, action={right_grip_action:+.2f}, "
            f"force=({right_force_inner:.1f},{right_force_outer:.1f}), f_min={right_force_min:.1f}, "
            f"f_avg={right_force_avg:.1f}, c_min={right_contact_min:.3f}, c_avg={right_contact_avg:.3f}, "
            f"local={right_local_grasp:.3f}, grip_rew={right_grasp_term:.3f}, lift_rew={right_lift_reward_raw:.3f})"
        )
        print(f"    secure_local={secure_local:.3f}")
        print(
            "  lift: "
            f"obj_z={object_height:.3f}, dz={object_height_delta:.3f}, hprog={lift_gate:.3f}, "
            f"L_lift={left_lift_reward_raw:.3f}, R_lift={right_lift_reward_raw:.3f}, "
            f"tilt={object_tilt_deg:.1f}deg, roll_pitch={success_roll_pitch_deg:.1f}deg, "
            f"yaw={success_yaw_deg:.1f}deg, tilt_x={object_tilt_excess:.3f}, tilt_pen={object_tilt_penalty:+.3f}, "
            f"hold_gate={hold_active_gate:.3f}, height_drop={hold_height_drop:.4f}, drop_pen={-hold_height_drop_penalty:+.3f}, "
            f"obj_x_err={x_error:.3f}, x_center_track={goal_tracking:.3f}, goal_err={target_delta:.3f}"
        )

    def _prepare_motion_context_act(self):
        """policy inference 전에 agent 내부의 signed EE motion intent를 최신 값으로 갱신한다."""

        motion_intents = self._current_motion_intents()
        if hasattr(self.agents, "set_current_motion_intents"):
            self.agents.set_current_motion_intents(motion_intents)
        return motion_intents

    def _apply_intent_share_strategy(
        self,
        shared_intents: dict[str, torch.Tensor],
        outputs,
        timestep: int,
        sharing_enabled: bool,
        warmup_active: bool,
    ):
        """Share the full paper context descriptor when intent sharing is enabled."""

        del outputs, timestep, warmup_active
        intent_dim = int(self._resolve_env_attr("intent_feature_dim", 0) or 0)

        if intent_dim <= 0:
            return shared_intents

        if not sharing_enabled:
            return {agent: torch.zeros_like(intent) for agent, intent in shared_intents.items()}

        return shared_intents

    def _mask_diagnostic_actor_states(self, states: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Zero only the received partner-message channels during evaluation."""

        mask = self.diagnostic_mask
        if mask == "none":
            return states
        communication_mode = str(self._resolve_env_attr("communication_mode", ""))
        if communication_mode != "motion_context":
            raise ValueError("--diagnostic_mask requires a motion_context environment and checkpoint.")
        own_dim = int(self._resolve_env_attr("own_observation_dim", 30))
        motion_dim = int(self._resolve_env_attr("motion_intent_dim", 3))
        context_dim = int(self._resolve_env_attr("motion_context_dim", 3))
        slot_dim = int(self._resolve_env_attr("communication_feature_dim", 30))
        if mask == "motion_zero":
            start, end = own_dim, own_dim + motion_dim
        elif mask == "context_zero":
            start, end = own_dim + motion_dim, own_dim + motion_dim + context_dim
        elif mask == "message_zero":
            start, end = own_dim, own_dim + slot_dim
        else:
            raise ValueError(f"Unknown diagnostic mask: {mask}")
        masked = {}
        for agent, agent_states in states.items():
            if agent_states.shape[-1] < own_dim + slot_dim:
                raise ValueError(f"{agent} actor state is too short for diagnostic masking: {agent_states.shape[-1]}")
            masked_states = agent_states.clone()
            masked_states[..., start:end] = 0.0
            masked[agent] = masked_states
        return masked

    def multi_agent_train(self) -> None:
        """Paper motion-context intent handoff를 포함한 multi-agent training loop를 실행한다."""

        assert self.num_simultaneous_agents == 1, "This method is not allowed for simultaneous agents"
        assert self.env.num_agents > 1, "This method is not allowed for single-agent"
        self._in_eval = False
        self._set_env_attr("training", True)
        self._set_env_attr("motion_context_update_scale", True)

        states, infos = self.env.reset()
        shared_states = self.env.state()

        for timestep in tqdm.tqdm(
            range(self.initial_timestep, self.timesteps), disable=self.disable_progressbar, file=sys.stdout
        ):
            self.agents.pre_interaction(timestep=timestep, timesteps=self.timesteps)

            warmup = self._intent_share_warmup_timesteps()
            sharing_mode = str(self._resolve_env_attr("sharing_mode", "motion_context_share"))
            communication_mode = str(self._resolve_env_attr("communication_mode", "motion_context"))
            sharing_enabled = communication_mode != "none" and sharing_mode != "no_share"
            warmup_active = warmup > 0 and timestep < warmup
            self._set_env_attr("training", True)
            self._set_env_attr("motion_context_update_scale", True)
            self._set_env_attr("freeze_gripper_open", False)
            self._set_env_attr("communication_enabled", bool(sharing_enabled))

            with torch.no_grad():
                # env가 proprioceptive z_t를 계산하고 trainer는 이를 다음 step의
                # 상대 팔 관측으로 넘긴다.
                self._prepare_motion_context_act()
                actions, _, outputs = self.agents.act(states, timestep=timestep, timesteps=self.timesteps)
                shared_intents = {agent: outputs[agent]["intent"] for agent in outputs}
                shared_intents = self._apply_intent_share_strategy(
                    shared_intents, outputs, timestep, sharing_enabled, warmup_active
                )
                self.env.set_pending_intents(
                    predicted_intents=shared_intents,
                    policy_states=states,
                    current_actions=actions,
                )

                next_states, rewards, terminated, truncated, infos = self.env.step(actions)
                shared_next_states = self.env.state()
                infos["shared_states"] = shared_states
                infos["shared_next_states"] = shared_next_states

                if not self.headless:
                    self.env.render()

                self.agents.record_transition(
                    states=states,
                    actions=actions,
                    rewards=rewards,
                    next_states=next_states,
                    terminated=terminated,
                    truncated=truncated,
                    infos=infos,
                    timestep=timestep,
                    timesteps=self.timesteps,
                )

                self._track_environment_info(infos)
                self._update_episode_history(rewards, terminated, truncated)
                self._emit_debug_summary(timestep, actions, outputs, infos)

            self.agents.post_interaction(timestep=timestep, timesteps=self.timesteps)

            if not self.env.agents:
                with torch.no_grad():
                    states, infos = self.env.reset()
                    shared_states = self.env.state()
            else:
                states = next_states
                shared_states = shared_next_states

    def multi_agent_eval(self) -> None:
        """deterministic/stochastic evaluation을 실행하고 필요하면 trace를 저장한다."""

        assert self.num_simultaneous_agents == 1, "This method is not allowed for simultaneous agents"
        assert self.env.num_agents > 1, "This method is not allowed for single-agent"
        communication_mode = str(self._resolve_env_attr("communication_mode", "motion_context"))
        calibration_enabled = bool(self._resolve_env_attr("motion_context_calibration_enabled", True))
        if (
            communication_mode in ("context_only", "motion_context", "motion_context_la", "motion_context_s")
            and calibration_enabled
            and not bool(self._resolve_env_attr("_motion_context_calibrated", False))
        ):
            raise RuntimeError(
                "Evaluation requires a checkpoint sidecar with completed adaptive Context calibration. "
                "The loaded environment state is not calibrated."
            )
        self._in_eval = True
        self._set_env_attr("training", False)
        self._set_env_attr("motion_context_update_scale", False)

        states, infos = self.env.reset()
        shared_states = self.env.state()

        for timestep in tqdm.tqdm(
            range(self.initial_timestep, self.timesteps), disable=self.disable_progressbar, file=sys.stdout
        ):
            self.agents.pre_interaction(timestep=timestep, timesteps=self.timesteps)

            warmup = self._intent_share_warmup_timesteps()
            sharing_mode = str(self._resolve_env_attr("sharing_mode", "motion_context_share"))
            communication_mode = str(self._resolve_env_attr("communication_mode", "motion_context"))
            sharing_enabled = communication_mode != "none" and sharing_mode != "no_share"
            warmup_active = warmup > 0 and timestep < warmup
            self._set_env_attr("training", False)
            self._set_env_attr("motion_context_update_scale", False)
            self._set_env_attr("freeze_gripper_open", False)
            self._set_env_attr("communication_enabled", bool(sharing_enabled))

            with torch.no_grad():
                # eval도 training과 같은 intent handoff를 따른다. 그래야 trace가 실제
                # deploy 시 communication path를 반영한다.
                self._prepare_motion_context_act()
                trace_getter = self._resolve_env_attr("get_paper_motion_context_trace_signals", None)
                trace_env_index = int(max(0, min(self.mode_trace_env_index, self.env.num_envs - 1)))
                pre_step_trace_signals = trace_getter(trace_env_index) if callable(trace_getter) else {}
                if self.save_mode_trace and not self._mode_trace["left_motion_context"]:
                    self._mode_trace["initial_physical_state"] = capture_physical_start(
                        self._resolve_env_attr, trace_env_index
                    )
                    self._mode_trace["initial_physical_state_schema"] = "openarm_physical_state/wxyz_v1"
                actor_states = self._mask_diagnostic_actor_states(states)
                outputs = self.agents.act(actor_states, timestep=timestep, timesteps=self.timesteps)
                shared_intents = {agent: outputs[2][agent]["intent"] for agent in outputs[2]}
                shared_intents = self._apply_intent_share_strategy(
                    shared_intents, outputs[2], timestep, sharing_enabled, warmup_active
                )
                actions = (
                    outputs[0]
                    if self.stochastic_evaluation
                    else {k: outputs[-1][k].get("mean_actions", outputs[0][k]) for k in outputs[-1]}
                )
                self.env.set_pending_intents(
                    predicted_intents=shared_intents,
                    policy_states=actor_states,
                    current_actions=actions,
                )

                next_states, rewards, terminated, truncated, infos = self.env.step(actions)
                shared_next_states = self.env.state()
                infos["shared_states"] = shared_states
                infos["shared_next_states"] = shared_next_states

                if not self.headless:
                    self.env.render()

                self.agents.record_transition(
                    states=states,
                    actions=actions,
                    rewards=rewards,
                    next_states=next_states,
                    terminated=terminated,
                    truncated=truncated,
                    infos=infos,
                    timestep=timestep,
                    timesteps=self.timesteps,
                )
                self._append_mode_trace(outputs[2], rewards, infos, pre_step_trace_signals, actor_states)
                self._update_episode_history(rewards, terminated, truncated)
                self._emit_debug_summary(timestep, actions, outputs[2], infos)

            terminal_info = self._trace_terminal_info(terminated, truncated)
            trace_done = bool(terminal_info["done"])
            trace_terminated = bool(terminal_info["terminated"])
            trace_truncated = bool(terminal_info["truncated"])
            if trace_done:
                self._mode_trace_last_done = terminal_info
                self._save_mode_trace(
                    terminated_at_end=trace_terminated,
                    truncated_at_end=trace_truncated,
                )
                if self.num_trace_episodes > 0 and self._mode_trace_episode_index >= self.num_trace_episodes:
                    break

            if not self.env.agents:
                with torch.no_grad():
                    states, infos = self.env.reset()
                    shared_states = self.env.state()
            else:
                states = next_states
                shared_states = shared_next_states
        if (
            self.num_trace_episodes <= 0 or self._mode_trace_episode_index < self.num_trace_episodes
        ) and self._mode_trace["left_motion_context"]:
            self._save_mode_trace(
                terminated_at_end=bool(self._mode_trace_last_done.get("terminated", False)),
                truncated_at_end=bool(self._mode_trace_last_done.get("truncated", False)),
            )


# Backward-compatible name for existing Hydra experiment metadata.
PaperIntentTrainer = MotionContextTrainer
