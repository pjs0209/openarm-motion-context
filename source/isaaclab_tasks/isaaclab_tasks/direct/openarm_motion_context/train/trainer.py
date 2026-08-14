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
    return f"[mx={vals[0]:+.2f}, my={vals[1]:+.2f}, mz={vals[2]:+.2f}]"


_COMMUNICATION_PAYLOAD_DIMS = {
    "none": 0,
    "motion_only": 3,
    "context_only": 3,
    "motion_context": 6,
    "previous_action": 8,
    "full_partner_observation": 30,
}


def _communication_layout(mode: str) -> list[str]:
    """Return the semantic layout of the active (unpadded) payload."""

    if mode == "motion_only":
        return ["signed_motion_x", "signed_motion_y", "signed_motion_z"]
    if mode == "context_only":
        return ["linear_activity", "angular_activity", "action_smoothness"]
    if mode == "motion_context":
        return [
            "signed_motion_x",
            "signed_motion_y",
            "signed_motion_z",
            "linear_activity",
            "angular_activity",
            "action_smoothness",
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
        self.num_trace_episodes = int(self.cfg.get("num_trace_episodes", 0))
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
            "object_z": [],
            "object_dz": [],
            "tip_dist": [],
            "lateral_error": [],
            "axis_alignment": [],
            "insertion_depth": [],
            "target_insertion_depth": [],
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
            "left_linear_activity": [],
            "left_angular_activity": [],
            "left_action_smoothness": [],
            "right_linear_activity": [],
            "right_angular_activity": [],
            "right_action_smoothness": [],
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
                if force_set or hasattr(env, attr_name):
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
        """Return env-side normalization state saved beside policy checkpoints."""

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

        step = int(self._resolve_env_attr("_motion_context_scale_update_step", -1) or -1)
        freeze_after = int(self._resolve_env_attr("motion_context_freeze_after_steps", 10000) or 10000)
        common_step = int(self._resolve_env_attr("common_step_counter", 0) or 0)
        scale_frozen = bool(
            self._resolve_env_attr("_motion_context_scale_frozen", common_step >= freeze_after)
        )
        return {
            "running_scale": running_scale,
            "scale_update_step": step,
            "common_step_counter": common_step,
            "scale_frozen": scale_frozen,
            "freeze_after_steps": freeze_after,
            "normalization": "ema_batch_p90",
            "scale_beta": float(self._resolve_env_attr("motion_context_scale_beta", 0.99) or 0.99),
            "scale_percentile": float(self._resolve_env_attr("motion_context_scale_percentile", 0.90) or 0.90),
            "norm_max": float(self._resolve_env_attr("motion_context_norm_max", 1.5) or 1.5),
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
        self._set_env_attr("_motion_context_scale_frozen", bool(state.get("scale_frozen", False)))

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

    def _append_mode_trace(self, outputs, rewards, infos, trace_signals: dict | None = None) -> None:
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
        for key in (
            "object_z",
            "object_dz",
            "tip_dist",
            "lateral_error",
            "axis_alignment",
            "insertion_depth",
            "target_insertion_depth",
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
            "left_linear_activity",
            "left_angular_activity",
            "left_action_smoothness",
            "right_linear_activity",
            "right_angular_activity",
            "right_action_smoothness",
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
        trace["terminal_flags"] = {
            "success": terminal_success,
            "drop": terminal_drop,
            "far": terminal_far,
            "tilt_fail": terminal_tilt_fail,
            "invalid": terminal_invalid,
        }

        # Agent/MAPPO owns the architecture debug values. Read through the public
        # getter so train and eval trace metadata stay identical.
        agent_debug = (
            self.agents.get_motion_context_debug()
            if hasattr(self.agents, "get_motion_context_debug")
            else {}
        )
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
                return float(self._resolve_env_attr(env_attr, default) or default)
            except Exception:
                return float(default)

        def _env_int(attr: str, default: int = 0) -> int:
            try:
                return int(self._resolve_env_attr(attr, default) or default)
            except Exception:
                return int(default)

        def _env_float(attr: str, default: float = 0.0) -> float:
            try:
                return float(self._resolve_env_attr(attr, default) or default)
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
        communication_slot_dim = _env_int("communication_feature_dim", _env_int("intent_feature_dim", 0))
        communication_payload_dim = min(
            int(_COMMUNICATION_PAYLOAD_DIMS.get(communication_mode, communication_slot_dim)),
            communication_slot_dim,
        )
        trace["metadata"] = {
            "trace_schema": "openarm_motion_context/v2",
            "timeline_convention": "state_message_action_t_reward_t_to_t_plus_1",
            "communication_delay_steps": 1,
            "seed": trace_seed,
            "context_dim": _env_int("motion_context_dim", 0),
            "motion_intent_dim": _env_int("motion_intent_dim", 0),
            "communication_mode": communication_mode,
            "communication_feature_dim": communication_slot_dim,
            "communication_slot_dim": communication_slot_dim,
            "communication_payload_dim": communication_payload_dim,
            "communication_payload_layout": _communication_layout(communication_mode),
            "intent_feature_dim": _env_int("intent_feature_dim", 0),
            "intent_dim": _env_int("intent_feature_dim", 0),
            "own_obs_dim": _env_int("own_observation_dim", 0),
            "actor_input_dim": _env_int("actor_input_dim", 0),
            "motion_intent_horizon": _env_int("motion_intent_horizon", 0),
            "motion_context_layout": [
                "signed_motion_x",
                "signed_motion_y",
                "signed_motion_z",
                "linear_activity",
                "angular_activity",
                "action_smoothness",
            ],
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
            "sharing_mode": _env_str("sharing_mode", ""),
            "method": communication_mode,
            "method_name": "Proprioceptive Motion Context Sharing",
            "reward": _env_str("intent_task_label", "openarm_task"),
            "context_normalization": "ema_batch_p90",
            "context_dt": _env_float("step_dt", 1.0),
            "context_layout": ["linear_activity", "angular_activity", "action_smoothness"],
            "context_scale_beta": _env_float("motion_context_scale_beta", 0.99),
            "context_scale_percentile": _env_float("motion_context_scale_percentile", 0.90),
            "context_norm_max": _env_float("motion_context_norm_max", 1.5),
            "context_update_scale": bool(self._resolve_env_attr("motion_context_update_scale", True)),
            "context_freeze_after_steps": _env_int("motion_context_freeze_after_steps", 10000),
            "context_running_scale": context_scale,
            "success_lateral_threshold": _env_float("success_lateral_threshold", 0.012),
            "success_axis_threshold": _env_float("success_axis_threshold", 0.85),
            "success_depth_threshold": _env_float("success_depth_threshold", 0.045),
            "target_insertion_depth": _env_float("target_insertion_depth", 0.05),
            "wall_penetration_lateral_threshold": _env_float("wall_penetration_lateral_threshold", 0.015),
            "wall_penetration_depth_threshold": _env_float("wall_penetration_depth_threshold", 0.0),
            "shared_intent_encoder": shared_intent_encoder,
            "context_classifier": False,
            "auxiliary_loss": False,
            "actor_backbone_shared": False,
            "action_head_shared": False,
            "partner_message_embed_dim": int(left_debug.get("partner_message_embed_dim", right_debug.get("partner_message_embed_dim", 0))),
            "partner_intent_embed_dim": int(left_debug.get("partner_intent_embed_dim", right_debug.get("partner_intent_embed_dim", 0))),
            "actor_action_input_dim": int(left_debug.get("actor_action_input_dim", right_debug.get("actor_action_input_dim", 0))),
            "gripper_head_input_dim": int(left_debug.get("gripper_head_input_dim", right_debug.get("gripper_head_input_dim", 0))),
            "architecture": (
                "agent_specific_actor_partner_context_encoder"
                if shared_intent_encoder
                else "no_intent"
            ),
            "actor_input_excludes_contact_reward_success_object_z": True,
            "deterministic_eval": not bool(getattr(self, "stochastic_evaluation", True)),
        }

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
        slot_dim = int(self._resolve_env_attr("communication_feature_dim", self._resolve_env_attr("intent_feature_dim", 0)) or 0)
        motion_dim = int(self._resolve_env_attr("motion_intent_dim", 3) or 3)
        context_dim = int(self._resolve_env_attr("motion_context_dim", 3) or 0)
        if communication_mode == "none":
            context_dim = 0
        payload_dim = min(int(_COMMUNICATION_PAYLOAD_DIMS.get(communication_mode, slot_dim)), slot_dim)
        actor_in_dim = own_dim + slot_dim
        phase = "eval" if self._in_eval else "train"
        log_info = infos.get("log", {})
        context_debug = self.agents.get_motion_context_debug() if hasattr(self.agents, "get_motion_context_debug") else {}
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
                    "  episode(last100): "
                    f"n={episode_summary['count']:.0f} "
                    f"success_rate={episode_summary['success_rate']:.2f} "
                    f"mean_return={episode_summary['mean_return']:.2f} "
                    f"mean_length={episode_summary['mean_length']:.1f} "
                    f"timeout={episode_summary['timeout']:.2f} "
                    f"drop={episode_summary['drop']:.2f} "
                    f"far={episode_summary['far']:.2f} "
                    f"tilt_fail={episode_summary['tilt_fail']:.2f} "
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
                print("  received_message: disabled")
            else:
                signals = {}
                if hasattr(self.env, "get_paper_motion_context_trace_signals"):
                    try:
                        signals = self.env.get_paper_motion_context_trace_signals(0)
                    except Exception:
                        signals = {}
                left_context = signals.get("left_motion_context", [0.0, 0.0, 0.0])
                right_context = signals.get("right_motion_context", [0.0, 0.0, 0.0])
                left_payload = _message_slice("left", 0, payload_dim)
                right_payload = _message_slice("right", 0, payload_dim)
                left_payload_norm = float(torch.linalg.vector_norm(torch.tensor(left_payload))) if left_payload else 0.0
                right_payload_norm = float(torch.linalg.vector_norm(torch.tensor(right_payload))) if right_payload else 0.0
                print(
                    "  message: "
                    f"L->R norm={left_payload_norm:.2f} R->L norm={right_payload_norm:.2f} "
                    f"active={payload_dim}/{slot_dim}"
                )
                if communication_mode in ("motion_only", "motion_context"):
                    print(
                        "  received_motion: "
                        f"L<-R={format_motion(right_payload[:motion_dim])} "
                        f"R<-L={format_motion(left_payload[:motion_dim])}"
                    )
                if communication_mode in ("context_only", "motion_context"):
                    context_start = motion_dim if communication_mode == "motion_context" else 0
                    left_received_context = right_payload[context_start : context_start + context_dim]
                    right_received_context = left_payload[context_start : context_start + context_dim]
                    print(
                        "  received_context: "
                        f"L<-R=[lin={float(left_received_context[0]):.2f}, ang={float(left_received_context[1]):.2f}, smooth={float(left_received_context[2]):.2f}] "
                        f"R<-L=[lin={float(right_received_context[0]):.2f}, ang={float(right_received_context[1]):.2f}, smooth={float(right_received_context[2]):.2f}]"
                    )
                    print(
                        "  outgoing_context: "
                        f"L=[lin={float(left_context[0]):.2f}, ang={float(left_context[1]):.2f}, smooth={float(left_context[2]):.2f}] "
                        f"R=[lin={float(right_context[0]):.2f}, ang={float(right_context[1]):.2f}, smooth={float(right_context[2]):.2f}]"
                    )
                raw_scale = self._resolve_env_attr("_motion_context_running_scale", None)
                if isinstance(raw_scale, torch.Tensor) and raw_scale.numel() >= 3:
                    scale = raw_scale[:3].detach().float().cpu().tolist()
                    print(
                        "  context_scale: "
                        f"lin={scale[0]:.4f} ang={scale[1]:.4f} action={scale[2]:.4f} "
                        f"freeze_after={int(self._resolve_env_attr('motion_context_freeze_after_steps', 10000) or 10000)}"
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
                lateral_error = _log("peg_hole/lateral_error")
                insertion_depth = _log("peg_hole/insertion_depth")
                target_depth = _log("peg_hole/target_insertion_depth", 0.025)
                axis_gate = _log("peg_hole/axis_gate", max(0.0, min(axis_align, 1.0)))
                lateral_gate = _log("peg_hole/lateral_gate", 1.0)
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
                    "  geometry_gate: "
                    f"tip_dist={_log('peg_hole/tip_dist'):.3f} "
                    f"lateral={lateral_error:.3f}/{lateral_threshold:.3f} "
                    f"depth={insertion_depth:.3f}/{depth_threshold:.3f} "
                    f"depth_progress={depth_progress:.2f} "
                    f"axis_gate={axis_gate:.2f} "
                    f"lateral_gate={lateral_gate:.2f} "
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
                goal_error_name = "y_error" if _has_log("y_error") else "xy_error"
                roll_pitch_deg = _log(
                    "success_roll_pitch_deg",
                    max(_log("success_roll_deg"), _log("success_pitch_deg")),
                )
                yaw_deg = _log("success_yaw_deg")
                dual_grasp_ok = _log("dual_grasp_gate")
                stable_now = _log("stable_now_ratio", min(height_ok, tilt_ok))
                strict_success = _log("strict_success_ratio", _log("success_ratio"))
                if dual_grasp_ok < 0.3:
                    bottleneck = "grasp"
                elif height_ok < 0.3:
                    bottleneck = "lift"
                elif stable_now < 0.3:
                    bottleneck = "stability"
                elif strict_success < 0.3:
                    bottleneck = "hold"
                else:
                    bottleneck = "solved"
                print(
                    "  reward: "
                    f"assigned_team={_log('team_reward', _log('left_reward')):+.2f} "
                    f"terms=reach:{_log('reward_reach_term'):+.2f} "
                    f"grasp:{_log('reward_grasp_term'):+.2f} "
                    f"lift:{_log('reward_lift_term'):+.2f} "
                    f"success:{_log('reward_success_term'):+.2f} "
                    f"action:-{_log('reward_action_term'):.4f} "
                    f"bottleneck={bottleneck}"
                )
                print(
                    "  conditions: "
                    f"reach=L:{_log('left_reach'):.2f}/R:{_log('right_reach'):.2f} "
                    f"orientation=L:{_log('left_orientation'):.2f}/R:{_log('right_orientation'):.2f} "
                    f"dual_grasp={dual_grasp_ok:.2f} "
                    f"height_ok={height_ok:.2f} "
                    f"tilt_ok={tilt_ok:.2f} "
                    f"stable_now={stable_now:.2f} "
                    f"strict_success={strict_success:.2f}"
                )
                print(
                    "  grasp_target_tf: "
                    f"actor_vs_gt_pos=L:{_log('left_actor_target_pos_error'):.6f}/"
                    f"R:{_log('right_actor_target_pos_error'):.6f}m "
                    f"max=L:{_log('left_actor_target_pos_error_max'):.6f}/"
                    f"R:{_log('right_actor_target_pos_error_max'):.6f}m "
                    f"rot=L:{_log('left_actor_target_rot_error_deg'):.3f}/"
                    f"R:{_log('right_actor_target_rot_error_deg'):.3f}deg"
                )
                print(
                    "  grasp_target_geometry: "
                    f"local_L=[{_log('left_target_local_x'):+.4f},"
                    f"{_log('left_target_local_y'):+.4f},{_log('left_target_local_z'):+.4f}] "
                    f"local_R=[{_log('right_target_local_x'):+.4f},"
                    f"{_log('right_target_local_y'):+.4f},{_log('right_target_local_z'):+.4f}] "
                    f"ee_to_actor=L:{_log('left_actor_dist'):.3f}/R:{_log('right_actor_dist'):.3f}m "
                    f"ee_to_gt=L:{_log('left_dist'):.3f}/R:{_log('right_dist'):.3f}m"
                )
                print(
                    "  grasp_target_world[env0]: "
                    f"box=[{_log('sample_object_x'):+.3f},{_log('sample_object_y'):+.3f},"
                    f"{_log('sample_object_z'):+.3f}] "
                    f"L=[{_log('sample_left_actor_target_x'):+.3f},"
                    f"{_log('sample_left_actor_target_y'):+.3f},"
                    f"{_log('sample_left_actor_target_z'):+.3f}] "
                    f"R=[{_log('sample_right_actor_target_x'):+.3f},"
                    f"{_log('sample_right_actor_target_y'):+.3f},"
                    f"{_log('sample_right_actor_target_z'):+.3f}]"
                )
                print(
                    "  hold: "
                    f"mean={_log('hold_count_mean'):.1f} "
                    f"p90={_log('hold_count_p90'):.1f} "
                    f"max={_log('hold_count_max'):.0f} "
                    f"required={_log('hold_required_steps'):.0f}"
                )
                print(
                    "  grasp_kinematic:\n"
                    f"    L(near={_log('left_near_collision_target'):.2f}, ori={_log('left_orientation'):.2f}, "
                    f"cmd={_log('left_close_command'):.2f}, closure={_log('left_closure'):.2f}, "
                    f"close_signal={_log('left_close_signal'):.2f}, "
                    f"inside={_log('left_target_inside_gripper'):.2f}, "
                    f"span={_log('left_target_inside_finger_span'):.2f}, "
                    f"center={_log('left_target_centered_between_fingers'):.2f}, "
                    f"midline_dist={_log('left_target_to_gripper_midline_dist'):.3f})\n"
                    f"    R(near={_log('right_near_collision_target'):.2f}, ori={_log('right_orientation'):.2f}, "
                    f"cmd={_log('right_close_command'):.2f}, closure={_log('right_closure'):.2f}, "
                    f"close_signal={_log('right_close_signal'):.2f}, "
                    f"inside={_log('right_target_inside_gripper'):.2f}, "
                    f"span={_log('right_target_inside_finger_span'):.2f}, "
                    f"center={_log('right_target_centered_between_fingers'):.2f}, "
                    f"midline_dist={_log('right_target_to_gripper_midline_dist'):.3f})"
                )
                print(
                    "  grasp_frame_check: "
                    f"tcp_from_finger_origin=L:{_log('left_tcp_from_finger_origin_dist'):.3f}/"
                    f"R:{_log('right_tcp_from_finger_origin_dist'):.3f}m "
                    f"target_to_tcp_plane=L:{_log('left_target_to_tcp_plane_dist'):.3f}/"
                    f"R:{_log('right_target_to_tcp_plane_dist'):.3f}m "
                    f"inside_tcp_plane=L:{_log('left_target_inside_tcp_plane'):.2f}/"
                    f"R:{_log('right_target_inside_tcp_plane'):.2f}"
                )
                print(
                    "  object: "
                    f"dz={_log('object_height_delta'):.3f} lift_progress={_log('lift'):.2f} "
                    f"grasp_gate={dual_grasp_ok:.2f} lift_gate={_log('dual_lift_gate'):.2f} "
                    f"tilt_lift={_log('tilt_aware_lift', _log('lift')):.2f} "
                    f"tilt={_log('object_tilt_deg'):.1f}deg "
                    f"roll_pitch={roll_pitch_deg:.1f}deg yaw={yaw_deg:.1f}deg "
                    f"tilt_pen={_log('tilt_penalty'):.2f} "
                    f"{goal_error_name}={_log(goal_error_name):.3f}"
                )
                _print_action_debug()
                if _has_log("ball_xy_error") or _has_log("ball_center_tracking"):
                    print(
                        "  ball: "
                        f"xy_err={_log('ball_xy_error'):.3f} "
                        f"center={_log('ball_center_tracking'):.2f}"
                    )
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
                f"context=[lin={float(left_context[0]):.2f}, ang={float(left_context[1]):.2f}, "
                f"smooth={float(left_context[2]):.2f}]"
            )
            print(
                f"    R self: motion={self._fmt_motion(outputs['right_arm'].get('motion_intent'))} "
                f"context=[lin={float(right_context[0]):.2f}, ang={float(right_context[1]):.2f}, "
                f"smooth={float(right_context[2]):.2f}]"
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
            "    z=[signed_EE_motion_3D, linear_activity, angular_activity, action_smoothness]\n"
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
        left_force_min = self._scalar(log_info.get("left_grasp_force_min"), default=min(left_force_inner, left_force_outer))
        right_force_min = self._scalar(log_info.get("right_grasp_force_min"), default=min(right_force_inner, right_force_outer))
        left_force_avg = self._scalar(log_info.get("left_grasp_force_avg"), default=0.5 * (left_force_inner + left_force_outer))
        right_force_avg = self._scalar(log_info.get("right_grasp_force_avg"), default=0.5 * (right_force_inner + right_force_outer))
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
        left_local_grasp = self._scalar(log_info.get("left_local_grasp"), default=self._scalar(log_info.get("left_grasp_reward")))
        right_local_grasp = self._scalar(log_info.get("right_local_grasp"), default=self._scalar(log_info.get("right_grasp_reward")))
        left_lift_reward_raw = self._scalar(log_info.get("left_lift_reward"))
        right_lift_reward_raw = self._scalar(log_info.get("right_lift_reward"))
        secure_local = self._scalar(log_info.get("secure_local"), default=self._scalar(log_info.get("secure_bilateral")))

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
        grasp_imbalance = self._scalar(log_info.get("grasp_imbalance"), default=abs(left_local_grasp - right_local_grasp))
        reward_gap = self._scalar(log_info.get("reward_gap"), default=abs(left_reward_total - right_reward_total))
        closure_gap = self._scalar(log_info.get("left_right_closure_gap"), default=abs(left_closure - right_closure))
        contact_gap = self._scalar(log_info.get("left_right_contact_gap"), default=abs(left_contact_min - right_contact_min))

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
                self.env.set_pending_intents(predicted_intents=shared_intents, policy_states=states)

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
                outputs = self.agents.act(states, timestep=timestep, timesteps=self.timesteps)
                shared_intents = {agent: outputs[2][agent]["intent"] for agent in outputs[2]}
                shared_intents = self._apply_intent_share_strategy(
                    shared_intents, outputs[2], timestep, sharing_enabled, warmup_active
                )
                self.env.set_pending_intents(predicted_intents=shared_intents, policy_states=states)
                actions = (
                    outputs[0]
                    if self.stochastic_evaluation
                    else {k: outputs[-1][k].get("mean_actions", outputs[0][k]) for k in outputs[-1]}
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
                self._append_mode_trace(outputs[2], rewards, infos, pre_step_trace_signals)
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
            (self.num_trace_episodes <= 0 or self._mode_trace_episode_index < self.num_trace_episodes)
            and self._mode_trace["left_motion_context"]
        ):
            self._save_mode_trace(
                terminated_at_end=bool(self._mode_trace_last_done.get("terminated", False)),
                truncated_at_end=bool(self._mode_trace_last_done.get("truncated", False)),
            )
