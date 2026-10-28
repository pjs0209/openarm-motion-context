"""Unlaunched coherent37D actor tracing; frozen original trainer stays intact.

Reuse the existing raw-state overlay in isolated globals allowing only own37.
Install on a NEW derived trainer class before construction, never on its parent
or a live trainer. No policy, reward, rollout or observation is changed.
"""

import inspect
import math

import torch

from scripts.studies import raw_actor_state_trace as raw

MARKER = "_camera_lead_trace_version"
FIELDS = (
    "ball_camera_lead_observation",
    "ball_camera_frame",
    "ball_camera_acquisition_time",
    "ball_camera_current_time",
    "ball_camera_control_step",
    "ball_camera_episode_length",
    "ball_camera_timestamp_dtype",
)


def own_rows(states, own, slot, env_index):
    if (
        own != 37
        or slot != 37
        or type(env_index) is not int
        or env_index < 0
        or not isinstance(states, dict)
        or set(states) != {"left_arm", "right_arm"}
    ):
        raise ValueError("Exact camera-lead own37/peer37 actual actor states required")
    rows, batch = {}, None
    for side, uid in (("left", "left_arm"), ("right", "right_arm")):
        value = states[uid]
        if (
            not isinstance(value, torch.Tensor)
            or not value.is_floating_point()
            or value.ndim != 2
            or value.shape[1] != 74
            or env_index >= value.shape[0]
            or not torch.isfinite(value[env_index]).all()
            or batch is not None
            and batch != value.shape[0]
        ):
            raise ValueError("Finite same-batch74D actual actor rows required")
        batch = value.shape[0]
        rows[side] = value[env_index, :37].detach().cpu().tolist()
    return rows


def validate_signals(signals, rows):
    if not isinstance(signals, dict) or any(key not in signals for key in FIELDS):
        raise ValueError("Coherent pre-step camera-lead signals required")
    camera = signals["ball_camera_lead_observation"]
    if (
        not isinstance(camera, list)
        or len(camera) != 7
        or not all(math.isfinite(v) for v in camera)
        or camera[3] not in (0, 1)
        or camera[6] not in (0, 1)
        or camera[:4] != signals.get("ball_camera_observation")
        or any(row[30:37] != camera for row in rows.values())
        or camera[6] == 0
        and any(camera[4:6])
        or camera[3] == 0
        and any(camera[:3] + camera[4:])
    ):
        raise ValueError("Delivered actor camera does not match measured/derived pre-step signals")
    for key in ("ball_camera_frame", "ball_camera_control_step", "ball_camera_episode_length"):
        if type(signals[key]) is not int or signals[key] < 0:
            raise ValueError("Nonnegative camera counters required")
    acquired, current = signals["ball_camera_acquisition_time"], signals["ball_camera_current_time"]
    if not all(math.isfinite(v) and v >= 0 for v in (acquired, current)) or current < acquired:
        raise ValueError("Ordered finite camera times required")
    if signals["ball_camera_timestamp_dtype"] not in ("torch.float32", "torch.float64"):
        raise ValueError("Recorded sensor timestamp precision required")


def install_trainer_class(cls):
    if hasattr(cls, MARKER) or cls.__name__ == "MotionContextTrainer":
        raise ValueError("New camera-lead derived trainer class required; never mutate original")
    # This reuses its original cadence/state validation without mutating its
    # module globals or extending any frozen source file.
    namespace = dict(raw.__dict__)
    namespace["own_rows"] = own_rows
    exec(compile(inspect.getsource(raw.install_trainer_class), __file__, "exec"), namespace)
    namespace["install_trainer_class"](cls)
    original_new, original_append = cls._new_mode_trace, cls._append_mode_trace

    def new_trace():
        trace = original_new()
        for key in (*FIELDS, "camera_lead_trace_version"):
            if key in trace:
                raise ValueError("Camera-lead trace already present")
            trace[key] = []
        return trace

    def append(self, outputs, rewards, infos, trace_signals=None, policy_states=None):
        before = len(self._mode_trace["left_motion_context"])
        active = (
            self.save_mode_trace
            and self._in_eval
            and (self.num_trace_episodes <= 0 or self._mode_trace_episode_index < self.num_trace_episodes)
        )
        if active:
            index = int(max(0, min(self.mode_trace_env_index, self.env.num_envs - 1)))
            rows = own_rows(policy_states, 37, 37, index)
            validate_signals(trace_signals, rows)
            if any(
                key not in self._mode_trace or len(self._mode_trace[key]) != before
                for key in (*FIELDS, "camera_lead_trace_version")
            ):
                raise ValueError("Install before construction; no partial camera-lead trace")
        result = original_append(self, outputs, rewards, infos, trace_signals, policy_states)
        after = len(self._mode_trace["left_motion_context"])
        if after == before + 1 and active:
            for key in FIELDS:
                value = trace_signals[key]
                self._mode_trace[key].append(list(value) if isinstance(value, list) else value)
            self._mode_trace["camera_lead_trace_version"].append(1)
        elif after != before:
            raise ValueError("Unexpected original trace cadence")
        return result

    cls._new_mode_trace = staticmethod(new_trace)
    cls._append_mode_trace = append
    setattr(cls, MARKER, 1)


def load_trainer_class():
    # Lazy import after AppLauncher, when selected by a future runtime config.
    from isaaclab_tasks.direct.openarm_motion_context.train.trainer import MotionContextTrainer

    class CameraLeadTraceTrainer(MotionContextTrainer):
        pass

    install_trainer_class(CameraLeadTraceTrainer)
    return CameraLeadTraceTrainer


def CameraLeadTraceTrainer(*args, **kwargs):
    """Runtime trainer factory; unused by every current GPU process."""
    return load_trainer_class()(*args, **kwargs)
