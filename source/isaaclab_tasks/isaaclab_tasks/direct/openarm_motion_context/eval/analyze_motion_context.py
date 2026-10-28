#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
from collections import defaultdict
from pathlib import Path
from statistics import NormalDist

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


CONTEXT_NAMES = ("translation_trend", "rotation_trend", "stability_trend")
MOTION_NAMES = ("motion_x", "motion_y", "motion_z")
METHOD_ORDER = (
    "none",
    "motion_only",
    "context_only",
    "motion_context",
    "motion_context_la",
    "motion_context_s",
    "previous_action",
    "full_partner_observation",
)
METHOD_LABELS = {
    "none": "No Communication",
    "motion_only": "Motion Only",
    "context_only": "Context Only",
    "motion_context": "Motion Context",
    "motion_context_la": "Motion + Tv/Tw",
    "motion_context_s": "Motion + Ts",
    "previous_action": "Previous Action",
    "full_partner_observation": "Full Observation",
}
MESSAGE_DIMS = {
    "none": 0,
    "motion_only": 3,
    "context_only": 3,
    "motion_context": 6,
    "motion_context_la": 6,
    "motion_context_s": 6,
    "previous_action": 8,
    "full_partner_observation": 30,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create paper-ready OpenArm motion-context tables and figures with Lift metrics matched to the simplified reward/success definition."
    )
    parser.add_argument("--trace_dir", type=str, required=True, help="Trace directory or experiment root.")
    parser.add_argument(
        "--include_trace_dir",
        action="append",
        default=[],
        help="Additional trace run to include in the same comparison; repeat for each run.",
    )
    parser.add_argument(
        "--analysis_name",
        type=str,
        default=None,
        help="Named combined output folder under --out_dir (keeps source trace runs untouched).",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="logs/motion_context_analysis",
        help="Base output directory. A numbered trace run is analyzed into the matching numbered result directory.",
    )
    parser.add_argument("--episode", type=int, default=None, help="Episode index to plot. Defaults to the first trace.")
    parser.add_argument("--smooth_window", type=int, default=10)
    parser.add_argument(
        "--task", type=str, default=None, help="Task family or registered task name for missing metadata."
    )
    parser.add_argument(
        "--method", type=str, default=None, help="Override communication mode for traces missing metadata."
    )
    parser.add_argument("--seed", type=int, default=None, help="Override seed for traces missing metadata.")
    parser.add_argument(
        "--ci", type=float, default=0.95, help="Confidence interval level for evaluation success-stability curves."
    )
    return parser.parse_args()


def load_trace(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        trace = json.load(f)
    if "episode_index" not in trace:
        stem = path.stem.rsplit("_", 1)[-1]
        trace["episode_index"] = int(stem) if stem.isdigit() else 0
    trace["_path"] = str(path)
    return trace


def load_traces(trace_dir: Path, episode: int | None) -> list[dict]:
    files = sorted(
        set(trace_dir.rglob("motion_context_trace_episode_*.json")) | set(trace_dir.rglob("mode_trace_episode_*.json"))
    )
    if not files:
        raise FileNotFoundError(f"No trace files found under: {trace_dir}")
    traces = [load_trace(path) for path in files]
    if episode is not None:
        traces = [trace for trace in traces if int(trace.get("episode_index", -1)) == int(episode)]
        if not traces:
            raise FileNotFoundError(f"No trace with episode_index={episode} under: {trace_dir}")
    return traces


def validate_trace(trace: dict) -> list[str]:
    """Return categorized schema findings without rejecting useful legacy traces."""

    issues: list[str] = []
    length = recorded_len(trace)
    meta = metadata(trace)
    method = infer_method(trace, None)
    if length <= 0:
        issues.append("REQUIRED: empty episode")
    if method == "unknown":
        issues.append("REQUIRED: communication method could not be inferred; add metadata or use --method")

    for key in ("reward_mean", "left_motion_context", "right_motion_context"):
        size = len(trace.get(key, []))
        if size not in (0, length):
            issues.append(f"RECOMMENDED: {key} has {size} samples, expected {length}")

    schema = str(meta.get("trace_schema", ""))
    if schema and schema not in (
        "openarm_motion_context/v1",
        "openarm_motion_context/v2",
        "openarm_motion_context/v3",
    ):
        issues.append(f"RECOMMENDED: unsupported trace schema {schema!r}")

    default_payload = MESSAGE_DIMS.get(method, 0)
    payload_dim = 0 if method == "none" else int(meta.get("communication_payload_dim", default_payload) or 0)
    if method in MESSAGE_DIMS:
        expected_dim = int(MESSAGE_DIMS[method])
        if payload_dim != expected_dim:
            issues.append(f"RECOMMENDED: payload dim {payload_dim} does not match {method} ({expected_dim})")

    for side in ("left", "right"):
        raw = np.asarray(trace.get(f"{side}_message", []), dtype=float)
        if raw.size and (raw.ndim != 2 or raw.shape[0] != length or raw.shape[1] < payload_dim):
            issues.append(
                f"RECOMMENDED: {side}_message shape {raw.shape} is incompatible with length={length}, payload={payload_dim}"
            )

    required_task_fields = (
        ("tip_dist", "lateral_error", "axis_alignment", "insertion_depth")
        if infer_task(trace, None) == "Peg"
        else (
            "object_z",
            "object_dz",
            "object_tilt_deg",
            "hprog",
            "left_target_dist",
            "right_target_dist",
            "left_closure",
            "right_closure",
        )
    )
    for key in required_task_fields:
        if len(trace.get(key, [])) != length:
            issues.append(f"RECOMMENDED: missing or incomplete task field {key}")
    if infer_task(trace, None) == "Lift":
        threshold_names = (
            ("lift_target_height", ("lift_target_height", "success_height_margin")),
            ("success_grasp_distance", ("success_grasp_distance", "grasp_distance_threshold")),
            ("success_min_closure", ("success_min_closure", "grasp_closure_threshold")),
            ("success_tilt_threshold_deg", ("success_tilt_threshold_deg", "success_tilt_deg")),
        )
        for label, keys in threshold_names:
            if not math.isfinite(optional_float_config(trace, keys)):
                issues.append(f"LEGACY_OPTIONAL: metadata {label} unavailable")
        if not math.isfinite(control_dt(trace)[0]):
            issues.append("LEGACY_OPTIONAL: control_dt unavailable; second-based timing will be NaN")

    recommended = (
        "left_motion",
        "right_motion",
        "left_motion_context",
        "right_motion_context",
        "left_ee_pos_b",
        "right_ee_pos_b",
        "left_action_variation_raw",
        "right_action_variation_raw",
        "context_linear_scale",
        "context_angular_scale",
        "context_action_variation_scale",
    )
    for key in recommended:
        if len(trace.get(key, [])) != length:
            issues.append(f"LEGACY_OPTIONAL: {key} unavailable; dependent metric will use fallback or NaN")
    return issues


def reserve_analysis_output_dir(trace_dir: Path, out_dir: Path) -> Path:
    """Match a numbered trace run, keeping repeat analyses as flat siblings."""

    trace_run = trace_dir.name if trace_dir.name.isdigit() else None
    method_name = trace_dir.parent.name if trace_run else trace_dir.name
    method_dir = out_dir if out_dir.name == method_name else out_dir / method_name
    method_dir.mkdir(parents=True, exist_ok=True)

    if trace_run:
        run_dir = method_dir / trace_run
        try:
            run_dir.mkdir()
            return run_dir
        except FileExistsError:
            # Reanalyzing the same trace must not overwrite the previous result.
            for index in range(1, 10000):
                rerun_dir = method_dir / f"{trace_run}_reanalysis_{index:03d}"
                try:
                    rerun_dir.mkdir()
                    return rerun_dir
                except FileExistsError:
                    continue
            raise RuntimeError(f"Too many analyses for trace run {trace_run}: {method_dir}")

    for index in range(1, 10000):
        run_dir = method_dir / f"{index:03d}"
        try:
            run_dir.mkdir()
            return run_dir
        except FileExistsError:
            continue
    raise RuntimeError(f"Too many analyses in: {method_dir}")


def metadata(trace: dict) -> dict:
    data = trace.get("metadata", {})
    return data if isinstance(data, dict) else {}


def trace_float_config(trace: dict, keys: tuple[str, ...], default: float) -> float:
    """Read a numeric task setting from metadata/top-level trace with a safe default."""

    meta = metadata(trace)
    for key in keys:
        for source in (meta, trace):
            if key in source:
                try:
                    value = float(source[key])
                except (TypeError, ValueError):
                    continue
                if math.isfinite(value):
                    return value
    return float(default)


def optional_float_config(trace: dict, keys: tuple[str, ...]) -> float:
    """Read a finite numeric setting, returning NaN when it was not recorded."""

    meta = metadata(trace)
    for key in keys:
        for source in (meta, trace):
            try:
                value = float(source[key])
            except (KeyError, TypeError, ValueError):
                continue
            if math.isfinite(value):
                return value
    return float("nan")


def control_dt(trace: dict) -> tuple[float, str]:
    """Resolve the policy/control period without embedding a task-specific constant."""

    direct = optional_float_config(trace, ("control_dt", "context_dt", "step_dt", "policy_dt"))
    if math.isfinite(direct) and direct > 0.0:
        return direct, "metadata_control_dt"
    sim_dt = optional_float_config(trace, ("sim_dt", "physics_dt"))
    decimation = optional_float_config(trace, ("decimation", "control_decimation"))
    if math.isfinite(sim_dt) and math.isfinite(decimation) and sim_dt > 0.0 and decimation > 0.0:
        return sim_dt * decimation, "metadata_sim_dt_x_decimation"
    return float("nan"), "unavailable"


def lift_thresholds(trace: dict) -> tuple[float, float, float, float]:
    """Return recorded Lift thresholds; unavailable values remain NaN.

    This intentionally avoids silently importing historical task constants into
    paper analysis. Legacy traces still retain progress and efficiency metrics.
    """

    return (
        optional_float_config(trace, ("lift_target_height", "success_height_margin")),
        optional_float_config(trace, ("success_grasp_distance", "grasp_distance_threshold")),
        optional_float_config(trace, ("success_min_closure", "grasp_closure_threshold")),
        optional_float_config(trace, ("success_tilt_threshold_deg", "success_tilt_deg")),
    )


def lift_orientation_threshold(trace: dict) -> tuple[float, str]:
    radians = optional_float_config(
        trace,
        ("success_grasp_orientation_rad", "grasp_orientation_threshold_rad"),
    )
    if math.isfinite(radians):
        return radians, "metadata_rad"
    degrees = optional_float_config(
        trace,
        ("success_grasp_orientation_deg", "grasp_orientation_threshold_deg"),
    )
    if math.isfinite(degrees):
        return math.radians(degrees), "metadata_deg"
    return float("nan"), "legacy_distance_closure_fallback"


def lift_grasp_masks(trace: dict, length: int) -> dict[str, np.ndarray]:
    """Reconstruct the simple deployment-observable grasp conditions used by Lift success.

    A side is grasp-valid when its EE is within the configured grasp-point distance
    and its measured gripper closure exceeds the configured minimum.  This keeps
    analysis consistent with the simplified environment instead of relying on
    legacy ``*_grasp_ok`` fields whose meaning changed across reward versions.
    """

    _, distance_threshold, closure_threshold, _ = lift_thresholds(trace)
    orientation_threshold, orientation_source = lift_orientation_threshold(trace)
    left_dist = scalar_series(trace, "left_target_dist", length)
    right_dist = scalar_series(trace, "right_target_dist", length)
    left_closure = scalar_series(trace, "left_closure", length)
    right_closure = scalar_series(trace, "right_closure", length)

    thresholds_available = math.isfinite(distance_threshold) and math.isfinite(closure_threshold)
    left_reach = left_dist <= distance_threshold if math.isfinite(distance_threshold) else np.zeros(length, dtype=bool)
    right_reach = (
        right_dist <= distance_threshold if math.isfinite(distance_threshold) else np.zeros(length, dtype=bool)
    )
    left_grasp = (
        left_reach & (left_closure >= closure_threshold) if thresholds_available else np.zeros(length, dtype=bool)
    )
    right_grasp = (
        right_reach & (right_closure >= closure_threshold) if thresholds_available else np.zeros(length, dtype=bool)
    )
    orientation_available = (
        math.isfinite(orientation_threshold)
        and len(trace.get("left_target_quat_error", [])) >= length
        and len(trace.get("right_target_quat_error", [])) >= length
    )
    if orientation_available:
        left_grasp &= scalar_series(trace, "left_target_quat_error", length) <= orientation_threshold
        right_grasp &= scalar_series(trace, "right_target_quat_error", length) <= orientation_threshold
        event_source = "distance_closure_orientation"
    elif len(trace.get("left_grasp_ok", [])) >= length and len(trace.get("right_grasp_ok", [])) >= length:
        # Legacy traces often stored the environment's complete grasp predicate
        # but omitted its numeric orientation threshold from metadata.
        left_grasp = bool_series(trace, "left_grasp_ok", length)
        right_grasp = bool_series(trace, "right_grasp_ok", length)
        event_source = "legacy_recorded_grasp_ok"
    else:
        event_source = "legacy_distance_closure_fallback" if thresholds_available else "unavailable"
    return {
        "left_reach": left_reach,
        "right_reach": right_reach,
        "dual_reach": left_reach & right_reach,
        "left_grasp": left_grasp,
        "right_grasp": right_grasp,
        "dual_grasp": left_grasp & right_grasp,
        "orientation_fallback": not orientation_available,
        "orientation_source": orientation_source if orientation_available else event_source,
        "event_source": event_source,
    }


def recorded_len(trace: dict) -> int:
    """Return the number of rows physically stored in a trace."""

    candidates = [
        len(trace.get("reward_mean", [])),
        len(trace.get("left_motion_context", [])),
        len(trace.get("left_motion", [])),
        len(trace.get("success_signal", [])),
        int(trace.get("episode_length", 0) or 0),
        int(trace.get("num_steps", 0) or 0),
    ]
    return max(candidates)


def episode_len(trace: dict) -> int:
    """Return rows used for analysis, trimming only legacy auto-reset tails."""

    length = recorded_len(trace)
    schema = str(metadata(trace).get("trace_schema", ""))
    return trim_reset_tail_length(trace, length) if schema in ("", "openarm_motion_context/v1") else length


def trim_reset_tail_length(trace: dict, length: int) -> int:
    """Drop trailing reset-buffer frames such as [0, 0, 1] context after done."""

    while length > 1:
        left = vector_series_raw(trace, "left_motion_context", 3, length)
        right = vector_series_raw(trace, "right_motion_context", 3, length)
        if left.shape[0] < length or right.shape[0] < length:
            break
        last_left = left[length - 1]
        last_right = right[length - 1]
        reset_context = np.allclose(last_left, np.asarray([0.0, 0.0, 1.0]), atol=1.0e-8) and np.allclose(
            last_right, np.asarray([0.0, 0.0, 1.0]), atol=1.0e-8
        )
        if not reset_context:
            break
        length -= 1
    return length


def vector_series_raw(trace: dict, key: str, width: int, length: int) -> np.ndarray:
    raw = np.asarray(trace.get(key, []), dtype=float)
    out = np.zeros((length, width), dtype=float)
    if raw.ndim == 2 and raw.shape[0] > 0:
        n = min(length, raw.shape[0])
        k = min(width, raw.shape[1])
        out[:n, :k] = raw[:n, :k]
    elif raw.ndim == 1 and raw.size >= width:
        out[0, :width] = raw[:width]
    return out


def scalar_series(trace: dict, key: str, length: int, fill: float = 0.0) -> np.ndarray:
    raw = np.asarray(trace.get(key, []), dtype=float)
    if raw.ndim > 1:
        raw = raw.reshape(raw.shape[0], -1)[:, 0]
    out = np.full(length, float(fill), dtype=float)
    n = min(length, raw.shape[0] if raw.ndim else 1)
    if n > 0 and raw.size:
        out[:n] = raw.reshape(-1)[:n]
    return out


def bool_series(trace: dict, key: str, length: int) -> np.ndarray:
    raw = trace.get(key, [])
    out = np.zeros(length, dtype=bool)
    if isinstance(raw, list):
        n = min(length, len(raw))
        if n:
            out[:n] = np.asarray(raw[:n], dtype=bool)
    return out


def vector_series(trace: dict, key: str, width: int, length: int) -> np.ndarray:
    return vector_series_raw(trace, key, width, length)


def context_series(trace: dict, side: str, length: int) -> np.ndarray:
    vector = vector_series(trace, f"{side}_motion_context", 3, length)
    if np.any(vector):
        return vector
    smoothness_key = f"{side}_stability_trend" if trace.get(f"{side}_stability_trend") else f"{side}_action_smoothness"
    return np.stack(
        [
            scalar_series(trace, f"{side}_translation_trend", length),
            scalar_series(trace, f"{side}_rotation_trend", length),
            scalar_series(trace, smoothness_key, length),
        ],
        axis=-1,
    )


def payload_layout(trace: dict, method: str) -> list[str]:
    if method == "none":
        return []
    layout = metadata(trace).get("communication_payload_layout", [])
    if isinstance(layout, list) and layout:
        return [str(name) for name in layout]
    if method == "motion_only":
        return list(MOTION_NAMES)
    if method == "context_only":
        return list(CONTEXT_NAMES)
    if method in ("motion_context", "motion_context_la", "motion_context_s"):
        return [*MOTION_NAMES, *CONTEXT_NAMES]
    if method == "previous_action":
        return [f"previous_action_{index}" for index in range(MESSAGE_DIMS[method])]
    if method == "full_partner_observation":
        return [f"partner_observation_{index}" for index in range(MESSAGE_DIMS[method])]
    return []


def payload_series(trace: dict, side: str, method: str, length: int) -> tuple[np.ndarray, list[str]]:
    """Read the raw partner slot seen by a policy actor, before its running scaler."""

    names = payload_layout(trace, method)
    width = len(names)
    if width == 0:
        return np.zeros((length, 0), dtype=float), names
    if has_vector_series(trace, f"{side}_policy_message", width, length):
        return vector_series(trace, f"{side}_policy_message", width, length), names
    raw_message = np.asarray(trace.get(f"{side}_message", []), dtype=float)
    if raw_message.ndim == 2 and raw_message.shape[0] > 0:
        return vector_series(trace, f"{side}_message", width, length), names
    motion = vector_series(trace, f"{side}_motion", 3, length)
    context = context_series(trace, side, length)
    if method == "motion_only":
        return motion, names
    if method == "context_only":
        return context, names
    if method in ("motion_context", "motion_context_la", "motion_context_s"):
        if method == "motion_context_la":
            context[:, 2] = 0.0
        elif method == "motion_context_s":
            context[:, :2] = 0.0
        return np.concatenate([motion, context], axis=-1), names
    return np.zeros((length, 0), dtype=float), []


def raw_payload_series(trace: dict, side: str, method: str, length: int) -> tuple[np.ndarray, list[str]]:
    """Reconstruct the outgoing physical descriptor before policy preprocessing."""

    names = payload_layout(trace, method)
    motion = vector_series(trace, f"{side}_motion", 3, length)
    context = context_series(trace, side, length)
    if method == "motion_only":
        return motion, names
    if method == "context_only":
        return context, names
    if method in ("motion_context", "motion_context_la", "motion_context_s"):
        if method == "motion_context_la":
            context[:, 2] = 0.0
        elif method == "motion_context_s":
            context[:, :2] = 0.0
        return np.concatenate([motion, context], axis=-1), names
    if method == "previous_action":
        arm = vector_series(trace, f"{side}_arm_action", 7, length)
        grip = scalar_series(trace, f"{side}_action_grip", length)[:, None]
        return np.concatenate([arm, grip], axis=-1), names
    # The complete raw 30D own observation was not stored by legacy traces.
    return np.zeros((length, 0), dtype=float), []


def saturation_ratio(trace: dict, method: str, feature: str, values: np.ndarray) -> float:
    """Measure saturation against the feature's actual configured bound."""

    if values.size == 0:
        return float("nan")
    if feature in MOTION_NAMES or feature.startswith("previous_action_"):
        bound = 1.0
    elif feature in CONTEXT_NAMES or feature == "action_smoothness":
        bound = 1.0
    elif method == "full_partner_observation":
        return float("nan")
    else:
        return float("nan")
    return safe_mean(np.abs(values) >= 0.99 * max(bound, 1.0e-6))


def smooth(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or values.shape[0] < window:
        return values
    kernel = np.ones(window, dtype=float) / float(window)
    if values.ndim == 1:
        return np.convolve(values, kernel, mode="same")
    return np.stack([np.convolve(values[:, i], kernel, mode="same") for i in range(values.shape[1])], axis=-1)


def safe_mean(values) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.mean(arr)) if arr.size else float("nan")


def safe_std(values) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0


def safe_percentile(values, percentile: float) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.percentile(arr, percentile)) if arr.size else float("nan")


def has_vector_series(trace: dict, key: str, width: int, length: int) -> bool:
    raw = np.asarray(trace.get(key, []), dtype=float)
    return raw.ndim == 2 and raw.shape[0] >= length and raw.shape[1] >= width


def ee_path_length(trace: dict, side: str, length: int) -> tuple[float, str]:
    for key in (f"{side}_ee_pos_b", f"{side}_ee_position_b", f"{side}_ee_pos_w"):
        if has_vector_series(trace, key, 3, length):
            positions = vector_series(trace, key, 3, length)
            delta = np.diff(positions, axis=0)
            finite = np.all(np.isfinite(delta), axis=1)
            return float(np.sum(np.linalg.norm(delta[finite], axis=-1))), key
    return float("nan"), "unavailable"


def integrated_angular_motion(trace: dict, side: str, length: int, dt: float) -> tuple[float, str, bool]:
    for key in (f"{side}_angular_speed_raw", f"{side}_ee_angular_speed_raw"):
        if len(trace.get(key, [])) >= length and math.isfinite(dt):
            speed = np.abs(scalar_series(trace, key, length))
            return float(np.sum(speed[np.isfinite(speed)]) * dt), key, False
    for key in (f"{side}_ee_quat_b", f"{side}_ee_quat_w"):
        if has_vector_series(trace, key, 4, length):
            quat = vector_series(trace, key, 4, length)
            norms = np.linalg.norm(quat, axis=1, keepdims=True)
            valid = np.isfinite(norms[:, 0]) & (norms[:, 0] > 1.0e-9)
            quat = quat / np.maximum(norms, 1.0e-9)
            dots = np.abs(np.sum(quat[1:] * quat[:-1], axis=1))
            pair_valid = valid[1:] & valid[:-1]
            angles = 2.0 * np.arccos(np.clip(dots[pair_valid], 0.0, 1.0))
            return float(np.sum(angles)), f"{key}_quaternion_difference", True
    return float("nan"), "unavailable", False


def action_variation(trace: dict, side: str, length: int) -> tuple[float, float, str, bool]:
    key = f"{side}_action_variation_raw"
    if len(trace.get(key, [])) >= length:
        values = np.abs(scalar_series(trace, key, length))
        values = values[np.isfinite(values)]
        return safe_mean(values), float(np.sum(values)) if values.size else float("nan"), key, False
    key = f"{side}_arm_action"
    if has_vector_series(trace, key, 7, length) and length > 1:
        values = np.linalg.norm(np.diff(vector_series(trace, key, 7, length), axis=0), axis=-1)
        return safe_mean(values), float(np.sum(values)), f"{key}_difference", True
    return float("nan"), float("nan"), "unavailable", False


def first_step(mask: np.ndarray) -> float:
    idx = np.flatnonzero(mask)
    return float(idx[0]) if idx.size else float("nan")


def first_post_reset_event(mask: np.ndarray) -> float:
    """Return the first event after discarding a stale true run at reset."""

    values = np.asarray(mask, dtype=bool)
    if not values.size:
        return float("nan")
    start = 0
    if values[0]:
        false_idx = np.flatnonzero(~values)
        if not false_idx.size:
            return float("nan")
        start = int(false_idx[0]) + 1
    event_idx = np.flatnonzero(values[start:])
    return float(start + event_idx[0]) if event_idx.size else float("nan")


def max_consecutive(mask: np.ndarray) -> int:
    best = 0
    run = 0
    for value in mask.astype(bool):
        run = run + 1 if value else 0
        best = max(best, run)
    return int(best)


def first_success_step(trace: dict, length: int) -> float:
    value = trace.get("success_step", None)
    if value is not None:
        try:
            return float(value)
        except (TypeError, ValueError):
            pass
    return first_step(bool_series(trace, "success_signal", length))


def bool_success(trace: dict) -> bool:
    if bool(trace.get("terminal_success", False)):
        return True
    if bool(trace.get("success", False)):
        return True
    return bool(np.any(bool_series(trace, "success_signal", episode_len(trace))))


def infer_seed(trace: dict, override: int | None) -> int:
    if override is not None:
        return int(override)
    meta = metadata(trace)
    for key in ("training_seed", "seed", "random_seed"):
        if key in meta:
            try:
                return int(meta[key])
            except (TypeError, ValueError):
                pass
    path = Path(str(trace.get("_path", "")))
    for part in path.parts[::-1]:
        match = re.search(r"(?:seed|s)[_-]?(\d+)", part, re.IGNORECASE)
        if match:
            return int(match.group(1))
    return 0


def normalize_method(value: str | None) -> str:
    text = str(value or "").strip().lower().replace("-", "_")
    aliases = {
        "no_intent": "none",
        "no_share": "none",
        "motion": "motion_only",
        "context": "context_only",
        "full_observation": "full_partner_observation",
        "full_obs": "full_partner_observation",
        "previous_action_share": "previous_action",
        "motion_context_share": "motion_context",
    }
    normalized = aliases.get(text, text)
    return normalized if normalized in METHOD_LABELS else "unknown"


def infer_method(trace: dict, override: str | None) -> str:
    if override is not None:
        return normalize_method(override)
    meta = metadata(trace)
    # A zero-message baseline may retain the full decoder for equal capacity.
    # Its architecture label must not turn it into a communicating method.
    if normalize_method(meta.get("sharing_mode")) == "none":
        return "none"
    for key in ("communication_mode", "method", "sharing_mode"):
        if key in meta:
            method = normalize_method(meta[key])
            if method in METHOD_LABELS:
                return method

    path_text = str(trace.get("_path", "")).lower()
    # Prefer the longer/more specific names before generic substrings.
    for method in (
        "full_partner_observation",
        "previous_action",
        "motion_context_la",
        "motion_context_s",
        "motion_context",
        "motion_only",
        "context_only",
        "none",
    ):
        if method in path_text:
            return method
    if "full" in path_text and "obs" in path_text:
        return "full_partner_observation"
    if "prev" in path_text and "action" in path_text:
        return "previous_action"
    if "none" in path_text or "no_comm" in path_text or "no_share" in path_text:
        return "none"
    return "unknown"


def infer_task(trace: dict, override: str | None) -> str:
    meta = metadata(trace)
    text = override or " ".join(str(meta.get(key, "")) for key in ("task", "checkpoint", "reward"))
    if not override:
        text += " " + str(trace.get("_path", ""))
    lower = text.lower()
    if "peg" in lower or "hole" in lower:
        return "Peg"
    if "lift" in lower or "klt" in lower or "re_" in lower:
        return "Lift"
    if override:
        raise ValueError(f"Unknown OpenArm task override: {override!r}; use Lift, Peg, or a registered task name")
    if "lateral_error" in trace or "tip_dist" in trace:
        return "Peg"
    return "Lift"


def auc_progress(trace: dict, task: str, length: int) -> float:
    """Return normalized task-progress AUC over an evaluation episode."""

    if length <= 0:
        return float("nan")
    if task == "Peg":
        progress = scalar_series(trace, "hprog", length)
        if not np.any(progress):
            depth = scalar_series(trace, "insertion_depth", length)
            target_depth = trace_float_config(
                trace,
                ("target_insertion_depth", "success_depth_threshold"),
                0.05,
            )
            progress = np.clip(depth / max(target_depth, 1.0e-6), 0.0, 1.0)
    else:
        progress = scalar_series(trace, "hprog", length)
        if not np.any(progress):
            dz = scalar_series(trace, "object_dz", length)
            lift_target_height, _, _, _ = lift_thresholds(trace)
            if not math.isfinite(lift_target_height) or lift_target_height <= 0.0:
                return float("nan")
            progress = np.clip(dz / lift_target_height, 0.0, 1.0)
    return float(np.trapz(np.clip(progress, 0.0, 1.0), dx=1.0) / max(length - 1, 1))


def classify_lift_failure(trace: dict, length: int, success: bool) -> str:
    """Classify Lift failures using the same simple grasp geometry as success."""

    if success:
        return "success"
    explicit = trace.get("termination_reason", metadata(trace).get("termination_reason", ""))
    if explicit in {"drop", "far", "tilt_fail", "invalid", "timeout", "unknown"}:
        return "invalid_state" if explicit == "invalid" else explicit
    if bool(trace.get("truncated_at_end", False)):
        return "timeout"
    if bool(np.any(bool_series(trace, "drop", length))):
        return "drop"
    if bool(np.any(bool_series(trace, "far", length))):
        return "far"
    if bool(np.any(bool_series(trace, "tilt_fail", length))):
        return "tilt_fail"
    if bool(np.any(bool_series(trace, "invalid", length))):
        return "invalid_state"
    if not length:
        return "invalid_state"

    reward = scalar_series(trace, "reward_mean", length)
    if np.any(~np.isfinite(reward)):
        return "invalid_state"

    hprog = scalar_series(trace, "hprog", length)
    tilt = scalar_series(trace, "object_tilt_deg", length)
    left_dist = scalar_series(trace, "left_target_dist", length)
    right_dist = scalar_series(trace, "right_target_dist", length)
    grasp = lift_grasp_masks(trace, length)
    _, grasp_distance, _, _ = lift_thresholds(trace)

    if np.nanmax(hprog) > 0.45 and hprog[-1] < 0.20:
        return "drop"
    max_tilt = optional_float_config(trace, ("max_tilt_deg", "termination_tilt_threshold_deg"))
    if math.isfinite(max_tilt) and np.nanmax(tilt) > max_tilt:
        return "excessive_tilt"
    if math.isfinite(grasp_distance) and (
        np.nanmin(left_dist) > grasp_distance or np.nanmin(right_dist) > grasp_distance
    ):
        return "far_from_target"
    if (np.any(grasp["left_grasp"]) or np.any(grasp["right_grasp"])) and not np.any(grasp["dual_grasp"]):
        return "single_grasp"
    return "timeout"


def classify_peg_failure(trace: dict, length: int, success: bool) -> str:
    if success:
        return "success"
    if not length:
        return "invalid_state"
    tip = scalar_series(trace, "tip_dist", length)
    lateral = scalar_series(trace, "lateral_error", length)
    axis = scalar_series(trace, "axis_alignment", length)
    depth = scalar_series(trace, "insertion_depth", length)
    meta = metadata(trace)
    lateral_threshold = float(meta.get("success_lateral_threshold", 0.012))
    axis_threshold = float(meta.get("success_axis_threshold", 0.85))
    depth_threshold = float(meta.get("success_depth_threshold", 0.045))
    wall_lateral = float(meta.get("wall_penetration_lateral_threshold", 0.015))
    wall_depth = float(meta.get("wall_penetration_depth_threshold", 0.0))
    side_wall = np.any((depth > wall_depth) & (lateral > wall_lateral))
    if side_wall:
        return "side_wall_penetration"
    if np.any(tip) and np.nanmin(tip) > 0.05:
        return "far_from_hole"
    if (np.any(axis) and np.nanmax(axis) < axis_threshold) or (
        np.any(lateral) and np.nanmin(lateral) > lateral_threshold
    ):
        return "alignment_fail"
    if np.nanmax(depth) < depth_threshold:
        return "insufficient_depth"
    return "timeout"


def summarize_trace(trace: dict, args) -> dict[str, float | int | str]:
    length = episode_len(trace)
    task = infer_task(trace, args.task)
    method = infer_method(trace, args.method)
    seed = infer_seed(trace, args.seed)
    success = bool_success(trace)
    reward = scalar_series(trace, "reward_mean", length)
    reward_left = scalar_series(trace, "reward_left", length)
    reward_right = scalar_series(trace, "reward_right", length)
    hprog = scalar_series(trace, "hprog", length)
    object_z = scalar_series(trace, "object_z", length)
    object_dz = scalar_series(trace, "object_dz", length)
    tilt = scalar_series(trace, "object_tilt_deg", length)
    left_dist = scalar_series(trace, "left_target_dist", length)
    right_dist = scalar_series(trace, "right_target_dist", length)
    left_closure = scalar_series(trace, "left_closure", length)
    right_closure = scalar_series(trace, "right_closure", length)
    hold = bool_series(trace, "hold_ok", length)
    lateral = scalar_series(trace, "lateral_error", length)
    axis = scalar_series(trace, "axis_alignment", length)
    depth = scalar_series(trace, "insertion_depth", length)
    tip = scalar_series(trace, "tip_dist", length)
    keypoint = scalar_series(trace, "keypoint_dist", length)
    left_action = vector_series(trace, "left_arm_action", 7, length)
    right_action = vector_series(trace, "right_arm_action", 7, length)

    if length > 1:
        left_delta_raw = np.diff(left_action, axis=0)
        right_delta_raw = np.diff(right_action, axis=0)
        left_action_delta_l2 = np.linalg.norm(left_delta_raw, axis=-1)
        right_action_delta_l2 = np.linalg.norm(right_delta_raw, axis=-1)
        left_action_rate_mse = np.mean(np.square(left_delta_raw), axis=-1)
        right_action_rate_mse = np.mean(np.square(right_delta_raw), axis=-1)
    else:
        left_action_delta_l2 = np.zeros(0)
        right_action_delta_l2 = np.zeros(0)
        left_action_rate_mse = np.zeros(0)
        right_action_rate_mse = np.zeros(0)

    left_motion = vector_series(trace, "left_motion", 3, length)
    right_motion = vector_series(trace, "right_motion", 3, length)
    left_context = context_series(trace, "left", length)
    right_context = context_series(trace, "right", length)
    success_step = first_success_step(trace, length) if success else float("nan")
    dt, dt_source = control_dt(trace)

    lift_fail = classify_lift_failure(trace, length, success)
    peg_fail = classify_peg_failure(trace, length, success)
    failure_reason = peg_fail if task == "Peg" else lift_fail
    mean_lifted_tilt = safe_mean(tilt[hprog > 0.2]) if np.any(hprog > 0.2) else 0.0
    meta = metadata(trace)
    lateral_threshold = float(meta.get("success_lateral_threshold", 0.012))
    axis_threshold = float(meta.get("success_axis_threshold", 0.85))
    depth_threshold = float(meta.get("success_depth_threshold", 0.045))
    wall_lateral = float(meta.get("wall_penetration_lateral_threshold", 0.015))
    wall_depth = float(meta.get("wall_penetration_depth_threshold", 0.0))

    lift_target_height, grasp_distance, min_closure, success_tilt_deg = lift_thresholds(trace)
    lift_masks = lift_grasp_masks(trace, length)
    left_reach_step = first_step(lift_masks["left_reach"]) if task == "Lift" else float("nan")
    right_reach_step = first_step(lift_masks["right_reach"]) if task == "Lift" else float("nan")
    left_grasp_step = first_post_reset_event(lift_masks["left_grasp"]) if task == "Lift" else float("nan")
    right_grasp_step = first_post_reset_event(lift_masks["right_grasp"]) if task == "Lift" else float("nan")
    dual_grasp_step = first_post_reset_event(lift_masks["dual_grasp"]) if task == "Lift" else float("nan")
    # A tiny reset-height fluctuation is not Lift onset. Require meaningful
    # height progress and the task's bilateral grasp feasibility together.
    lift_onset_mask = (hprog >= 0.2) & lift_masks["dual_grasp"]
    if math.isfinite(dual_grasp_step):
        lift_onset_mask[: int(dual_grasp_step)] = False
    lift_step = first_step(lift_onset_mask) if task == "Lift" else float("nan")

    left_path, left_path_source = ee_path_length(trace, "left", length)
    right_path, right_path_source = ee_path_length(trace, "right", length)
    left_angular, left_angular_source, left_angular_fallback = integrated_angular_motion(trace, "left", length, dt)
    right_angular, right_angular_source, right_angular_fallback = integrated_angular_motion(trace, "right", length, dt)
    left_action_mean, left_action_integral, left_action_source, left_action_fallback = action_variation(
        trace, "left", length
    )
    right_action_mean, right_action_integral, right_action_source, right_action_fallback = action_variation(
        trace, "right", length
    )

    def finite_gap(left: float, right: float) -> float:
        return abs(left - right) if math.isfinite(left) and math.isfinite(right) else float("nan")

    def finite_interval(start: float, end: float) -> float:
        return end - start if math.isfinite(start) and math.isfinite(end) else float("nan")

    row = {
        "task": task,
        "method": METHOD_LABELS.get(method, method),
        "communication_mode": method,
        "diagnostic_mask": str(metadata(trace).get("diagnostic_mask", trace.get("diagnostic_mask", "none"))),
        "msg_dim": 0
        if method == "none"
        else int(metadata(trace).get("communication_payload_dim", MESSAGE_DIMS.get(method, 0))),
        "seed": seed,
        "episode_id": int(trace.get("episode_index", 0)),
        "episode_path": str(trace.get("_path", "")),
        "checkpoint": str(metadata(trace).get("checkpoint", "")),
        "success": int(success),
        "termination_reason": failure_reason,
        "episode_return": float(np.sum(reward)) if length else 0.0,
        "mean_return": safe_mean(reward),
        "final_return": float(reward[-1]) if length else 0.0,
        "left_return": float(np.sum(reward_left)) if length else 0.0,
        "right_return": float(np.sum(reward_right)) if length else 0.0,
        "episode_length": int(length),
        "time_to_success": success_step,
        "control_dt": dt,
        "control_dt_source": dt_source,
        "time_to_success_sec": success_step * dt if math.isfinite(success_step) and math.isfinite(dt) else float("nan"),
        "auc": auc_progress(trace, task, length),
        "invalid": int(failure_reason == "invalid_state"),
        "lift_target_height": lift_target_height,
        "success_grasp_distance": grasp_distance,
        "success_min_closure": min_closure,
        "success_tilt_threshold_deg": success_tilt_deg,
        "initial_object_z": float(object_z[0]) if length else float("nan"),
        "final_object_z": float(object_z[-1]) if length else float("nan"),
        "initial_object_dz": float(object_dz[0]) if length else float("nan"),
        "final_object_dz": float(object_dz[-1]) if length else float("nan"),
        "max_object_height_delta": float(np.nanmax(object_dz)) if length else float("nan"),
        "final_object_height_delta": float(object_dz[-1]) if length else float("nan"),
        "max_object_tilt_deg": float(np.nanmax(tilt)) if length else 0.0,
        "mean_lifted_tilt_deg": float(mean_lifted_tilt),
        "min_left_target_dist": float(np.nanmin(left_dist)) if length else float("nan"),
        "min_right_target_dist": float(np.nanmin(right_dist)) if length else float("nan"),
        "final_left_target_dist": float(left_dist[-1]) if length else float("nan"),
        "final_right_target_dist": float(right_dist[-1]) if length else float("nan"),
        "first_left_reach_step": left_reach_step,
        "first_right_reach_step": right_reach_step,
        "reach_sync_gap": finite_gap(left_reach_step, right_reach_step),
        "first_left_grasp_transition_step": left_grasp_step,
        "first_right_grasp_transition_step": right_grasp_step,
        # Legacy aliases are retained for downstream notebooks.
        "first_left_grasp_step": left_grasp_step,
        "first_right_grasp_step": right_grasp_step,
        "grasp_sync_gap": finite_gap(left_grasp_step, right_grasp_step),
        "first_dual_grasp_step": dual_grasp_step,
        "first_lift_step": lift_step,
        "dual_grasp_to_lift": finite_interval(dual_grasp_step, lift_step),
        "lift_to_success": finite_interval(lift_step, success_step),
        "grasp_transition_source": lift_masks.get("event_source", "not_applicable")
        if task == "Lift"
        else "not_applicable",
        "grasp_orientation_fallback": int(lift_masks.get("orientation_fallback", False)) if task == "Lift" else 0,
        "success_step": success_step,
        "max_hold_steps": max_consecutive(hold | bool_series(trace, "success_signal", length)),
        # ``mean_action_rate`` now matches the environment action-rate penalty:
        # mean((a_t - a_{t-1})^2).  L2 action change is retained separately.
        "mean_action_rate": safe_mean(np.concatenate([left_action_rate_mse, right_action_rate_mse])),
        "mean_action_rate_mse": safe_mean(np.concatenate([left_action_rate_mse, right_action_rate_mse])),
        "mean_action_delta_l2": safe_mean(np.concatenate([left_action_delta_l2, right_action_delta_l2])),
        "left_ee_path_length": left_path,
        "right_ee_path_length": right_path,
        "mean_ee_path_length": safe_mean([left_path, right_path]),
        "total_ee_path_length": left_path + right_path
        if math.isfinite(left_path) and math.isfinite(right_path)
        else float("nan"),
        "ee_path_source": f"L:{left_path_source};R:{right_path_source}",
        "left_integrated_angular_motion": left_angular,
        "right_integrated_angular_motion": right_angular,
        "mean_integrated_angular_motion": safe_mean([left_angular, right_angular]),
        "integrated_angular_motion_source": f"L:{left_angular_source};R:{right_angular_source}",
        "integrated_angular_motion_fallback": int(left_angular_fallback or right_angular_fallback),
        "mean_action_variation": safe_mean([left_action_mean, right_action_mean]),
        "integrated_action_variation": safe_mean([left_action_integral, right_action_integral]),
        "action_variation_source": f"L:{left_action_source};R:{right_action_source}",
        "action_variation_fallback": int(left_action_fallback or right_action_fallback),
        "mean_joint_velocity": float("nan"),
        "mean_grasp_imbalance": safe_mean(scalar_series(trace, "grasp_imbalance", length)),
        "mean_closure_gap": safe_mean(np.abs(left_closure - right_closure)),
        "initial_tip_distance": float(tip[0]) if length and np.any(tip) else float("nan"),
        "initial_lateral_error": float(lateral[0]) if length and np.any(lateral) else float("nan"),
        "initial_axis_alignment": float(axis[0]) if length and np.any(axis) else float("nan"),
        "initial_insertion_depth": float(depth[0]) if length and np.any(depth) else float("nan"),
        "min_tip_distance": float(np.nanmin(tip)) if length and np.any(tip) else float("nan"),
        "min_lateral_error": float(np.nanmin(lateral)) if length and np.any(lateral) else float("nan"),
        "max_axis_alignment": float(np.nanmax(axis)) if length and np.any(axis) else float("nan"),
        "max_insertion_depth": float(np.nanmax(depth)) if length and np.any(depth) else float("nan"),
        "final_keypoint_distance": float(keypoint[-1]) if length and np.any(keypoint) else float("nan"),
        "first_alignment_step": first_step((lateral < lateral_threshold) & (axis > axis_threshold)),
        "first_insertion_step": first_step(depth > 0.0),
        "first_success_condition_step": first_step(
            (lateral < lateral_threshold) & (axis > axis_threshold) & (depth > depth_threshold)
        ),
        "side_wall_penetration": int(np.any((depth > wall_depth) & (lateral > wall_lateral))),
        "left_motion_norm_mean": safe_mean(np.linalg.norm(left_motion, axis=-1)),
        "right_motion_norm_mean": safe_mean(np.linalg.norm(right_motion, axis=-1)),
        "left_context_mean": safe_mean(left_context.reshape(-1)),
        "right_context_mean": safe_mean(right_context.reshape(-1)),
    }
    for side, context in (("left", left_context), ("right", right_context)):
        for idx, name in enumerate(CONTEXT_NAMES):
            row[f"{side}_{name}_mean"] = safe_mean(context[:, idx])
            row[f"{side}_{name}_final"] = float(context[-1, idx]) if length else float("nan")
    return row


def csv_fieldnames(rows: list[dict]) -> list[str]:
    keys: list[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                keys.append(key)
    return keys


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    keys = csv_fieldnames(rows)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def group_rows(rows: list[dict], keys: tuple[str, ...]) -> dict[tuple, list[dict]]:
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row.get(key) for key in keys)].append(row)
    return dict(groups)


def summarize_seed(rows: list[dict]) -> list[dict]:
    metric_map = {
        "Success Rate": ("success", False),
        "Mean Return": ("mean_return", False),
        "Episode Return": ("episode_return", False),
        "AUC": ("auc", False),
        "Time to Success": ("time_to_success", True),
        "Time to Success Sec": ("time_to_success_sec", True),
        "Reach Gap": ("reach_sync_gap", False),
        "Grasp Gap": ("grasp_sync_gap", False),
        "Dual Grasp to Lift": ("dual_grasp_to_lift", False),
        "Lift to Success": ("lift_to_success", True),
        "Mean EE Path Length All": ("mean_ee_path_length", False),
        "Mean EE Path Length Success": ("mean_ee_path_length", True),
        "Integrated Angular Motion All": ("mean_integrated_angular_motion", False),
        "Integrated Angular Motion Success": ("mean_integrated_angular_motion", True),
        "Mean Action Variation": ("mean_action_variation", False),
        "Integrated Action Variation": ("integrated_action_variation", False),
    }
    out = []
    for (task, method, mode, msg_dim, seed), group in sorted(
        group_rows(rows, ("task", "method", "communication_mode", "msg_dim", "seed")).items()
    ):
        row_out = {
            "Task": task,
            "Method": method,
            "Communication Mode": mode,
            "Msg Dim": msg_dim,
            "Seed": seed,
            "Episodes": len(group),
            "Successful Episodes": sum(int(row["success"]) for row in group),
            "Mean Episode Length": safe_mean([row["episode_length"] for row in group]),
        }
        for output_name, (source_name, success_only) in metric_map.items():
            selected = [row for row in group if not success_only or int(row["success"]) == 1]
            row_out[output_name] = safe_mean([row.get(source_name, float("nan")) for row in selected])
        out.append(row_out)
    return out


def fmt_mean_std(mean: float, std: float) -> str:
    if not math.isfinite(mean):
        return ""
    return f"{mean:.3f} +/- {std:.3f}"


def summarize_method(seed_rows: list[dict]) -> list[dict]:
    metrics = (
        "Success Rate",
        "Time to Success",
        "Time to Success Sec",
        "Mean Return",
        "Episode Return",
        "AUC",
        "Reach Gap",
        "Grasp Gap",
        "Dual Grasp to Lift",
        "Lift to Success",
        "Mean EE Path Length All",
        "Mean EE Path Length Success",
        "Integrated Angular Motion All",
        "Integrated Angular Motion Success",
        "Mean Action Variation",
        "Integrated Action Variation",
    )
    out = []
    for (task, method, mode, msg_dim), group in sorted(
        group_rows(seed_rows, ("Task", "Method", "Communication Mode", "Msg Dim")).items()
    ):
        row_out = {
            "Task": task,
            "Method": method,
            "Communication Mode": mode,
            "Msg Dim": msg_dim,
            "Seeds": len(group),
            "Episodes": sum(int(row.get("Episodes", 0)) for row in group),
            "Successful Episodes": sum(int(row.get("Successful Episodes", 0)) for row in group),
        }
        for metric in metrics:
            values = [row.get(metric, float("nan")) for row in group]
            mean = safe_mean(values)
            std = safe_std(values)
            row_out[f"{metric} Mean"] = mean
            row_out[f"{metric} Std"] = std
            row_out[metric] = fmt_mean_std(mean, std)
        # Backward-compatible return columns.
        row_out["Return Mean"] = row_out["Mean Return Mean"]
        row_out["Return Std"] = row_out["Mean Return Std"]
        row_out["Return"] = row_out["Mean Return"]
        out.append(row_out)
    by_task = defaultdict(dict)
    for row in out:
        by_task[row["Task"]][row["Communication Mode"]] = row
    for row in out:
        task_methods = by_task[row["Task"]]
        proposed = task_methods.get("motion_context")
        none = task_methods.get("none")
        comparisons = {
            "None": none,
            "Motion Only": task_methods.get("motion_only"),
            "Context Only": task_methods.get("context_only"),
            "Full Obs": task_methods.get("full_partner_observation"),
        }
        for label, baseline in comparisons.items():
            row[f"Motion Context - {label} Success pp"] = (
                100.0 * (proposed["Success Rate Mean"] - baseline["Success Rate Mean"])
                if proposed and baseline
                else float("nan")
            )
            for metric, short in (
                ("Time to Success Sec Mean", "Time to Success Sec Delta"),
                ("Grasp Gap Mean", "Grasp Gap Delta"),
                ("Mean EE Path Length All Mean", "EE Path Length Delta"),
            ):
                row[f"Motion Context - {label} {short}"] = (
                    proposed[metric] - baseline[metric]
                    if proposed
                    and baseline
                    and math.isfinite(float(proposed[metric]))
                    and math.isfinite(float(baseline[metric]))
                    else float("nan")
                )
    return out


def ablation_summary(seed_rows: list[dict]) -> list[dict]:
    wanted = ("none", "motion_only", "context_only", "motion_context", "motion_context_la", "motion_context_s")
    out = []
    by_task_mode = group_rows(seed_rows, ("Task", "Communication Mode"))
    for task in sorted({row["Task"] for row in seed_rows}):
        row = {"Task": task}
        for mode in wanted:
            group = by_task_mode.get((task, mode), [])
            row[METHOD_LABELS[mode]] = safe_mean([item["Success Rate"] for item in group]) if group else float("nan")
        proposed = row.get("Motion Context", float("nan"))
        motion = row.get("Motion Only", float("nan"))
        context = row.get("Context Only", float("nan"))
        none = row.get("No Communication", float("nan"))
        row["Motion Context - Motion Only"] = proposed - motion if math.isfinite(proposed + motion) else float("nan")
        row["Motion Context - Context Only"] = proposed - context if math.isfinite(proposed + context) else float("nan")
        row["Motion Context - None"] = proposed - none if math.isfinite(proposed + none) else float("nan")
        out.append(row)
    return out


def failure_breakdown(rows: list[dict]) -> list[dict]:
    """Break down failure causes conditional on failure.

    Fractions therefore sum to one across failure reasons for each task/method
    whenever at least one failed episode exists.  Success is intentionally not
    included in the denominator; overall outcome rates live in
    ``outcome_performance.csv``.
    """

    out = []
    failed_rows = [row for row in rows if int(row.get("success", 0)) == 0]
    for (task, method, reason), group in sorted(
        group_rows(failed_rows, ("task", "method", "termination_reason")).items()
    ):
        total_failures = len([row for row in failed_rows if row["task"] == task and row["method"] == method])
        out.append(
            {
                "Task": task,
                "Method": method,
                "Failure Reason": reason,
                "Count": len(group),
                "Failure Count": total_failures,
                "Fraction": len(group) / max(total_failures, 1),
            }
        )
    return out


def outcome_performance(rows: list[dict]) -> list[dict]:
    """Separate cumulative return from per-step reward for success/failure audits."""

    out = []
    for (task, method, success), group in sorted(group_rows(rows, ("task", "method", "success")).items()):
        out.append(
            {
                "Task": task,
                "Method": method,
                "Outcome": "Success" if int(success) else "Failure",
                "Episodes": len(group),
                "Success Rate": safe_mean([row["success"] for row in group]),
                "Cumulative Return Mean": safe_mean([row["episode_return"] for row in group]),
                "Per-Step Reward Mean": safe_mean([row["mean_return"] for row in group]),
                "Episode Length Mean": safe_mean([row["episode_length"] for row in group]),
            }
        )
    return out


def coordination_timing(rows: list[dict]) -> list[dict]:
    out = []
    for row in rows:
        out.append(
            {
                "Task": row["task"],
                "Method": row["method"],
                "Communication Mode": row["communication_mode"],
                "Seed": row["seed"],
                "Episode": row["episode_id"],
                "Success": row["success"],
                "Left Reach Step": row["first_left_reach_step"],
                "Right Reach Step": row["first_right_reach_step"],
                "Reach Gap": row["reach_sync_gap"],
                "Left Grasp Transition Step": row["first_left_grasp_transition_step"],
                "Right Grasp Transition Step": row["first_right_grasp_transition_step"],
                "Grasp Gap": row["grasp_sync_gap"],
                "Dual Grasp Step": row["first_dual_grasp_step"],
                "Lift Step": row["first_lift_step"],
                "Success Step": row["success_step"],
                "Dual Grasp to Lift": row["dual_grasp_to_lift"],
                "Lift to Success": row["lift_to_success"],
                "Grasp Transition Source": row["grasp_transition_source"],
                "Orientation Fallback": row["grasp_orientation_fallback"],
            }
        )
    return out


def motion_efficiency(rows: list[dict]) -> list[dict]:
    keys = (
        "left_ee_path_length",
        "right_ee_path_length",
        "mean_ee_path_length",
        "total_ee_path_length",
        "left_integrated_angular_motion",
        "right_integrated_angular_motion",
        "mean_integrated_angular_motion",
        "mean_action_variation",
        "integrated_action_variation",
        "ee_path_source",
        "integrated_angular_motion_source",
        "integrated_angular_motion_fallback",
        "action_variation_source",
        "action_variation_fallback",
    )
    return [
        {
            "Task": row["task"],
            "Method": row["method"],
            "Communication Mode": row["communication_mode"],
            "Seed": row["seed"],
            "Episode": row["episode_id"],
            "Success": row["success"],
            **{key: row.get(key, float("nan")) for key in keys},
        }
        for row in rows
    ]


def paper_metric_summary(method_rows: list[dict]) -> list[dict]:
    out = []
    for row in method_rows:
        out.append(
            {
                "Task": row["Task"],
                "Method": row["Method"],
                "Communication Mode": row["Communication Mode"],
                "Msg Dim": row["Msg Dim"],
                "Seeds": row["Seeds"],
                "Success Rate Mean": row["Success Rate Mean"],
                "Success Rate Std": row["Success Rate Std"],
                "Time to Success [s] Mean": row["Time to Success Sec Mean"],
                "Time to Success [s] Std": row["Time to Success Sec Std"],
                "Reach Sync Gap [step] Mean": row["Reach Gap Mean"],
                "Reach Sync Gap [step] Std": row["Reach Gap Std"],
                "Grasp Sync Gap [step] Mean": row["Grasp Gap Mean"],
                "Grasp Sync Gap [step] Std": row["Grasp Gap Std"],
                "EE Path Length [m] Mean": row["Mean EE Path Length All Mean"],
                "EE Path Length [m] Std": row["Mean EE Path Length All Std"],
                "Integrated Angular Motion [rad] Mean": row["Integrated Angular Motion All Mean"],
                "Integrated Angular Motion [rad] Std": row["Integrated Angular Motion All Std"],
                "Progress AUC Mean": row["AUC Mean"],
                "Progress AUC Std": row["AUC Std"],
            }
        )
    return out


def message_statistics(traces: list[dict], args, *, policy_input: bool = False) -> list[dict]:
    """Summarize raw outgoing descriptors or raw received actor slots."""

    rows = []
    for trace in traces:
        length = episode_len(trace)
        task = infer_task(trace, args.task)
        method = infer_method(trace, args.method)
        seed = infer_seed(trace, args.seed)
        for side in ("left", "right"):
            if policy_input:
                features, names = payload_series(trace, side, method, length)
                representation = "policy_input_raw_received_pre_scaler"
            else:
                features, names = raw_payload_series(trace, side, method, length)
                representation = "raw_outgoing_physical"
            for idx, name in enumerate(names):
                values = features[:, idx] if features.size else np.zeros(0)
                rows.append(
                    {
                        "Task": task,
                        "Method": METHOD_LABELS.get(method, method),
                        "Communication Mode": method,
                        "Seed": seed,
                        "Representation": representation,
                        "Side": side,
                        "Feature": name,
                        "Mean": safe_mean(values),
                        "Std": safe_std(values),
                        "P5": safe_percentile(values, 5),
                        "P50": safe_percentile(values, 50),
                        "P95": safe_percentile(values, 95),
                        "Saturation Ratio": saturation_ratio(trace, method, name, values),
                        "Zero Ratio": safe_mean(np.abs(values) < 1.0e-6),
                        "Positive Signal Ratio": safe_mean(values > 0.05) if name in CONTEXT_NAMES else float("nan"),
                        "Count": int(values.shape[0]),
                        "Negative Signal Ratio": safe_mean(values < -0.05) if name in CONTEXT_NAMES else float("nan"),
                    }
                )
    return rows


def success_predicate_rates(traces: list[dict], args) -> list[dict]:
    """Report each recorded Lift success gate without treating absent legacy fields as false."""

    keys = (
        "left_distance_ok",
        "right_distance_ok",
        "left_orientation_ok",
        "right_orientation_ok",
        "left_closure_ok",
        "right_closure_ok",
        "left_span_ok",
        "right_span_ok",
        "height_ok",
        "tilt_ok",
        "left_grasp_ok",
        "right_grasp_ok",
        "dual_grasp_ok",
    )
    rows = []
    for trace in traces:
        if infer_task(trace, args.task) != "Lift":
            continue
        length = episode_len(trace)
        row = {
            "episode_path": str(trace.get("_path", "")),
            "episode_index": int(trace.get("episode_index", 0)),
            "communication_mode": infer_method(trace, args.method),
            "diagnostic_mask": str(metadata(trace).get("diagnostic_mask", trace.get("diagnostic_mask", "none"))),
            "success": int(bool_success(trace)),
        }
        for key in keys:
            values = trace.get(key, [])
            row[f"{key}_rate"] = safe_mean(values[:length]) if len(values) >= length and length else float("nan")
        hold_values = trace.get("success_hold_count", [])
        row["max_success_hold_count"] = (
            max(hold_values[:length]) if len(hold_values) >= length and length else float("nan")
        )
        rows.append(row)
    return rows


def plot_message_action_timeline(trace: dict, path: Path) -> None:
    """Plot outgoing Motion/Context beside action changes and wrist angular motion."""

    length = episode_len(trace)
    if length == 0 or infer_method(trace, None) not in ("motion_context", "motion_context_la", "motion_context_s"):
        return
    dt, _ = control_dt(trace)
    time = np.arange(length, dtype=float) * dt if math.isfinite(dt) else np.arange(length, dtype=float)
    x_label = "time [s]" if math.isfinite(dt) else "control step"
    fig, axes = plt.subplots(4, 1, figsize=(14, 11), sharex=True)
    for side, style in (("left", "-"), ("right", "--")):
        payload, _ = raw_payload_series(trace, side, infer_method(trace, None), length)
        motion, context = payload[:, :3], payload[:, 3:6]
        for index, name in enumerate(("mx", "my", "mz")):
            axes[0].plot(time, motion[:, index], style, label=f"{side} {name}", alpha=0.8)
        for index, name in enumerate(("L", "A", "S")):
            axes[1].plot(time, context[:, index], style, label=f"{side} {name}", alpha=0.8)
        actions = vector_series(trace, f"{side}_arm_action", 7, length)
        delta = np.r_[0.0, np.linalg.norm(np.diff(actions, axis=0), axis=1)]
        axes[2].plot(time, delta, style, label=f"{side} |Δa|", alpha=0.8)
        angular = scalar_series(trace, f"{side}_angular_speed_raw", length)
        axes[3].plot(time, angular, style, label=f"{side} |ω|", alpha=0.8)
    for ax, ylabel in zip(axes, ("signed motion", "context", "action change", "angular speed [rad/s]")):
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
        ax.legend(ncol=6, fontsize=8)
    axes[-1].set_xlabel(x_label)
    fig.suptitle(f"Motion Context diagnostics — episode {int(trace.get('episode_index', 0)):03d}")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def lift_phase_masks(trace: dict, length: int) -> dict[str, np.ndarray]:
    """Return analysis-only, mutually exclusive Lift pseudo-phases."""

    hprog = scalar_series(trace, "hprog", length)
    tilt = scalar_series(trace, "object_tilt_deg", length)
    grasp = lift_grasp_masks(trace, length)
    _, _, _, tilt_threshold = lift_thresholds(trace)

    tilt_valid = tilt <= tilt_threshold if math.isfinite(tilt_threshold) else bool_series(trace, "tilt_ok", length)
    candidates = {
        "stabilize": (hprog >= 1.0) & tilt_valid,
        "lift": hprog > 0.0,
        "grasp_transition": grasp["dual_grasp"],
        "alignment": grasp["dual_reach"],
        "approach": np.ones(length, dtype=bool),
    }
    assigned = np.zeros(length, dtype=bool)
    masks = {}
    for phase in ("stabilize", "lift", "grasp_transition", "alignment", "approach"):
        masks[phase] = candidates[phase] & ~assigned
        assigned |= masks[phase]
    return {phase: masks[phase] for phase in ("approach", "alignment", "grasp_transition", "lift", "stabilize")}


def peg_phase_masks(trace: dict, length: int) -> dict[str, np.ndarray]:
    tip = scalar_series(trace, "tip_dist", length)
    lateral = scalar_series(trace, "lateral_error", length)
    axis = scalar_series(trace, "axis_alignment", length)
    depth = scalar_series(trace, "insertion_depth", length)
    meta = metadata(trace)
    lateral_threshold = float(meta.get("success_lateral_threshold", 0.012))
    axis_threshold = float(meta.get("success_axis_threshold", 0.85))
    depth_threshold = float(meta.get("success_depth_threshold", 0.045))
    approach = tip > 0.05
    align = (tip <= 0.05) & ((lateral > lateral_threshold) | (axis < axis_threshold))
    insert = (depth > 0.0) & (depth <= depth_threshold)
    hold = depth > depth_threshold
    return {
        "approach": approach,
        "align": align,
        "insert": insert,
        "hold": hold,
    }


def message_phase_statistics(traces: list[dict], args) -> list[dict]:
    rows = []
    buckets = defaultdict(list)
    for trace in traces:
        length = episode_len(trace)
        task = infer_task(trace, args.task)
        method = infer_method(trace, args.method)
        seed = infer_seed(trace, args.seed)
        masks = peg_phase_masks(trace, length) if task == "Peg" else lift_phase_masks(trace, length)
        for side in ("left", "right"):
            features, names = raw_payload_series(trace, side, method, length)
            for phase, mask in masks.items():
                if not np.any(mask):
                    continue
                for idx, name in enumerate(names):
                    phase_label = LIFT_PHASE_LABELS.get(phase, phase.title()) if task == "Lift" else phase.title()
                    buckets[(task, METHOD_LABELS.get(method, method), method, seed, side, phase_label, name)].extend(
                        features[mask, idx].tolist()
                    )
    for (task, method_label, mode, seed, side, phase, feature), values in sorted(buckets.items()):
        rows.append(
            {
                "Task": task,
                "Method": method_label,
                "Communication Mode": mode,
                "Seed": seed,
                "Side": side,
                "Phase": phase,
                "Feature": feature,
                "Mean": safe_mean(values),
                "Std": safe_std(values),
                "Count": len(values),
            }
        )
    return rows


def success_failure_message_comparison(traces: list[dict], args) -> list[dict]:
    rows = []
    buckets = defaultdict(list)
    for trace in traces:
        length = episode_len(trace)
        task = infer_task(trace, args.task)
        method = infer_method(trace, args.method)
        outcome = "success" if bool_success(trace) else "failure"
        for side in ("left", "right"):
            features, names = raw_payload_series(trace, side, method, length)
            for idx, name in enumerate(names):
                buckets[(task, METHOD_LABELS.get(method, method), method, outcome, side, name)].extend(
                    features[:, idx].tolist()
                )
    for (task, method_label, mode, outcome, side, feature), values in sorted(buckets.items()):
        rows.append(
            {
                "Task": task,
                "Method": method_label,
                "Communication Mode": mode,
                "Outcome": outcome,
                "Side": side,
                "Feature": feature,
                "Mean": safe_mean(values),
                "Std": safe_std(values),
                "Count": len(values),
            }
        )
    return rows


LIFT_PHASE_ORDER = ("approach", "alignment", "grasp_transition", "lift", "stabilize")
LIFT_PHASE_LABELS = {
    "approach": "Approach",
    "alignment": "Alignment",
    "grasp_transition": "Grasp Transition",
    "lift": "Lift",
    "stabilize": "Stabilize",
}


def context_phase_signatures(traces: list[dict], args) -> list[dict]:
    """Aggregate the canonical context by analysis-only Lift pseudo-phase."""

    buckets = defaultdict(list)
    episodes = defaultdict(set)
    totals = defaultdict(int)
    for trace in traces:
        if infer_task(trace, args.task) != "Lift":
            continue
        length = episode_len(trace)
        phases = lift_phase_masks(trace, length)
        method = infer_method(trace, args.method)
        seed = infer_seed(trace, args.seed)
        episode_id = int(trace.get("episode_index", 0))
        for side in ("left", "right"):
            context = context_series(trace, side, length)
            trend_context = metadata(trace).get("context_normalization") == "relative_ema_trend"
            totals[(method, seed, side, trend_context)] += length
            for phase in LIFT_PHASE_ORDER:
                mask = phases[phase]
                if not np.any(mask):
                    continue
                buckets[(method, seed, side, trend_context, phase)].append(context[mask])
                episodes[(method, seed, side, trend_context, phase)].add(episode_id)

    def activity_level(value: float) -> str:
        # Fixed normalized-value thresholds; no phase labels are fitted.
        if not math.isfinite(value):
            return "Unavailable"
        if value < 0.5:
            return "Low"
        if value < 1.0:
            return "Medium"
        return "High"

    def smoothness_level(value: float) -> str:
        if not math.isfinite(value):
            return "Unavailable"
        if value < 0.4:
            return "Low"
        if value < 0.7:
            return "Medium"
        return "High"

    def trend_level(value: float, positive: str, negative: str) -> str:
        if not math.isfinite(value):
            return "Unavailable"
        if value > 0.05:
            return positive
        if value < -0.05:
            return negative
        return "Steady"

    rows = []
    for method, seed, side, trend_context in sorted(totals):
        for phase in LIFT_PHASE_ORDER:
            chunks = buckets.get((method, seed, side, trend_context, phase), [])
            values = np.concatenate(chunks, axis=0) if chunks else np.zeros((0, 3), dtype=float)
            linear_mean, angular_mean, smoothness_mean = (
                [safe_mean(values[:, index]) for index in range(3)] if len(values) else [float("nan")] * 3
            )
            linear_std, angular_std, smoothness_std = (
                [safe_std(values[:, index]) for index in range(3)] if len(values) else [float("nan")] * 3
            )
            sample_count = int(values.shape[0])
            episode_count = len(episodes[(method, seed, side, trend_context, phase)])
            phase_fraction = sample_count / max(totals[(method, seed, side, trend_context)], 1)
            support = "LOW_SUPPORT" if phase_fraction < 0.01 or episode_count < 3 else "OK"
            rows.append(
                {
                    "Task": "Lift",
                    "Method": METHOD_LABELS.get(method, method),
                    "Communication Mode": method,
                    "Seed": seed,
                    "Side": side,
                    "Phase": LIFT_PHASE_LABELS[phase],
                    "Phase Key": phase,
                    "Linear Mean": float(linear_mean),
                    "Linear Std": float(linear_std),
                    "Angular Mean": float(angular_mean),
                    "Angular Std": float(angular_std),
                    "Smoothness Mean": float(smoothness_mean),
                    "Smoothness Std": float(smoothness_std),
                    "Linear Level": trend_level(float(linear_mean), "Increasing", "Decreasing")
                    if trend_context
                    else activity_level(float(linear_mean)),
                    "Angular Level": trend_level(float(angular_mean), "Increasing", "Decreasing")
                    if trend_context
                    else activity_level(float(angular_mean)),
                    "Smoothness Level": trend_level(float(smoothness_mean), "Stabilizing", "Destabilizing")
                    if trend_context
                    else smoothness_level(float(smoothness_mean)),
                    "Qualitative Signature": " / ".join(
                        (
                            trend_level(float(linear_mean), "Increasing", "Decreasing"),
                            trend_level(float(angular_mean), "Increasing", "Decreasing"),
                            trend_level(float(smoothness_mean), "Stabilizing", "Destabilizing"),
                        )
                    )
                    if trend_context
                    else " ".join(
                        (
                            activity_level(float(linear_mean))[0],
                            activity_level(float(angular_mean))[0],
                            smoothness_level(float(smoothness_mean))[0],
                        )
                    ),
                    "Trend Positive Ratio": [safe_mean(values[:, i] > 0.05) for i in range(3)]
                    if trend_context and len(values)
                    else [],
                    "Trend Negative Ratio": [safe_mean(values[:, i] < -0.05) for i in range(3)]
                    if trend_context and len(values)
                    else [],
                    "Context Type": "Signed Trend" if trend_context else "Absolute Level",
                    "Level Definition": "signed-trend:mean<-0.05=negative,abs(mean)<=0.05=steady,mean>0.05=positive"
                    if trend_context
                    else "activity:<0.5=L,<1.0=M,else=H;smoothness:<0.4=L,<0.7=M,else=H",
                    "Sample Count": sample_count,
                    "Episode Count": episode_count,
                    "Phase Fraction": phase_fraction,
                    "Support": support,
                    "Count": sample_count,
                    "Analysis Role": "analysis_only_pseudo_phase",
                }
            )
    return rows


def motion_phase_signatures(traces: list[dict], args) -> list[dict]:
    buckets = defaultdict(list)
    episodes = defaultdict(set)
    totals = defaultdict(int)
    for trace in traces:
        if infer_task(trace, args.task) != "Lift":
            continue
        length = episode_len(trace)
        phases = lift_phase_masks(trace, length)
        method = infer_method(trace, args.method)
        seed = infer_seed(trace, args.seed)
        episode_id = int(trace.get("episode_index", 0))
        for side in ("left", "right"):
            motion = vector_series(trace, f"{side}_motion", 3, length)
            totals[(method, seed, side)] += length
            for phase in LIFT_PHASE_ORDER:
                mask = phases[phase]
                if np.any(mask):
                    buckets[(method, seed, side, phase)].append(motion[mask])
                    episodes[(method, seed, side, phase)].add(episode_id)
    rows = []
    for method, seed, side in sorted(totals):
        for phase in LIFT_PHASE_ORDER:
            chunks = buckets.get((method, seed, side, phase), [])
            values = np.concatenate(chunks, axis=0) if chunks else np.zeros((0, 3), dtype=float)
            count = int(values.shape[0])
            episode_count = len(episodes[(method, seed, side, phase)])
            fraction = count / max(totals[(method, seed, side)], 1)
            rows.append(
                {
                    "Task": "Lift",
                    "Method": METHOD_LABELS.get(method, method),
                    "Communication Mode": method,
                    "Seed": seed,
                    "Side": side,
                    "Phase": LIFT_PHASE_LABELS[phase],
                    "Phase Key": phase,
                    "Motion X Mean": safe_mean(values[:, 0]),
                    "Motion X Std": safe_std(values[:, 0]),
                    "Motion Y Mean": safe_mean(values[:, 1]),
                    "Motion Y Std": safe_std(values[:, 1]),
                    "Motion Z Mean": safe_mean(values[:, 2]),
                    "Motion Z Std": safe_std(values[:, 2]),
                    "Motion XY Norm Mean": safe_mean(np.linalg.norm(values[:, :2], axis=1)),
                    "Motion Norm Mean": safe_mean(np.linalg.norm(values, axis=1)),
                    "Sample Count": count,
                    "Episode Count": episode_count,
                    "Phase Fraction": fraction,
                    "Support": "LOW_SUPPORT" if fraction < 0.01 or episode_count < 3 else "OK",
                    "Count": count,
                    "Analysis Role": "analysis_only_pseudo_phase",
                }
            )
    return rows


def context_phase_probe(traces: list[dict], args) -> tuple[list[dict], list[dict]]:
    """Compare three representations using episode-level nearest-centroid LOO."""

    grouped = defaultdict(list)
    for trace in traces:
        if infer_task(trace, args.task) != "Lift":
            continue
        method = infer_method(trace, args.method)
        seed = infer_seed(trace, args.seed)
        length = episode_len(trace)
        masks = lift_phase_masks(trace, length)
        labels = np.full(length, "approach", dtype=object)
        for phase in LIFT_PHASE_ORDER:
            labels[masks[phase]] = phase
        episode_representations = {}
        for representation in ("motion_3d", "context_3d", "motion_context_6d"):
            side_features = []
            for side in ("left", "right"):
                motion = vector_series(trace, f"{side}_motion", 3, length)
                context = context_series(trace, side, length)
                if representation == "motion_3d":
                    features = motion
                elif representation == "context_3d":
                    features = context
                else:
                    features = np.concatenate([motion, context], axis=-1)
                side_features.append(features)
            # Both arms from a held-out episode remain together in the test set.
            episode_representations[representation] = np.concatenate(side_features, axis=0)
        grouped[(method, seed)].append((episode_representations, np.concatenate([labels, labels], axis=0)))

    summary_rows = []
    recall_rows = []
    for (method, seed), episodes in sorted(grouped.items()):
        if len(episodes) < 2:
            continue
        for representation in ("motion_3d", "context_3d", "motion_context_6d"):
            confusion = defaultdict(int)
            fold_balanced_accuracy = []
            samples = 0
            for test_idx, (test_repr, test_y) in enumerate(episodes):
                train = [episode for idx, episode in enumerate(episodes) if idx != test_idx]
                train_x = np.concatenate([episode[0][representation] for episode in train], axis=0)
                train_y = np.concatenate([episode[1] for episode in train], axis=0)
                test_x = test_repr[representation]
                mean = np.mean(train_x, axis=0)
                std = np.std(train_x, axis=0)
                std = np.where(std > 1.0e-6, std, 1.0)
                train_x = (train_x - mean) / std
                test_x = (test_x - mean) / std
                available = [phase for phase in LIFT_PHASE_ORDER if np.any(train_y == phase)]
                if not available or test_x.shape[0] == 0:
                    continue
                centroids = np.stack([np.mean(train_x[train_y == phase], axis=0) for phase in available])
                predictions = np.asarray(available, dtype=object)[
                    np.argmin(np.linalg.norm(test_x[:, None, :] - centroids[None, :, :], axis=-1), axis=1)
                ]
                fold_recalls = []
                for phase in LIFT_PHASE_ORDER:
                    phase_mask = test_y == phase
                    if np.any(phase_mask):
                        fold_recalls.append(float(np.mean(predictions[phase_mask] == phase)))
                if fold_recalls:
                    fold_balanced_accuracy.append(float(np.mean(fold_recalls)))
                for actual, predicted in zip(test_y, predictions):
                    confusion[(str(actual), str(predicted))] += 1
                samples += int(test_x.shape[0])

            summary_rows.append(
                {
                    "Task": "Lift",
                    "Method": METHOD_LABELS.get(method, method),
                    "Communication Mode": method,
                    "Seed": seed,
                    "Representation": representation,
                    "Balanced Accuracy Mean": safe_mean(fold_balanced_accuracy),
                    "Balanced Accuracy Std": safe_std(fold_balanced_accuracy),
                    "Episodes": len(fold_balanced_accuracy),
                    "Samples": samples,
                    "Split": "leave_one_episode_out",
                    "Classifier": "nearest_centroid_train_standardized",
                }
            )
            for phase in LIFT_PHASE_ORDER:
                support = sum(confusion[(phase, predicted)] for predicted in LIFT_PHASE_ORDER)
                recall_rows.append(
                    {
                        "Task": "Lift",
                        "Method": METHOD_LABELS.get(method, method),
                        "Communication Mode": method,
                        "Seed": seed,
                        "Representation": representation,
                        "Phase": LIFT_PHASE_LABELS[phase],
                        "Phase Key": phase,
                        "Recall": confusion[(phase, phase)] / support if support else float("nan"),
                        "Support": support,
                        "Split": "leave_one_episode_out",
                    }
                )
    by_method_seed = group_rows(summary_rows, ("Communication Mode", "Seed"))
    for group in by_method_seed.values():
        values = {row["Representation"]: row["Balanced Accuracy Mean"] for row in group}
        combined = values.get("motion_context_6d", float("nan"))
        motion = values.get("motion_3d", float("nan"))
        context = values.get("context_3d", float("nan"))
        for row in group:
            row["Motion Context Gain Over Motion"] = (
                combined - motion if math.isfinite(combined) and math.isfinite(motion) else float("nan")
            )
            row["Motion Context Gain Over Context"] = (
                combined - context if math.isfinite(combined) and math.isfinite(context) else float("nan")
            )
            row["Leakage Guard"] = "entire_episode_held_out"
    return summary_rows, recall_rows


def context_success_failure_diagnostic(traces: list[dict], args) -> list[dict]:
    """Compare episode-level context means between successful and failed trials."""

    buckets = defaultdict(list)
    for trace in traces:
        length = episode_len(trace)
        task = infer_task(trace, args.task)
        method = infer_method(trace, args.method)
        outcome = "success" if bool_success(trace) else "failure"
        for side in ("left", "right"):
            context = context_series(trace, side, length)
            for index, feature in enumerate(CONTEXT_NAMES):
                buckets[(task, method, outcome, side, feature)].append(safe_mean(context[:, index]))
    rows = []
    for (task, method, outcome, side, feature), values in sorted(buckets.items()):
        rows.append(
            {
                "Task": task,
                "Communication Mode": method,
                "Outcome": outcome,
                "Side": side,
                "Feature": feature,
                "Episode Mean": safe_mean(values),
                "Episode Std": safe_std(values),
                "Episodes": len(values),
            }
        )
    return rows


def plot_lift_context_phase_signature(rows: list[dict], out_path: Path) -> None:
    """Plot left/right phase signatures in one compact paper-facing figure."""

    columns = ("Linear Mean", "Angular Mean", "Smoothness Mean")
    if any(row.get("Communication Mode") == "motion_context" for row in rows):
        rows = [row for row in rows if row.get("Communication Mode") == "motion_context"]
    trend_context = any(row.get("Context Type") == "Signed Trend" for row in rows)
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.2), sharey=True)
    images = []
    for ax, side in zip(axes, ("left", "right")):
        matrix = np.full((len(LIFT_PHASE_ORDER), len(columns)), np.nan, dtype=float)
        for row_idx, phase in enumerate(LIFT_PHASE_ORDER):
            phase_rows = [row for row in rows if row["Side"] == side and row.get("Phase Key") == phase]
            for col_idx, column in enumerate(columns):
                matrix[row_idx, col_idx] = safe_mean([row[column] for row in phase_rows])
        image = ax.imshow(
            matrix,
            aspect="auto",
            cmap="coolwarm" if trend_context else "viridis",
            vmin=-1.0 if trend_context else 0.0,
            vmax=1.0 if trend_context else 1.5,
        )
        images.append(image)
        ax.set_xticks(
            range(len(CONTEXT_NAMES)), ["T_v", "T_ω", "T_S"] if trend_context else ["Linear", "Angular", "Smoothness"]
        )
        ax.set_yticks(range(len(LIFT_PHASE_ORDER)), [LIFT_PHASE_LABELS[phase] for phase in LIFT_PHASE_ORDER])
        ax.set_title(f"{side.title()} Arm")
        for row_idx in range(matrix.shape[0]):
            for col_idx in range(matrix.shape[1]):
                value = matrix[row_idx, col_idx]
                if np.isfinite(value):
                    ax.text(col_idx, row_idx, f"{value:.2f}", ha="center", va="center", color="white")
    fig.suptitle("Lift Motion Context Phase Signature (Analysis Proxy)")
    fig.colorbar(
        images[-1],
        ax=axes,
        label="Mean signed trend" if trend_context else "Mean normalized context",
        fraction=0.025,
        pad=0.03,
    )
    fig.subplots_adjust(left=0.10, right=0.90, bottom=0.12, top=0.82, wspace=0.10)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_motion_context_phase_summary(motion_rows: list[dict], context_rows: list[dict], out_path: Path) -> None:
    """Visualize the paper framing: Motion says where; Context says how."""

    if any(row.get("Communication Mode") == "motion_context" for row in context_rows):
        context_rows = [row for row in context_rows if row.get("Communication Mode") == "motion_context"]
        motion_rows = [row for row in motion_rows if row.get("Communication Mode") == "motion_context"]
    trend_context = any(row.get("Context Type") == "Signed Trend" for row in context_rows)
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.6), sharey=True)
    features = ("Motion Z Mean", "Linear Mean", "Angular Mean", "Smoothness Mean")
    labels = (
        ("Motion Z", "T_v", "T_ω", "T_S")
        if trend_context
        else ("Motion Z", "Linear Activity", "Angular Activity", "Motion Smoothness")
    )
    for ax, side in zip(axes, ("left", "right")):
        matrix = np.full((len(LIFT_PHASE_ORDER), len(features)), np.nan)
        for i, phase in enumerate(LIFT_PHASE_ORDER):
            motion_group = [r for r in motion_rows if r["Side"] == side and r.get("Phase Key") == phase]
            context_group = [r for r in context_rows if r["Side"] == side and r.get("Phase Key") == phase]
            matrix[i, 0] = safe_mean([r["Motion Z Mean"] for r in motion_group])
            for j, feature in enumerate(features[1:], start=1):
                matrix[i, j] = safe_mean([r[feature] for r in context_group])
        image = ax.imshow(matrix, aspect="auto", cmap="coolwarm", vmin=-1.0, vmax=1.0 if trend_context else 1.5)
        ax.set_xticks(range(len(labels)), labels, rotation=18, ha="right")
        ax.set_yticks(range(len(LIFT_PHASE_ORDER)), [LIFT_PHASE_LABELS[p] for p in LIFT_PHASE_ORDER])
        ax.set_title(f"{side.title()} Arm")
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                if np.isfinite(matrix[i, j]):
                    ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=8)
    fig.suptitle("Lift Motion + Context Phase Summary (Analysis-Only Proxy)")
    fig.colorbar(
        image,
        ax=axes,
        fraction=0.025,
        pad=0.03,
        label="Mean signed feature" if trend_context else "Mean normalized feature",
    )
    fig.subplots_adjust(left=0.11, right=0.90, bottom=0.23, top=0.82, wspace=0.10)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def evaluation_success_stability(rows: list[dict], ci: float) -> list[dict]:
    """Cumulative evaluation success by evaluation-episode order.

    This is a stability/consistency visualization only.  It is *not* a training
    learning curve because the x-axis is evaluation episode order, not training
    transitions or checkpoint steps.
    """

    z = NormalDist().inv_cdf(0.5 + ci / 2.0)
    out = []
    by_task_method_seed = group_rows(rows, ("task", "method", "seed"))
    seed_curves = defaultdict(list)
    for (task, method, seed), group in by_task_method_seed.items():
        ordered = sorted(group, key=lambda row: int(row["episode_id"]))
        cumulative = []
        for idx, row in enumerate(ordered, start=1):
            cumulative.append(float(row["success"]))
            seed_curves[(task, method, idx)].append(float(np.mean(cumulative)))
    for (task, method, episode), values in sorted(seed_curves.items()):
        arr = np.asarray(values, dtype=float)
        mean = float(np.mean(arr))
        stderr = float(np.std(arr, ddof=1) / math.sqrt(arr.size)) if arr.size > 1 else 0.0
        out.append(
            {
                "Task": task,
                "Method": method,
                "Evaluation Episode": int(episode),
                "Mean Success": mean,
                "CI Lower": max(0.0, mean - z * stderr),
                "CI Upper": min(1.0, mean + z * stderr),
                "Seeds": int(arr.size),
                "Source": "evaluation_episode_order_not_training_steps",
            }
        )
    return out


def plot_overview(trace: dict, out_path: Path, smooth_window: int) -> None:
    length = episode_len(trace)
    steps = np.arange(length)
    left_context = smooth(context_series(trace, "left", length), smooth_window)
    right_context = smooth(context_series(trace, "right", length), smooth_window)
    left_motion = vector_series(trace, "left_motion", 3, length)
    right_motion = vector_series(trace, "right_motion", 3, length)
    reward = smooth(scalar_series(trace, "reward_mean", length), smooth_window)
    hprog = scalar_series(trace, "hprog", length)
    object_z = scalar_series(trace, "object_z", length)
    left_closure = scalar_series(trace, "left_closure", length)
    right_closure = scalar_series(trace, "right_closure", length)
    lateral = scalar_series(trace, "lateral_error", length)
    axis = scalar_series(trace, "axis_alignment", length)
    depth = scalar_series(trace, "insertion_depth", length)
    tilt = scalar_series(trace, "object_tilt_deg", length)

    fig, axes = plt.subplots(6, 1, figsize=(16, 18), sharex=True)
    method = METHOD_LABELS.get(infer_method(trace, None), infer_method(trace, None))
    fig.suptitle(
        f"OpenArm Motion Context Overview: {method}, episode {trace.get('episode_index', 0)}",
        fontsize=16,
    )

    axes[0].plot(steps, left_motion[:, 2], label="L motion z")
    axes[0].plot(steps, right_motion[:, 2], label="R motion z")
    axes[0].plot(steps, np.linalg.norm(left_motion[:, :2], axis=-1), linestyle="--", label="L motion xy norm")
    axes[0].plot(steps, np.linalg.norm(right_motion[:, :2], axis=-1), linestyle="--", label="R motion xy norm")
    axes[0].axhline(0.0, color="black", linewidth=0.7, alpha=0.5)
    axes[0].set_title("Signed EE Motion Components")
    axes[0].set_ylabel("normalized motion")
    axes[0].legend(loc="upper right")

    for idx, name in enumerate(CONTEXT_NAMES):
        axes[1].plot(steps, left_context[:, idx], label=f"L {name}")
        axes[1].plot(steps, right_context[:, idx], linestyle="--", label=f"R {name}")
    axes[1].set_title("Motion Context")
    axes[1].set_ylabel("value")
    axes[1].set_ylim(-0.05, 1.55)
    axes[1].legend(loc="upper right", ncol=2)

    axes[2].plot(steps, left_closure, label="L closure")
    axes[2].plot(steps, right_closure, label="R closure")
    axes[2].set_title("Gripper Closure")
    axes[2].legend(loc="upper right")

    axes[3].plot(steps, object_z, label="object z / peg tip z")
    axes[3].plot(steps, hprog, label="hprog")
    axes[3].set_title("Task Progress")
    axes[3].legend(loc="upper right")

    if np.any(lateral) or np.any(axis) or np.any(depth):
        axes[4].plot(steps, lateral, label="lateral")
        axes[4].plot(steps, axis, label="axis")
        axes[4].plot(steps, depth, label="depth")
        axes[4].set_title("Peg-In-Hole Metrics")
    else:
        axes[4].plot(steps, tilt, label="tilt deg")
        axes[4].set_title("Lift Tilt")
    axes[4].legend(loc="upper right")

    axes[5].plot(steps, reward, label="reward mean")
    axes[5].set_title("Reward")
    axes[5].set_xlabel("step")
    axes[5].legend(loc="upper right")

    for ax in axes:
        ax.grid(True, alpha=0.25)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_context_summary(rows: list[dict], out_path: Path) -> None:
    if not rows:
        return
    method_rows = group_rows(rows, ("communication_mode",))
    methods = [method for method in METHOD_ORDER if (method,) in method_rows]
    x = np.arange(len(methods))
    width = 0.35
    fig, axes = plt.subplots(1, len(CONTEXT_NAMES), figsize=(16, 5), sharey=True)
    for axis, name in zip(axes, CONTEXT_NAMES):
        left = [safe_mean([float(row[f"left_{name}_mean"]) for row in method_rows[(method,)]]) for method in methods]
        right = [safe_mean([float(row[f"right_{name}_mean"]) for row in method_rows[(method,)]]) for method in methods]
        axis.bar(x - width / 2, left, width, label="L")
        axis.bar(x + width / 2, right, width, label="R")
        axis.set_xticks(x, [METHOD_LABELS.get(method, method) for method in methods], rotation=20, ha="right")
        axis.set_title(name.replace("_", " ").title())
        axis.grid(True, axis="y", alpha=0.25)
    axes[0].set_ylabel("episode mean")
    axes[0].legend()
    fig.suptitle("Proprioceptive Motion Context by Communication Method (Supplementary Diagnostic)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_ablation(rows: list[dict], out_path: Path) -> None:
    if not rows:
        return
    methods = ("No Communication", "Motion Only", "Context Only", "Motion Context")
    available_set = {
        method for method in methods if any(math.isfinite(float(row.get(method, float("nan")))) for row in rows)
    }
    if len(available_set) < 2:
        if out_path.exists():
            out_path.unlink()
        print("[INFO] Skipped ablation.png: fewer than two communication methods have evaluation data.")
        return
    tasks = [row["Task"] for row in rows]
    x = np.arange(len(tasks))
    available = [method for method in methods if method in available_set]
    width = 0.8 / len(available)
    fig, ax = plt.subplots(figsize=(11, 5))
    for i, method in enumerate(available):
        values = [row.get(method, float("nan")) for row in rows]
        ax.bar(x + (i - (len(available) - 1) / 2.0) * width, values, width, label=method)
    ax.set_xticks(x, tasks)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("success rate")
    ax.set_title("Communication Ablation")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_metric_ablation(
    rows: list[dict],
    out_path: Path,
    mean_column: str,
    std_column: str,
    title: str,
    ylabel: str,
) -> None:
    """Plot available methods only; uncertainty always reflects seed variation."""

    available = [
        mode
        for mode in METHOD_ORDER
        if any(
            row["Communication Mode"] == mode and math.isfinite(float(row.get(mean_column, float("nan"))))
            for row in rows
        )
    ]
    if not available:
        print(f"[INFO] Skipped {out_path.name}: metric unavailable.")
        return
    tasks = sorted({row["Task"] for row in rows})
    fig, axes = plt.subplots(1, len(tasks), figsize=(max(6.5, 5.2 * len(tasks)), 4.8), squeeze=False)
    for ax, task in zip(axes[0], tasks):
        task_rows = {row["Communication Mode"]: row for row in rows if row["Task"] == task}
        modes = [
            mode
            for mode in available
            if mode in task_rows and math.isfinite(float(task_rows[mode].get(mean_column, float("nan"))))
        ]
        values = [float(task_rows[mode][mean_column]) for mode in modes]
        errors = [float(task_rows[mode].get(std_column, 0.0)) for mode in modes]
        seeds = [int(task_rows[mode].get("Seeds", 1)) for mode in modes]
        x = np.arange(len(modes))
        bars = ax.bar(x, values, color="tab:blue", alpha=0.85)
        for i, (value, error, seed_count) in enumerate(zip(values, errors, seeds)):
            if seed_count > 1 and math.isfinite(error):
                ax.errorbar(i, value, yerr=error, color="black", capsize=4, fmt="none")
        ax.set_xticks(x, [METHOD_LABELS[mode] for mode in modes], rotation=22, ha="right")
        ax.set_title(task)
        ax.set_ylabel(ylabel)
        ax.grid(True, axis="y", alpha=0.25)
        if mean_column == "Success Rate Mean":
            ax.set_ylim(0.0, 1.0)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_failure_breakdown(rows: list[dict], out_dir: Path) -> None:
    by_task = defaultdict(list)
    for row in rows:
        by_task[row["Task"]].append(row)
    for task, task_rows in by_task.items():
        methods = sorted({row["Method"] for row in task_rows})
        reasons = sorted({row["Failure Reason"] for row in task_rows if row["Failure Reason"] != "success"})
        if not methods or not reasons:
            continue
        if len(methods) < 2 or len(reasons) < 2:
            print(
                f"[WARNING] {task} failure breakdown has low information: methods={len(methods)}, reasons={len(reasons)}"
            )
        x = np.arange(len(methods))
        bottom = np.zeros(len(methods), dtype=float)
        fig, ax = plt.subplots(figsize=(12, 5))
        for reason in reasons:
            values = []
            for method in methods:
                match = [row for row in task_rows if row["Method"] == method and row["Failure Reason"] == reason]
                values.append(float(match[0]["Fraction"]) if match else 0.0)
            ax.bar(x, values, bottom=bottom, label=reason)
            bottom += np.asarray(values, dtype=float)
        ax.set_xticks(x, methods, rotation=20, ha="right")
        ax.set_ylim(0.0, 1.0)
        ax.set_ylabel("fraction")
        ax.set_title(f"Failure Breakdown: {task}")
        ax.grid(True, axis="y", alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / f"failure_breakdown_{task.lower()}.png", dpi=150)
        plt.close(fig)


def plot_evaluation_success_stability(rows: list[dict], out_dir: Path) -> None:
    by_task = defaultdict(list)
    for row in rows:
        by_task[row["Task"]].append(row)
    for task, task_rows in by_task.items():
        methods = sorted({row["Method"] for row in task_rows})
        if not methods:
            continue
        fig, ax = plt.subplots(figsize=(10, 5))
        for method in methods:
            group = sorted(
                [row for row in task_rows if row["Method"] == method],
                key=lambda row: row["Evaluation Episode"],
            )
            if not group:
                continue
            x = np.asarray([row["Evaluation Episode"] for row in group], dtype=float)
            mean = np.asarray([row["Mean Success"] for row in group], dtype=float)
            lo = np.asarray([row["CI Lower"] for row in group], dtype=float)
            hi = np.asarray([row["CI Upper"] for row in group], dtype=float)
            ax.plot(x, mean, label=method)
            ax.fill_between(x, lo, hi, alpha=0.15)
        ax.set_ylim(0.0, 1.0)
        ax.set_xlabel("evaluation episode order")
        ax.set_ylabel("cumulative evaluation success rate")
        ax.set_title(f"Evaluation Success Stability: {task}\nEvaluation episode order; not training steps")
        ax.grid(True, alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / f"evaluation_success_stability_{task.lower()}.png", dpi=150)
        plt.close(fig)


def print_readiness(
    method_rows: list[dict],
    message_rows: list[dict],
    probe_rows: list[dict],
    context_signature_rows: list[dict],
    episode_rows: list[dict],
) -> None:
    print("\n=== Paper Readiness: Motion says where; Context says how ===")
    print("[DATA]")
    print(
        f"methods={len({row['Communication Mode'] for row in method_rows})} "
        f"max_seeds_per_method={max((int(row.get('Seeds', 0)) for row in method_rows), default=0)} "
        f"method_seed_runs={sum(int(row.get('Seeds', 0)) for row in method_rows)} "
        f"episodes={len(episode_rows)}"
    )
    for row in method_rows:
        print(
            f"{row['Task']:>4} | {row['Method']:<18} | success={row['Success Rate']} "
            f"| t_success={row['Time to Success Sec']} s | path={row['Mean EE Path Length All']} m"
        )
    by_task_mode = {(row["Task"], row["Communication Mode"]): row for row in method_rows}
    for task in sorted({row["Task"] for row in method_rows}):
        proposed = by_task_mode.get((task, "motion_context"))
        none = by_task_mode.get((task, "none"))
        print(f"[PERFORMANCE] {task}")
        if proposed and none:
            print(
                "[COORDINATION] Motion Context - None: "
                f"success={100.0 * (proposed['Success Rate Mean'] - none['Success Rate Mean']):+.1f} pp, "
                f"time={proposed['Time to Success Sec Mean'] - none['Time to Success Sec Mean']:+.3f} s, "
                f"reach_gap={proposed['Reach Gap Mean'] - none['Reach Gap Mean']:+.2f} step, "
                f"grasp_gap={proposed['Grasp Gap Mean'] - none['Grasp Gap Mean']:+.2f} step, "
                f"path={proposed['Mean EE Path Length All Mean'] - none['Mean EE Path Length All Mean']:+.3f} m"
            )
        if proposed and proposed["Success Rate Mean"] >= 0.95:
            print("[WARNING] Success rate is saturated; prioritize timing and coordination metrics.")

    saturation = [
        float(row["Saturation Ratio"])
        for row in message_rows
        if str(row["Communication Mode"]) == "motion_context" and math.isfinite(float(row["Saturation Ratio"]))
    ]
    zero = [float(row["Zero Ratio"]) for row in message_rows if str(row["Communication Mode"]) == "motion_context"]
    max_sat = max(saturation) if saturation else 0.0
    max_zero = max(zero) if zero else 0.0
    print(f"[MESSAGE] max_saturation={max_sat:.3f} max_zero={max_zero:.3f}")
    print("[REPRESENTATION]")
    for row in probe_rows:
        print(
            f"  {row['Communication Mode']} {row['Representation']}: "
            f"balanced_accuracy={row['Balanced Accuracy Mean']:.3f} +/- {row['Balanced Accuracy Std']:.3f}"
        )
    modes = {row["Communication Mode"] for row in method_rows}
    for mode in ("motion_only", "context_only"):
        if mode not in modes:
            print(f"[WARNING] missing {METHOD_LABELS[mode]} ablation")
    if any(int(row.get("Seeds", 0)) < 2 for row in method_rows):
        print("[WARNING] single-seed result exists; error bars are not inferential.")
    if any(int(row.get("Episodes", 0)) < 10 for row in method_rows):
        print("[WARNING] fewer than 10 evaluation episodes for at least one method.")
    low_support = [row for row in context_signature_rows if row.get("Support") == "LOW_SUPPORT"]
    if low_support:
        print(f"[WARNING] {len(low_support)} phase signature row(s) have LOW_SUPPORT.")
    print("[NOTE] Evaluation success stability uses evaluation episode order, not training transitions.")
    print("[NOTE] Phase separability is descriptive, not causal evidence of performance.")


def validate_results(
    traces: list[dict],
    episode_rows: list[dict],
    seed_rows: list[dict],
    method_rows: list[dict],
    failure_rows: list[dict],
    probe_rows: list[dict],
    args,
) -> list[str]:
    """Run inexpensive invariants on actual loaded traces and derived metrics."""

    checks = []
    for trace in traces:
        length = episode_len(trace)
        if infer_task(trace, args.task) == "Lift":
            masks = lift_phase_masks(trace, length)
            occupancy = np.sum(np.stack(list(masks.values()), axis=0), axis=0)
            assert np.all(occupancy == 1), "Lift phase masks are not mutually exclusive/exhaustive"
        for side in ("left", "right"):
            raw = np.asarray(trace.get(f"{side}_motion", []), dtype=float)
            if raw.ndim == 2 and raw.shape[1] >= 3 and np.any(raw < 0.0):
                loaded = vector_series(trace, f"{side}_motion", 3, length)
                assert np.any(loaded < 0.0), "signed motion was lost"
    checks.extend(("phase_masks_mutually_exclusive", "signed_motion_preserved"))

    for row in episode_rows:
        for key in ("left_ee_path_length", "right_ee_path_length", "mean_integrated_angular_motion"):
            value = float(row.get(key, float("nan")))
            assert not math.isfinite(value) or value >= -1.0e-10, f"negative physical efficiency metric: {key}"
        if int(row["success"]):
            assert math.isfinite(float(row["time_to_success"])), "successful episode lacks time-to-success"
        else:
            assert not math.isfinite(float(row["time_to_success"])), "failed episode has time-to-success"
    checks.extend(("physical_metrics_nonnegative", "success_time_semantics"))

    for (_, _), group in group_rows(failure_rows, ("Task", "Method")).items():
        assert abs(sum(float(row["Fraction"]) for row in group) - 1.0) < 1.0e-6
    checks.append("failure_fractions_sum_to_one")
    assert all(row.get("Leakage Guard") == "entire_episode_held_out" for row in probe_rows)
    checks.append("probe_episode_leakage_guard")
    assert seed_rows and method_rows
    checks.append("seed_then_method_aggregation")
    return checks


def ensure_no_share_inputs_zero(traces: list[dict]) -> None:
    """Reject a NoShare label contradicted by recorded actor message inputs."""
    for trace in traces:
        if infer_method(trace, None) != "none":
            continue
        for side in ("left", "right"):
            received = np.asarray(trace.get(f"{side}_policy_message", []), dtype=float)
            if received.size and (not np.isfinite(received).all() or np.any(np.abs(received) > 1e-7)):
                raise ValueError("NoShare trace contains nonzero or nonfinite actor message inputs")


def ensure_comparable_success_protocols(traces: list[dict]) -> None:
    """Reject pooled Peg scores from incompatible physical success definitions."""
    signatures = set()
    for trace in traces:
        if infer_task(trace, None) != "Peg":
            continue
        meta = metadata(trace)
        values = []
        for key in (
            "success_depth_threshold",
            "success_lateral_threshold",
            "success_axis_threshold",
            "hold_required_steps",
            "target_insertion_depth",
            "success_radial_clearance",
            "peg_collision_radius",
            "success_depth_overrun_tolerance",
        ):
            values.append(round(trace_float_config(trace, (key,), 0.0), 6))
        signatures.add((meta.get("success_protocol", "legacy_marker_depth_v1"), *values))
    if len(signatures) > 1:
        raise ValueError("Different Peg success protocols cannot be pooled; analyze each protocol separately")


def main() -> None:
    args = parse_args()
    trace_dir = Path(args.trace_dir)
    trace_dirs = [trace_dir, *(Path(path) for path in args.include_trace_dir)]
    traces = []
    seen_paths = set()
    for source_dir in trace_dirs:
        for trace in load_traces(source_dir, args.episode):
            source_path = str(Path(trace["_path"]).resolve())
            if source_path not in seen_paths:
                traces.append(trace)
                seen_paths.add(source_path)
    ensure_no_share_inputs_zero(traces)
    ensure_comparable_success_protocols(traces)
    rows = [summarize_trace(trace, args) for trace in traces]
    validation_rows = []
    for trace, episode_row in zip(traces, rows):
        issues = validate_trace(trace)
        required = [issue for issue in issues if issue.startswith("REQUIRED:")]
        recommended = [issue for issue in issues if issue.startswith("RECOMMENDED:")]
        optional = [issue for issue in issues if issue.startswith("LEGACY_OPTIONAL:")]
        validation_rows.append(
            {
                "episode_path": str(trace.get("_path", "")),
                "valid": int(not required),
                "issue_count": len(issues),
                "required_issue_count": len(required),
                "recommended_issue_count": len(recommended),
                "legacy_optional_issue_count": len(optional),
                "issues": "; ".join(issues),
                "grasp_transition_source": episode_row.get("grasp_transition_source", "not_applicable"),
                "integrated_angular_motion_source": episode_row.get("integrated_angular_motion_source", "unavailable"),
                "action_variation_source": episode_row.get("action_variation_source", "unavailable"),
                "control_dt_source": episode_row.get("control_dt_source", "unavailable"),
            }
        )
    invalid_count = sum(not bool(row["valid"]) for row in validation_rows)
    if invalid_count:
        print(f"[WARN] {invalid_count}/{len(traces)} trace(s) have schema issues; see trace_validation.csv")
    unknown_method_count = sum(infer_method(trace, args.method) == "unknown" for trace in traces)
    if unknown_method_count:
        raise ValueError(
            f"{unknown_method_count} trace(s) have unknown communication method. "
            "Add communication_mode/method metadata or pass --method explicitly."
        )
    seed_rows = summarize_seed(rows)
    method_rows = summarize_method(seed_rows)
    ablation_rows = ablation_summary(seed_rows)
    failure_rows = failure_breakdown(rows)
    outcome_rows = outcome_performance(rows)
    timing_rows = coordination_timing(rows)
    efficiency_rows = motion_efficiency(rows)
    paper_rows = paper_metric_summary(method_rows)
    message_rows = message_statistics(traces, args)
    policy_input_rows = message_statistics(traces, args, policy_input=True)
    phase_rows = message_phase_statistics(traces, args)
    success_failure_rows = success_failure_message_comparison(traces, args)
    context_signature_rows = context_phase_signatures(traces, args)
    motion_signature_rows = motion_phase_signatures(traces, args)
    context_probe_rows, context_probe_recall_rows = context_phase_probe(traces, args)
    context_outcome_rows = context_success_failure_diagnostic(traces, args)
    stability_rows = evaluation_success_stability(rows, args.ci)
    predicate_rows = success_predicate_rates(traces, args)

    if args.analysis_name:
        if Path(args.analysis_name).name != args.analysis_name or args.analysis_name in (".", ".."):
            raise ValueError("--analysis_name must be a single folder name, without path separators")
        out_dir = reserve_analysis_output_dir(Path(args.analysis_name), Path(args.out_dir))
    else:
        out_dir = reserve_analysis_output_dir(trace_dir, Path(args.out_dir))
    write_csv(out_dir / "eval_episodes.csv", rows)
    write_csv(out_dir / "trace_validation.csv", validation_rows)
    write_csv(out_dir / "summary_by_seed.csv", seed_rows)
    write_csv(out_dir / "summary_by_method.csv", method_rows)
    write_csv(out_dir / "ablation_summary.csv", ablation_rows)
    write_csv(out_dir / "failure_breakdown.csv", failure_rows)
    write_csv(out_dir / "outcome_performance.csv", outcome_rows)
    write_csv(out_dir / "coordination_timing.csv", timing_rows)
    write_csv(out_dir / "motion_efficiency.csv", efficiency_rows)
    write_csv(out_dir / "paper_metric_summary.csv", paper_rows)
    write_csv(out_dir / "message_raw_statistics.csv", message_rows)
    write_csv(out_dir / "message_policy_input_raw_statistics.csv", policy_input_rows)
    # Backward-compatible filename now contains raw, physically interpretable values.
    write_csv(out_dir / "message_statistics.csv", message_rows)
    write_csv(out_dir / "message_phase_statistics.csv", phase_rows)
    write_csv(out_dir / "success_failure_message_comparison.csv", success_failure_rows)
    write_csv(out_dir / "context_phase_signature.csv", context_signature_rows)
    write_csv(out_dir / "motion_phase_signature.csv", motion_signature_rows)
    write_csv(out_dir / "context_phase_probe.csv", context_probe_rows)
    write_csv(out_dir / "context_phase_probe_per_phase_recall.csv", context_probe_recall_rows)
    write_csv(out_dir / "context_success_failure_diagnostic.csv", context_outcome_rows)
    write_csv(out_dir / "evaluation_success_stability.csv", stability_rows)
    write_csv(out_dir / "success_predicate_rates.csv", predicate_rows)

    for row in predicate_rows:
        fields = (
            "left_orientation_ok",
            "right_orientation_ok",
            "left_distance_ok",
            "right_distance_ok",
            "left_closure_ok",
            "right_closure_ok",
            "left_span_ok",
            "right_span_ok",
            "height_ok",
            "tilt_ok",
        )
        rates = " ".join(
            f"{key}={row[f'{key}_rate']:.0%}" if math.isfinite(row[f"{key}_rate"]) else f"{key}=N/A" for key in fields
        )
        print(f"[PREDICATES episode={row['episode_index']:03d}] {rates} hold_max={row['max_success_hold_count']}")

    legacy_overview = out_dir / "overview.png"
    if legacy_overview.exists():
        legacy_overview.unlink()
    traces_by_method = group_rows(
        [{"trace": trace, "method": infer_method(trace, args.method)} for trace in traces],
        ("method",),
    )
    for (method,), group in sorted(traces_by_method.items()):
        trace = min(group, key=lambda item: int(item["trace"].get("episode_index", 0)))["trace"]
        plot_overview(trace, out_dir / f"overview_{method}.png", max(int(args.smooth_window), 1))
        if method == "motion_context":
            plot_message_action_timeline(trace, out_dir / "message_action_timeline.png")
    plot_context_summary(rows, out_dir / "context_summary.png")
    plot_ablation(ablation_rows, out_dir / "ablation.png")
    plot_metric_ablation(
        method_rows,
        out_dir / "ablation_success_rate.png",
        "Success Rate Mean",
        "Success Rate Std",
        "Task Performance: Success Rate (Higher Is Better)",
        "success rate",
    )
    plot_metric_ablation(
        method_rows,
        out_dir / "ablation_time_to_success.png",
        "Time to Success Sec Mean",
        "Time to Success Sec Std",
        "Temporal Coordination: Time to Success (Lower Is Better)",
        "successful episodes: seconds",
    )
    plot_metric_ablation(
        method_rows,
        out_dir / "ablation_grasp_sync.png",
        "Grasp Gap Mean",
        "Grasp Gap Std",
        "Bimanual Coordination: Grasp Transition Gap (Lower Is Better)",
        "step gap",
    )
    plot_metric_ablation(
        method_rows,
        out_dir / "ablation_motion_efficiency.png",
        "Mean EE Path Length All Mean",
        "Mean EE Path Length All Std",
        "Motion Efficiency: Mean EE Path Length (Lower Is Better)",
        "path length [m]",
    )
    plot_failure_breakdown(failure_rows, out_dir)
    plot_evaluation_success_stability(stability_rows, out_dir)
    if context_signature_rows:
        plot_lift_context_phase_signature(
            context_signature_rows,
            out_dir / "context_phase_signature_lift.png",
        )
    if context_signature_rows and motion_signature_rows:
        plot_motion_context_phase_summary(
            motion_signature_rows,
            context_signature_rows,
            out_dir / "motion_context_phase_summary_lift.png",
        )
    checks = validate_results(traces, rows, seed_rows, method_rows, failure_rows, context_probe_rows, args)
    print("[VALIDATION] " + ", ".join(checks))
    print_readiness(method_rows, message_rows, context_probe_rows, context_signature_rows, rows)

    print(f"\n[OK] analyzed {len(traces)} trace(s)")
    print(f"[OK] wrote {out_dir}")


if __name__ == "__main__":
    main()
