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


CONTEXT_NAMES = ("linear_activity", "angular_activity", "action_smoothness")
MOTION_NAMES = ("motion_x", "motion_y", "motion_z")
METHOD_ORDER = (
    "none",
    "motion_only",
    "context_only",
    "motion_context",
    "previous_action",
    "full_partner_observation",
)
METHOD_LABELS = {
    "none": "None",
    "motion_only": "Motion Only",
    "context_only": "Context Only",
    "motion_context": "Motion Context",
    "previous_action": "Previous Action",
    "full_partner_observation": "Full Observation",
}
MESSAGE_DIMS = {
    "none": 0,
    "motion_only": 3,
    "context_only": 3,
    "motion_context": 6,
    "previous_action": 8,
    "full_partner_observation": 30,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Create paper-ready tables and figures from paper motion-context traces.")
    parser.add_argument("--trace_dir", type=str, required=True, help="Trace directory or experiment root.")
    parser.add_argument(
        "--out_dir",
        type=str,
        default="logs/paper_motion_context_analysis",
        help="Base output directory. The trace directory name is appended automatically.",
    )
    parser.add_argument("--episode", type=int, default=None, help="Episode index to plot. Defaults to the first trace.")
    parser.add_argument("--smooth_window", type=int, default=10)
    parser.add_argument("--task", type=str, default=None, help="Override task label for traces missing metadata.")
    parser.add_argument("--method", type=str, default=None, help="Override communication mode for traces missing metadata.")
    parser.add_argument("--seed", type=int, default=None, help="Override seed for traces missing metadata.")
    parser.add_argument("--ci", type=float, default=0.95, help="Confidence interval level for learning curves.")
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
    files = sorted(trace_dir.rglob("mode_trace_episode_*.json"))
    if not files:
        raise FileNotFoundError(f"No trace files found under: {trace_dir}")
    traces = [load_trace(path) for path in files]
    if episode is not None:
        traces = [trace for trace in traces if int(trace.get("episode_index", -1)) == int(episode)]
        if not traces:
            raise FileNotFoundError(f"No trace with episode_index={episode} under: {trace_dir}")
    return traces


def resolve_output_dir(trace_dir: Path, out_dir: Path) -> Path:
    trace_name = trace_dir.resolve().name
    return out_dir if out_dir.name == trace_name else out_dir / trace_name


def metadata(trace: dict) -> dict:
    data = trace.get("metadata", {})
    return data if isinstance(data, dict) else {}


def episode_len(trace: dict) -> int:
    candidates = [
        len(trace.get("reward_mean", [])),
        len(trace.get("left_motion_context", [])),
        len(trace.get("left_motion", [])),
        len(trace.get("success_signal", [])),
        int(trace.get("episode_length", 0) or 0),
        int(trace.get("num_steps", 0) or 0),
    ]
    length = max(candidates)
    return trim_reset_tail_length(trace, length)


def trim_reset_tail_length(trace: dict, length: int) -> int:
    """Drop trailing reset-buffer frames such as [0, 0, 1] context after done."""

    while length > 1:
        left = vector_series_raw(trace, "left_motion_context", 3, length)
        right = vector_series_raw(trace, "right_motion_context", 3, length)
        if left.shape[0] < length or right.shape[0] < length:
            break
        last_left = left[length - 1]
        last_right = right[length - 1]
        reset_context = (
            np.allclose(last_left, np.asarray([0.0, 0.0, 1.0]), atol=1.0e-8)
            and np.allclose(last_right, np.asarray([0.0, 0.0, 1.0]), atol=1.0e-8)
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
    return np.stack([scalar_series(trace, f"{side}_{name}", length) for name in CONTEXT_NAMES], axis=-1)


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


def first_step(mask: np.ndarray) -> float:
    idx = np.flatnonzero(mask)
    return float(idx[0]) if idx.size else float("nan")


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
    for key in ("seed", "random_seed"):
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
    return aliases.get(text, text if text in METHOD_LABELS else "motion_context")


def infer_method(trace: dict, override: str | None) -> str:
    if override is not None:
        return normalize_method(override)
    meta = metadata(trace)
    for key in ("communication_mode", "method", "sharing_mode"):
        if key in meta:
            method = normalize_method(meta[key])
            if method in METHOD_LABELS:
                return method
    path_text = str(trace.get("_path", "")).lower()
    for method in METHOD_ORDER:
        if method in path_text:
            return method
    if "full" in path_text and "obs" in path_text:
        return "full_partner_observation"
    if "prev" in path_text and "action" in path_text:
        return "previous_action"
    if "none" in path_text or "no_comm" in path_text:
        return "none"
    return "motion_context"


def infer_task(trace: dict, override: str | None) -> str:
    if override:
        return override
    meta = metadata(trace)
    text = " ".join(str(meta.get(key, "")) for key in ("task", "checkpoint", "reward"))
    text += " " + str(trace.get("_path", ""))
    lower = text.lower()
    if "peg" in lower or "hole" in lower:
        return "Peg"
    if "lift" in lower or "klt" in lower or "re_" in lower:
        return "Lift"
    if "lateral_error" in trace or "tip_dist" in trace:
        return "Peg"
    return "Lift"


def auc_progress(trace: dict, task: str, length: int) -> float:
    if length <= 0:
        return float("nan")
    if task == "Peg":
        progress = scalar_series(trace, "hprog", length)
        if not np.any(progress):
            depth = scalar_series(trace, "insertion_depth", length)
            target_depth = float(trace.get("target_insertion_depth", 0.05) or 0.05)
            progress = np.clip(depth / max(target_depth, 1.0e-6), 0.0, 1.0)
    else:
        progress = scalar_series(trace, "hprog", length)
        if not np.any(progress):
            dz = scalar_series(trace, "object_dz", length)
            progress = np.clip(dz / 0.13, 0.0, 1.0)
    return float(np.trapz(np.clip(progress, 0.0, 1.0), dx=1.0) / max(length - 1, 1))


def classify_lift_failure(trace: dict, length: int, success: bool) -> str:
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
    invalid = False
    reward = scalar_series(trace, "reward_mean", length)
    if np.any(~np.isfinite(reward)):
        invalid = True
    if invalid:
        return "invalid_state"
    hprog = scalar_series(trace, "hprog", length)
    tilt = scalar_series(trace, "object_tilt_deg", length)
    left_dist = scalar_series(trace, "left_target_dist", length)
    right_dist = scalar_series(trace, "right_target_dist", length)
    left_grasp = bool_series(trace, "left_grasp_ok", length)
    right_grasp = bool_series(trace, "right_grasp_ok", length)
    dual_grasp = bool_series(trace, "dual_grasp_ok", length)
    if np.nanmax(hprog) > 0.45 and hprog[-1] < 0.20:
        return "drop"
    if np.nanmax(tilt) > 30.0:
        return "excessive_tilt"
    if np.nanmin(left_dist) > 0.10 or np.nanmin(right_dist) > 0.10:
        return "far_from_target"
    if (np.any(left_grasp) or np.any(right_grasp)) and not np.any(dual_grasp):
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
    side_wall = np.any((depth > 0.0) & (lateral > 0.015))
    if side_wall:
        return "side_wall_penetration"
    if np.any(tip) and np.nanmin(tip) > 0.05:
        return "far_from_hole"
    if (np.any(axis) and np.nanmax(axis) < 0.80) or (np.any(lateral) and np.nanmin(lateral) > 0.015):
        return "alignment_fail"
    if np.nanmax(depth) < 0.03:
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
    left_grasp = bool_series(trace, "left_grasp_ok", length)
    right_grasp = bool_series(trace, "right_grasp_ok", length)
    dual_grasp = bool_series(trace, "dual_grasp_ok", length)
    hold = bool_series(trace, "hold_ok", length)
    lateral = scalar_series(trace, "lateral_error", length)
    axis = scalar_series(trace, "axis_alignment", length)
    depth = scalar_series(trace, "insertion_depth", length)
    tip = scalar_series(trace, "tip_dist", length)
    keypoint = scalar_series(trace, "keypoint_dist", length)
    left_action = vector_series(trace, "left_arm_action", 7, length)
    right_action = vector_series(trace, "right_arm_action", 7, length)
    left_action_delta = np.linalg.norm(np.diff(left_action, axis=0), axis=-1) if length > 1 else np.zeros(0)
    right_action_delta = np.linalg.norm(np.diff(right_action, axis=0), axis=-1) if length > 1 else np.zeros(0)
    left_motion = vector_series(trace, "left_motion", 3, length)
    right_motion = vector_series(trace, "right_motion", 3, length)
    left_context = context_series(trace, "left", length)
    right_context = context_series(trace, "right", length)
    success_step = first_success_step(trace, length)

    lift_fail = classify_lift_failure(trace, length, success)
    peg_fail = classify_peg_failure(trace, length, success)
    failure_reason = peg_fail if task == "Peg" else lift_fail
    mean_lifted_tilt = safe_mean(tilt[hprog > 0.2]) if np.any(hprog > 0.2) else 0.0
    lateral_threshold = 0.008
    axis_threshold = 0.92
    depth_threshold = float(trace.get("target_insertion_depth", 0.05) or 0.05) * 0.90

    row = {
        "task": task,
        "method": METHOD_LABELS.get(method, method),
        "communication_mode": method,
        "msg_dim": int(MESSAGE_DIMS.get(method, 6)),
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
        "auc": auc_progress(trace, task, length),
        "invalid": int(failure_reason == "invalid_state"),
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
        "first_left_reach_step": first_step(left_dist < 0.08),
        "first_right_reach_step": first_step(right_dist < 0.08),
        "first_left_grasp_step": first_step(left_grasp | (left_closure > 0.35)),
        "first_right_grasp_step": first_step(right_grasp | (right_closure > 0.35)),
        "first_dual_grasp_step": first_step(dual_grasp | ((left_closure > 0.35) & (right_closure > 0.35))),
        "first_lift_step": first_step(hprog > 0.20),
        "success_step": success_step,
        "max_hold_steps": max_consecutive(hold | bool_series(trace, "success_signal", length)),
        "mean_action_rate": safe_mean(np.concatenate([left_action_delta, right_action_delta])),
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
        "side_wall_penetration": int(np.any((depth > 0.0) & (lateral > 0.015))),
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
    out = []
    for (task, method, mode, msg_dim, seed), group in sorted(
        group_rows(rows, ("task", "method", "communication_mode", "msg_dim", "seed")).items()
    ):
        success = [float(row["success"]) for row in group]
        tts = [float(row["time_to_success"]) for row in group if math.isfinite(float(row["time_to_success"]))]
        out.append(
            {
                "Task": task,
                "Method": method,
                "Communication Mode": mode,
                "Msg Dim": msg_dim,
                "Seed": seed,
                "Episodes": len(group),
                "Success Rate": safe_mean(success),
                "Mean Return": safe_mean([row["mean_return"] for row in group]),
                "Episode Return": safe_mean([row["episode_return"] for row in group]),
                "AUC": safe_mean([row["auc"] for row in group]),
                "Time to Success": safe_mean(tts),
                "Mean Episode Length": safe_mean([row["episode_length"] for row in group]),
            }
        )
    return out


def fmt_mean_std(mean: float, std: float) -> str:
    if not math.isfinite(mean):
        return ""
    return f"{mean:.3f} +/- {std:.3f}"


def summarize_method(seed_rows: list[dict]) -> list[dict]:
    out = []
    for (task, method, mode, msg_dim), group in sorted(
        group_rows(seed_rows, ("Task", "Method", "Communication Mode", "Msg Dim")).items()
    ):
        success = [row["Success Rate"] for row in group]
        tts = [row["Time to Success"] for row in group]
        ret = [row["Mean Return"] for row in group]
        auc = [row["AUC"] for row in group]
        out.append(
            {
                "Task": task,
                "Method": method,
                "Communication Mode": mode,
                "Msg Dim": msg_dim,
                "Seeds": len(group),
                "Success Rate Mean": safe_mean(success),
                "Success Rate Std": safe_std(success),
                "Success Rate": fmt_mean_std(safe_mean(success), safe_std(success)),
                "Time to Success Mean": safe_mean(tts),
                "Time to Success Std": safe_std(tts),
                "Time to Success": fmt_mean_std(safe_mean(tts), safe_std(tts)),
                "Return Mean": safe_mean(ret),
                "Return Std": safe_std(ret),
                "Return": fmt_mean_std(safe_mean(ret), safe_std(ret)),
                "AUC Mean": safe_mean(auc),
                "AUC Std": safe_std(auc),
                "AUC": fmt_mean_std(safe_mean(auc), safe_std(auc)),
            }
        )
    by_task = defaultdict(dict)
    for row in out:
        by_task[row["Task"]][row["Communication Mode"]] = row
    for row in out:
        task_methods = by_task[row["Task"]]
        proposed = task_methods.get("motion_context")
        none = task_methods.get("none")
        full = task_methods.get("full_partner_observation")
        if proposed and none:
            row["Motion Context - None Success pp"] = 100.0 * (
                proposed["Success Rate Mean"] - none["Success Rate Mean"]
            )
        else:
            row["Motion Context - None Success pp"] = float("nan")
        if proposed and full:
            row["Motion Context - Full Obs Success pp"] = 100.0 * (
                proposed["Success Rate Mean"] - full["Success Rate Mean"]
            )
        else:
            row["Motion Context - Full Obs Success pp"] = float("nan")
    return out


def ablation_summary(seed_rows: list[dict]) -> list[dict]:
    wanted = ("none", "motion_only", "context_only", "motion_context")
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
        none = row.get("None", float("nan"))
        row["Motion Context - Motion Only"] = proposed - motion if math.isfinite(proposed + motion) else float("nan")
        row["Motion Context - Context Only"] = proposed - context if math.isfinite(proposed + context) else float("nan")
        row["Motion Context - None"] = proposed - none if math.isfinite(proposed + none) else float("nan")
        out.append(row)
    return out


def failure_breakdown(rows: list[dict]) -> list[dict]:
    out = []
    for (task, method, reason), group in sorted(group_rows(rows, ("task", "method", "termination_reason")).items()):
        total = len([row for row in rows if row["task"] == task and row["method"] == method])
        out.append(
            {
                "Task": task,
                "Method": method,
                "Failure Reason": reason,
                "Count": len(group),
                "Fraction": len(group) / max(total, 1),
            }
        )
    return out


def coordination_timing(rows: list[dict]) -> list[dict]:
    out = []
    for (task, method), group in sorted(group_rows(rows, ("task", "method")).items()):
        reach_gap = [
            abs(float(row["first_left_reach_step"]) - float(row["first_right_reach_step"]))
            for row in group
            if math.isfinite(float(row["first_left_reach_step"]))
            and math.isfinite(float(row["first_right_reach_step"]))
        ]
        grasp_gap = [
            abs(float(row["first_left_grasp_step"]) - float(row["first_right_grasp_step"]))
            for row in group
            if math.isfinite(float(row["first_left_grasp_step"]))
            and math.isfinite(float(row["first_right_grasp_step"]))
        ]
        out.append(
            {
                "Task": task,
                "Method": method,
                "Left Reach Step": safe_mean([row["first_left_reach_step"] for row in group]),
                "Right Reach Step": safe_mean([row["first_right_reach_step"] for row in group]),
                "Reach Gap": safe_mean(reach_gap),
                "Left Grasp Step": safe_mean([row["first_left_grasp_step"] for row in group]),
                "Right Grasp Step": safe_mean([row["first_right_grasp_step"] for row in group]),
                "Grasp Gap": safe_mean(grasp_gap),
                "Dual Grasp Step": safe_mean([row["first_dual_grasp_step"] for row in group]),
                "Lift Step": safe_mean([row["first_lift_step"] for row in group]),
                "Success Step": safe_mean([row["success_step"] for row in group]),
            }
        )
    return out


def message_statistics(traces: list[dict], args) -> list[dict]:
    rows = []
    for trace in traces:
        length = episode_len(trace)
        task = infer_task(trace, args.task)
        method = infer_method(trace, args.method)
        seed = infer_seed(trace, args.seed)
        for side in ("left", "right"):
            motion = vector_series(trace, f"{side}_motion", 3, length)
            context = context_series(trace, side, length)
            features = np.concatenate([motion, context], axis=-1) if length else np.zeros((0, 6))
            names = [*MOTION_NAMES, *CONTEXT_NAMES]
            for idx, name in enumerate(names):
                values = features[:, idx] if features.size else np.zeros(0)
                rows.append(
                    {
                        "Task": task,
                        "Method": METHOD_LABELS.get(method, method),
                        "Communication Mode": method,
                        "Seed": seed,
                        "Side": side,
                        "Feature": name,
                        "Mean": safe_mean(values),
                        "Std": safe_std(values),
                        "Saturation Ratio": safe_mean(np.abs(values) >= 0.99),
                        "Zero Ratio": safe_mean(np.abs(values) < 1.0e-6),
                        "Count": int(values.shape[0]),
                    }
                )
    return rows


def lift_phase_masks(trace: dict, length: int) -> dict[str, np.ndarray]:
    hprog = scalar_series(trace, "hprog", length)
    left_dist = scalar_series(trace, "left_target_dist", length)
    right_dist = scalar_series(trace, "right_target_dist", length)
    left_closure = scalar_series(trace, "left_closure", length)
    right_closure = scalar_series(trace, "right_closure", length)
    approach = (left_dist > 0.08) | (right_dist > 0.08)
    grasp = ((left_dist <= 0.08) & (right_dist <= 0.08)) | ((left_closure > 0.25) | (right_closure > 0.25))
    lift = hprog > 0.2
    stabilize = hprog > 0.8
    return {
        "approach": approach & ~lift,
        "grasp": grasp & ~lift,
        "lift": lift & ~stabilize,
        "stabilize": stabilize,
    }


def peg_phase_masks(trace: dict, length: int) -> dict[str, np.ndarray]:
    tip = scalar_series(trace, "tip_dist", length)
    lateral = scalar_series(trace, "lateral_error", length)
    axis = scalar_series(trace, "axis_alignment", length)
    depth = scalar_series(trace, "insertion_depth", length)
    approach = tip > 0.05
    align = (tip <= 0.05) & ((lateral > 0.008) | (axis < 0.92))
    insert = (depth > 0.0) & (depth <= 0.045)
    hold = depth > 0.045
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
            features = np.concatenate(
                [vector_series(trace, f"{side}_motion", 3, length), context_series(trace, side, length)],
                axis=-1,
            )
            for phase, mask in masks.items():
                if not np.any(mask):
                    continue
                for idx, name in enumerate([*MOTION_NAMES, *CONTEXT_NAMES]):
                    buckets[(task, METHOD_LABELS.get(method, method), method, seed, side, phase, name)].extend(
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
            features = np.concatenate(
                [vector_series(trace, f"{side}_motion", 3, length), context_series(trace, side, length)],
                axis=-1,
            )
            for idx, name in enumerate([*MOTION_NAMES, *CONTEXT_NAMES]):
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


def cumulative_eval_success(rows: list[dict], ci: float) -> list[dict]:
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
                "Step": int(episode),
                "Mean Success": mean,
                "CI Lower": max(0.0, mean - z * stderr),
                "CI Upper": min(1.0, mean + z * stderr),
                "Seeds": int(arr.size),
                "Source": "eval_trace_episode_order",
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
    fig.suptitle(f"Paper Motion Context Overview: episode {trace.get('episode_index', 0)}", fontsize=16)

    axes[0].plot(steps, np.linalg.norm(left_motion, axis=-1), label="L signed motion norm")
    axes[0].plot(steps, np.linalg.norm(right_motion, axis=-1), label="R signed motion norm")
    axes[0].set_title("Signed EE Motion Intent")
    axes[0].set_ylabel("norm")
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
    labels = list(CONTEXT_NAMES)
    x = np.arange(len(labels))
    width = 0.35
    left = [safe_mean([float(row[f"left_{name}_mean"]) for row in rows]) for name in labels]
    right = [safe_mean([float(row[f"right_{name}_mean"]) for row in rows]) for name in labels]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width / 2, left, width, label="L")
    ax.bar(x + width / 2, right, width, label="R")
    ax.set_xticks(x, labels)
    ax.set_ylim(0.0, 1.55)
    ax.set_title("Mean Proprioceptive Motion Context")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_ablation(rows: list[dict], out_path: Path) -> None:
    if not rows:
        return
    methods = ("None", "Motion Only", "Context Only", "Motion Context")
    tasks = [row["Task"] for row in rows]
    x = np.arange(len(tasks))
    width = 0.18
    fig, ax = plt.subplots(figsize=(11, 5))
    for i, method in enumerate(methods):
        values = [row.get(method, float("nan")) for row in rows]
        ax.bar(x + (i - 1.5) * width, values, width, label=method)
    ax.set_xticks(x, tasks)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("success rate")
    ax.set_title("Communication Ablation")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
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


def plot_cumulative_eval_success(rows: list[dict], out_dir: Path) -> None:
    by_task = defaultdict(list)
    for row in rows:
        by_task[row["Task"]].append(row)
    for task, task_rows in by_task.items():
        methods = sorted({row["Method"] for row in task_rows})
        if not methods:
            continue
        fig, ax = plt.subplots(figsize=(10, 5))
        for method in methods:
            group = sorted([row for row in task_rows if row["Method"] == method], key=lambda row: row["Step"])
            if not group:
                continue
            x = np.asarray([row["Step"] for row in group], dtype=float)
            mean = np.asarray([row["Mean Success"] for row in group], dtype=float)
            lo = np.asarray([row["CI Lower"] for row in group], dtype=float)
            hi = np.asarray([row["CI Upper"] for row in group], dtype=float)
            ax.plot(x, mean, label=method)
            ax.fill_between(x, lo, hi, alpha=0.15)
        ax.set_ylim(0.0, 1.0)
        ax.set_xlabel("evaluation episode order")
        ax.set_ylabel("cumulative success rate")
        ax.set_title(f"Cumulative Evaluation Success: {task}")
        ax.grid(True, alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / f"cumulative_eval_success_{task.lower()}.png", dpi=150)
        plt.close(fig)


def print_readiness(method_rows: list[dict], message_rows: list[dict], ablation_rows: list[dict]) -> None:
    print("\n=== Paper Analysis Summary ===")
    for row in method_rows:
        print(
            f"{row['Task']:>4} | {row['Method']:<18} | success={row['Success Rate']} "
            f"| return={row['Return']} | auc={row['AUC']}"
        )
    by_task_mode = {(row["Task"], row["Communication Mode"]): row for row in method_rows}
    for task in sorted({row["Task"] for row in method_rows}):
        proposed = by_task_mode.get((task, "motion_context"))
        none = by_task_mode.get((task, "none"))
        full = by_task_mode.get((task, "full_partner_observation"))
        if proposed and none:
            delta = 100.0 * (proposed["Success Rate Mean"] - none["Success Rate Mean"])
            print(f"[COMPARE] {task}: Motion Context - None = {delta:+.1f} pp")
        if proposed and full:
            delta = 100.0 * (proposed["Success Rate Mean"] - full["Success Rate Mean"])
            print(f"[COMPARE] {task}: Motion Context - Full Observation = {delta:+.1f} pp")

    saturation = [float(row["Saturation Ratio"]) for row in message_rows if str(row["Communication Mode"]) == "motion_context"]
    zero = [float(row["Zero Ratio"]) for row in message_rows if str(row["Communication Mode"]) == "motion_context"]
    max_sat = max(saturation) if saturation else 0.0
    max_zero = max(zero) if zero else 0.0
    if max_sat > 0.50 or max_zero > 0.95:
        status = "REVIEW"
        reason = "message features look saturated or degenerate"
    elif any(
        math.isfinite(float(row.get("Motion Context - None", float("nan"))))
        and float(row.get("Motion Context - None", 0.0)) <= 0.0
        for row in ablation_rows
    ):
        status = "REVIEW"
        reason = "Motion Context does not outperform None in at least one task"
    else:
        status = "READY"
        reason = "main traces are analyzable and message statistics are non-degenerate"
    print(f"[MESSAGE] max_saturation={max_sat:.3f} max_zero={max_zero:.3f}")
    print(f"[{status}] {reason}")
    print("Note: cumulative_eval_success.csv uses evaluation trace order, not training transitions.")


def main() -> None:
    args = parse_args()
    trace_dir = Path(args.trace_dir)
    out_dir = resolve_output_dir(trace_dir, Path(args.out_dir))
    out_dir.mkdir(parents=True, exist_ok=True)

    traces = load_traces(trace_dir, args.episode)
    rows = [summarize_trace(trace, args) for trace in traces]
    seed_rows = summarize_seed(rows)
    method_rows = summarize_method(seed_rows)
    ablation_rows = ablation_summary(seed_rows)
    failure_rows = failure_breakdown(rows)
    timing_rows = coordination_timing(rows)
    message_rows = message_statistics(traces, args)
    phase_rows = message_phase_statistics(traces, args)
    success_failure_rows = success_failure_message_comparison(traces, args)
    curve_rows = cumulative_eval_success(rows, args.ci)

    write_csv(out_dir / "eval_episodes.csv", rows)
    write_csv(out_dir / "episode_summary.csv", rows)
    write_csv(out_dir / "summary_by_seed.csv", seed_rows)
    write_csv(out_dir / "summary_by_method.csv", method_rows)
    write_csv(out_dir / "ablation_summary.csv", ablation_rows)
    write_csv(out_dir / "failure_breakdown.csv", failure_rows)
    write_csv(out_dir / "coordination_timing.csv", timing_rows)
    write_csv(out_dir / "message_statistics.csv", message_rows)
    write_csv(out_dir / "message_phase_statistics.csv", phase_rows)
    write_csv(out_dir / "success_failure_message_comparison.csv", success_failure_rows)
    write_csv(out_dir / "cumulative_eval_success.csv", curve_rows)

    plot_overview(traces[0], out_dir / "overview.png", max(int(args.smooth_window), 1))
    plot_context_summary(rows, out_dir / "context_summary.png")
    plot_ablation(ablation_rows, out_dir / "ablation.png")
    plot_failure_breakdown(failure_rows, out_dir)
    plot_cumulative_eval_success(curve_rows, out_dir)
    print_readiness(method_rows, message_rows, ablation_rows)

    print(f"\n[OK] analyzed {len(traces)} trace(s)")
    print(f"[OK] wrote {out_dir}")


if __name__ == "__main__":
    main()
