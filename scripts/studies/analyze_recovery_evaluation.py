#!/usr/bin/env python3
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Summarize all scored episodes separately from those exposed to a wrist pulse."""

import argparse
import csv
import json
import math
from pathlib import Path


def _alignment_recovery(trace, onset_s, end_s, dt, hold_s):
    """Measure regained alignment only after observed loss from an aligned start.

    This geometric measure is distinct from completion time and net force.
    Unobserved loss and an unaligned pre-event state have no recovery-time value.
    """
    if "PegInHole" in trace.get("metadata", {}).get("task", ""):
        metadata = trace["metadata"]
        keys = ("shaft_radial_envelope", "axis_alignment", "dual_grasp_ok")
        if not all(key in trace for key in keys):
            return {"status": "unavailable", "seconds": None}
        valid = [
            radial <= metadata["success_radial_clearance"]
            and axis >= metadata["success_axis_threshold"]
            and bool(grasp)
            for radial, axis, grasp in zip(*(trace[key] for key in keys))
        ]
    elif all(key in trace for key in ("tilt_ok", "dual_grasp_ok")):
        valid = [bool(tilt) and bool(grasp) for tilt, grasp in zip(trace["tilt_ok"], trace["dual_grasp_ok"])]
    else:
        return {"status": "unavailable", "seconds": None}
    before = min(int(math.floor(onset_s / dt + 1e-9)), len(valid) - 1)
    if before < 0 or not valid[before]:
        return {"status": "not_aligned_before_event", "seconds": None}
    lost = next((i for i, value in enumerate(valid) if onset_s < i * dt <= end_s + dt and not value), None)
    if lost is None:
        return {"status": "no_observed_alignment_loss", "seconds": None}
    hold = max(1, math.ceil(hold_s / dt - 1e-9))
    start = max(lost + 1, math.ceil(end_s / dt - 1e-9))
    regained = next((i for i in range(start, len(valid) - hold + 1) if all(valid[i : i + hold])), None)
    return {
        "status": "recovered" if regained is not None else "not_recovered",
        "seconds": regained * dt - end_s if regained is not None else None,
    }


def summarize_run(directory: Path, recovery_hold_s=0.2):
    """Audit exposure/impulse accounting; retain failures before the scheduled onset."""
    directory = Path(directory)
    manifest = json.loads((directory / "wrist_disturbance.json").read_text())
    resets = json.loads((directory / "evaluation_reset_scenarios.json").read_text())
    if manifest["status"] != "complete" or resets["status"] != "complete":
        raise ValueError("Cannot summarize an incomplete evaluation as completed")
    files = sorted(directory.glob("motion_context_trace_episode_*.json"))
    if len(files) != resets["expected_episodes"]:
        raise ValueError("Scored episode count differs from declared reset protocol")
    schedule = manifest["schedule"]
    dt = schedule["physics_dt"]
    onset, end = schedule["onset_tick"], schedule["end_tick"]
    rows = []
    for path in files:
        trace = json.loads(path.read_text())
        index = trace["trace_episode_index"]
        record = manifest["episodes"][index]
        control_dt = trace["metadata"]["context_dt"]
        ticks = record["physics_ticks"]
        if abs(ticks * dt - trace["num_steps"] * control_dt) > 1e-6:
            raise ValueError("Trace length differs from recorded physics-step count")
        expected_window = max(0, min(ticks, end) - onset)
        if record["window_ticks"] != expected_window:
            raise ValueError("Pulse-window accounting disagrees with observed episode duration")
        expected_impulse = [value * expected_window * dt for value in record["event"]["force_w"]]
        if any(abs(a - b) > 1e-4 for a, b in zip(record["input_impulse_w_ns"], expected_impulse)):
            raise ValueError("Backend input impulse differs from scheduled delivered pulse")
        success = trace["terminal_success"]
        if not isinstance(success, bool):
            raise ValueError("Terminal task success must be explicitly recorded as a boolean")
        reached = ticks > onset
        row = {
            "episode_index": index,
            "scenario_seed": trace["metadata"]["evaluation_scenario_seed"],
            "initial_state_sha256": trace["metadata"]["evaluation_scenario_initial_state_sha256"],
            "success": success,
            "reached_onset": reached,
            "complete_pulse_window": ticks >= end,
            "observed_window_s": expected_window * dt,
            "input_impulse_w_ns": record["input_impulse_w_ns"],
            "terminal_reason": trace["termination_reason"],
            "completion_after_window_s": ticks * dt - end * dt if success and ticks >= end else None,
            "alignment_recovery": _alignment_recovery(trace, onset * dt, end * dt, control_dt, recovery_hold_s)
            if reached
            else {"status": "event_not_reached", "seconds": None},
        }
        rows.append(row)
    exposed = [row for row in rows if row["reached_onset"]]
    return {
        "directory": str(directory),
        "episodes": len(rows),
        "successes": sum(row["success"] for row in rows),
        "reached_onset": len(exposed),
        "successes_given_onset": sum(row["success"] for row in exposed),
        "pre_onset_terminations": len(rows) - len(exposed),
        "complete_pulse_windows": sum(row["complete_pulse_window"] for row in rows),
        "force_n": schedule["requested"]["force_n"],
        "recovery_hold_s": recovery_hold_s,
        "interpretation": (
            "Completion, conditional completion, and geometric recovery are separate; "
            "pilot episodes are not independent training runs."
        ),
        "rows": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directories", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    results = [summarize_run(path) for path in args.directories]
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "summary.json").write_text(json.dumps(results, indent=2, allow_nan=False) + "\n")
    with (args.output / "summary.csv").open("w", newline="") as stream:
        fields = [key for key in results[0] if key not in ("rows", "interpretation")]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: value for key, value in result.items() if key in fields} for result in results)
    for result in results:
        print(
            f"{Path(result['directory']).name}: {result['successes']}/{result['episodes']} overall; "
            f"{result['successes_given_onset']}/{result['reached_onset']} given onset; "
            f"{result['pre_onset_terminations']} terminated before onset"
        )


if __name__ == "__main__":
    main()
