#!/usr/bin/env python3
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Execute a declared recovery pilot sequentially, preserving sources and exact commands."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import zipfile
from pathlib import Path


def fingerprints(paths):
    return {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    os.chdir(root)
    plan = json.loads(args.manifest.read_text())
    names = [row["name"] for row in plan["cases"]]
    if (
        not names
        or len(set(names)) != len(names)
        or any(Path(name).name != name or name in (".", "..") for name in names)
    ):
        raise ValueError("Pilot case names must be unique directory basenames")
    args.output.mkdir(parents=True, exist_ok=False)
    play = Path("scripts/reinforcement_learning/skrl/play_openarm_bimanual_direct.py")
    paths = sorted(Path("source/isaaclab_tasks/isaaclab_tasks/direct/openarm_motion_context").rglob("*.py")) + [
        play,
        Path(__file__).relative_to(root),
    ]
    source_hashes = fingerprints(paths)
    with zipfile.ZipFile(args.output / "sources.zip", "x", zipfile.ZIP_DEFLATED) as archive:
        for path in paths:
            archive.write(path, path)
    state = {
        "status": "running",
        "scope": "diagnostic pilot, not confirmatory performance evidence",
        "plan": plan,
        "source_sha256": source_hashes,
        "cases": [],
    }
    state_path = args.output / "lifecycle.json"
    state_path.write_text(json.dumps(state, indent=2) + "\n")
    environment = {**os.environ, "OMP_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "4"}
    try:
        for row in plan["cases"]:
            if fingerprints(paths) != source_hashes:
                raise RuntimeError("Source changed during the declared pilot; start a new campaign")
            checkpoint = Path(row["checkpoint"])
            if not checkpoint.is_file():
                raise FileNotFoundError(checkpoint)
            pulse = args.output / f"{row['name']}_pulse.json"
            pulse.write_text(json.dumps(row["pulse"], indent=2) + "\n")
            directory = args.output / row["name"]
            command = [
                sys.executable,
                str(play),
                "--task",
                row["task"],
                "--num_envs",
                "1",
                "--checkpoint",
                str(checkpoint),
                "--headless",
                "--deterministic_eval",
                "--seed",
                str(row["seed"]),
                "--num_steps",
                str(row.get("max_steps", row["episodes"] * 1200)),
                "--num_eval_episodes",
                str(row["episodes"]),
                "--scenario_seed_start",
                str(row["scenario_seed_start"]),
                "--save_motion_context_trace",
                "--motion_context_trace_dir",
                str(directory),
                "--wrist_disturbance",
                str(pulse),
            ]
            record = {
                "name": row["name"],
                "status": "running",
                "command": command,
                "checkpoint_sha256": fingerprints([checkpoint])[str(checkpoint)],
                "started_unix": time.time(),
            }
            state["cases"].append(record)
            state_path.write_text(json.dumps(state, indent=2) + "\n")
            print(f"Starting {row['name']}", flush=True)
            with (args.output / f"{row['name']}.log").open("x") as stream:
                completed = subprocess.run(
                    command, env=environment, stdout=stream, stderr=subprocess.STDOUT, check=False
                )
            record.update(
                status="complete" if completed.returncode == 0 else "failed",
                returncode=completed.returncode,
                finished_unix=time.time(),
            )
            state_path.write_text(json.dumps(state, indent=2) + "\n")
            if completed.returncode:
                raise RuntimeError(f"Pilot {row['name']} failed; see its separate log")
        state["status"] = "complete"
    except BaseException:
        state["status"] = "incomplete"
        raise
    finally:
        state_path.write_text(json.dumps(state, indent=2) + "\n")


if __name__ == "__main__":
    main()
