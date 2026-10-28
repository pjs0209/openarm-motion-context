#!/usr/bin/env python3
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Declare and run independent, matched NoShare/MotionOnly/Twist/Trend training."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import yaml

CONDITIONS = {
    "noshare": ("motion_context", "no_share", "trend"),
    "motiononly": ("motion_only", "intent_share", "trend"),
    "signedtwist": ("motion_context", "intent_share", "signed_twist"),
    "full6d": ("motion_context", "intent_share", "trend"),
}
PREFIX = "isaaclab_tasks.direct.openarm_motion_context"


def _shared_configuration(run):
    """Compare resolved saved configurations, allowing only declared message/run fields."""
    with (run / "params/env.yaml").open() as stream:
        env = yaml.load(stream, Loader=yaml.FullLoader)
    with (run / "params/agent.yaml").open() as stream:
        agent = yaml.load(stream, Loader=yaml.FullLoader)
    for key in ("communication_mode", "sharing_mode", "motion_context_representation"):
        del env[key]
    del agent["models"]["policy"]["communication_mode"]
    del agent["agent"]["experiment"]["experiment_name"]
    evidence = json.loads((run / "main_comparison_model.json").read_text())
    return {"env": env, "agent": agent, "models": evidence["models"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true", help="Otherwise write the expanded plan only")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    os.chdir(root)
    manifest = json.loads(args.manifest.read_text())
    if set(manifest) != {"name", "scope", "seeds", "num_envs", "timesteps", "tasks"}:
        raise ValueError("Unexpected or missing manifest fields")
    if not manifest["name"].replace("_", "").isalnum():
        raise ValueError("Campaign name must be alphanumeric with optional underscores")
    if (
        not manifest["seeds"]
        or len(set(manifest["seeds"])) != len(manifest["seeds"])
        or any(type(seed) is not int or seed < 0 for seed in manifest["seeds"])
    ):
        raise ValueError("Declare unique nonnegative training seeds")
    if manifest["num_envs"] < 1 or manifest["timesteps"] < 32 or manifest["timesteps"] % 32:
        raise ValueError("Use positive world count and complete 32-step PPO rollout windows")
    if not manifest["tasks"] or set(manifest["tasks"]) - {"lift", "peg"}:
        raise ValueError("Declare Lift and/or Peg task profiles")
    cases = []
    for task, profile in manifest["tasks"].items():
        overrides = profile["common_overrides"]
        keys = [value.split("=", 1)[0].lstrip("+") for value in overrides]
        reserved = {
            "env.communication_mode",
            "env.sharing_mode",
            "env.motion_context_representation",
            "env.observe_control_state",
            "agent.runtime.agent_builder",
            "agent.runtime.main_comparison_task",
            "agent.trainer.timesteps",
            "agent.agent.experiment.checkpoint_interval",
            "seed",
            "env.seed",
        }
        if (
            len(set(keys)) != len(keys)
            or any("=" not in value for value in overrides)
            or set(keys) & reserved
            or any(value.startswith("--") for value in overrides)
        ):
            raise ValueError("Common overrides cannot replace campaign/message fields or CLI options")
        for seed in manifest["seeds"]:
            for condition, (mode, sharing, representation) in CONDITIONS.items():
                name = f"{manifest['name']}_{task}_{condition}_seed{seed}"
                command = [
                    sys.executable,
                    "scripts/reinforcement_learning/skrl/train_openarm_bimanual_direct.py",
                    "--task",
                    profile["task"],
                    "--num_envs",
                    str(manifest["num_envs"]),
                    "--seed",
                    str(seed),
                    "--mode",
                    sharing,
                    "--headless",
                    "--experiment_tag",
                    name,
                    "--debug_interval",
                    str(min(512, manifest["timesteps"])),
                    *overrides,
                    f"env.communication_mode={mode}",
                    f"env.motion_context_representation={representation}",
                    "env.observe_control_state=true",
                    f"agent.runtime.agent_builder={PREFIX}.models.main_comparison:build_main_comparison_agent",
                    f"+agent.runtime.main_comparison_task={task}",
                    f"agent.trainer.timesteps={manifest['timesteps']}",
                    f"agent.agent.experiment.checkpoint_interval={manifest['timesteps']}",
                ]
                cases.append(
                    {
                        "name": name,
                        "task": task,
                        "seed": seed,
                        "condition": condition,
                        "log_group": profile["log_group"],
                        "command": command,
                    }
                )
    args.output.mkdir(parents=True, exist_ok=False)
    paths = sorted(Path("source/isaaclab_tasks/isaaclab_tasks/direct/openarm_motion_context").rglob("*.py"))
    paths += sorted(
        Path("source/isaaclab_tasks/isaaclab_tasks/direct/openarm_motion_context/train/configs").glob("*.yaml")
    )
    paths += [
        Path(__file__).relative_to(root),
        Path("scripts/reinforcement_learning/skrl/train_openarm_bimanual_direct.py"),
        Path("scripts/reinforcement_learning/skrl/play_openarm_bimanual_direct.py"),
    ]

    def fingerprints():
        return {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}

    source = fingerprints()
    state = {"status": "declared", "manifest": manifest, "cases": cases, "source_sha256": source}

    def save():
        (args.output / "lifecycle.json").write_text(json.dumps(state, indent=2) + "\n")

    with zipfile.ZipFile(args.output / "sources.zip", "x", zipfile.ZIP_DEFLATED) as archive:
        for path in paths:
            archive.write(path, path)
    save()
    if not args.execute:
        print(f"Declared {len(cases)} independent runs in {args.output}")
        return
    references = {}
    environment = {**os.environ, "OMP_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "4"}
    state["status"] = "running"
    try:
        for case in cases:
            if fingerprints() != source:
                raise RuntimeError("Sources changed; declare a new campaign")
            case.update(status="running", started_unix=time.time())
            save()
            print(f"Starting {case['name']}", flush=True)
            with (args.output / f"{case['name']}.log").open("x") as stream:
                result = subprocess.run(
                    case["command"], env=environment, stdout=stream, stderr=subprocess.STDOUT, check=False
                )
            case["returncode"] = result.returncode
            if result.returncode:
                raise RuntimeError(f"Training failed: {case['name']}")
            runs = list((root / "logs/current/skrl" / case["log_group"]).glob(f"*_{case['name']}"))
            if len(runs) != 1:
                raise RuntimeError(f"Ambiguous training directory: {runs}")
            run = runs[0]
            checkpoint = run / "checkpoints" / f"agent_{manifest['timesteps']}.pt"
            if not checkpoint.is_file():
                raise RuntimeError("Final checkpoint missing")
            actual = _shared_configuration(run)
            key = (case["task"], case["seed"])
            if key in references and actual != references[key]:
                raise RuntimeError("Resolved common settings, model shapes or initial weights differ between messages")
            references[key] = actual
            case.update(
                status="complete",
                finished_unix=time.time(),
                run_dir=str(run),
                checkpoint=str(checkpoint),
                checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                common_configuration_and_initial_models_match=True,
            )
            save()
        state["status"] = "complete"
    except BaseException:
        state["status"] = "incomplete"
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
