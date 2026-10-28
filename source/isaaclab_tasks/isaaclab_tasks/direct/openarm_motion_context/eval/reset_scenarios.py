# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Evaluation-only reset seeds independent of Gaussian sampling and rollout length."""

import hashlib
import json
from pathlib import Path
from types import MethodType

PROTOCOL = "openarm_torch_reset_scenarios/v1"


def _state_digest(state):
    return hashlib.sha256(json.dumps(state, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def install_reset_scenarios(task, seed_start: int) -> list[dict]:
    """Seed each single-world task reset while preserving the caller's Torch RNG.

    Install on the unwrapped task immediately before evaluation's first reset.
    The task still performs its own reset, including randomization and pregrasp
    placement. This does not replay physical states, change observations, seed
    the physics solver, or guarantee determinism of later contact dynamics.
    NumPy/Python randomization is outside this Torch-only protocol.
    """
    import torch

    from ..train.evaluation_state import capture_physical_start

    if task.num_envs != 1:
        raise ValueError("Reset scenarios require one evaluation world")
    if not isinstance(seed_start, int) or isinstance(seed_start, bool) or not 0 <= seed_start < 2**32:
        raise ValueError("Scenario seed start must be an integer in [0, 2**32)")
    if hasattr(task, "_scenario_reset_records"):
        raise ValueError("Reset scenarios are already installed")
    device = torch.device(task.device)
    devices = (
        [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
    )
    original = task._reset_idx
    records = []

    def reset(self, env_ids):
        if env_ids is not None and len(env_ids) == 0:
            return original(env_ids)
        seed = seed_start + len(records)
        with torch.random.fork_rng(devices=devices):
            torch.set_rng_state(torch.Generator(device="cpu").manual_seed(seed).get_state())
            if devices:
                torch.cuda.set_rng_state(torch.Generator(device=device).manual_seed(seed).get_state(), device=device)
            result = original(env_ids)
            state = capture_physical_start(lambda name, default=None: getattr(self, name, default), 0)
        records.append(
            {"scenario_index": len(records), "scenario_seed": seed, "initial_state_sha256": _state_digest(state)}
        )
        return result

    task._scenario_reset_records = records
    task._reset_idx = MethodType(reset, task)
    return records


def finalize_reset_scenarios(trace_dir, records, seed_start: int, expected_episodes: int, eval_returned: bool):
    """Label only new scenario traces after checking their observed physical starts.

    Native automatic reset creates one extra, unscored start after the final
    terminal transition. Preserve that record without counting it as an episode.
    """
    directory = Path(trace_dir)
    files = sorted(directory.glob("motion_context_trace_episode_*.json"))
    traces = [(path, json.loads(path.read_text())) for path in files]
    for _, trace in traces:
        index = int(trace["trace_episode_index"])
        if (
            index >= len(records)
            or _state_digest(trace["initial_physical_state"]) != records[index]["initial_state_sha256"]
        ):
            raise ValueError("Scenario trace does not match its recorded reset state")
    for path, trace in traces:
        index = int(trace["trace_episode_index"])
        trace["metadata"].update(
            {
                "evaluation_reset_protocol": PROTOCOL,
                "evaluation_scenario_seed_start": seed_start,
                "evaluation_scenario_seed": records[index]["scenario_seed"],
                "evaluation_scenario_index": index,
                "evaluation_scenario_initial_state_sha256": records[index]["initial_state_sha256"],
            }
        )
        path.write_text(json.dumps(trace, indent=2) + "\n")
    complete = (
        eval_returned
        and len(traces) == expected_episodes
        and all(trace["terminated_at_end"] or trace["truncated_at_end"] for _, trace in traces)
    )
    manifest = {
        "protocol": PROTOCOL,
        "seed_start": seed_start,
        "expected_episodes": expected_episodes,
        "scored_traces": len(traces),
        "status": "complete" if complete else "incomplete",
        "reset_records": records,
        "extra_terminal_reset_is_unscored": True,
        "solver_determinism_claimed": False,
    }
    (directory / "evaluation_reset_scenarios.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if eval_returned and not complete:
        raise ValueError("Scenario evaluation ended before all requested episodes completed")
