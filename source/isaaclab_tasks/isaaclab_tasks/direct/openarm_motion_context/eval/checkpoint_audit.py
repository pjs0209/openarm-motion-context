# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Inspect saved policy distributions on identical observed inputs, without physics replay."""

import hashlib
import json
from pathlib import Path

import torch


def audit_checkpoint_distributions(
    agent, directory: Path, checkpoints: list[Path], reference: Path, switch_epsilon: float
):
    """Use each checkpoint's real policy and scaler on one recorded input corpus.

    Old policies see states visited by the reference policy. These counterfactual
    outputs are not evaluations of old-policy task performance. The reference
    checkpoint must reproduce the recorded rollout means before results are saved.
    """
    directory = Path(directory)
    trace = json.loads((directory / "motion_context_trace_episode_000.json").read_text())
    if trace["metadata"]["diagnostic_mask"] != "none":
        raise ValueError("Checkpoint audit requires unmasked observations")
    count = trace["num_steps"]
    records = [json.loads(line) for line in (directory / "actor_observations.jsonl").read_text().splitlines()][:count]
    if len(records) != count or any(record["episode_control_step"] != i for i, record in enumerate(records)):
        raise ValueError("Observation corpus must be the complete first scored episode in control-step order")
    paths = [Path(path).resolve() for path in checkpoints]
    reference = reference.resolve()
    if reference not in paths or len(set(paths)) != len(paths):
        raise ValueError("Include the reference checkpoint once in a unique checkpoint list")
    if any(path.parent != reference.parent or not path.is_file() for path in paths):
        raise ValueError("Audit checkpoints must belong to the same recorded training run")
    corpus = {
        uid: torch.tensor([record["actor_observations"][uid] for record in records], device=agent.device)
        for uid in ("left_arm", "right_arm")
    }
    results = []
    reference_error = {}
    try:
        for path in paths:
            agent.load(str(path))
            row = {"checkpoint": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "arms": {}}
            with torch.no_grad():
                for side, uid in (("left", "left_arm"), ("right", "right_arm")):
                    policy = agent.models[uid]["policy"]
                    standardized = agent._state_preprocessor[uid](corpus[uid], train=False)
                    mean, log_std, _ = policy.compute({"states": standardized}, role="policy")
                    # Match GaussianMixin's actual variance clamp, rather than
                    # scoring a distribution inferred from executed clipped commands.
                    if policy._g_clip_log_std:
                        log_std = log_std.clamp(policy._g_log_std_min, policy._g_log_std_max)
                    sigma = log_std[-1].exp()
                    gripper = mean[:, -1]
                    close_probability = torch.special.ndtr((-switch_epsilon - gripper) / sigma)
                    row["arms"][side] = {
                        "gripper_mean_min": gripper.min().item(),
                        "gripper_mean_max": gripper.max().item(),
                        "gripper_mean_average": gripper.mean().item(),
                        "gripper_std": sigma.item(),
                        "mean_selects_close_fraction": (gripper < -switch_epsilon).float().mean().item(),
                        "sample_close_probability_average": close_probability.mean().item(),
                        "mean_above_execution_max_fraction": (gripper > 1).float().mean().item(),
                    }
                    if path == reference:
                        expected = torch.tensor(trace[f"{side}_policy_mean_action"], device=agent.device)
                        error = (mean - expected).abs().max().item()
                        reference_error[side] = error
                        if not torch.isfinite(mean).all() or error > 1e-4:
                            raise RuntimeError(
                                f"Reference policy/scaler does not reproduce real rollout means: {error}"
                            )
            results.append(row)
    finally:
        agent.load(str(reference))
    output = {
        "scope": (
            "counterfactual checkpoint distributions on the reference policy's first-episode raw states, "
            "not task success"
        ),
        "reference_checkpoint": str(reference),
        "observed_states": count,
        "switch_epsilon": switch_epsilon,
        "reference_mean_max_error": reference_error,
        "checkpoints": results,
    }
    (directory / "checkpoint_distributions.json").write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    return output
