# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""External-force protocols must preserve physical time and unambiguous units."""

import copy

import pytest
import torch

from isaaclab_tasks.direct.openarm_motion_context.eval.wrist_disturbance import pulse_schedule


def test_pulse_protocol_quantizes_seconds_and_rejects_ambiguous_force_vectors():
    config = {
        "onset_s": 0.501,
        "duration_s": 0.2,
        "force_n": 2.0,
        "events": [{"side": "left", "direction_w": [0, 1, 0]}, {"side": "right", "direction_w": [0, -1, 0]}],
    }
    before = torch.get_rng_state().clone()
    schedule = pulse_schedule(config, 1 / 120)
    assert schedule["onset_tick"] == 61
    assert schedule["end_tick"] == 85
    assert schedule["effective_duration_s"] == pytest.approx(0.2)
    assert schedule["events"][0]["force_w"] == [0, 2, 0]
    assert schedule["events"][1]["force_w"] == [0, -2, 0]
    control = pulse_schedule({**config, "force_n": 0.0}, 1 / 120)
    assert control["events"][0]["force_w"] == [0, 0, 0]
    assert control["onset_tick"] == 61 and control["end_tick"] == 85
    assert torch.equal(before, torch.get_rng_state())
    ambiguous = copy.deepcopy(config)
    ambiguous["events"][0]["direction_w"] = [0, 2, 0]
    with pytest.raises(ValueError, match="unit vector"):
        pulse_schedule(ambiguous, 1 / 120)
    with pytest.raises(ValueError, match="finite numbers"):
        pulse_schedule({**config, "force_n": float("nan")}, 1 / 120)


def test_recovery_summary_keeps_pre_event_failures_and_ignores_unscored_reset(tmp_path):
    """Overall and exposed denominators differ; partial pulses retain their actual impulse."""
    import json
    import runpy
    from pathlib import Path

    analyzer = runpy.run_path(
        str(Path(__file__).resolve().parents[3] / "scripts/studies/analyze_recovery_evaluation.py")
    )
    manifest = {
        "status": "complete",
        "schedule": {
            "physics_dt": 1 / 120,
            "onset_tick": 60,
            "end_tick": 84,
            "requested": {"force_n": 1.0},
        },
        "episodes": [],
    }
    # Fail before onset, fail halfway through the pulse, succeed after release.
    for index, (ticks, window, impulse, success) in enumerate(
        [(20, 0, 0.0, False), (72, 12, 0.1, False), (180, 24, 0.2, True)]
    ):
        manifest["episodes"].append(
            {
                "physics_ticks": ticks,
                "window_ticks": window,
                "input_impulse_w_ns": [0, impulse, 0],
                "event": {"force_w": [0, 1, 0]},
            }
        )
        trace = {
            "trace_episode_index": index,
            "num_steps": ticks // 2,
            "terminal_success": success,
            "termination_reason": "success" if success else "drop",
            "metadata": {
                "context_dt": 1 / 60,
                "evaluation_scenario_seed": 42100 + index,
                "evaluation_scenario_initial_state_sha256": f"start-{index}",
            },
        }
        (tmp_path / f"motion_context_trace_episode_{index:03d}.json").write_text(json.dumps(trace))
    manifest["episodes"].append({"physics_ticks": 0})  # Final automatic reset is not a scored trial.
    (tmp_path / "wrist_disturbance.json").write_text(json.dumps(manifest))
    (tmp_path / "evaluation_reset_scenarios.json").write_text(
        json.dumps({"status": "complete", "expected_episodes": 3})
    )
    result = analyzer["summarize_run"](tmp_path)
    assert (result["successes"], result["episodes"]) == (1, 3)
    assert (result["successes_given_onset"], result["reached_onset"]) == (1, 2)
    assert result["pre_onset_terminations"] == 1
    assert result["complete_pulse_windows"] == 1
    assert result["rows"][1]["observed_window_s"] == pytest.approx(0.1)
    assert result["rows"][2]["completion_after_window_s"] == pytest.approx(0.8)
