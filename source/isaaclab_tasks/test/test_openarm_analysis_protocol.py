# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Communication methods may share a table only under one physical success protocol."""

from copy import deepcopy
from types import SimpleNamespace

import pytest

from isaaclab_tasks.direct.openarm_motion_context.eval import analyze_motion_context as analysis
from isaaclab_tasks.direct.openarm_motion_context.eval.analyze_motion_context import ensure_comparable_success_protocols


@pytest.mark.parametrize("change", [None, "protocol", "threshold"])
def test_analysis_preserves_method_comparison_but_rejects_mixed_success_definitions(change):
    trace = {
        "metadata": {
            "task": "Peg",
            "communication_mode": "none",
            "success_protocol": "circle_cad_seating_v1",
            "success_depth_threshold": 0.018,
        }
    }
    partner = deepcopy(trace)
    partner["metadata"]["communication_mode"] = "motion_context"
    if change == "protocol":
        partner["metadata"]["success_protocol"] = "legacy_marker_depth_v1"
    elif change == "threshold":
        partner["metadata"]["success_depth_threshold"] = 0.0225
    if change is None:
        ensure_comparable_success_protocols([trace, partner])
    else:
        with pytest.raises(ValueError, match="success protocols"):
            ensure_comparable_success_protocols([trace, partner])


@pytest.mark.parametrize(
    "sharing,mask,expected",
    [
        ("no_share", "none", "none"),
        ("intent_share", "message_zero", "motion_context"),
        ("intent_share", "none", "motion_context"),
    ],
)
def test_zero_message_baseline_keeps_architecture_but_reports_no_communication(sharing, mask, expected):
    trace = {
        "num_steps": 2,
        "reward_mean": [0.0, 0.0],
        "metadata": {
            "task": "Lift",
            "communication_mode": "motion_context",
            "sharing_mode": sharing,
            "diagnostic_mask": mask,
            "communication_payload_dim": 6,
            "context_dt": 1 / 60,
        },
    }
    assert analysis.infer_method(trace, None) == expected
    row = analysis.summarize_trace(trace, SimpleNamespace(method=None, task=None, seed=None))
    assert row["communication_mode"] == expected
    assert row["msg_dim"] == (0 if sharing == "no_share" else 6)
    if sharing == "no_share":
        trace["metadata"]["communication_payload_layout"] = ["x"] * 6
        assert analysis.payload_layout(trace, expected) == []


@pytest.mark.parametrize("value", [0.0, 1e-3, float("nan")])
def test_no_share_analysis_checks_actual_actor_inputs(value):
    trace = {
        "metadata": {"sharing_mode": "no_share", "communication_mode": "motion_context"},
        "left_policy_message": [[value] + [0.0] * 29],
    }
    if value == 0.0:
        analysis.ensure_no_share_inputs_zero([trace])
    else:
        with pytest.raises(ValueError, match="NoShare trace"):
            analysis.ensure_no_share_inputs_zero([trace])


@pytest.mark.parametrize(
    "name,family",
    [
        ("Lift", "Lift"),
        ("lift", "Lift"),
        ("lift_with_ball", "Lift"),
        ("Isaac-OpenArm-LiftWithBall-v0", "Lift"),
        ("Peg", "Peg"),
        ("peg_in_hole_circle", "Peg"),
        ("Isaac-OpenArm-PegInHole-Circle-Seated-v0", "Peg"),
    ],
)
def test_analysis_task_overrides_select_the_metric_family(name, family):
    assert analysis.infer_task({}, name) == family


def test_analysis_rejects_unrecognized_task_overrides():
    with pytest.raises(ValueError, match="Unknown OpenArm task"):
        analysis.infer_task({}, "Cartpole")
