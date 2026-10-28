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
