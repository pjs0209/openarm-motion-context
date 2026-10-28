# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Evaluation start records must preserve task identity, quaternion order and RNG."""

import json
from types import SimpleNamespace

import pytest
import torch

from isaaclab_tasks.direct.openarm_motion_context.eval.reset_scenarios import (
    finalize_reset_scenarios,
    install_reset_scenarios,
)
from isaaclab_tasks.direct.openarm_motion_context.train.evaluation_state import capture_physical_start


@pytest.mark.parametrize("bodies", [("object",), ("object", "ball"), ("peg", "hole")])
def test_pre_action_start_is_an_independent_nonmutating_record(bodies):
    state = torch.zeros(2, 13)
    state[:, 6] = 1.0  # Native XYZW, converted to saved WXYZ by the boundary.
    state[1, 0] = 3.0
    robot = SimpleNamespace(
        data=SimpleNamespace(
            root_state_w=state,
            joint_pos=torch.arange(36.0).reshape(2, 18),
            joint_vel=torch.zeros(2, 18),
        )
    )
    assets = {"robot": robot, **{name: SimpleNamespace(data=SimpleNamespace(root_state_w=state)) for name in bodies}}
    rng = torch.get_rng_state().clone()
    record = capture_physical_start(assets.get, 1)
    assert set(record) == {"robot_root_state_w", "robot_joint_pos", "robot_joint_vel"} | {
        f"{name}_root_state_w" for name in bodies
    }
    assert record["robot_root_state_w"][:7] == [3.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
    assert record["robot_joint_pos"] == list(range(18, 36))
    assert torch.equal(rng, torch.get_rng_state())
    torch.testing.assert_close(state[:, 6], torch.ones(2))
    state[1, 0] = 99.0
    robot.data.joint_pos.zero_()
    assert record["robot_root_state_w"][0] == 3.0
    assert record["robot_joint_pos"][0] == 18.0


def test_reset_scenarios_isolate_policy_rng_and_audit_terminal_autoreset(tmp_path):
    """Different rollout draw counts must yield the same scored physical starts."""

    class ResetTask:
        num_envs, device = 1, "cpu"

        def __init__(self):
            state = torch.zeros(1, 13)
            state[:, 6] = 1
            self.robot = SimpleNamespace(
                data=SimpleNamespace(root_state_w=state, joint_pos=torch.zeros(1, 18), joint_vel=torch.zeros(1, 18))
            )

        def _reset_idx(self, ids):
            # This supplies only an ordinary stochastic task reset. The real
            # installer owns seeding, isolation, indexing and start recording.
            self.robot.data.joint_pos.copy_(torch.rand_like(self.robot.data.joint_pos))

    short, long = ResetTask(), ResetTask()
    records = install_reset_scenarios(short, 42000)
    other = install_reset_scenarios(long, 42000)
    starts = []
    for index in range(3):  # Two scored episodes and the final automatic reset.
        for task, draws in ((short, 1), (long, 117)):
            torch.rand(draws * (index + 1))
            before = torch.get_rng_state().clone()
            task._reset_idx(None)
            assert torch.equal(before, torch.get_rng_state()), "Reset changed policy RNG"
        torch.testing.assert_close(short.robot.data.joint_pos, long.robot.data.joint_pos, rtol=0, atol=0)
        starts.append(capture_physical_start(lambda name, default=None: getattr(short, name, default), 0))
    assert records == other
    assert starts[0]["robot_joint_pos"] != starts[1]["robot_joint_pos"]
    for index in range(2):
        trace = {
            "trace_episode_index": index,
            "initial_physical_state": starts[index],
            "terminated_at_end": True,
            "truncated_at_end": False,
            "metadata": {},
        }
        (tmp_path / f"motion_context_trace_episode_{index:03d}.json").write_text(json.dumps(trace))
    finalize_reset_scenarios(tmp_path, records, 42000, 2, True)
    manifest = json.loads((tmp_path / "evaluation_reset_scenarios.json").read_text())
    assert manifest["status"] == "complete" and manifest["scored_traces"] == 2
    assert len(manifest["reset_records"]) == 3  # Extra reset is not a third episode.
    with pytest.raises(ValueError, match="before all requested episodes"):
        finalize_reset_scenarios(tmp_path, records, 42000, 3, True)
    incomplete = json.loads((tmp_path / "evaluation_reset_scenarios.json").read_text())
    assert incomplete["status"] == "incomplete" and incomplete["scored_traces"] == 2
    finalize_reset_scenarios(tmp_path, records, 42000, 2, True)
    paths = sorted(tmp_path.glob("motion_context_trace_episode_*.json"))
    assert [json.loads(path.read_text())["metadata"]["evaluation_scenario_seed"] for path in paths] == [42000, 42001]
    changed = json.loads(paths[1].read_text())
    changed["initial_physical_state"]["robot_joint_pos"][0] += 0.1
    paths[1].write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="recorded reset state"):
        finalize_reset_scenarios(tmp_path, records, 42000, 2, True)
