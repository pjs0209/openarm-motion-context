# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Read-only control-state records and guards for diagnostic evaluation."""

import json
from pathlib import Path
from types import MethodType

import torch

from ..legacy_api import read_data


class ControlTimingAudit:
    """Observe action handoff and reject temporal mutations caused by trace reads."""

    def __init__(self, task, directory: Path):
        self.task = task
        self.stream = (Path(directory) / "control_timing.jsonl").open("x")
        self.observation_stream = (Path(directory) / "actor_observations.jsonl").open("x")
        self.original_pre = task._pre_physics_step
        self.original_trace = task.get_paper_motion_context_trace_signals
        self.original_observations = task._get_observations

        def trace(env, index=0):
            before = self.temporal_state()
            result = self.original_trace(index)
            after = self.temporal_state()
            changed = [
                key
                for key in before.keys() | after.keys()
                if key not in before or key not in after or not torch.equal(before[key], after[key])
            ]
            if changed:
                raise RuntimeError(f"Trace collector modified motion context state: {changed}")
            return result

        def pre(env, actions):
            before = self.control_state()
            result = self.original_pre(actions)
            record = {
                "common_step": int(task.common_step_counter),
                "episode_control_step": int(task.episode_length_buf[0]),
                "control_dt": float(task.cfg.sim.dt * task.cfg.decimation),
                "before_action": before,
                "after_action": self.control_state(),
                "context_cache_step": int(getattr(task, "_motion_context_cache_step", -1)),
                "history_index": int(getattr(task, "_motion_context_trend_history_index", -1)),
                "outgoing_messages": {
                    key: value[0].detach().cpu().flatten().tolist() for key, value in task.communication_buffer.items()
                },
            }
            self.stream.write(json.dumps(record, allow_nan=False) + "\n")
            return result

        def observations(env):
            result = self.original_observations()
            own = task.own_observation_dim
            for side, sender in (("left", "right"), ("right", "left")):
                received = result[f"{side}_arm"][:, own:]
                expected = task.communication_buffer[f"{sender}_intent"].reshape(task.num_envs, -1)
                if not torch.equal(received, expected):
                    raise RuntimeError("Actor's raw partner slot differs from the handed-off message")
            self.observation_stream.write(
                json.dumps(
                    {
                        "common_step": int(task.common_step_counter),
                        "episode_control_step": int(task.episode_length_buf[0]),
                        "boundary": "task raw actor observation before diagnostic masking or normalization",
                        "actor_observations": {key: value[0].detach().cpu().tolist() for key, value in result.items()},
                    },
                    allow_nan=False,
                )
                + "\n"
            )
            return result

        task._pre_physics_step = MethodType(pre, task)
        task.get_paper_motion_context_trace_signals = MethodType(trace, task)
        task._get_observations = MethodType(observations, task)

    def temporal_state(self):
        return {
            name: value.detach().clone() if isinstance(value, torch.Tensor) else torch.as_tensor(value)
            for name, value in vars(self.task).items()
            if name.startswith("_motion_context_") and isinstance(value, (torch.Tensor, int, float, bool))
        }

    def control_state(self):
        task = self.task
        joints = read_data(task.robot.data, "joint_pos")
        result = {}
        for side in ("left", "right"):
            ids = getattr(task, f"{side}_arm_joint_ids")
            result[side] = {
                "joint_pos": joints[0, ids].detach().cpu().tolist(),
                "arm_target": getattr(task, f"_{side}_arm_incremental_target")[0].detach().cpu().tolist(),
                "gripper_position_command": getattr(task, f"_{side}_gripper_position_command")[0]
                .detach()
                .cpu()
                .tolist(),
                "gripper_closed": bool(getattr(task, f"_{side}_gripper_closed")[0, 0]),
            }
        return result

    def close(self):
        self.stream.close()
        self.observation_stream.close()
        self.task._pre_physics_step = self.original_pre
        self.task.get_paper_motion_context_trace_signals = self.original_trace
        self.task._get_observations = self.original_observations
