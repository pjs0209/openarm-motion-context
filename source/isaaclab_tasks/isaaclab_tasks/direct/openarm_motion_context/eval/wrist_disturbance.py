# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Opt-in single-world Newton wrist pulses, with independently seeded task resets.

Forces are applied through the asset wrench composer at the selected link's
center of mass, in world coordinates. The diagnostic records the actual Newton
input buffer after asset submission. These are external solver inputs, not
contact forces, measured net forces, or proof of realistic hardware response.
"""

import json
import math
from pathlib import Path
from types import MethodType

import torch

from ..legacy_api import read_data, tensor

PROTOCOL = "openarm_wrist_pulse/v1"


def pulse_schedule(config: dict, physics_dt: float) -> dict:
    """Validate and discretize a deterministic, episode-indexed pulse protocol.

    Onset and duration are rounded upwards to whole physics steps. Scheduling
    consumes no random draws; changing policy rollout length cannot change the
    side or direction assigned to a scored episode. Zero N is the matched
    instrumented control, with the same nominal pulse window.
    """
    required = {"onset_s", "duration_s", "force_n", "events"}
    if set(config) != required:
        raise ValueError(f"Wrist pulse requires exactly {sorted(required)}")
    values = [config[name] for name in ("onset_s", "duration_s", "force_n")]
    if not all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in values):
        raise ValueError("Pulse times and force must be finite numbers")
    onset, duration, force = values
    if onset < 0 or duration <= 0 or force < 0 or not math.isfinite(physics_dt) or physics_dt <= 0:
        raise ValueError("Require onset >= 0, duration > 0, force >= 0 and physics_dt > 0")
    events = config["events"]
    if not isinstance(events, list) or not events:
        raise ValueError("At least one side/direction event is required")
    validated = []
    for event in events:
        if (
            not isinstance(event, dict)
            or set(event) != {"side", "direction_w"}
            or event["side"] not in ("left", "right")
        ):
            raise ValueError("Events require side left/right and direction_w")
        direction = event["direction_w"]
        if (
            not isinstance(direction, list)
            or len(direction) != 3
            or not all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in direction)
        ):
            raise ValueError("direction_w must be a finite three-vector")
        norm = math.sqrt(sum(v * v for v in direction))
        if abs(norm - 1) > 1e-6:
            raise ValueError("direction_w must be a unit vector")
        validated.append({"side": event["side"], "force_w": [v * force for v in direction]})
    onset_ticks = math.ceil(onset / physics_dt - 1e-10)
    duration_ticks = max(1, math.ceil(duration / physics_dt - 1e-10))
    return {
        "protocol": PROTOCOL,
        "physics_dt": physics_dt,
        "onset_tick": onset_ticks,
        "end_tick": onset_ticks + duration_ticks,
        "effective_onset_s": onset_ticks * physics_dt,
        "effective_duration_s": duration_ticks * physics_dt,
        "events": validated,
        "requested": config,
    }


class WristDisturbance:
    """Own opt-in evaluation hooks, per-substep force readback and episode records."""

    def __init__(self, task, config: dict, directory: Path):
        from isaaclab_newton.physics import NewtonCfg

        if task.num_envs != 1 or not isinstance(task.cfg.sim.physics, NewtonCfg):
            raise ValueError("Wrist diagnostics currently require one Newton evaluation world")
        if getattr(task, "training", False):
            raise ValueError("Install wrist diagnostics only for evaluation")
        if hasattr(task, "_wrist_disturbance"):
            raise ValueError("Wrist disturbance is already installed")
        self.task = task
        self.schedule = pulse_schedule(config, float(task.cfg.sim.dt))
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.path = self.directory / "wrist_disturbance.json"
        if self.path.exists():
            raise FileExistsError(self.path)
        self.body_ids = []
        self.body_names = []
        # TCP helper links may be massless/fixed. Force the physical wrist links.
        for side in ("left", "right"):
            name = f"openarm_{side}_link7"
            ids, names = task.robot.find_bodies(f"^{name}$")
            if len(ids) != 1:
                raise ValueError(f"Expected exactly one wrist body {name}, got {names}")
            self.body_ids.append(ids[0])
            self.body_names.append(name)
        if task.robot.data.has_body_ordering:
            mapping = tensor(task.robot._body_user_to_backend_map()).cpu().tolist()
            self.backend_ids = [mapping[i] for i in self.body_ids]
        else:
            self.backend_ids = self.body_ids
        self.forces = torch.zeros((1, 2, 3), device=task.device)
        self.tick, self.pending, self.records = 0, False, []
        self.episode = -1
        self._stream = (self.directory / "wrist_disturbance_substeps.jsonl").open("x")
        self._original_apply = task._apply_action
        self._original_reset = task._reset_idx
        self._original_write = task.robot.write_data_to_sim

        def apply(env):
            self._original_apply()
            self.submit()

        def reset(env, ids):
            if ids is not None and len(ids) == 0:
                return self._original_reset(ids)
            result = self._original_reset(ids)
            self.begin_episode()
            return result

        def write(robot):
            result = self._original_write()
            if self.pending:
                self.record_submission()
            return result

        task._apply_action = MethodType(apply, task)
        task._reset_idx = MethodType(reset, task)
        task.robot.write_data_to_sim = MethodType(write, task.robot)
        task._wrist_disturbance = self
        self._write_summary("running")

    def begin_episode(self):
        if self.pending:
            raise RuntimeError("Reset occurred before a submitted force was recorded")
        self.tick = 0
        self.episode += 1
        event = self.schedule["events"][self.episode % len(self.schedule["events"])]
        self.records.append(
            {
                "episode_index": self.episode,
                "event": event,
                "physics_ticks": 0,
                "window_ticks": 0,
                "nonzero_force_ticks": 0,
                "input_impulse_w_ns": [0.0, 0.0, 0.0],
                "max_input_error_n": 0.0,
            }
        )

    def submit(self):
        if self.episode < 0 or self.pending:
            raise RuntimeError("Pulse requires an episode reset and one asset submission per physics step")
        self.forces.zero_()
        event = self.records[-1]["event"]
        active = self.schedule["onset_tick"] <= self.tick < self.schedule["end_tick"]
        if active:
            self.forces[0, 0 if event["side"] == "left" else 1] = self.forces.new_tensor(event["force_w"])
        # Submit zero outside the window as well, so release is explicit at the
        # backend boundary and a stale external force cannot persist unnoticed.
        self.task.robot.instantaneous_wrench_composer.add_forces_and_torques_index(
            forces=self.forces, body_ids=self.body_ids, is_global=True
        )
        self.pending = True

    def record_submission(self):
        robot = self.task.robot
        actual = tensor(robot.data._sim_bind_body_external_wrench)[0, self.backend_ids].clone()
        expected = self.forces[0]
        error = (actual[:, :3] - expected).abs().max().item()
        if not torch.isfinite(actual).all() or error > 1e-4 or actual[:, 3:].abs().max().item() > 1e-4:
            raise RuntimeError(f"Newton wrist input differs from requested world-frame COM force: {actual}, {expected}")
        point = read_data(robot.data, "body_com_pos_w")[0, self.body_ids].cpu().tolist()
        data = {
            "episode_index": self.episode,
            "physics_tick": self.tick,
            "time_s": self.tick * self.schedule["physics_dt"],
            "body_names": self.body_names,
            "application_point_w_m": point,
            "submitted_force_w_n": expected.cpu().tolist(),
            "backend_input_wrench_w": actual.cpu().tolist(),
        }
        self._stream.write(json.dumps(data, allow_nan=False) + "\n")
        record = self.records[-1]
        record["physics_ticks"] += 1
        record["window_ticks"] += int(self.schedule["onset_tick"] <= self.tick < self.schedule["end_tick"])
        record["nonzero_force_ticks"] += int(expected.abs().max().item() > 0)
        impulse = actual[:, :3].sum(0).cpu().tolist()
        record["input_impulse_w_ns"] = [
            old + value * self.schedule["physics_dt"] for old, value in zip(record["input_impulse_w_ns"], impulse)
        ]
        record["max_input_error_n"] = max(record["max_input_error_n"], error)
        self.tick += 1
        self.pending = False

    def _write_summary(self, status):
        self.path.write_text(
            json.dumps(
                {
                    "status": status,
                    "schedule": self.schedule,
                    "application": "world-frame force at selected wrist link center of mass; zero external torque",
                    "input_readback": "Newton body_f after asset submission, before solver integration; force XYZ then torque XYZ",
                    "solver_num_substeps": self.task.cfg.sim.physics.num_substeps,
                    "episodes": self.records,
                },
                indent=2,
                allow_nan=False,
            )
            + "\n"
        )

    def close(self, completed: bool):
        self._stream.close()
        self._write_summary("complete" if completed and not self.pending else "incomplete")
        self.task._apply_action = self._original_apply
        self.task._reset_idx = self._original_reset
        self.task.robot.write_data_to_sim = self._original_write
