"""Opt-in trace/asset instrumentation; model and reward builders stay explicit."""

import hashlib
import json
from pathlib import Path

from isaaclab_tasks.direct.openarm_motion_context.train.trainer import MotionContextTrainer
from scripts.studies.paper_motion_context_v2_contract import NAME, validate_physical_state
from isaaclab_tasks.direct.openarm_motion_context.legacy_api import (
    read_data,
    write_root_pose,
    set_joint_position,
    tensor,
)


def asset_fingerprints(stage):
    result = {}
    for layer in stage.GetUsedLayers():
        if layer.anonymous:
            continue
        identifier = str(layer.identifier)
        path = Path(layer.realPath)
        local = path.is_file()
        content = path.read_bytes() if local else layer.ExportToString().encode("utf-8")
        result[identifier] = {
            "sha256": hashlib.sha256(content).hexdigest(),
            "encoding": "source_bytes" if local else "loaded_layer_usda",
        }
    if not result:
        raise RuntimeError("No loaded asset layers were fingerprinted")
    return result


class PaperMotionContextTrainer(MotionContextTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        import omni.usd

        self._paper_assets = asset_fingerprints(omni.usd.get_context().get_stage())
        log_dir = self._resolve_env_attr("cfg").log_dir
        if log_dir:
            path = Path(log_dir) / "params/paper_runtime_assets.json"
            with path.open("x") as stream:
                json.dump(self._paper_assets, stream, indent=2)
        print(
            f"[PAPER_MOTION_CONTEXT_V2] loaded_asset_layers={len(self._paper_assets)}; native matched builders",
            flush=True,
        )

    def _resolve_env_attr(self, name, default=None):
        resolved = super()._resolve_env_attr(name, default)
        if name != "get_paper_motion_context_trace_signals" or not callable(resolved):
            return resolved

        def getter(index=0):
            signals = dict(resolved(index))
            if not (
                getattr(self, "save_mode_trace", False)
                and getattr(self, "_in_eval", False)
                and not self._mode_trace["left_motion_context"]
            ):
                return signals
            resolve = super(PaperMotionContextTrainer, self)._resolve_env_attr
            robot, obj, ball = resolve("robot"), resolve("object"), resolve("ball")
            tensors = {
                "robot_root_state_w": read_data(robot.data, "root_state_w"),
                "robot_joint_pos": read_data(robot.data, "joint_pos"),
                "robot_joint_vel": read_data(robot.data, "joint_vel"),
                "object_root_state_w": read_data(obj.data, "root_state_w"),
            }
            if ball is not None:
                tensors["ball_root_state_w"] = read_data(ball.data, "root_state_w")
            state = {key: value[index].detach().float().cpu().tolist() for key, value in tensors.items()}
            validate_physical_state(state, "ball" if ball is not None else "lift")
            signals["paper_initial_physical_state"] = state
            return signals

        return getter

    def _append_mode_trace(self, outputs, rewards, infos, trace_signals=None, policy_states=None):
        before = len(self._mode_trace["left_motion_context"])
        super()._append_mode_trace(outputs, rewards, infos, trace_signals, policy_states)
        if before == 0 and self._mode_trace["left_motion_context"]:
            if trace_signals is None or "paper_initial_physical_state" not in trace_signals:
                raise RuntimeError("First transition has no recorded pre-action physical state")
            self._mode_trace["initial_physical_state"] = trace_signals["paper_initial_physical_state"]
            self._mode_trace["paper_protocol"] = NAME
            self._mode_trace["paper_runtime_assets"] = self._paper_assets
            self._mode_trace["paper_success_goal"] = self._resolve_env_attr("_ball_goal_variant", "native_lift")
