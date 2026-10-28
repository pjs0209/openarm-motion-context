"""Explicit unclipped value regression for the project's real custom updater.

Unlike native MAPPO, MotionContextMAPPO already reads per-agent dict flags.
Do NOT replace that dict with a bool. This is a config-only value-loss ablation
with fresh critic identity buffers, not a repair of the native flag issue.
"""

import hashlib
import inspect

import torch
from scripts.studies import bounded_all_control as base

MARKER = "_project_unclipped_value_regression_version"


def _validate(module, state, prefix, *args):
    value = state.get(prefix + MARKER)
    if (
        value is None
        or value.numel() != 1
        or not torch.isfinite(value).all()
        or not torch.equal(value, value.new_tensor(1.0))
    ):
        raise ValueError("Project unclipped-critic identity mismatch")


def install(agent):
    update = getattr(agent, "_full6d_original_update", agent._update)
    function = getattr(update, "__func__", None)
    code = inspect.getsource(function) if function is not None else ""
    if (
        agent.cfg.get("clip_predicted_values") is not False
        or not isinstance(agent._clip_predicted_values, dict)
        or set(agent._clip_predicted_values) != set(agent.possible_agents)
        or any(value is not False for value in agent._clip_predicted_values.values())
        or getattr(update, "__self__", None) is not agent
        or getattr(function, "__qualname__", "") != "MotionContextMAPPO._update"
        or not any(line.strip() == "if self._clip_predicted_values[uid]:" for line in code.splitlines())
    ):
        raise ValueError("Requires saved false and actual project per-agent updater")
    for modules in agent.models.values():
        if hasattr(modules["value"], MARKER):
            raise ValueError("Already installed")
    for modules in agent.models.values():
        value = modules["value"]
        value.register_buffer(MARKER, next(value.parameters()).new_tensor(1.0))
        value.register_load_state_dict_pre_hook(_validate)
    # Preserve the correct per-agent dictionary and all update wrappers.
    return {
        "kind": "project_unclipped_value_regression_v1",
        "effective_project_update_sha256": hashlib.sha256(code.encode()).hexdigest(),
        "limits": "Actual project updater already honors dict flags. This disables default value clipping, not a native-bug repair or proven policy improvement.",
    }


def build_project_unclipped_value_bounded_all_mappo_agent(env, experiment_cfg):
    agent = base.build_bounded_all_control_mappo_agent(env, experiment_cfg)
    install(agent)
    return agent


def build_fresh_camera_project_unclipped_value_bounded_all_mappo_agent(env, experiment_cfg):
    agent = base.build_fresh_camera_bounded_all_control_mappo_agent(env, experiment_cfg)
    install(agent)
    return agent
