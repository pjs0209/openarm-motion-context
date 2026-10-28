"""Uninstalled read-only evaluation tracing of actual pre-normalizer own inputs.

Existing traces already store the matching raw peer slot, but omit own inputs.
Install only in a fresh evaluator process, before constructing its trainer.
No observation reconstruction, privileged critic data or algorithm changes.
"""

import inspect

import torch

MARKER = "_raw_actor_state_trace_version"


def own_rows(states, own, slot, env_index):
    if (
        type(own) is not int
        or own not in (30, 34)
        or type(slot) is not int
        or slot not in (0, own)
        or type(env_index) is not int
        or env_index < 0
        or not isinstance(states, dict)
        or set(states) != {"left_arm", "right_arm"}
    ):
        raise ValueError("Exact own/peer layout and two actual actor states required")
    result = {}
    batch = None
    for side, uid in (("left", "left_arm"), ("right", "right_arm")):
        state = states[uid]
        if (
            not isinstance(state, torch.Tensor)
            or not state.is_floating_point()
            or state.ndim != 2
            or state.shape[1] != own + slot
            or env_index >= state.shape[0]
            or not torch.isfinite(state[env_index]).all()
            or batch is not None
            and state.shape[0] != batch
        ):
            raise ValueError("Finite exact-width actor rows with matching batches required")
        batch = state.shape[0]
        result[side] = state[env_index, :own].detach().cpu().tolist()
    return result


def install_trainer_class(cls):
    if hasattr(cls, MARKER) or cls._append_mode_trace.__qualname__ != "MotionContextTrainer._append_mode_trace":
        raise ValueError("Fresh original MotionContextTrainer class required")
    original_new, original_append = cls._new_mode_trace, cls._append_mode_trace
    parameters = tuple(inspect.signature(original_append).parameters)
    if parameters != ("self", "outputs", "rewards", "infos", "trace_signals", "policy_states"):
        raise ValueError("Known actual policy_states trace interface required")

    def new_trace():
        trace = original_new()
        for name in ("left_policy_own_observation", "right_policy_own_observation", "actor_state_trace_version"):
            if name in trace:
                raise ValueError("Own actor trace already supplied by original trainer")
            trace[name] = []
        return trace

    def append(self, outputs, rewards, infos, trace_signals=None, policy_states=None):
        active = (
            self.save_mode_trace
            and self._in_eval
            and (self.num_trace_episodes <= 0 or self._mode_trace_episode_index < self.num_trace_episodes)
        )
        before = len(self._mode_trace["left_motion_context"])
        rows = None
        if active:
            own = int(self._resolve_env_attr("own_observation_dim", 30))
            slot = int(self._resolve_env_attr("communication_feature_dim", 0) or 0)
            index = int(max(0, min(self.mode_trace_env_index, self.env.num_envs - 1)))
            rows = own_rows(policy_states, own, slot, index)
            if any(
                name not in self._mode_trace or len(self._mode_trace[name]) != before
                for name in ("left_policy_own_observation", "right_policy_own_observation", "actor_state_trace_version")
            ):
                raise ValueError("Install only before trainer construction; no partial own trace")
        result = original_append(self, outputs, rewards, infos, trace_signals, policy_states)
        after = len(self._mode_trace["left_motion_context"])
        if after == before + 1 and rows is not None:
            for side in ("left", "right"):
                self._mode_trace[side + "_policy_own_observation"].append(rows[side])
            self._mode_trace["actor_state_trace_version"].append(1)
        elif after != before:
            raise ValueError("Unexpected original trace append cadence")
        return result

    setattr(cls, MARKER, 1)
    cls._new_mode_trace = staticmethod(new_trace)
    cls._append_mode_trace = append
