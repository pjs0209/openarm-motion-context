"""Opt-in Lift H5 curriculum applied to the *received* hand-crafted message.

LiftEnv.set_pending_intents intentionally ignores the trainer's predicted_intents
and recomputes Motion Context from the just-applied action. Therefore scaling
predicted_intents (the old delayed-sharing candidate) has no effect. This
trainer scales the recomputed six message channels in the task buffer after
that method returns. Own proprioception, reward, physics, actor and raw
Gaussian scoring remain unchanged. NoShare remains a zero-message control.
"""

from __future__ import annotations

from types import MethodType

import torch

from scripts.studies.delayed_motion_sharing import share_gain


def find_task(env):
    queue, visited = [env], set()
    while queue:
        current = queue.pop(0)
        if current is None or id(current) in visited:
            continue
        visited.add(id(current))
        fields = vars(current) if hasattr(current, "__dict__") else {}
        if "communication_buffer" in fields and "set_pending_intents" in dir(current):
            return current
        for name in ("_env", "_unwrapped", "env", "unwrapped"):
            try:
                child = getattr(current, name, None)
            except Exception:
                child = None
            if child is not None and child is not current:
                queue.append(child)
    raise ValueError("Motion Context task with owned communication buffer required")


def make_effective_delayed_trainer_class(base):
    class EffectiveDelayedMotionSharingTrainer(base):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            warmup = self.cfg.get("motion_share_warmup_timesteps")
            ramp = self.cfg.get("motion_share_ramp_timesteps")
            if type(warmup) is not int or warmup < 0 or type(ramp) is not int or ramp < 1:
                raise ValueError("Explicit nonnegative communication warmup and positive ramp required")
            self.intent_share_warmup_timesteps = warmup
            self._motion_share_ramp_timesteps = ramp
            self._effective_motion_share_gain = 1.0
            self._effective_motion_share_probe_seen = set()
            self._effective_motion_share_calls = 0
            task = find_task(self.env)
            if getattr(task, "_effective_delayed_share_installed", False):
                raise ValueError("Effective delayed sharing already installed")
            original = task.set_pending_intents

            def apply_then_scale(task_self, predicted_intents, policy_states, current_actions=None):
                original(predicted_intents, policy_states, current_actions)
                self._effective_motion_share_calls += 1
                gain = self._effective_motion_share_gain
                stage = "warmup" if gain == 0 else "full" if gain == 1 else "ramp"
                raw_norm = None
                if stage not in self._effective_motion_share_probe_seen:
                    raw_norm = sum(
                        float(task_self.communication_buffer[key].abs().sum())
                        for key in ("left_intent", "right_intent")
                    )
                if self._effective_motion_share_calls <= 3:
                    print(
                        f"[EFFECTIVE_SHARE_CALL] n={self._effective_motion_share_calls} "
                        f"stage={stage} raw_l1={raw_norm}",
                        flush=True,
                    )
                for key in ("left_intent", "right_intent"):
                    message = task_self.communication_buffer[key]
                    if not isinstance(message, torch.Tensor) or message.ndim != 3 or message.shape[-1] < 6:
                        raise ValueError("Expected recomputed 6D Motion Trend buffer")
                    if message.shape[-1] > 6 and torch.any(message[..., 6:] != 0):
                        raise ValueError("Expected zero padding after 6D Motion Trend payload")
                    message.mul_(gain)
                task_self._effective_delayed_share_gain = gain
                if raw_norm is not None and raw_norm > 1e-6:
                    print(
                        f"[EFFECTIVE_SHARE] {stage} gain={gain:.6f} "
                        f"raw_l1={raw_norm:.6f} received_l1={raw_norm * gain:.6f}",
                        flush=True,
                    )
                    self._effective_motion_share_probe_seen.add(stage)

            scheduled_handoff = MethodType(apply_then_scale, task)
            task.set_pending_intents = scheduled_handoff
            # train_openarm_bimanual_direct.py copies the bound handoff onto
            # the skrl wrapper before constructing the trainer. Rebind both.
            self.env.set_pending_intents = scheduled_handoff
            task._effective_delayed_share_installed = True

        def _apply_intent_share_strategy(self, shared_intents, outputs, timestep, sharing_enabled, warmup_active):
            result = super()._apply_intent_share_strategy(
                shared_intents, outputs, timestep, sharing_enabled, warmup_active
            )
            self._effective_motion_share_gain = (
                share_gain(
                    timestep,
                    self.intent_share_warmup_timesteps,
                    self._motion_share_ramp_timesteps,
                    evaluation=bool(self._in_eval),
                )
                if sharing_enabled
                else 0.0
            )
            return result

    return EffectiveDelayedMotionSharingTrainer


def __getattr__(name):
    if name != "EffectiveDelayedMotionSharingTrainer":
        raise AttributeError(name)
    from isaaclab_tasks.direct.openarm_motion_context.train.trainer import MotionContextTrainer

    return make_effective_delayed_trainer_class(MotionContextTrainer)
