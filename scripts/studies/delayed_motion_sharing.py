"""Opt-in task-agnostic communication curriculum for unchanged Full6D Trend.

Train the local controller with a zero partner slot, then smoothly reveal the
same hand-crafted 6D message. No reward, actor size, physics or sensor change.
The matched NoShare policy receives the same trainer/config but always zero.
Only the named runtime trainer symbol triggers the Isaac Lab import.
"""


def share_gain(timestep, warmup, ramp, *, evaluation=False):
    if (
        type(timestep) is not int
        or timestep < 0
        or type(warmup) is not int
        or warmup < 0
        or type(ramp) is not int
        or ramp < 1
        or type(evaluation) is not bool
    ):
        raise ValueError("Nonnegative integer timestep/warmup and positive ramp required")
    if evaluation:
        return 1.0
    return max(0.0, min(1.0, (timestep - warmup) / ramp))


def make_delayed_trainer_class(base):
    class DelayedMotionSharingTrainer(base):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            warmup = self.cfg.get("motion_share_warmup_timesteps")
            ramp = self.cfg.get("motion_share_ramp_timesteps")
            if type(warmup) is not int or warmup < 0 or type(ramp) is not int or ramp < 1:
                raise ValueError("Explicit nonnegative communication warmup and positive ramp required")
            self.intent_share_warmup_timesteps = warmup
            self._motion_share_ramp_timesteps = ramp

        def _apply_intent_share_strategy(self, shared_intents, outputs, timestep, sharing_enabled, warmup_active):
            result = super()._apply_intent_share_strategy(
                shared_intents, outputs, timestep, sharing_enabled, warmup_active
            )
            if not sharing_enabled:
                return result
            gain = share_gain(
                timestep,
                self.intent_share_warmup_timesteps,
                self._motion_share_ramp_timesteps,
                evaluation=bool(self._in_eval),
            )
            return {agent: intent * gain for agent, intent in result.items()}

    return DelayedMotionSharingTrainer


def __getattr__(name):
    if name != "DelayedMotionSharingTrainer":
        raise AttributeError(name)
    from isaaclab_tasks.direct.openarm_motion_context.train.trainer import MotionContextTrainer

    return make_delayed_trainer_class(MotionContextTrainer)
