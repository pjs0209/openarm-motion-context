"""Disabled legacy seeded output-dropout trainer.

Its masks never reached the actual environment communication buffer.
Use Full6DBufferDropoutTrainer for new training experiments.
"""

from __future__ import annotations

from .trainer import MotionContextTrainer


class Full6DMessageDropoutSeededTrainer(MotionContextTrainer):
    """Deprecated: masking predicted_intents never reached actor observations."""

    def __init__(self, *args, **kwargs):
        raise RuntimeError(
            "This trainer masked an ignored predicted_intents argument. "
            "Use full6d_buffer_dropout:Full6DBufferDropoutTrainer for training "
            "or the standard MotionContextTrainer to evaluate old checkpoints."
        )
