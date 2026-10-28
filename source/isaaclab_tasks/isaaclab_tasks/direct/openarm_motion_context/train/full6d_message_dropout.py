"""Message-erasure helper and a disabled legacy output-dropout trainer.

The environment ignores predicted_intents, so the legacy trainer did not
mask messages actually consumed by actors. Use Full6DBufferDropoutTrainer.
The pure helper below remains shared by the corrected buffer implementation.
"""

from __future__ import annotations

import torch

from .trainer import MotionContextTrainer


def drop_partner_messages(
    messages: dict[str, torch.Tensor],
    probability: float,
    generator: torch.Generator,
) -> dict[str, torch.Tensor]:
    """Mask complete per-environment messages without changing their shapes."""

    if not 0.0 <= probability <= 1.0:
        raise ValueError(f"message dropout probability must be in [0, 1], got {probability}")
    if probability == 0.0:
        return messages
    masked = {}
    for agent_id, message in messages.items():
        keep_shape = (message.shape[0],) + (1,) * (message.ndim - 1)
        keep = torch.rand(keep_shape, device=message.device, generator=generator) >= probability
        masked[agent_id] = message * keep.to(dtype=message.dtype)
    return masked


class Full6DMessageDropoutTrainer(MotionContextTrainer):
    """Deprecated: masking predicted_intents never reached actor observations."""

    def __init__(self, *args, **kwargs):
        raise RuntimeError(
            "This trainer masked an ignored predicted_intents argument. "
            "Use full6d_buffer_dropout:Full6DBufferDropoutTrainer for training "
            "or the standard MotionContextTrainer to evaluate old checkpoints."
        )
