"""Whole-message dropout at the actual environment communication handoff.

The environment recomputes its proprioceptive descriptor inside
set_pending_intents and ignores predicted_intents. Masking trainer outputs is
therefore ineffective. This trainer masks the generated communication_buffer
after that hook and before env.step constructs the next actor observation.
"""

from __future__ import annotations

import torch

from .full6d_message_dropout import drop_partner_messages
from .trainer import MotionContextTrainer


class Full6DBufferDropoutTrainer(MotionContextTrainer):
    """Apply training-only erasure to messages actually consumed by actors."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.message_dropout_probability = float(self.cfg.get("message_dropout_probability", 0.5))
        if not 0.0 <= self.message_dropout_probability <= 1.0:
            raise ValueError("message_dropout_probability must be in [0, 1]")
        self.message_dropout_seed = int(self.cfg.get("message_dropout_seed", 17041))
        self._buffer_dropout_generator = None
        self._buffer_dropout_rows = 0
        self._buffer_dropout_erased_rows = 0
        self._buffer_dropout_handoffs = 0
        self._original_message_handoff = self.env.set_pending_intents
        self.env.set_pending_intents = self._handoff_with_buffer_dropout
        print(
            f"[INFO] actual_buffer_message_dropout_probability={self.message_dropout_probability} "
            f"seed={self.message_dropout_seed}"
        )

    def _handoff_with_buffer_dropout(self, *args, **kwargs):
        result = self._original_message_handoff(*args, **kwargs)
        if self._in_eval or self.message_dropout_probability == 0.0:
            return result
        if not bool(self._resolve_env_attr("communication_enabled", True)):
            return result
        buffers = self._resolve_env_attr("communication_buffer", None)
        if not isinstance(buffers, dict) or set(buffers) != {"left_intent", "right_intent"}:
            raise RuntimeError("Buffer dropout requires the actual left/right communication_buffer")
        first = buffers["left_intent"]
        if self._buffer_dropout_generator is None:
            self._buffer_dropout_generator = torch.Generator(device=first.device)
            self._buffer_dropout_generator.manual_seed(self.message_dropout_seed)
        masked = drop_partner_messages(buffers, self.message_dropout_probability, self._buffer_dropout_generator)
        for key, tensor in masked.items():
            buffers[key].copy_(tensor)
        self._buffer_dropout_handoffs += 1
        # Sample actual buffer rows periodically to verify the live path without
        # synchronizing CUDA every control step. Startup descriptors can be zero
        # naturally; this diagnostic is not an exact Bernoulli-mask counter.
        if self._buffer_dropout_handoffs % 500 == 0:
            rows = sum(tensor.shape[0] for tensor in buffers.values())
            erased = sum(
                int((tensor.flatten(start_dim=1).abs().sum(dim=1) == 0).sum().item()) for tensor in buffers.values()
            )
            self._buffer_dropout_rows += rows
            self._buffer_dropout_erased_rows += erased
            print(
                f"[BUFFER_DROPOUT] handoffs={self._buffer_dropout_handoffs} "
                f"actual_zero_rows={erased}/{rows} "
                f"sampled_zero_fraction={self._buffer_dropout_erased_rows / self._buffer_dropout_rows:.3f}"
            )
        return result
