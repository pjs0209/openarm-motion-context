"""Fixed-width communication message construction helpers."""

import torch

MOTION_CONTEXT_MODES = ("motion_context", "motion_context_la", "motion_context_s")


def motion_context_payload(motion: torch.Tensor, context: torch.Tensor, mode: str) -> torch.Tensor:
    """Build a 6D payload while keeping ablated context channels at fixed positions."""

    if mode not in MOTION_CONTEXT_MODES:
        raise ValueError(f"Unsupported motion-context payload mode: {mode!r}")
    if motion.shape[-1] != 3 or context.shape[-1] != 3:
        raise ValueError("Motion and context must each have three channels.")
    if mode == "motion_context_la":
        context = torch.cat((context[..., :2], torch.zeros_like(context[..., 2:3])), dim=-1)
    elif mode == "motion_context_s":
        context = torch.cat((torch.zeros_like(context[..., :2]), context[..., 2:3]), dim=-1)
    return torch.cat((motion, context), dim=-1)


def pad_message(payload: torch.Tensor, width: int, mode: str) -> torch.Tensor:
    """Place a compact payload in the fixed actor communication slot."""

    if payload.shape[-1] > width:
        raise RuntimeError(
            f"communication_mode={mode!r} payload width {payload.shape[-1]} exceeds slot width {width}."
        )
    slot = torch.zeros((*payload.shape[:-1], width), device=payload.device, dtype=payload.dtype)
    if payload.shape[-1] > 0:
        slot[..., : payload.shape[-1]] = payload
    return torch.nan_to_num(slot, nan=0.0, posinf=0.0, neginf=0.0)
