"""Fixed-width communication message construction helpers."""

import torch


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
