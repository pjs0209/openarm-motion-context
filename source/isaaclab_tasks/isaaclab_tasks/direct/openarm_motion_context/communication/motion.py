"""Signed short-horizon end-effector motion descriptor."""

import torch


def signed_motion_intent(
    ee_linear_velocity_base: torch.Tensor,
    control_dt: float,
    horizon: int,
    scale: float,
) -> torch.Tensor:
    """Return normalized signed EE displacement over a short horizon."""

    displacement = ee_linear_velocity_base * (float(control_dt) * float(horizon))
    return torch.clamp(displacement / max(float(scale), 1.0e-6), -1.0, 1.0)
