"""Task-agnostic proprioceptive motion-context descriptor."""

import torch


def motion_context_from_raw(
    linear_speed: torch.Tensor,
    angular_speed: torch.Tensor,
    action_change: torch.Tensor,
    running_scale: torch.Tensor,
    norm_max: float = 1.5,
) -> torch.Tensor:
    """Build [linear activity, angular activity, action smoothness]."""

    scale = running_scale.clamp_min(1.0e-6)
    maximum = max(float(norm_max), 1.0e-6)
    linear_normalized = torch.clamp(linear_speed / scale[0], 0.0, maximum)
    angular_normalized = torch.clamp(angular_speed / scale[1], 0.0, maximum)
    action_normalized = torch.clamp(action_change / scale[2], 0.0, maximum)
    context = torch.cat(
        [
            1.0 - torch.exp(-linear_normalized),
            1.0 - torch.exp(-angular_normalized),
            torch.exp(-action_normalized),
        ],
        dim=-1,
    )
    return torch.clamp(torch.nan_to_num(context, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
