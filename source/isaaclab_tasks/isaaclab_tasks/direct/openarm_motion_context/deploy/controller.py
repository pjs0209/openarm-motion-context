"""Incremental arm and binary gripper command helpers."""

import torch


def incremental_joint_target(
    current_target: torch.Tensor,
    arm_action: torch.Tensor,
    default_position: torch.Tensor,
    step_scale: float,
    target_limit: float,
) -> torch.Tensor:
    """Apply one bounded incremental joint-position action."""

    return torch.clamp(
        current_target + float(step_scale) * arm_action,
        min=default_position - float(target_limit),
        max=default_position + float(target_limit),
    )


def update_gripper_closed(current: bool, command: torch.Tensor, epsilon: float) -> bool:
    """Apply the policy's persistent open/close switch semantics."""

    value = float(command)
    if value < -float(epsilon):
        return True
    if value > float(epsilon):
        return False
    return current
