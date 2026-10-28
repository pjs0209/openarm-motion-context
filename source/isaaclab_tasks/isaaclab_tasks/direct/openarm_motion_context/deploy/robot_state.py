"""Real-robot state containers used by deployment."""

from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass
class ArmState:
    q_target: torch.Tensor
    target_initialized: bool = False
    gripper_closed: bool = False
    gripper_position_command: float | None = None
    prev_action: torch.Tensor | None = None
    prev_ee_pos: torch.Tensor | None = None
    prev_ee_quat: torch.Tensor | None = None
    ee_lin_vel: torch.Tensor | None = None
    ee_ang_speed: float = 0.0
    motion_stamp: float | None = None
    action_variation_ema: torch.Tensor | None = None
    action_variation_valid: bool = False
    motion_signal_ema: torch.Tensor | None = None
    motion_signal_history: list[torch.Tensor] = field(default_factory=list)


@dataclass
class JointSnapshot:
    stamp: float
    msg: Any
