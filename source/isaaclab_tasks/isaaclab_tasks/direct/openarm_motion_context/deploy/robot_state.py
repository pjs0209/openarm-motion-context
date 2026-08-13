"""Real-robot state containers used by deployment."""

from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class ArmState:
    q_target: torch.Tensor
    target_initialized: bool = False
    gripper_closed: bool = False
    prev_action: torch.Tensor | None = None
    prev_ee_pos: torch.Tensor | None = None
    prev_ee_quat: torch.Tensor | None = None
    ee_lin_vel: torch.Tensor | None = None
    ee_ang_speed: float = 0.0


@dataclass
class JointSnapshot:
    stamp: float
    msg: Any
