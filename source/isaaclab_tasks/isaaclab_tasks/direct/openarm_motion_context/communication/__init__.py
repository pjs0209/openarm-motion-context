"""Deterministic motion-context communication descriptors."""

from .message_builder import pad_message
from .motion import signed_motion_intent
from .motion_context import (
    full_action_variation,
    motion_context_calibration_diagnostics,
    motion_context_from_raw,
    motion_context_sharing_active,
    relative_motion_trend,
    reset_motion_trend_state,
    update_motion_trend_context,
    update_motion_context_scale_and_calibration,
    update_variation_ema,
)

__all__ = [
    "full_action_variation",
    "motion_context_calibration_diagnostics",
    "motion_context_from_raw",
    "motion_context_sharing_active",
    "relative_motion_trend",
    "reset_motion_trend_state",
    "update_motion_trend_context",
    "pad_message",
    "signed_motion_intent",
    "update_motion_context_scale_and_calibration",
    "update_variation_ema",
]
