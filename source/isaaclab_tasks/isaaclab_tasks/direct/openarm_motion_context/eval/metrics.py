"""Shared evaluation metadata for communication ablations."""

COMMUNICATION_MESSAGE_DIMS = {
    "none": 0,
    "motion_only": 3,
    "context_only": 3,
    "motion_context": 6,
    "previous_action": 8,
    "full_partner_observation": 30,
}
