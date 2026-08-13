"""Motion-context checkpoint sidecar naming helpers."""

import os


def motion_context_sidecar_path(checkpoint_path: str) -> str:
    """Return the normalization-state sidecar path for a policy checkpoint."""

    root, extension = os.path.splitext(checkpoint_path)
    return f"{root}_motion_context{extension or '.pt'}"
