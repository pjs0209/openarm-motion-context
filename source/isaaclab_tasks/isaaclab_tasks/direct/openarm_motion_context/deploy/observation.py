"""Checkpoint-compatible actor observation normalization."""

from typing import Any

import torch


class ObservationNormalizer:
    """Apply the state-preprocessor statistics stored in a policy checkpoint."""

    def __init__(self, checkpoint: Any, agent: str, obs_dim: int, *, device: torch.device, enabled: bool = True):
        self.enabled = enabled
        self.loaded = not enabled
        self.mean = torch.zeros((obs_dim,), dtype=torch.float32, device=device)
        self.std = torch.ones((obs_dim,), dtype=torch.float32, device=device)
        if not enabled:
            return
        state = self._extract_state(checkpoint, agent)
        if state is None:
            return
        mean = self._first_tensor(state, ("mean", "_mean", "running_mean"))
        variance = self._first_tensor(state, ("running_variance", "variance", "var", "_variance", "running_var"))
        std = self._first_tensor(state, ("std", "_std", "running_std"))
        mean_valid = mean is not None and mean.numel() >= obs_dim
        std_valid = std is not None and std.numel() >= obs_dim
        variance_valid = variance is not None and variance.numel() >= obs_dim
        if mean_valid:
            self.mean = mean.flatten()[:obs_dim].to(device=device, dtype=torch.float32)
        if std_valid:
            self.std = std.flatten()[:obs_dim].to(device=device, dtype=torch.float32).clamp_min(1.0e-6)
        elif variance_valid:
            self.std = torch.sqrt(
                variance.flatten()[:obs_dim].to(device=device, dtype=torch.float32).clamp_min(1.0e-12)
            )
        self.loaded = bool(mean_valid and (std_valid or variance_valid))

    @staticmethod
    def _first_tensor(state: dict[str, Any], names: tuple[str, ...]) -> torch.Tensor | None:
        for name in names:
            value = state.get(name)
            if isinstance(value, torch.Tensor):
                return value.detach()
        return None

    @staticmethod
    def _extract_state(checkpoint: Any, agent: str) -> dict[str, Any] | None:
        if not isinstance(checkpoint, dict):
            return None
        agent_modules = checkpoint.get(agent)
        if isinstance(agent_modules, dict):
            state = agent_modules.get("state_preprocessor")
            if isinstance(state, dict):
                return state
        for key in (
            "state_preprocessor",
            "_state_preprocessor",
            "state_preprocessors",
            "_state_preprocessors",
            "preprocessors",
        ):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                if agent in value and isinstance(value[agent], dict):
                    return value[agent]
                compound_key = f"{agent}/state_preprocessor"
                if compound_key in value and isinstance(value[compound_key], dict):
                    return value[compound_key]
        for key, value in checkpoint.items():
            if isinstance(key, str) and agent in key and "preprocessor" in key and isinstance(value, dict):
                return value
        return None

    def __call__(self, states: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return states
        return (states - self.mean) / self.std.clamp_min(1.0e-6)
