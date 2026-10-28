"""Unlaunched camera-history prototype; no simulator truth or future input.

One update per control observation. Estimate box-frame XY velocity from a
short finite-difference window, then forecast XY. Keep measured Z/visibility
and the four-channel layout. This is not an installed environment adapter.
Resets and occlusions discard history; callers must explicitly reset rows at
episode boundaries. A replay score is not trained balancing performance.
"""

import math

import torch

from scripts.studies.rendered_ball_velocity import RenderedBallVelocity


class CameraXYForecast:
    def __init__(self, *, dt, history_steps=6, lookahead_seconds=0.15, position_scale=0.2, speed_limit=1.0):
        if (
            type(history_steps) is not int
            or history_steps < 1
            or not all(math.isfinite(v) and v > 0 for v in (dt, lookahead_seconds, position_scale, speed_limit))
        ):
            raise ValueError("Positive finite camera timing/scales and integer history required")
        self.dt = dt
        self.history_steps = history_steps
        self.lookahead_seconds = lookahead_seconds
        self.position_scale = position_scale
        self.speed_limit = speed_limit
        self._estimator = None

    def reset(self, mask):
        if self._estimator is None:
            raise ValueError("No allocated camera history to reset")
        if (
            mask.shape != self._estimator.consecutive.shape
            or mask.dtype != torch.bool
            or mask.device != self._estimator.history.device
        ):
            raise ValueError("Same-device boolean batch reset mask required")
        self._estimator.reset(mask)

    @torch.no_grad()
    def update(self, camera):
        if (
            camera.ndim != 2
            or camera.shape[1] != 4
            or camera.shape[0] < 1
            or not camera.is_floating_point()
            or not torch.isfinite(camera).all()
            or not ((camera[:, 3] == 0) | (camera[:, 3] == 1)).all()
            or (camera[:, :3].abs() > 5).any()
            or (camera[camera[:, 3] == 0, :3] != 0).any()
        ):
            raise ValueError("Finite bounded XYZ plus binary visibility; hidden XYZ must be zero")
        if self._estimator is None:
            self._estimator = RenderedBallVelocity(
                camera.shape[0], self.dt, horizon=self.history_steps, tau=0.0, device=camera.device, dtype=camera.dtype
            )
        elif (
            camera.shape[0] != self._estimator.history.shape[1]
            or camera.device != self._estimator.history.device
            or camera.dtype != self._estimator.history.dtype
        ):
            raise ValueError("Camera batch/device/dtype changed without a new forecaster")
        visible = camera[:, 3].bool()
        # An occlusion breaks the derivative window; never carry unseen motion.
        velocity, ready = self._estimator.update(camera[:, :2] * self.position_scale, visible)
        speed = torch.linalg.vector_norm(velocity, dim=1, keepdim=True)
        velocity = velocity * (self.speed_limit / speed.clamp_min(self.speed_limit))
        velocity = torch.where(ready[:, None], velocity, torch.zeros_like(velocity))
        result = camera.clone()
        result[:, :2] = (camera[:, :2] + velocity * self.lookahead_seconds / self.position_scale).clamp(-5, 5)
        return result, ready
