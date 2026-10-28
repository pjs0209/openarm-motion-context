"""Candidate camera-only velocity estimator; NOT installed in current actors.

Inputs are camera-derived box-relative XY positions and visibility only.
Missing detections and per-environment resets invalidate historical windows.
The output is a lagged, smoothed velocity, not instantaneous ground truth.
"""

import math

import torch


class RenderedBallVelocity:
    def __init__(self, num_envs, dt, horizon=10, tau=0.1, device="cpu", dtype=torch.float32):
        if (
            type(num_envs) is not int
            or num_envs < 1
            or type(horizon) is not int
            or horizon < 1
            or not math.isfinite(dt)
            or dt <= 0
            or not math.isfinite(tau)
            or tau < 0
        ):
            raise ValueError("Invalid camera velocity timing/batch")
        self.dt, self.horizon = float(dt), horizon
        self.alpha = 1.0 if tau == 0 else -math.expm1(-dt / tau)
        self.history = torch.zeros(horizon + 1, num_envs, 2, device=device, dtype=dtype)
        self.consecutive = torch.zeros(num_envs, device=device, dtype=torch.long)
        self.filtered = torch.zeros(num_envs, 2, device=device, dtype=dtype)
        self.previous_valid = torch.zeros(num_envs, device=device, dtype=torch.bool)
        self.index = 0

    def reset(self, env_ids=None):
        selected = slice(None) if env_ids is None else env_ids
        self.history[:, selected] = 0
        self.consecutive[selected] = 0
        self.filtered[selected] = 0
        self.previous_valid[selected] = False

    @torch.no_grad()
    def update(self, positions_xy, visible):
        if (
            positions_xy.shape != self.filtered.shape
            or visible.shape != self.consecutive.shape
            or visible.dtype != torch.bool
            or positions_xy.device != self.filtered.device
            or visible.device != self.filtered.device
            or not torch.isfinite(positions_xy[visible]).all()
        ):
            raise ValueError("Invalid camera position/visibility batch")
        positions = torch.where(visible[:, None], positions_xy.to(self.history), 0.0)
        past = self.history[(self.index - self.horizon) % len(self.history)].clone()
        self.history[self.index] = positions
        self.index = (self.index + 1) % len(self.history)
        self.consecutive = torch.where(visible, self.consecutive + 1, 0)
        valid = self.consecutive >= self.horizon + 1
        averaged = (positions - past) / (self.horizon * self.dt)
        warmed = self.filtered + self.alpha * (averaged - self.filtered)
        estimate = torch.where(self.previous_valid[:, None], warmed, averaged)
        self.filtered = torch.where(valid[:, None], estimate, 0.0)
        self.previous_valid = valid
        return self.filtered.clone(), valid.clone()


class FrameAwareCameraVelocity:
    """Prototype using camera frame IDs/acquisition times, not control ticks.

    Positions, visibility, IDs and times must all come from camera sensing.
    Re-reading an image holds the last estimate; it is not a new sample.
    This class is not installed in an environment or scored by legacy traces
    (those traces did not record frame IDs). A live integration must supply
    current_times/max_age to invalidate overly old images, and reconstruct
    position in a box frame aligned to the image acquisition time.
    """

    def __init__(self, num_envs, tau=0.0, device="cpu", dtype=torch.float32):
        if type(num_envs) is not int or num_envs < 1 or not math.isfinite(tau) or tau < 0:
            raise ValueError("Invalid camera batch/smoothing time")
        self.tau = float(tau)
        self.position = torch.zeros(num_envs, 2, device=device, dtype=dtype)
        self.time = torch.zeros(num_envs, device=device, dtype=torch.float64)
        self.frame = torch.full((num_envs,), -1, device=device, dtype=torch.long)
        self.velocity = torch.zeros_like(self.position)
        self.valid = torch.zeros(num_envs, device=device, dtype=torch.bool)

    def reset(self, env_ids=None):
        selected = slice(None) if env_ids is None else env_ids
        self.position[selected] = 0
        self.time[selected] = 0
        self.frame[selected] = -1
        self.velocity[selected] = 0
        self.valid[selected] = False

    @torch.no_grad()
    def update(self, positions_xy, visible, frame_ids, acquisition_times, *, current_times=None, max_age=None):
        shape, device = self.frame.shape, self.frame.device
        if (
            positions_xy.shape != self.position.shape
            or visible.shape != shape
            or visible.dtype != torch.bool
            or frame_ids.shape != shape
            or frame_ids.dtype != torch.long
            or acquisition_times.shape != shape
            or any(value.device != device for value in (positions_xy, visible, frame_ids, acquisition_times))
            or not torch.isfinite(positions_xy[visible]).all()
            or not torch.isfinite(acquisition_times).all()
            or (frame_ids < 0).any()
            or (acquisition_times < 0).any()
        ):
            raise ValueError("Invalid camera position/frame/timestamp batch")
        if (current_times is None) != (max_age is None):
            raise ValueError("Camera current times and maximum age must be supplied together")
        if current_times is not None:
            if (
                current_times.shape != shape
                or current_times.device != device
                or not torch.isfinite(current_times).all()
                or not math.isfinite(max_age)
                or max_age <= 0
                or (current_times < acquisition_times).any()
            ):
                raise ValueError("Invalid camera current-time/age specification")
            # Stale images break history even if their frame ID was previously
            # valid. Re-reading them must not keep a usable velocity alive.
            visible = visible & ((current_times - acquisition_times) <= max_age)
        rollback = visible & (self.frame >= 0) & ((frame_ids < self.frame) | (acquisition_times < self.time))
        repeated = visible & (frame_ids == self.frame) & ~rollback
        if (acquisition_times[repeated] != self.time[repeated]).any():
            raise ValueError("Repeated image must retain its acquisition timestamp")
        fresh = visible & (frame_ids != self.frame) & ~rollback
        existing = fresh & (self.frame >= 0)
        if (acquisition_times[existing] <= self.time[existing]).any():
            raise ValueError("New images require strictly increasing acquisition times")
        # Dropout and reset-like counter/time rollback must break velocity history.
        self.reset(torch.nonzero(~visible | rollback, as_tuple=True)[0])
        fresh = visible & (frame_ids != self.frame)
        existing = fresh & (self.frame >= 0)
        elapsed = (acquisition_times[existing] - self.time[existing]).to(self.position.dtype)
        instantaneous = (positions_xy[existing] - self.position[existing]) / elapsed[:, None]
        alpha = torch.ones_like(elapsed) if self.tau == 0 else -torch.expm1(-elapsed / self.tau)
        smoothed = self.velocity[existing] + alpha[:, None] * (instantaneous - self.velocity[existing])
        self.velocity[existing] = torch.where(self.valid[existing, None], smoothed, instantaneous)
        self.valid[existing] = True
        self.position[fresh] = positions_xy[fresh].to(self.position.dtype)
        self.time[fresh] = acquisition_times[fresh].double()
        self.frame[fresh] = frame_ids[fresh]
        return self.velocity.clone(), self.valid.clone()
