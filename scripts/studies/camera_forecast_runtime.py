"""Uninstalled control-sampled forecast runtime core; camera inputs only.

The original measured XYZ/visibility is never replaced. Return a separate XY
lead and validity for a future observation adapter. Sensor frame IDs and times
check freshness; they are not treated as new samples on repeated callbacks.
This is not image-time-aligned instantaneous velocity and not a GPU-tested
environment adapter. Call once per observed control tick (duplicates cache).
"""

import math

import torch

from scripts.studies.camera_xy_forecast import CameraXYForecast


class CameraForecastRuntime:
    def __init__(self, *, dt, position_scale=0.2, max_image_age=0.05):
        if not math.isfinite(max_image_age) or max_image_age <= 0:
            raise ValueError("Positive finite maximum image age required")
        self.forecaster = CameraXYForecast(dt=dt, position_scale=position_scale)
        self.max_image_age = max_image_age
        self._last = None

    @torch.no_grad()
    def update(self, camera, *, control_step, episode_lengths, frame_ids, acquisition_times, current_times):
        if type(control_step) is not int or control_step < 0:
            raise ValueError("Nonnegative integer control tick required")
        if camera.ndim != 2 or camera.shape[1] != 4 or not camera.is_floating_point():
            raise ValueError("Four-channel floating camera observation required")
        shape = (camera.shape[0],)
        if (
            any(
                value.shape != shape or value.device != camera.device
                for value in (episode_lengths, frame_ids, acquisition_times, current_times)
            )
            or episode_lengths.dtype != torch.long
            or frame_ids.dtype != torch.long
            or (episode_lengths < 0).any()
            or (frame_ids < 0).any()
            or not acquisition_times.is_floating_point()
            or not current_times.is_floating_point()
            or acquisition_times.dtype != current_times.dtype
            or not torch.isfinite(acquisition_times).all()
            or not torch.isfinite(current_times).all()
            or (acquisition_times < 0).any()
            or (current_times < acquisition_times).any()
            or not torch.isfinite(camera).all()
            or (camera[:, :3].abs() > 5).any()
            or not ((camera[:, 3] == 0) | (camera[:, 3] == 1)).all()
            or (camera[camera[:, 3] == 0, :3] != 0).any()
        ):
            raise ValueError("Matched finite sensor counters/times and bounded camera input required")
        reset = episode_lengths <= 1
        if self._last is not None:
            old = self._last
            if (
                camera.shape != old["camera"].shape
                or camera.dtype != old["camera"].dtype
                or camera.device != old["camera"].device
                or control_step < old["step"]
            ):
                raise ValueError("Runtime batch/device/dtype changed or control tick rolled back")
            reset |= episode_lengths < old["lengths"]
            reset |= (frame_ids < old["frames"]) | (acquisition_times < old["acquisition"])
            repeated = (frame_ids == old["frames"]) & ~reset
            if (acquisition_times[repeated] != old["acquisition"][repeated]).any():
                raise ValueError("Repeated image must retain its acquisition timestamp")
            fresh = (frame_ids > old["frames"]) & ~reset
            if (acquisition_times[fresh] <= old["acquisition"][fresh]).any():
                raise ValueError("New images require increasing acquisition timestamps")
            if control_step == old["step"]:
                unchanged = ~reset
                if (
                    not torch.equal(camera[unchanged], old["camera"][unchanged])
                    or not torch.equal(frame_ids[unchanged], old["frames"][unchanged])
                    or not torch.equal(episode_lengths[unchanged], old["lengths"][unchanged])
                    or not torch.equal(current_times[unchanged], old["current"][unchanged])
                ):
                    raise ValueError("Camera/counters changed inside a cached control tick")
                # A reset at the same tick must not advance other environments.
                self.forecaster.reset(reset)
                lead, ready = old["lead"].clone(), old["ready"].clone()
                lead[reset], ready[reset] = 0.0, False
                self._remember(
                    camera, control_step, episode_lengths, frame_ids, acquisition_times, current_times, lead, ready
                )
                return lead.clone(), ready.clone()
            if control_step != old["step"] + 1:
                reset = torch.ones_like(reset)  # Do not invent unobserved fixed-dt samples.
            self.forecaster.reset(reset)
        fresh_enough = current_times - acquisition_times <= self.max_image_age
        visible = camera[:, 3].bool() & fresh_enough & ~reset
        usable = torch.where(visible[:, None], camera, torch.zeros_like(camera))
        prediction, ready = self.forecaster.update(usable)
        lead = prediction[:, :2] - usable[:, :2]
        self._remember(camera, control_step, episode_lengths, frame_ids, acquisition_times, current_times, lead, ready)
        return lead.clone(), ready.clone()

    def _remember(self, camera, step, lengths, frames, acquisition, current, lead, ready):
        self._last = {
            "camera": camera.clone(),
            "step": step,
            "lengths": lengths.clone(),
            "frames": frames.clone(),
            "acquisition": acquisition.clone(),
            "current": current.clone(),
            "lead": lead.clone(),
            "ready": ready.clone(),
        }
