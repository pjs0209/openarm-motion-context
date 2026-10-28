* Restored the high-PD OpenArm robot's intended gravity exclusion in MJWarp using
  explicit robot-only compensation while preserving scene-object gravity.
  The setting was recorded in training configurations. Playback of older
  configurations retained the uncompensated model unless explicitly overridden.
