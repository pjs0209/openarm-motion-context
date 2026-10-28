* Restored saved OpenArm environment and agent configurations during checkpoint playback,
  retaining explicit scalar overrides and rejecting implicit physics backend changes.
  Disabled evaluation TensorBoard/checkpoint writes into the recorded training run.
  Recorded resolved evaluation configs alongside traces and refused to overwrite existing episodes.
* Raised the OpenArm Newton constraint capacity to cover full-action PPO contact poses
  that exceeded the previous small-action preflight capacity.
* Added a CAD-preserving SDF for the static OpenArm table rear frame to eliminate
  spurious deeply penetrating mesh contacts and first-step box impulses in the validated reset probe.
