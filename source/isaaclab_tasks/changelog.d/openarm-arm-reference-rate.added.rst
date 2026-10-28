* Added an opt-in Lift/WithBall controller that advanced arm position references
  at the declared per-joint velocity rates once per control tick. Preserved
  gripper control, raw Gaussian policies, observations, rewards, physics-substep
  targets and task-owned reset state. Identified the protocol in saved policy
  buffers and the runtime builder without changing frozen task defaults or
  existing checkpoints. Reference rates do not impose hard solver velocity or
  contact impulse limits.
