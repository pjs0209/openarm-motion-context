* Added a separately checkpointed Lift/WithBall controller that constrained arm
  references to authored joint limits and an optional tracking-error bound.
  Raw Gaussian actions, gripper commands and physical states were preserved.
  Added real-wrapper regression coverage and native motor-effort diagnostics.
