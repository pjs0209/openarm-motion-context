* Added a separate OpenArm Circle task with two free 0.2 kg parts, gripper contact
  and friction instead of grasp welds, and eight policy actions per arm. Pregrasp
  poses were initialized only on reset; CAD seating success required both
  commanded grasps to retain their parts, and part loss was reported as a drop.
  Existing welded task registrations and checkpoints were preserved. Native
  hold, transport and release diagnostics and environment-browser support were added.
