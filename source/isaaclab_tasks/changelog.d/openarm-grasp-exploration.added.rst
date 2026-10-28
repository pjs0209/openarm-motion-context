* Added an opt-in Gaussian gripper initialization for the OpenArm Lift reset
  curriculum. It initialized only the gripper output mean toward closing while
  retaining trainable weights, the existing controller and the unchanged task
  observations, rewards, success conditions and six-dimensional partner message.
