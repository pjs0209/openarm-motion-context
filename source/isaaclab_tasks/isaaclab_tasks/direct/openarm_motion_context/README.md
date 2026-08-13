# OpenArm Motion Context

This package contains the paper implementation of Proprioceptive Motion
Context Sharing for OpenArm bimanual manipulation.

- `envs`: Isaac Lab task definitions and reward/state logic.
- `perception`: AprilTag transforms, multi-camera fusion, and grasp targets.
- `communication`: signed EE motion, motion context, and message construction.
- `models`: decentralized actors and centralized critic.
- `train`: MAPPO trainer, checkpoint state, and YAML configurations.
- `eval`: communication ablations, metrics, and trace analysis.
- `deploy`: ROS2 inference, calibration, and real-robot configuration.
- `common`: shared transform, observation, and action contracts.
- `scripts`: short command wrappers for training, evaluation, and deployment.

The Gym task IDs remain `Isaac-OpenArm-Re-Lift-Paper-v0` and
`Isaac-OpenArm-PegInHole-Fixed-Paper-v0` for checkpoint and command
compatibility.
