# Newton Lift deployment

This ROS 2 adapter loads the same two decentralized actors used by the current
Newton Lift task. It restores model dimensions, 60 Hz control timing,
incremental action scales, 20-step initial gripper grace, observation scaling,
checkpoint normalization/clipping, and Full 6D Trend H5 settings from the run's
`params/env.yaml` and `params/agent.yaml`. Explicit ROS topics and physical
camera/tag calibration remain deployment settings.

Keep a checkpoint together with its parent run's `params` directory. Historical
checkpoints without that configuration are rejected. The exported configuration
defaults to dry-run. A short smoke checkpoint tests loading; it is not a trained
manipulation policy.

## Software environment

Use Ubuntu 22.04, ROS Humble and system Python 3.10 for deployment. The
[repository quick start](../../../../../../README.md) covers prerequisites and
clone instructions. Isaac Lab training uses a separate Python 3.12 environment.
From the standalone repository root:

```bash
bash scripts/deploy/setup.sh
source scripts/deploy/env.sh
```

Setup builds the pinned OpenArm V1.0 bringup, hardware/CAN library and robot
description, plus RealSense and AprilTag sources, into `.openarm-ros`. It also
creates `.venv-openarm-deploy`. The deployment computer does not need Isaac Lab.
Use `--robot-only` for controller checks without cameras. Motor/CAN configuration,
camera serials, measured calibration and a compatible checkpoint remain
robot-specific. Use a ROS distribution and Python ABI that match.

A fresh clone still needs a trained Lift checkpoint with its run's `params`
directory, measured camera/tag and hand-eye calibration, connected OpenArm V1.0
hardware, and configured `can0`/`can1`. The template configuration defaults to
dry-run. Export the checkpoint below, verify live state and TF, then use the
explicit `--publish` option to send commands.

## Browser 3D monitor

The [dashboard guide](../../../../../../scripts/deploy/dashboard/README.md)
describes the local browser monitor for measured/target robot poses,
box/ball/peg/hole estimates, grasp targets, joint errors and observation health.
It runs alongside the ROS bridge, and can be previewed with explicit synthetic
data using `--demo`. Ball/Peg/Hole display inputs do not add real perception or
extend the Lift policy adapter.

## Export and validate a checkpoint

```bash
uv run --no-project --python .venv-openarm-deploy/bin/python python \
  source/isaaclab_tasks/isaaclab_tasks/direct/openarm_motion_context/deploy/deploy_node.py \
  --config source/isaaclab_tasks/isaaclab_tasks/direct/openarm_motion_context/deploy/configs/lift_real.yaml \
  --checkpoint /absolute/path/to/run/checkpoints/agent_60000.pt \
  --export-config /absolute/path/to/deployment/lift.yaml
```

This validates both actors and their normalization statistics without importing
Isaac Sim or ROS. It writes `lift.yaml` plus three camera-specific detector
configuration files. Use `--validate-checkpoint` for validation without export.

Lift observations are 30 own features plus the 30-wide partner slot. Only the
first six partner slots carry `[m_x,m_y,m_z,T_v,T_omega,T_S]`; the remaining
slots are zero. Communication is delayed by one policy tick. ROS joint-state
header timestamps own freshness checks and velocity differencing. TF for the
end effector and each camera is queried at its corresponding measurement time.
After an observation gap the temporal/message buffers restart.

## Cameras and tags

The user reports standard 36h11 prints including one white cell around the
pattern: tag 0 is 50 x 50 mm; tags 1 and 2 are 20 x 20 mm. The official
[36h11 definition](https://github.com/AprilRobotics/apriltag/blob/master/tag36h11.c)
has an 8-cell black-border width and 10-cell total width. Therefore exported
`apriltag_ros` sizes are **0.040, 0.016, 0.016 m**, respectively. These sizes
derive from the reported print convention; they are not physical measurements.
Recheck if the print has been cropped or padded. Tag size sets estimated metric
distance; the outer paper width must not be passed as the detector edge size.

Each camera needs its own detector process, rectified image and matching
CameraInfo. Use its exported YAML and publish detections on the matching
`topics.apriltag` topic. The nine TF tag children are unique across cameras;
sharing one tag TF child between three parents produces ambiguous transforms.
The included CAD hand-eye transforms are starting values, not measured physical
calibration. Keep the camera frame names consistent with the RealSense launch.

The included `scripts/deploy/bringup.launch.py` starts all three cameras,
rectifiers and detectors with matching frames/topics when `start_cameras:=true`.
Pass the exported `deployment_config` and three distinct serials as shown in
the repository quick start. It reads detector parameters from the exported
`<config-stem>_<camera>_apriltag.yaml` files.

## Robot commands

Use OpenArm **V1.0**, not the upstream launch default V2.0:

```bash
source scripts/deploy/env.sh
ros2 launch ./scripts/deploy/bringup.launch.py use_fake_hardware:=false \
  right_can_interface:=can0 left_can_interface:=can1
```

The arm command is a seven-element Float64MultiArray in joint1-to-joint7 order,
held between policy ticks. Grippers use single-point JointTrajectory commands
to `finger_joint1`; the second finger follows its mimic mechanism. This removes
the old two-point, 80-150 ms trajectory delay. Joint-trajectory arm transport is
still available if its matching controller and topics are selected.

The exported checkpoint contract also restores `gripper_target_speed_limit`.
A positive value advances the gripper reference at that speed once per policy
tick; zero retains historical step commands. The reference restarts at the
open target with the existing grace period after a control gap. This bounds
command speed, not contact-induced motion or hardware tracking error.

Before policy control, use `deploy/task_start.py --config /path/to/lift.yaml`
to validate the measured state. Add `--execute` for its eight-second smooth
move to the checkpoint's start pose. Then run `deploy/deploy_node.py --config
/path/to/lift.yaml --dry-run`. Use `--publish` to enable robot commands after
physical calibration and the task pose have been checked. Dry-run is the default.

The bridge checks live `/robot_description` joint limits before enabling
commands, clamps outgoing positions to those limits, and holds on stale or
non-finite observations, missing TF, excessive target distance or tracking
error. A tracking failure is reported rather than silently changing the policy's
increment scale. These physical guards are deployment interventions; they are
not claims that the real actuator dynamics equal the simulator.

## Verification and limits

On 2026-10-27, the checkpoint loader and both actors ran under Python 3.10.12,
PyTorch 2.11.0 CPU and skrl 1.4.3. The ROS bridge initialized in an isolated
localhost domain with correct arm/gripper message types and no motor commands.
CPU regressions compare the complete 30D ROS observation with the simulator's
builder under a translated, rotated robot root, and normalization with the
training RunningStandardScaler, including clipping and zero variance.

Compared against the official V1 example URDF, the Newton USD joint-frame FK at
zero and configured start poses differed by about **1.414 mm** in hand position
and less than **1e-6 degrees** in orientation. The residual is a shoulder-origin
offset of +/-1 mm sideways and 1 mm vertically. This is not a measured robot
calibration or a full-workspace accuracy certification.

Historical simulation profiles used high position gains and disabled robot
gravity. Current research profiles also record explicit Newton gravity
compensation; inspect the saved checkpoint contract rather than assuming one
gravity setting for every policy. Upstream V1 motor gains differ, and the hardware
write path does not establish equivalent tracking or compensation. Identify
those real properties before claiming sim-to-real task success; do not blindly
copy simulated gains to motors.
No real lift success or hardware deployment performance is claimed here.

This adapter currently accepts **Lift Trend** checkpoints. WithBall additionally
requires a real ball perception adapter, and Circle V13 uses physical welds in
simulation that require a validated physical grasp adapter. Their checkpoints
are rejected rather than treated as Lift policies.

Upstream references inspected:

- [OpenArm ROS2 8f23aea](https://github.com/enactic/openarm_ros2/tree/8f23aeabb372c256f7807b9a74d48e035fc68386)
- [OpenArm description 1801c61](https://github.com/enactic/openarm_description/tree/1801c61247593fe2bf30cdbde36279aa559e04aa)
- [apriltag_ros detector parameters](https://github.com/christianrauch/apriltag_ros/blob/master/cfg/tags_36h11.yaml)
