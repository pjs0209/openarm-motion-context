# OpenArm Motion Context

## Simulation research source

The repository now also includes the current Lift, LiftWithBall, contact-grasp
Circle Peg, experimental actor/control variants, evaluation tools, regressions,
and pinned compatibility code. See [research setup](docs/RESEARCH_SETUP.md) for
the Isaac Lab 3 / Newton source-overlay installer and the separate licensed USD
asset requirements. Training logs and new large checkpoints are not uploaded.
This source publication does not certify task success or Full 6D superiority.
See [current research status and checks](docs/CURRENT_RESEARCH_STATUS.md) and
[source/checkpoint provenance](docs/CURRENT_RESEARCH_PROVENANCE.json).

The current controlled Lift/Peg comparison is documented in the
[recovery protocol](scripts/studies/RECOVERY_PROTOCOL.rst): independently trained
NoShare, MotionOnly, Signed Twist and Full6D Trend with the same Actor/Critic
and own controller-state inputs. The 512-step configuration is an integration
check, not a success-rate result.

The following quick start is the independent **ROS deployment** workflow.

OpenArm V1.0 bimanual deployment with decentralized MAPPO actors and a compact
partner message: `[m_x, m_y, m_z, translation trend, rotation trend, stability trend]`.
The Lift actor uses a 30D own observation and a 30-wide partner slot containing
six populated values, with one policy tick of communication delay.

## Clone and install

Use **Ubuntu 22.04, ROS 2 Humble and system Python 3.10** for deployment.
Isaac Lab and a GPU are not required on the robot computer. Install ROS Humble
using the [official instructions](https://docs.ros.org/en/humble/Installation/Ubuntu-Install-Debs.html)
and install [uv](https://docs.astral.sh/uv/getting-started/installation/) plus
Node.js 18+ with npm for the local browser dashboard.
Then install the build tools:

```bash
sudo apt-get update
sudo apt-get install -y git build-essential cmake python3-dev \
  python3-colcon-common-extensions python3-rosdep
# Run init once per computer; skip it if rosdep is already initialized.
sudo rosdep init
rosdep update --rosdistro humble

git clone https://github.com/pjs0209/openarm-motion-context.git
cd openarm-motion-context
bash scripts/deploy/setup.sh
source scripts/deploy/env.sh
```

Setup initializes the pinned deployment submodules automatically, installs ROS
dependencies, builds `.openarm-ros`, and creates `.venv-openarm-deploy`. It includes
OpenArm ROS2/CAN/description, RealSense ROS, AprilTag ROS/messages and image
rectification. It leaves the historical Isaac Lab submodule uninitialized.
Upstream optional MoveIt, GUI, Gazebo and ZED dependencies are excluded from
this V1 deployment build. The RealSense SDK is installed through rosdep.

For robot controllers without cameras, use `bash scripts/deploy/setup.sh --robot-only`.
Use `--skip-rosdep` only when the required system dependencies are already installed.
After pulling updates, rerun setup to update submodules and rebuild. Source
`scripts/deploy/env.sh` in every new terminal from this repository's root.

## Browser 3D dashboard

The included [local dashboard](scripts/deploy/dashboard/README.md) displays
measured and target robot meshes, box/ball/peg/hole poses, grasp targets,
per-camera freshness and per-joint errors. After setup:

```bash
source scripts/deploy/env.sh
uv run --no-project --python .venv-openarm-deploy/bin/python python \
  scripts/deploy/openarm_scene.py \
  --config source/isaaclab_tasks/isaaclab_tasks/direct/openarm_motion_context/deploy/configs/lift_real.yaml
```

Open **http://127.0.0.1:8765**. Add `--demo` for a clearly labelled synthetic
preview without connecting to ROS or motors. For live policies, use the exported
deployment YAML as `--config`. The monitor observes the existing ROS graph and
provides no motor-control buttons or HTTP command endpoint. Box poses come from
the Lift bridge; real ball/peg/hole poses need separate perception adapters.

## Check robot controllers

The included launch defaults to fake hardware and starts no cameras:

```bash
source scripts/deploy/env.sh
ros2 launch ./scripts/deploy/bringup.launch.py
```

In another terminal:

```bash
source scripts/deploy/env.sh
ros2 control list_controllers
ros2 topic echo /joint_states --once
```

Expect the joint-state broadcaster, left/right forward position controllers
and left/right gripper controllers to be active. The launch fixes the robot
model to V1.0 and uses the upstream controller configuration, without RViz.
To start the real robot after configuring SocketCAN and motor IDs for your
hardware, set `use_fake_hardware:=false`. Right arm defaults to `can0`, left to
`can1`, and CAN-FD is enabled; override `right_can_interface`,
`left_can_interface` and `can_fd` to match the robot. Follow the
[OpenArm hardware setup](https://docs.openarm.dev/) for bus and motor calibration.

## Export a Lift checkpoint

Supply a trained Newton **Lift Trend** checkpoint together with its run's
`params/env.yaml` and `params/agent.yaml`:

```text
/path/to/run/
  checkpoints/agent_60000.pt
  params/env.yaml
  params/agent.yaml
```

```bash
uv run --no-project --python .venv-openarm-deploy/bin/python python \
  scripts/deploy/openarm_lift_ros2.py \
  --config scripts/deploy/openarm_lift_deploy.yaml \
  --checkpoint /path/to/run/checkpoints/agent_60000.pt \
  --export-config deployment/lift.yaml
```

The export loads both actors and normalizers, restores their saved settings,
and writes `deployment/lift.yaml` plus three `lift_*_apriltag.yaml` files.
Use `--validate-checkpoint` to validate without exporting. The older models
in `artifacts/` are historical and are not validated for this adapter. This
repository supplies source software; a compatible trained policy is a separate
input. The new research overlay includes the current task code, but licensed
assets and the matching Isaac Lab installation remain separate inputs.

## Start cameras and run the policy

Measure camera mounting and tag/grasp transforms in `deployment/lift.yaml`.
The included CAD transforms are initial estimates. Use the serial number of
each physical camera to fix its role:

```bash
source scripts/deploy/env.sh
ros2 launch ./scripts/deploy/bringup.launch.py \
  use_fake_hardware:=false start_cameras:=true \
  deployment_config:=deployment/lift.yaml \
  left_wrist_serial:=LEFT_SERIAL right_wrist_serial:=RIGHT_SERIAL chest_serial:=CHEST_SERIAL
```

This starts the robot, three color cameras, three rectifiers and three AprilTag
detectors. Frame names and detection topics match the supplied deployment
configuration; detector parameters come from the checkpoint export. The launch
requires three distinct camera serials. Policy inference runs separately:

```bash
source scripts/deploy/env.sh
# Check measured state; add --execute for the smooth move to the checkpoint start pose.
uv run --no-project --python .venv-openarm-deploy/bin/python python \
  scripts/deploy/openarm_move_to_task_start.py --config deployment/lift.yaml
uv run --no-project --python .venv-openarm-deploy/bin/python python \
  scripts/deploy/openarm_lift_ros2.py --config deployment/lift.yaml --dry-run
# After calibration and start-pose checks, replace --dry-run with --publish.
```

Reported 36h11 print sizes include the white margin: 50 mm for tag 0, 20 mm
for tags 1/2. Exported detector black-border edge sizes are 40/16/16 mm. Check
these against the physical prints. The bridge holds commands when joint or
camera measurements are stale, TF is missing, or tracking/target limits fail.

See the [deployment guide](source/isaaclab_tasks/isaaclab_tasks/direct/openarm_motion_context/deploy/README.md)
for policy conventions, calibration details and existing model-comparison evidence.
Real camera streaming, motor tracking, gravity compensation and manipulation
success require validation on the physical robot. WithBall and welded Peg
checkpoints need separate perception/grasp adapters and are rejected here.

## Portable checks

Validated on 2026-10-28 with ROS Humble/Python 3.10:

- All eight deployment source packages built, and a fresh CPU Python environment installed.
- Five controllers activated on mock hardware; a position command returned through joint states.
- Synthetic raw images passed through all three rectifiers and produced tag 0 detections on the configured topics.
- A local Lift smoke checkpoint loaded both actors and exported the deployment/detector configurations.
- Three portable deployment regressions and focused formatting/lint checks passed.

These checks use mock hardware and synthetic images; they do not establish physical task success.

```bash
uv pip install --python .venv-openarm-deploy/bin/python pytest
uv run --no-project --python .venv-openarm-deploy/bin/python python -m pytest \
  source/isaaclab_tasks/test/test_openarm_deployment_runtime.py -q
```

## License

See [LICENSE](LICENSE). Upstream submodules retain their original licenses.
