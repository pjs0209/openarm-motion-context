# OpenArm deployment dashboard

A local browser view of the V1 CAD robot, measured and target joint poses,
box/ball/peg/hole estimates, grasp targets, camera freshness and joint errors.
The dashboard observes ROS topics. Its HTTP server exposes no motor-command API.

## Start

In the standalone deployment repository, `setup.sh` installs the locked browser
dependencies. Node.js 18+ and npm are required during setup; inference and viewing
do not need Isaac Lab. For a source checkout, install the browser dependencies:

```bash
npm ci --prefix scripts/deploy/dashboard
```

Source ROS Humble and the OpenArm ROS workspace, then run with the ROS Python
interpreter. From a configured standalone checkout:

```bash
source scripts/deploy/env.sh
uv run --no-project --python .venv-openarm-deploy/bin/python python \
  scripts/deploy/openarm_scene.py \
  --config source/isaaclab_tasks/isaaclab_tasks/direct/openarm_motion_context/deploy/configs/lift_real.yaml
```

Open **http://127.0.0.1:8765**. Use the exported `deployment/lift.yaml` as
`--config` when its joint names or command topics differ from the template.
Start the hardware bringup and policy in their usual separate terminals.
The monitor does not start either of them. All JS and CAD assets are served
locally, with no runtime CDN requests. `--port` changes the local port.

Add `--demo` to preview the UI without robot/camera data or a checkpoint. Demo
generates synthetic data inside the HTTP process, does not initialize ROS,
and is labelled **DEMO / SYNTHETIC** throughout the screen. It still uses the
installed OpenArm CAD assets.

## What the screen means

| Display               | Input and meaning                                                                                                    |
| --------------------- | -------------------------------------------------------------------------------------------------------------------- |
| Solid robot           | Measured `topics.joint_states`, using the official V1 URDF for FK                                                    |
| Mint robot            | Final setpoints received on the configured arm/gripper command topics                                                |
| Dry-run overlay       | `/openarm/deploy/policy_target` while a fresh control heartbeat says `dry_run`; these values were not sent to motors |
| TCP arrows            | Difference between measured and displayed target FK at the official URDF TCPs                                        |
| Joint table           | Measured and target angles in radians; absolute error per joint                                                      |
| Box and grasp targets | The Lift bridge's existing AprilTag fusion, with the oldest contributing detection timestamp                         |
| Ball, peg and hole    | `geometry_msgs/PoseStamped` inputs from a separate real perception adapter                                           |
| Perception cards      | Detection-array timestamp/count for each configured camera; not a camera image feed                                  |

A command's last trajectory waypoint is its displayed final setpoint, not a
claim about the intermediate trajectory. With partial commands, uncommanded
joints retain measured values in the overlay. The last published reference is
held until replaced, while measurement freshness is checked independently.

The bridge publishes a dry-run candidate after the same available URDF clamps
used for outgoing commands. The displayed preview does not activate controllers.
The control badge reports `dry_run`, `publishing`, `held`, or `monitor` when no
fresh policy heartbeat is available. A stopped HTTP/ROS stream is marked offline;
the last measured robot is dimmed and object estimates are hidden.

## Object inputs

Default topics are:

```text
/openarm/deploy/objects/box
/openarm/deploy/objects/ball
/openarm/deploy/objects/peg
/openarm/deploy/objects/hole
```

Publish `geometry_msgs/PoseStamped` with the **measurement timestamp**, a valid
frame ID, position in metres and normalized XYZW quaternion. TF must contain
that frame at the measurement time. The server transforms poses to
`base_frame`. Missing, stale, future, invalid or untransformable poses are not
drawn. The live UI never fills missing objects with demo coordinates.

The current Lift bridge supplies the box and left/right grasp targets. It does
not provide real ball/peg/hole perception, and the UI does not add policy
support for WithBall or Peg tasks. Their cards remain `NO DATA` until a real
adapter supplies poses. Run either this box source or an external one for the
box topic, so conflicting publishers cannot describe different estimates.

Primitive shapes are visual proxies centered at the supplied pose. Adjust their
display dimensions and topics in the deployment YAML to match the physical
objects. For example:

```yaml
visualization:
  rate_hz: 15
  max_pose_age_s: 0.5
  objects:
    box:
      topic: /openarm/deploy/objects/box
      size: [0.412, 0.3, 0.25] # Full X/Y/Z dimensions in metres; measure your object.
    ball:
      size: [0.03, 0.03, 0.03] # Diameter, not radius.
```

The primitive dimensions are not used by inference or safety checks. A live
object marker, a small joint error, or a green heartbeat is not evidence of a
successful physical manipulation task.

## Dependencies

Three.js (MIT) and urdf-loader (Apache-2.0) are pinned in `package-lock.json`.
Their original licenses remain in the installed packages. OpenArm CAD assets
come from the pinned `openarm_description` submodule and retain its license.
