# Pinned Upstream Dependencies

These repositories are Git submodules, not copied source trees. Clone this
project with `--recurse-submodules` or run:

```bash
git submodule update --init --recursive
```

| Directory | Purpose | Required for |
| --- | --- | --- |
| `openarm_ros2` | OpenArm v1.0 ROS 2 description, hardware and controllers | real robot |
| `openarm_can` | OpenArm CAN communication dependency | real robot |
| `realsense-ros` | D435i ROS 2 driver and optical TF frames | real robot |
| `apriltag_ros` | AprilTag detector and `apriltag_msgs` output | real robot |
| `IsaacLab` | Official Isaac Lab base checkout | simulation/training |

Isaac Lab is pinned to the official base commit immediately before the local
OpenArm customization. The customization itself is maintained by this
repository's `source/` and `scripts/` overlay.

Each submodule remains governed by its upstream license. OpenArm USD and mesh
workspace artifacts are intentionally handled separately because their source
licenses must be verified before redistribution.
