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

Isaac Lab is pinned to the current Isaac Lab 3 base commit
`e3f7d9ff1f5d7f8280de3ece9000b7117f66bcf4`. The customization itself is maintained
by this repository's `source/`, `scripts/` and `compat/` overlay; see
[research setup](../docs/RESEARCH_SETUP.md) for the explicit preview/apply installer.

Each submodule remains governed by its upstream license. OpenArm USD and mesh
workspace artifacts are intentionally handled separately because their source
licenses must be verified before redistribution.
