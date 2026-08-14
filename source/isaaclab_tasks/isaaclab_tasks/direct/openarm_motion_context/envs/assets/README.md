# Assets

OpenArm USD assets live in the Isaac Lab asset extension and are separated by
responsibility:

```text
source/isaaclab_assets/data/Robots/OpenArm
source/isaaclab_assets/data/Sensors/OpenArm
source/isaaclab_assets/data/Environments/OpenArm
  lift/
  peg_in_hole/
source/isaaclab_assets/data/Objects/OpenArm
  peg_in_hole/
```

This directory is reserved for small task-local calibration or generated
artifacts. It must not contain duplicate robot, environment, object, or sensor
USDs.
