# OpenArm Motion Context

Isaac Lab implementation of **Proprioceptive Motion Context Sharing** for
OpenArm bimanual manipulation.

Each decentralized actor receives its own 30D observation and, depending on
the ablation, a compact partner message containing signed end-effector motion
and task-agnostic motion context:

```text
z = [signed EE motion (3D), linear activity, angular activity, action smoothness]
```

The repository contains DirectMARL lift and fixed peg-in-hole tasks, MAPPO
models and training code, communication ablations, AprilTag-based grasp target
estimation, trace analysis, and ROS 2 deployment utilities.

## Requirements

- Isaac Lab checkout compatible with the local 2026 API
- Isaac Sim and a CUDA-capable GPU for simulation
- `skrl`, PyTorch, Gymnasium, NumPy, Matplotlib and PyYAML
- ROS 2, `apriltag_msgs`, TF2 and OpenArm interfaces for real deployment

## Install into Isaac Lab

Clone this repository with its pinned upstream dependencies, then copy the
project overlay into the bundled Isaac Lab checkout:

```bash
git clone --recurse-submodules https://github.com/pjs0209/openarm-motion-context.git
cd openarm-motion-context
cp -a source/. third_party/IsaacLab/source/
cp -a scripts/. third_party/IsaacLab/scripts/
```

If the repository was cloned without `--recurse-submodules`, initialize the
dependencies afterward with `git submodule update --init --recursive`.

All implementation and configuration lives under
`isaaclab_tasks.direct.openarm_motion_context`.

## Assets

OpenArm assets are grouped by responsibility in the Isaac Lab asset extension:

```text
source/isaaclab_assets/data/Robots/OpenArm
source/isaaclab_assets/data/Sensors/OpenArm
source/isaaclab_assets/data/Environments/OpenArm/{lift,peg_in_hole}
source/isaaclab_assets/data/Objects/OpenArm/peg_in_hole
```

The OpenArm USD assets are workspace artifacts and are not duplicated inside
the upstream dependency submodules. Copy the validated asset directories from
the training workstation into `third_party/IsaacLab/source/isaaclab_assets/data`
or use the following overrides when the assets live elsewhere:

```bash
export OPENARM_ROBOT_ASSET_DIR=/absolute/path/to/Robots/OpenArm
export OPENARM_SENSOR_ASSET_DIR=/absolute/path/to/Sensors/OpenArm
export OPENARM_ENVIRONMENT_ASSET_DIR=/absolute/path/to/Environments/OpenArm
export OPENARM_OBJECT_ASSET_DIR=/absolute/path/to/Objects/OpenArm
```

The legacy `OPENARM_ASSET_DIR` variable is retained only for workspaces where
all asset categories still share one directory.

## Train

Lift:

```bash
./isaaclab.sh -p scripts/reinforcement_learning/skrl/train_openarm_bimanual_direct.py \
  --task Isaac-OpenArm-Lift-v0 \
  --experiment_tag motion_context \
  --num_envs 128 \
  --headless \
  env.communication_mode=motion_context
```

Peg-in-hole:

```bash
./isaaclab.sh -p scripts/reinforcement_learning/skrl/train_openarm_bimanual_direct.py \
  --task Isaac-OpenArm-PegInHole-v0 \
  --experiment_tag motion_context \
  --num_envs 128 \
  --headless \
  env.communication_mode=motion_context
```

Available communication ablations:

```text
none
motion_only
context_only
motion_context
previous_action
full_partner_observation
```

## Evaluate and Save Traces

```bash
./isaaclab.sh -p scripts/reinforcement_learning/skrl/play_openarm_bimanual_direct.py \
  --task Isaac-OpenArm-Lift-v0 \
  --num_envs 1 \
  --num_steps 9000 \
  --num_eval_episodes 10 \
  --deterministic_eval \
  --save_motion_context_trace \
  --motion_context_trace_dir logs/motion_context_traces/lift_motion_context \
  --checkpoint /path/to/best_agent.pt \
  env.communication_mode=motion_context
```

Analyze traces without launching Isaac Sim:

```bash
python scripts/analyze_motion_context_trace.py \
  --trace_dir logs/motion_context_traces/lift_motion_context \
  --out_dir logs/motion_context_analysis
```

The analyzer writes episode, seed, method, failure, timing, payload-statistics,
and schema-validation CSV files plus paper-ready overview and ablation figures.

## Real-Robot Deployment

Deployment configuration and calibration must be checked for the real robot.
In particular, verify `T_box_tag`, grasp targets, camera optical frames, joint
names, command topics, action limits, and checkpoint normalization statistics.

```bash
./isaaclab.sh -p -m \
  isaaclab_tasks.direct.openarm_motion_context.deploy.deploy_node \
  --config source/isaaclab_tasks/isaaclab_tasks/direct/openarm_motion_context/deploy/configs/lift_real_calibrated.yaml
```

Camera TF consistency check:

```bash
./isaaclab.sh -p \
  source/isaaclab_tasks/isaaclab_tasks/direct/openarm_motion_context/deploy/camera_tf_check.py \
  --config source/isaaclab_tasks/isaaclab_tasks/direct/openarm_motion_context/deploy/configs/lift_real_calibrated.yaml
```

Use `--dry-run` during initial ROS subscription, TF and safety validation.

## License

Code is provided under the BSD-3-Clause license inherited from Isaac Lab. USD,
mesh, camera and robot assets are separate artifacts and may have different
license requirements.
