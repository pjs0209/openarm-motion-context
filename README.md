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

Clone this repository next to or inside your Isaac Lab workspace, then copy its
overlay while preserving relative paths:

```bash
git clone https://github.com/pjs0209/openarm-motion-context.git
cd /path/to/IsaacLab
cp -a /path/to/openarm-motion-context/source/. source/
cp -a /path/to/openarm-motion-context/scripts/. scripts/
```

All implementation and configuration lives under
`isaaclab_tasks.direct.openarm_motion_context`. The old
`isaaclab_tasks.direct.paper` compatibility wrappers are intentionally not
included in this standalone repository.

## Assets

USD assets are not included because the current local files contain unresolved
external references and need a separate license/relinking pass. Set
`OPENARM_ASSET_DIR` to a directory containing:

```text
openarm_robot_with_camera_fixed.usda
openarm_env_box.usd
openarm_env_peg_in_hole.usd
```

The robot USD must also be able to resolve its referenced camera and robot
assets.

```bash
export OPENARM_ASSET_DIR=/absolute/path/to/assets/openarm
```

When the repository is overlaid into an `IsaacLab` workspace, the task first
tries the standard sibling path `../assets/openarm`. Set the environment
variable explicitly when assets use a different layout. If neither location is
valid, the task fails immediately with the paths and required filenames.

## Train

Lift:

```bash
./isaaclab.sh -p scripts/reinforcement_learning/skrl/train_openarm_bimanual_direct.py \
  --task Isaac-OpenArm-Re-Lift-Paper-v0 \
  --intent_variant share_intent \
  --experiment_tag motion_context \
  --num_envs 128 \
  --headless \
  env.communication_mode=motion_context
```

Peg-in-hole:

```bash
./isaaclab.sh -p scripts/reinforcement_learning/skrl/train_openarm_bimanual_direct.py \
  --task Isaac-OpenArm-PegInHole-Fixed-Paper-v0 \
  --intent_variant share_intent \
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
  --task Isaac-OpenArm-Re-Lift-Paper-v0 \
  --intent_variant share_intent \
  --num_envs 1 \
  --num_steps 9000 \
  --num_eval_episodes 10 \
  --deterministic_eval \
  --save_mode_trace \
  --mode_trace_dir logs/paper_motion_context_traces/lift_motion_context \
  --checkpoint /path/to/best_agent.pt \
  env.communication_mode=motion_context
```

Analyze traces without launching Isaac Sim:

```bash
python scripts/analyze_paper_motion_context_trace.py \
  --trace_dir logs/paper_motion_context_traces/lift_motion_context \
  --out_dir logs/paper_motion_context_analysis
```

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
