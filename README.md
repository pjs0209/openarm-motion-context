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

## Clone and Install

Clone this repository with every pinned dependency:

```bash
cd ~
git clone --recurse-submodules https://github.com/pjs0209/openarm-motion-context.git
cd openarm-motion-context
git submodule update --init --recursive
```

Apply the project overlay and install Isaac Lab:

```bash
cp -a source/. third_party/IsaacLab/source/
cp -a scripts/. third_party/IsaacLab/scripts/
cd third_party/IsaacLab
./isaaclab.sh --install
cd ../..
```

Run the two `cp` commands again after pulling new motion-context code. They
refresh the local Isaac Lab working tree without changing pinned upstream
history.

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
cd ~/openarm-motion-context/third_party/IsaacLab
./isaaclab.sh -p scripts/reinforcement_learning/skrl/train_openarm_bimanual_direct.py \
  --task Isaac-OpenArm-Lift-v0 \
  --experiment_tag motion_context \
  --num_envs 128 \
  --headless \
  env.communication_mode=motion_context
```

Peg-in-hole:

```bash
cd ~/openarm-motion-context/third_party/IsaacLab
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
cd ~/openarm-motion-context/third_party/IsaacLab
./isaaclab.sh -p scripts/reinforcement_learning/skrl/play_openarm_bimanual_direct.py \
  --task Isaac-OpenArm-Lift-v0 \
  --num_envs 1 \
  --num_steps 9000 \
  --num_eval_episodes 10 \
  --deterministic_eval \
  --save_motion_context_trace \
  --motion_context_trace_dir logs/motion_context_traces/lift_motion_context \
  --checkpoint ../../artifacts/checkpoints/lift_motion_context/best_agent.pt \
  env.communication_mode=motion_context
```

Analyze traces without launching Isaac Sim:

```bash
cd ~/openarm-motion-context/third_party/IsaacLab
./isaaclab.sh -p scripts/analyze_motion_context_trace.py \
  --trace_dir logs/motion_context_traces/lift_motion_context \
  --out_dir logs/motion_context_analysis
```

The analyzer writes episode, seed, method, failure, timing, payload-statistics,
and schema-validation CSV files plus paper-ready overview and ablation figures.

## Real-Robot Deployment

Deployment configuration and calibration must be checked for the real robot.
In particular, verify `T_box_tag`, grasp targets, camera optical frames, joint
names, command topics, action limits, and checkpoint normalization statistics.

### Build the ROS 2 workspace

The commands below use ROS 2 Humble. Change the distro path if the robot uses a
different ROS 2 release.

```bash
source /opt/ros/humble/setup.bash
mkdir -p ~/openarm_robot_ws/src
cd ~/openarm_robot_ws/src
ln -sfn ~/openarm-motion-context/third_party/openarm_ros2 openarm_ros2
ln -sfn ~/openarm-motion-context/third_party/openarm_can openarm_can
ln -sfn ~/openarm-motion-context/third_party/realsense-ros realsense-ros
ln -sfn ~/openarm-motion-context/third_party/apriltag_ros apriltag_ros
cd ~/openarm_robot_ws
rosdep install --from-paths src --ignore-src -r -y
colcon build --symlink-install
```

Source ROS and this workspace in every real-robot terminal:

```bash
source /opt/ros/humble/setup.bash
source ~/openarm_robot_ws/install/setup.bash
```

Configure `can0` and `can1` according to the OpenArm v1.0 hardware guide before
starting real hardware. Confirm which interface belongs to each arm; swapping
them also swaps the commanded task pose.

### Terminal 1: OpenArm bringup

```bash
source /opt/ros/humble/setup.bash
source ~/openarm_robot_ws/install/setup.bash
ros2 launch openarm_bringup openarm.bimanual.launch.py \
  arm_type:=openarm_v1.0 \
  use_fake_hardware:=false \
  robot_controller:=joint_trajectory_controller \
  right_can_interface:=can0 \
  left_can_interface:=can1
```

Wait until the joint-state broadcaster, both arm controllers, and both gripper
controllers are active. Check them in another sourced terminal:

```bash
ros2 control list_controllers
ros2 topic echo /joint_states --once
```

### Move from bringup zero to the task start pose

OpenArm v1.0's hardware plugin moves all arm joints to zero during activation.
After bringup and controller activation, first validate that joint states and the
configured task pose are available. This command does not move the robot:

```bash
cd ~/openarm-motion-context
source /opt/ros/humble/setup.bash
source ~/openarm_robot_ws/install/setup.bash
python3 scripts/deploy/openarm_move_to_task_start.py \
  --config source/isaaclab_tasks/isaaclab_tasks/direct/openarm_motion_context/deploy/configs/lift_real_calibrated.yaml
```

Clear the workspace, keep the emergency stop ready, and then explicitly execute
the eight-second transition:

```bash
python3 scripts/deploy/openarm_move_to_task_start.py \
  --config source/isaaclab_tasks/isaaclab_tasks/direct/openarm_motion_context/deploy/configs/lift_real_calibrated.yaml \
  --duration 8.0 \
  --execute
```

Only start policy deployment after the command reports that both arms reached
the configured pose. The target joint values come from the same
`default_joint_pos` used to center incremental actions during deployment.

### Terminal 2: cameras and AprilTag detection

Start all three D435i drivers using their fixed serial numbers and launch one
AprilTag detector for each color stream. The resulting topics and optical
frames must match `lift_real_calibrated.yaml`:

```text
/left_d435i/tag_detections   -> realsense_d435i_left_color_optical_frame
/right_d435i/tag_detections  -> realsense_d435i_right_color_optical_frame
/chest_d435i/tag_detections  -> realsense_d435i_color_optical_frame
```

Check that detections and the chest camera transform are present:

```bash
ros2 topic echo /left_d435i/tag_detections --once
ros2 topic echo /right_d435i/tag_detections --once
ros2 topic echo /chest_d435i/tag_detections --once
ros2 run tf2_ros tf2_echo openarm_body_link0 realsense_d435i_color_optical_frame
```

Run the USD-to-real camera mount consistency check:

```bash
cd ~/openarm-motion-context
source /opt/ros/humble/setup.bash
source ~/openarm_robot_ws/install/setup.bash
third_party/IsaacLab/isaaclab.sh -p \
  source/isaaclab_tasks/isaaclab_tasks/direct/openarm_motion_context/deploy/camera_tf_check.py \
  --config source/isaaclab_tasks/isaaclab_tasks/direct/openarm_motion_context/deploy/configs/lift_real_calibrated.yaml
```

### Terminal 3: policy deployment

Start with dry-run mode. It performs subscriptions and inference but does not
publish robot commands:

```bash
cd ~/openarm-motion-context
source /opt/ros/humble/setup.bash
source ~/openarm_robot_ws/install/setup.bash
third_party/IsaacLab/isaaclab.sh -p -m \
  isaaclab_tasks.direct.openarm_motion_context.deploy.deploy_node \
  --config source/isaaclab_tasks/isaaclab_tasks/direct/openarm_motion_context/deploy/configs/lift_real_calibrated.yaml \
  --checkpoint artifacts/checkpoints/lift_motion_context/best_agent.pt \
  --dry-run
```

Only after joint ordering, transforms, tag IDs, target positions, context scale,
and action limits are verified should command publishing be enabled:

```bash
cd ~/openarm-motion-context
source /opt/ros/humble/setup.bash
source ~/openarm_robot_ws/install/setup.bash
third_party/IsaacLab/isaaclab.sh -p -m \
  isaaclab_tasks.direct.openarm_motion_context.deploy.deploy_node \
  --config source/isaaclab_tasks/isaaclab_tasks/direct/openarm_motion_context/deploy/configs/lift_real_calibrated.yaml \
  --checkpoint artifacts/checkpoints/lift_motion_context/best_agent.pt \
  --publish
```

Do not run the task-start mover and policy publisher at the same time. Keep the
physical emergency stop ready throughout initialization and deployment.

## License

Code is provided under the BSD-3-Clause license inherited from Isaac Lab. USD,
mesh, camera and robot assets are separate artifacts and may have different
license requirements.
