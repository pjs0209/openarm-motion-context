#!/usr/bin/env bash
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ROBOT_ONLY=false
SKIP_ROSDEP=false
for argument in "$@"; do
  case "${argument}" in
    --robot-only) ROBOT_ONLY=true ;;
    --skip-rosdep) SKIP_ROSDEP=true ;;
    --help|-h)
      echo "Usage: bash scripts/deploy/setup.sh [--robot-only] [--skip-rosdep]"
      echo "Build pinned ROS sources and install the ROS Python environment. Requires ROS Humble and uv."
      exit 0 ;;
    *) echo "Unknown argument: ${argument}" >&2; exit 2 ;;
  esac
done

if [[ ! -f /opt/ros/humble/setup.bash ]]; then
  echo "Install ROS 2 Humble on Ubuntu 22.04 first; see the deployment README." >&2
  exit 1
fi
for program in git colcon uv; do
  command -v "${program}" >/dev/null || { echo "Missing command: ${program}" >&2; exit 1; }
done
if [[ "${SKIP_ROSDEP}" == false ]]; then
  command -v rosdep >/dev/null || { echo "Missing command: rosdep" >&2; exit 1; }
fi

# ROS setup hooks may reference unset environment variables.
set +u
source /opt/ros/humble/setup.bash
set -u
cd "${REPO_ROOT}"
modules=(third_party/openarm_ros2 third_party/openarm_can third_party/openarm_description)
paths=(third_party/openarm_ros2/openarm_hardware third_party/openarm_ros2/openarm_bringup
       third_party/openarm_can third_party/openarm_description)
packages=(openarm_can openarm_hardware openarm_description openarm_bringup)
if [[ "${ROBOT_ONLY}" == false ]]; then
  modules+=(third_party/realsense-ros third_party/apriltag_ros third_party/apriltag_msgs)
  paths+=(third_party/realsense-ros/realsense2_camera third_party/realsense-ros/realsense2_camera_msgs
          third_party/apriltag_ros third_party/apriltag_msgs)
  packages+=(realsense2_camera realsense2_camera_msgs apriltag_ros apriltag_msgs)
fi
git submodule sync --recursive -- "${modules[@]}"
git submodule update --init --recursive -- "${modules[@]}"

if [[ "${SKIP_ROSDEP}" == false ]]; then
  # These upstream optional GUI, MoveIt, Gazebo and ZED packages are unused by our V1 launch.
  rosdep install --from-paths "${paths[@]}" --ignore-src --rosdistro humble -y \
    --dependency-types build --dependency-types buildtool --dependency-types build_export --dependency-types exec \
    --skip-keys "openarm_bimanual_moveit_config rviz2 joint_state_publisher_gui joint_state_publisher ros_gz zed_description"
  # The pinned CAN package uses CLI11 but does not declare it in package.xml.
  system_packages=(libcli11-dev)
  if [[ "${ROBOT_ONLY}" == false ]]; then
    system_packages+=(ros-humble-image-proc)
  fi
  sudo apt-get install -y "${system_packages[@]}"
fi

colcon --log-base .openarm-ros/log build --base-paths "${paths[@]}" \
  --build-base .openarm-ros/build --install-base .openarm-ros/install \
  --symlink-install --packages-select "${packages[@]}" \
  --cmake-args -DBUILD_TESTING=OFF -DPython3_EXECUTABLE=/usr/bin/python3 -DPYTHON_EXECUTABLE=/usr/bin/python3

uv venv --allow-existing --python /usr/bin/python3 --system-site-packages .venv-openarm-deploy
uv pip install --python .venv-openarm-deploy/bin/python \
  -r source/isaaclab_tasks/isaaclab_tasks/direct/openarm_motion_context/deploy/requirements.txt
echo "Setup complete. In each terminal: source scripts/deploy/env.sh"
