#!/usr/bin/env bash
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Source this file; it must update the calling shell's ROS and Python environment.
_openarm_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
if [[ ! -f "${_openarm_root}/.openarm-ros/install/setup.bash" || ! -f "${_openarm_root}/.venv-openarm-deploy/bin/activate" ]]; then
  echo "Run bash scripts/deploy/setup.sh first." >&2
  unset _openarm_root
  return 1
fi
source /opt/ros/humble/setup.bash
source "${_openarm_root}/.openarm-ros/install/setup.bash"
source "${_openarm_root}/.venv-openarm-deploy/bin/activate"
unset _openarm_root
