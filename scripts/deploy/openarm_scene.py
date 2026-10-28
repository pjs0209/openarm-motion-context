#!/usr/bin/env python3
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Launch the local browser dashboard without importing Isaac Lab or a policy."""

import runpy
from pathlib import Path

if __name__ == "__main__":
    runpy.run_path(
        str(
            Path(__file__).resolve().parents[2]
            / "source/isaaclab_tasks/isaaclab_tasks/direct/openarm_motion_context/deploy/visualization.py"
        ),
        run_name="__main__",
    )
