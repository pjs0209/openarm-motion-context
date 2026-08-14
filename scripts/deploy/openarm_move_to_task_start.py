#!/usr/bin/env python3
"""Launcher for the guarded real-robot task-start motion.

Load the implementation directly so ROS system Python does not import Isaac
Lab simulation dependencies through the task package initializer.
"""

import runpy
from pathlib import Path


if __name__ == "__main__":
    module_path = (
        Path(__file__).resolve().parents[2]
        / "source/isaaclab_tasks/isaaclab_tasks/direct/openarm_motion_context/deploy/task_start.py"
    )
    runpy.run_path(str(module_path), run_name="__main__")
