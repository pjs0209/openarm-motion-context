#!/usr/bin/env python3
"""Launcher for the real-robot deployment node without Isaac Sim imports."""

import runpy
from pathlib import Path


if __name__ == "__main__":
    module_path = (
        Path(__file__).resolve().parents[2]
        / "source/isaaclab_tasks/isaaclab_tasks/direct/openarm_motion_context/deploy/deploy_node.py"
    )
    runpy.run_path(str(module_path), run_name="__main__")
