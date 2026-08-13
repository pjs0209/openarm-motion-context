#!/usr/bin/env python3
"""Compatibility launcher for the reorganized calibration exporter."""

import runpy


if __name__ == "__main__":
    runpy.run_module(
        "isaaclab_tasks.direct.openarm_motion_context.deploy.calibration",
        run_name="__main__",
    )
