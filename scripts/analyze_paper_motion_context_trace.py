#!/usr/bin/env python3
"""Standalone launcher for the reorganized motion-context analysis."""

from pathlib import Path
import runpy


_ANALYZER = (
    Path(__file__).resolve().parents[1]
    / "source/isaaclab_tasks/isaaclab_tasks/direct/openarm_motion_context/eval/analyze_motion_context.py"
)


def main() -> None:
    """Run the pure log analyzer without importing Isaac Sim task packages."""

    namespace = runpy.run_path(str(_ANALYZER))
    namespace["main"]()


if __name__ == "__main__":
    main()
