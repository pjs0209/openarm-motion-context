"""Evaluation entry point using the shared Isaac Lab play runner.

Use ``scripts/eval.sh`` for command-line evaluation. Trace aggregation and
paper figures are implemented in :mod:`analyze_motion_context`.
"""

from .analyze_motion_context import main

__all__ = ["main"]
