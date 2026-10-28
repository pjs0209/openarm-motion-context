# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Circle seating protocol derived from the bundled CAD, separate from V13.

The 28 mm-radius shaft protrudes 20 mm beyond an 80 mm-wide shoulder. The
hole is 25 mm deep, with a minimum bore radius of 30.7336 mm. The shoulder
therefore seats at 20 mm; the historical 22.5 mm success depth requires
interpenetration. All methods must use this same protocol when compared.
"""

from isaaclab.utils import configclass

from .peg_circle_v13_cfg import OpenArmFixedCirclePegAndHoleV13EnvCfg


@configclass
class OpenArmFixedCirclePegAndHoleSeatedEnvCfg(OpenArmFixedCirclePegAndHoleV13EnvCfg):
    """Require 90% seating, a whole-shaft fit and at most 1 mm axial overshoot."""

    insertion_depth_limit: float = 0.020
    success_protocol: str = "circle_cad_seating_v1"
    success_depth_threshold: float = 0.018
    # Round the 2.7336 mm CAD radial clearance downward, leaving 0.2336 mm.
    success_lateral_threshold: float = 0.0025
    success_radial_clearance: float = 0.0025
    peg_collision_radius: float = 0.028
    success_depth_overrun_tolerance: float = 0.001
    depth_overrun_std: float = 0.001
    wall_penetration_lateral_threshold: float = 0.0035
