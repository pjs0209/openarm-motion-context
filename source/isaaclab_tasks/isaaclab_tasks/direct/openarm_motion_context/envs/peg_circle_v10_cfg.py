"""Circle v10: discourage insertion far beyond the physical hole bottom.

This version changes only the depth reward relative to Circle v9. The
physical scene, success predicate, actor, and other reward terms are intact.
"""

from isaaclab.utils import configclass

from .peg_circle_v9_cfg import OpenArmFixedCirclePegAndHoleV9EnvCfg


@configclass
class OpenArmFixedCirclePegAndHoleV10EnvCfg(OpenArmFixedCirclePegAndHoleV9EnvCfg):
    """Target the measured hole bottom instead of rewarding deeper overshoot."""

    depth_overrun_std = 0.025
