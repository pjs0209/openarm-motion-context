"""Circle v9 candidate: recover approach signal while retaining tilt pressure.

The v8 power-eight shaping increased training axis alignment but suppressed
the lateral/approach reward away from the mouth. This version changes only
that one reward hyperparameter relative to the replay-checked v7 preset.
"""

from isaaclab.utils import configclass

from .peg_in_hole_env import OpenArmFixedCirclePegAndHoleV7EnvCfg


@configclass
class OpenArmFixedCirclePegAndHoleV9EnvCfg(OpenArmFixedCirclePegAndHoleV7EnvCfg):
    """Moderate axis quality shaping; physics and success guard match v7/v8."""

    alignment_shaping_power = 2.0
