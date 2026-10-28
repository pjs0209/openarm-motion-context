"""Circle v14: keep v13 geometry and rewards, reduce unsafe reset noise.

The v13 0.01 rad reset noise failed the 64-environment physical alignment
gate at seed 46. An exploratory 0.005 rad candidate failed at seed 49.
The 0.003 rad candidate passed the same 50-step/reset-at-25 gate for seeds
42--49. Its registered task is versioned separately so v13 artifacts remain
reproducible and its RL performance can be evaluated independently.
"""

from isaaclab.utils import configclass

from .peg_circle_v13_cfg import OpenArmFixedCirclePegAndHoleV13EnvCfg


@configclass
class OpenArmFixedCirclePegAndHoleV14EnvCfg(OpenArmFixedCirclePegAndHoleV13EnvCfg):
    reset_arm_joint_noise_rad: float = 0.003
