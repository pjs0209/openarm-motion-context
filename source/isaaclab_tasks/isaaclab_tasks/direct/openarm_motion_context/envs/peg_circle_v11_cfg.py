"""Circle v11: reward holding a well-aligned insertion near target depth."""

from isaaclab.utils import configclass

from .peg_circle_v10_cfg import OpenArmFixedCirclePegAndHoleV10EnvCfg


@configclass
class OpenArmFixedCirclePegAndHoleV11EnvCfg(OpenArmFixedCirclePegAndHoleV10EnvCfg):
    """Keep v10 physics and guard; add a late-stage settling incentive."""

    reward_insert_settle_scale = 8.0
    insert_settle_axis_width = 0.006
    insert_settle_speed_std = 0.05
