"""Circle v12: strengthen the safe pre-insertion approach inherited from v11."""

from isaaclab.utils import configclass

from .peg_circle_v11_cfg import OpenArmFixedCirclePegAndHoleV11EnvCfg


@configclass
class OpenArmFixedCirclePegAndHoleV12EnvCfg(OpenArmFixedCirclePegAndHoleV11EnvCfg):
    """One reward-scale change; retain settling, physics and success guards."""

    reward_preinsert_scale = 6.0
