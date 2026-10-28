"""Circle v13: physically oriented opposed grasps, separate from historical v7.

The reset pose is a joint-limit-constrained kinematic candidate. It must pass
the simulator's fixed-joint, grasp-side and collision preflight before training.
"""

import copy
import math
import os

from isaaclab.assets import ArticulationCfg
from isaaclab.utils import configclass

from .peg_in_hole_env import OpenArmCirclePegInHoleSceneCfg, OpenArmFixedCirclePegAndHoleV7EnvCfg
from .robot_cfg import OPENARM_ROBOT_ASSET_DIR, OPEN_ARM_HIGH_PD_CFG


def _opposed_grasp_robot_cfg() -> ArticulationCfg:
    robot = copy.deepcopy(OPEN_ARM_HIGH_PD_CFG)
    robot.prim_path = "{ENV_REGEX_NS}/Robot"
    robot.spawn.activate_contact_sensors = False
    robot.spawn.usd_path = os.path.join(OPENARM_ROBOT_ASSET_DIR, "openarm_robot_circle_v13.usda")
    # Opposed TCP axes: left points across the gap (-Y), right points back
    # toward the hole body (+Y). The insertion/depth axis is -Y on both parts.
    # Offline USD joint-frame search kept both TCPs within 9 mm of v7 reset.
    left_deg = (-58.81, -46.31, 52.09, 54.21, 24.91, -5.24, -86.5)
    right_deg = (58.81, 46.31, -52.09, 54.21, -24.91, 5.24, 86.5)
    for side, angles in (("left", left_deg), ("right", right_deg)):
        for number, degrees in enumerate(angles, 1):
            robot.init_state.joint_pos[f"openarm_{side}_joint{number}"] = math.radians(degrees)
    # Keep robot links from passing through one another in this new physics
    # version; the fixed-joint connected-body collision setting is separate.
    robot.spawn.articulation_props.enabled_self_collisions = True
    return robot


@configclass
class OpenArmCirclePegInHoleV13SceneCfg(OpenArmCirclePegInHoleSceneCfg):
    robot: ArticulationCfg = _opposed_grasp_robot_cfg()


@configclass
class OpenArmFixedCirclePegAndHoleV13EnvCfg(OpenArmFixedCirclePegAndHoleV7EnvCfg):
    """V7 rewards/actions with a front-facing hole and backside right grasp."""

    scene: OpenArmCirclePegInHoleV13SceneCfg = OpenArmCirclePegInHoleV13SceneCfg(num_envs=64, env_spacing=3.0)
    # The opposed/backside grasp has less reset-orientation margin than V7.
    # A 64-env fixed-joint preflight with the inherited 0.03 rad noise
    # exceeded the entrance-lateral gate; 0.01 rad passed before and after
    # an explicit reset while preserving randomized initial joint poses.
    reset_arm_joint_noise_rad: float = 0.01
    require_backside_hole_grasp: bool = True
