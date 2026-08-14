# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""OpenArm proprioceptive motion-context tasks and deployment tools."""

import gymnasium as gym

from . import train


_LIFT_REGISTRATION = {
    "entry_point": f"{__name__}.envs.lift_env:OpenArmReIncrementalBimanualLiftEnv",
    "disable_env_checker": True,
    "kwargs": {
        "env_cfg_entry_point": f"{__name__}.envs.lift_env:OpenArmReIncrementalBimanualLiftEnvCfg",
        "skrl_mappo_cfg_entry_point": f"{train.__name__}.configs:lift_mappo.yaml",
    },
}

_PEG_REGISTRATION = {
    "entry_point": f"{__name__}.envs.peg_in_hole_env:OpenArmFixedPegAndHoleEnv",
    "disable_env_checker": True,
    "kwargs": {
        "env_cfg_entry_point": f"{__name__}.envs.peg_in_hole_env:OpenArmFixedPegAndHoleEnvCfg",
        "skrl_mappo_cfg_entry_point": f"{train.__name__}.configs:peg_in_hole_mappo.yaml",
    },
}

gym.register(id="Isaac-OpenArm-Lift-v0", **_LIFT_REGISTRATION)
gym.register(id="Isaac-OpenArm-PegInHole-v0", **_PEG_REGISTRATION)
