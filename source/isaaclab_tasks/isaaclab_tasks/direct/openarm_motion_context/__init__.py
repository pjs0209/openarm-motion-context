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

def _peg_registration(env_cfg_name: str) -> dict:
    """Build one peg geometry registration while sharing policy and environment code."""

    return {
        "entry_point": f"{__name__}.envs.peg_in_hole_env:OpenArmFixedPegAndHoleEnv",
        "disable_env_checker": True,
        "kwargs": {
            "env_cfg_entry_point": f"{__name__}.envs.peg_in_hole_env:{env_cfg_name}",
            "skrl_mappo_cfg_entry_point": f"{train.__name__}.configs:peg_in_hole_mappo.yaml",
        },
    }

gym.register(id="Isaac-OpenArm-Lift-v0", **_LIFT_REGISTRATION)
gym.register(
    id="Isaac-OpenArm-LiftWithBall-v0",
    entry_point=f"{__name__}.envs.lift_with_ball_env:OpenArmLiftWithBallEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.envs.lift_with_ball_env:OpenArmLiftWithBallEnvCfg",
        "skrl_mappo_cfg_entry_point": f"{train.__name__}.configs:lift_mappo.yaml",
    },
)
gym.register(id="Isaac-OpenArm-PegInHole-v0", **_peg_registration("OpenArmFixedPegAndHoleEnvCfg"))
gym.register(id="Isaac-OpenArm-PegInHole-Arrow-v0", **_peg_registration("OpenArmFixedPegAndHoleEnvCfg"))
gym.register(
    id="Isaac-OpenArm-PegInHole-Circle-v0",
    **_peg_registration("OpenArmFixedCirclePegAndHoleEnvCfg"),
)
gym.register(
    id="Isaac-OpenArm-PegInHole-Circle-V7-v0",
    **_peg_registration("OpenArmFixedCirclePegAndHoleV7EnvCfg"),
)
gym.register(
    id="Isaac-OpenArm-PegInHole-Circle-V8-v0",
    **_peg_registration("OpenArmFixedCirclePegAndHoleV8EnvCfg"),
)
gym.register(
    id="Isaac-OpenArm-PegInHole-Circle-V9-v0",
    entry_point=f"{__name__}.envs.peg_in_hole_env:OpenArmFixedPegAndHoleEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.envs.peg_circle_v9_cfg:OpenArmFixedCirclePegAndHoleV9EnvCfg",
        "skrl_mappo_cfg_entry_point": f"{train.__name__}.configs:peg_in_hole_mappo.yaml",
    },
)
gym.register(
    id="Isaac-OpenArm-PegInHole-Circle-V10-v0",
    entry_point=f"{__name__}.envs.peg_in_hole_env:OpenArmFixedPegAndHoleEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.envs.peg_circle_v10_cfg:OpenArmFixedCirclePegAndHoleV10EnvCfg",
        "skrl_mappo_cfg_entry_point": f"{train.__name__}.configs:peg_in_hole_mappo.yaml",
    },
)
gym.register(
    id="Isaac-OpenArm-PegInHole-Circle-V11-v0",
    entry_point=f"{__name__}.envs.peg_in_hole_env:OpenArmFixedPegAndHoleEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.envs.peg_circle_v11_cfg:OpenArmFixedCirclePegAndHoleV11EnvCfg",
        "skrl_mappo_cfg_entry_point": f"{train.__name__}.configs:peg_in_hole_mappo.yaml",
    },
)
gym.register(
    id="Isaac-OpenArm-PegInHole-Hexagon-v0",
    **_peg_registration("OpenArmFixedHexagonPegAndHoleEnvCfg"),
)
gym.register(
    id="Isaac-OpenArm-PegInHole-Circle-V12-v0",
    entry_point=f"{__name__}.envs.peg_in_hole_env:OpenArmFixedPegAndHoleEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.envs.peg_circle_v12_cfg:OpenArmFixedCirclePegAndHoleV12EnvCfg",
        "skrl_mappo_cfg_entry_point": f"{train.__name__}.configs:peg_in_hole_mappo.yaml",
    },
)
gym.register(
    id="Isaac-OpenArm-PegInHole-Circle-V13-v0",
    entry_point=f"{__name__}.envs.peg_in_hole_env:OpenArmFixedPegAndHoleEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.envs.peg_circle_v13_cfg:OpenArmFixedCirclePegAndHoleV13EnvCfg",
        "skrl_mappo_cfg_entry_point": f"{train.__name__}.configs:peg_in_hole_mappo.yaml",
    },
)
gym.register(
    id="Isaac-OpenArm-PegInHole-Circle-V14-v0",
    entry_point=f"{__name__}.envs.peg_in_hole_env:OpenArmFixedPegAndHoleEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.envs.peg_circle_v14_cfg:OpenArmFixedCirclePegAndHoleV14EnvCfg",
        "skrl_mappo_cfg_entry_point": f"{train.__name__}.configs:peg_in_hole_mappo.yaml",
    },
)
gym.register(
    id="Isaac-OpenArm-PegInHole-Circle-Seated-v0",
    entry_point=f"{__name__}.envs.peg_in_hole_env:OpenArmFixedPegAndHoleEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.envs.peg_circle_seated_cfg:OpenArmFixedCirclePegAndHoleSeatedEnvCfg",
        "skrl_mappo_cfg_entry_point": f"{train.__name__}.configs:peg_in_hole_mappo.yaml",
    },
)
gym.register(
    id="Isaac-OpenArm-PegInHole-Circle-Grasp-v0",
    entry_point=f"{__name__}.envs.peg_contact_grasp:OpenArmContactCirclePegAndHoleEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.envs.peg_contact_grasp:OpenArmContactCirclePegAndHoleEnvCfg",
        "skrl_mappo_cfg_entry_point": f"{train.__name__}.configs:peg_in_hole_mappo.yaml",
    },
)
