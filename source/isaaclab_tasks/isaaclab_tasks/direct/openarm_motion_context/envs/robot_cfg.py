"""양팔 lift 환경에서 사용하는 OpenArm articulation 설정.

task는 양쪽 7-DoF arm과 팔마다 gripper command 1개를 제어한다. USD path,
초기 joint pose, actuator limit, PD gain을 이 파일에 모아두면 reward/observation
코드가 low-level robot tuning을 알 필요가 없다.
"""

from __future__ import annotations

import math
import os
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg


def _resolve_openarm_asset_dirs() -> tuple[str, str, str, str]:
    """Resolve separate robot, environment, object, and sensor asset directories."""

    legacy = os.environ.get("OPENARM_ASSET_DIR")
    robot_override = os.environ.get("OPENARM_ROBOT_ASSET_DIR")
    environment_override = os.environ.get("OPENARM_ENVIRONMENT_ASSET_DIR")
    object_override = os.environ.get("OPENARM_OBJECT_ASSET_DIR")
    sensor_override = os.environ.get("OPENARM_SENSOR_ASSET_DIR")

    from isaaclab_assets import ISAACLAB_ASSETS_DATA_DIR

    data_dir = Path(ISAACLAB_ASSETS_DATA_DIR)
    packaged_dirs = {
        "robot": data_dir / "Robots" / "OpenArm",
        "environment": data_dir / "Environments" / "OpenArm",
        "object": data_dir / "Objects" / "OpenArm",
        "sensor": data_dir / "Sensors" / "OpenArm",
    }
    overrides = {
        "robot": robot_override,
        "environment": environment_override,
        "object": object_override,
        "sensor": sensor_override,
    }

    # OPENARM_ASSET_DIR is a deprecated compatibility option. It is accepted
    # only when it already follows the new category layout; a stale flat asset
    # directory must not shadow the packaged assets bundled with this task.
    legacy_root = Path(legacy).expanduser() if legacy else None
    legacy_dirs = {
        "robot": legacy_root / "Robots" / "OpenArm" if legacy_root else None,
        "environment": legacy_root / "Environments" / "OpenArm" if legacy_root else None,
        "object": legacy_root / "Objects" / "OpenArm" if legacy_root else None,
        "sensor": legacy_root / "Sensors" / "OpenArm" if legacy_root else None,
    }

    resolved_dirs: dict[str, Path] = {}
    for kind, packaged_dir in packaged_dirs.items():
        override = overrides[kind]
        if override:
            resolved_dirs[kind] = Path(override).expanduser()
        elif packaged_dir.is_dir():
            resolved_dirs[kind] = packaged_dir
        elif legacy_dirs[kind] is not None and legacy_dirs[kind].is_dir():
            resolved_dirs[kind] = legacy_dirs[kind]
        else:
            resolved_dirs[kind] = packaged_dir

    robot_dir = resolved_dirs["robot"]
    environment_dir = resolved_dirs["environment"]
    object_dir = resolved_dirs["object"]
    sensor_dir = resolved_dirs["sensor"]

    required = {
        "robot": (robot_dir, ("openarm_robot_with_camera.usda",)),
        "environment": (
            environment_dir,
            ("lift/openarm_env_box.usd", "peg_in_hole/openarm_env_peg_in_hole.usd"),
        ),
        "object": (object_dir, ("peg_in_hole/peg_usd.usdc", "peg_in_hole/hole_usd.usdc")),
        "sensor": (sensor_dir, ("cameras/realsense_d435i.usd",)),
    }
    missing = [
        f"{kind}: {directory / filename}"
        for kind, (directory, filenames) in required.items()
        for filename in filenames
        if not (directory / filename).is_file()
    ]
    if missing:
        raise RuntimeError("Missing OpenArm assets:\n  " + "\n  ".join(missing))
    return (
        str(robot_dir.resolve()),
        str(environment_dir.resolve()),
        str(object_dir.resolve()),
        str(sensor_dir.resolve()),
    )


(
    OPENARM_ROBOT_ASSET_DIR,
    OPENARM_ENVIRONMENT_ASSET_DIR,
    OPENARM_OBJECT_ASSET_DIR,
    OPENARM_SENSOR_ASSET_DIR,
) = _resolve_openarm_asset_dirs()


_START_POSE_DEG = {
    # object를 중심으로 양팔이 mirror된 초기 자세다. gripper를 미리 닫지 않으면서
    # policy가 접근 가능한 초기 configuration을 제공한다.
    "openarm_left_joint1": -30.0,
    "openarm_left_joint2": -20.0,
    "openarm_left_joint3": 20.0,
    "openarm_left_joint4": 130.0,
    "openarm_left_joint5": -5.5,
    "openarm_left_joint6": -3.0,
    "openarm_left_joint7": 89.0,
    "openarm_left_finger_joint1": 0.04,
    "openarm_right_joint1": 30.0,
    "openarm_right_joint2": 20.0,
    "openarm_right_joint3": -20.0,
    "openarm_right_joint4": 130.0,
    "openarm_right_joint5": 5.5,
    "openarm_right_joint6": 3.0,
    "openarm_right_joint7": -89.0,
    "openarm_right_finger_joint1": 0.04,
}

_START_POSE = {
    # arm joint는 읽기 쉽게 degree로 적은 뒤 radian으로 변환한다.
    # finger joint 값은 USD joint position에서 쓰는 단위 그대로 둔다.
    key: value if "finger" in key else math.radians(value) for key, value in _START_POSE_DEG.items()
}

OPEN_ARM_CFG = ArticulationCfg(
    # USD asset이 link/joint name의 source of truth다. task logic은 이 이름으로
    # left/right arm과 finger state를 slicing한다.
    spawn=sim_utils.UsdFileCfg(
        usd_path=os.path.join(OPENARM_ROBOT_ASSET_DIR, "openarm_robot_with_camera.usda"),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            # velocity iteration을 0보다 크게 두면 stiff position drive에서 penetration artifact를 줄이는 데 도움이 된다.
            solver_velocity_iteration_count=1,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "openarm_left_joint1": _START_POSE["openarm_left_joint1"],
            "openarm_left_joint2": _START_POSE["openarm_left_joint2"],
            "openarm_left_joint3": _START_POSE["openarm_left_joint3"],
            "openarm_left_joint4": _START_POSE["openarm_left_joint4"],
            "openarm_left_joint5": _START_POSE["openarm_left_joint5"],
            "openarm_left_joint6": _START_POSE["openarm_left_joint6"],
            "openarm_left_joint7": _START_POSE["openarm_left_joint7"],
            "openarm_right_joint1": _START_POSE["openarm_right_joint1"],
            "openarm_right_joint2": _START_POSE["openarm_right_joint2"],
            "openarm_right_joint3": _START_POSE["openarm_right_joint3"],
            "openarm_right_joint4": _START_POSE["openarm_right_joint4"],
            "openarm_right_joint5": _START_POSE["openarm_right_joint5"],
            "openarm_right_joint6": _START_POSE["openarm_right_joint6"],
            "openarm_right_joint7": _START_POSE["openarm_right_joint7"],
            "openarm_left_finger_joint.*": 0.04,
            "openarm_right_finger_joint.*": 0.04,
        },
    ),
    actuators={
        "openarm_arm": ImplicitActuatorCfg(
            # arm action dimension은 아래 14개 joint에 대응한다.
            # gripper action dimension은 별도 gripper actuator에서 처리한다.
            joint_names_expr=[
                "openarm_left_joint[1-7]",
                "openarm_right_joint[1-7]",
            ],
            velocity_limit_sim={
                "openarm_left_joint[1-2]": 2.175,
                "openarm_right_joint[1-2]": 2.175,
                "openarm_left_joint[3-4]": 2.175,
                "openarm_right_joint[3-4]": 2.175,
                "openarm_left_joint[5-7]": 2.61,
                "openarm_right_joint[5-7]": 2.61,
            },
            effort_limit_sim={
                "openarm_left_joint[1-2]": 40.0,
                "openarm_right_joint[1-2]": 40.0,
                "openarm_left_joint[3-4]": 27.0,
                "openarm_right_joint[3-4]": 27.0,
                "openarm_left_joint[5-7]": 7.0,
                "openarm_right_joint[5-7]": 7.0,
            },
            stiffness={
                "openarm_left_joint1": 700.0,
                "openarm_right_joint1": 700.0,
                "openarm_left_joint2": 400.0,
                "openarm_right_joint2": 400.0,
                "openarm_left_joint3": 220.0,
                "openarm_right_joint3": 220.0,
                "openarm_left_joint4": 220.0,
                "openarm_right_joint4": 220.0,
                "openarm_left_joint[5-7]": 50.0,
                "openarm_right_joint[5-7]": 50.0,
            },
            damping={
                "openarm_left_joint1": 70.0,
                "openarm_right_joint1": 70.0,
                "openarm_left_joint2": 40.0,
                "openarm_right_joint2": 40.0,
                "openarm_left_joint3": 22.0,
                "openarm_right_joint3": 22.0,
                "openarm_left_joint4": 22.0,
                "openarm_right_joint4": 22.0,
                "openarm_left_joint[5-7]": 6.0,
                "openarm_right_joint[5-7]": 6.0,
            },
        ),
        "openarm_gripper": ImplicitActuatorCfg(
            # 양쪽 gripper는 같은 actuator gain을 쓴다. reward/observation은 결과 finger
            # position/contact를 읽을 뿐, env action 적용 단계 밖에서 이 joint를 직접 command하지 않는다.
            joint_names_expr=[
                "openarm_left_finger_joint.*",
                "openarm_right_finger_joint.*",
            ],
            velocity_limit_sim=0.2,
            effort_limit_sim=333.33,
            stiffness=2e3,
            damping=1e2,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)


OPEN_ARM_HIGH_PD_CFG = OPEN_ARM_CFG.copy()
"""lift task에서 사용하는 high-PD variant.

기본 asset gain은 simulation sanity check에는 편하지만, 이 task는 multi-env GPU
simulation에서 더 강한 position tracking이 유리하다. 이 copy는 joint name/limit은
그대로 유지하고 gravity/gain만 안정적인 control 쪽으로 바꾼다.
"""
OPEN_ARM_HIGH_PD_CFG.spawn.rigid_props.disable_gravity = True
OPEN_ARM_HIGH_PD_CFG.actuators["openarm_arm"].stiffness = {
    "openarm_left_joint[1-2]": 400.0,
    "openarm_right_joint[1-2]": 400.0,
    "openarm_left_joint[3-4]": 320.0,
    "openarm_right_joint[3-4]": 320.0,
    "openarm_left_joint[5-7]": 220.0,
    "openarm_right_joint[5-7]": 220.0,
}
OPEN_ARM_HIGH_PD_CFG.actuators["openarm_arm"].damping = {
    "openarm_left_joint[1-2]": 80.0,
    "openarm_right_joint[1-2]": 80.0,
    "openarm_left_joint[3-4]": 64.0,
    "openarm_right_joint[3-4]": 64.0,
    "openarm_left_joint[5-7]": 40.0,
    "openarm_right_joint[5-7]": 40.0,
}
OPEN_ARM_HIGH_PD_CFG.actuators["openarm_gripper"].stiffness = 2e3
OPEN_ARM_HIGH_PD_CFG.actuators["openarm_gripper"].damping = 1e2
