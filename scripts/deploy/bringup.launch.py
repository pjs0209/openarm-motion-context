# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Start OpenArm V1 controllers and optional three-camera AprilTag perception."""

from pathlib import Path

import xacro
import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import ComposableNodeContainer, Node
from launch_ros.descriptions import ComposableNode


def _start(context):
    def value(name):
        return LaunchConfiguration(name).perform(context)

    description_share = Path(get_package_share_directory("openarm_description"))
    description = xacro.process_file(
        str(description_share / "assets/robot/openarm_v1.0/urdf/openarm_v10.urdf.xacro"),
        mappings={
            "arm_type": "openarm_v1.0",
            "bimanual": "true",
            "ros2_control": "true",
            "use_fake_hardware": value("use_fake_hardware"),
            "left_can_interface": value("left_can_interface"),
            "right_can_interface": value("right_can_interface"),
            "can_fd": value("can_fd"),
        },
    ).toxml()
    controllers = str(
        Path(get_package_share_directory("openarm_bringup")) / "config/controllers/openarm_bimanual_controllers.yaml"
    )
    actions = [
        Node(
            package="robot_state_publisher",
            executable="robot_state_publisher",
            parameters=[{"robot_description": description}],
        ),
        Node(
            package="controller_manager",
            executable="ros2_control_node",
            parameters=[{"robot_description": description}, controllers],
            output="screen",
        ),
        Node(
            package="controller_manager",
            executable="spawner",
            arguments=[
                "joint_state_broadcaster",
                "left_forward_position_controller",
                "right_forward_position_controller",
                "left_gripper_controller",
                "right_gripper_controller",
                "--controller-manager-timeout",
                "60",
            ],
            output="screen",
        ),
    ]
    if value("start_cameras") != "true":
        return actions

    config_path = Path(value("deployment_config")).expanduser().resolve()
    config = yaml.safe_load(config_path.read_text())
    layout = {
        "left_wrist": ("left_d435i", "realsense_d435i_left"),
        "right_wrist": ("right_d435i", "realsense_d435i_right"),
        "chest": ("chest_d435i", "realsense_d435i"),
    }
    serials = [value(f"{camera}_serial").strip() for camera in layout]
    if any(not serial for serial in serials) or len(set(serials)) != 3:
        raise ValueError("Set three distinct left_wrist_serial, right_wrist_serial and chest_serial camera arguments")
    camera_launch = Path(get_package_share_directory("realsense2_camera")) / "launch/rs_launch.py"
    for (camera, (namespace, name)), serial in zip(layout.items(), serials):
        if config["frames"]["cameras"][camera] != f"{name}_color_optical_frame":
            raise ValueError(f"Unexpected {camera} optical frame in {config_path}")
        detector_config = config_path.with_name(f"{config_path.stem}_{camera}_apriltag.yaml")
        if not detector_config.is_file():
            raise FileNotFoundError(f"Export the checkpoint configuration first: {detector_config}")
        image_base = f"/{namespace}/{name}/color"
        actions.extend(
            [
                IncludeLaunchDescription(
                    PythonLaunchDescriptionSource(str(camera_launch)),
                    launch_arguments={
                        "camera_namespace": namespace,
                        "camera_name": name,
                        "serial_no": f"_{serial}",
                        "enable_color": "true",
                        "enable_depth": "false",
                        "enable_infra1": "false",
                        "enable_infra2": "false",
                    }.items(),
                ),
                ComposableNodeContainer(
                    package="rclcpp_components",
                    executable="component_container",
                    name=f"{camera}_perception",
                    namespace=namespace,
                    composable_node_descriptions=[
                        ComposableNode(
                            package="image_proc",
                            plugin="image_proc::RectifyNode",
                            name="rectify",
                            namespace=namespace,
                            remappings=[
                                ("image", f"{image_base}/image_raw"),
                                ("camera_info", f"{image_base}/camera_info"),
                                ("image_rect", f"{image_base}/image_rect"),
                            ],
                        ),
                        ComposableNode(
                            package="apriltag_ros",
                            plugin="AprilTagNode",
                            name="apriltag",
                            namespace=namespace,
                            parameters=[str(detector_config)],
                            remappings=[
                                ("image_rect", f"{image_base}/image_rect"),
                                ("camera_info", f"{image_base}/camera_info"),
                                ("detections", config["topics"]["apriltag"][camera]),
                            ],
                        ),
                    ],
                    output="screen",
                ),
            ]
        )
    return actions


def generate_launch_description():
    """Keep fake hardware as the default; hardware use requires an explicit argument."""
    arguments = [
        DeclareLaunchArgument("use_fake_hardware", default_value="true", choices=["true", "false"]),
        DeclareLaunchArgument("start_cameras", default_value="false", choices=["true", "false"]),
        DeclareLaunchArgument("left_can_interface", default_value="can1"),
        DeclareLaunchArgument("right_can_interface", default_value="can0"),
        DeclareLaunchArgument("can_fd", default_value="true", choices=["true", "false"]),
        DeclareLaunchArgument("deployment_config", default_value="deployment/lift.yaml"),
        *[
            DeclareLaunchArgument(f"{camera}_serial", default_value="")
            for camera in ("left_wrist", "right_wrist", "chest")
        ],
    ]
    return LaunchDescription([*arguments, OpaqueFunction(function=_start)])
