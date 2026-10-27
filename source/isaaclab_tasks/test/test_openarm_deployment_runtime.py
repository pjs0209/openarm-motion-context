# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Portable deployment regressions; ROS and Isaac Sim are not required."""

import importlib
import runpy
from pathlib import Path

import pytest
import torch
import yaml

runtime = runpy.run_path(
    str(Path(__file__).parents[1] / "isaaclab_tasks/direct/openarm_motion_context/deploy/deploy_node.py")
)
ObservationNormalizer = runtime["ObservationNormalizer"]
apriltag_detector_parameters = runtime["apriltag_detector_parameters"]
joint_position_limits = runtime["joint_position_limits"]
message_is_fresh = runtime["message_is_fresh"]
RunningStandardScaler = importlib.import_module("skrl.resources.preprocessors.torch").RunningStandardScaler


def test_deploy_normalization_matches_training_including_clipping_and_zero_variance():
    scaler = RunningStandardScaler(size=3, device="cpu")
    scaler.running_mean = torch.tensor([1.0, 0.0, -2.0], dtype=torch.float64)
    scaler.running_variance = torch.tensor([4.0, 0.0, 1e-20], dtype=torch.float64)
    scaler.current_count.fill_(100)
    checkpoint = {"left_arm": {"state_preprocessor": scaler.state_dict()}}
    normalizer = ObservationNormalizer(checkpoint, "left_arm", 3, device=torch.device("cpu"))
    states = torch.tensor([[15.0, 1e-10, -1.9999999], [-15.0, -1.0, -2.0]])
    assert normalizer.loaded
    torch.testing.assert_close(normalizer(states), scaler(states, train=False))
    checkpoint["left_arm"]["state_preprocessor"]["running_variance"][0] = -1
    assert not ObservationNormalizer(checkpoint, "left_arm", 3, device=torch.device("cpu")).loaded


def test_detector_exports_metric_sizes_and_distinct_tf_children_for_every_camera():
    path = Path(__file__).parents[1] / "isaaclab_tasks/direct/openarm_motion_context/deploy/configs/lift_real.yaml"
    config = yaml.safe_load(path.read_text())
    frames = []
    for camera in config["frames"]["cameras"]:
        tag = apriltag_detector_parameters(config, camera)["/**"]["ros__parameters"]["tag"]
        assert tag["ids"] == [0, 1, 2]
        assert tag["sizes"] == [0.04, 0.016, 0.016]
        frames.extend(tag["frames"])
    assert len(set(frames)) == 9


def test_live_urdf_limits_follow_controller_order_and_reject_invalid_data():
    urdf = (
        '<robot><joint name="b"><limit lower="-2" upper="3"/></joint>'
        '<joint name="a"><limit lower="0" upper="1"/></joint></robot>'
    )
    assert joint_position_limits(urdf, ["a", "b"]) == ([0.0, -2.0], [1.0, 3.0])
    for invalid in (urdf.replace('lower="0"', 'lower="nan"'), "<robot>", "<robot/>"):
        with pytest.raises(ValueError):
            joint_position_limits(invalid, ["a"])
    assert message_is_fresh(10.0, 9.9, 0.25)
    assert not message_is_fresh(10.0, 9.0, 0.25)
    assert not message_is_fresh(10.0, 10.1, 0.25)
