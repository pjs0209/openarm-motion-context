# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Protect the browser export's physical units, frame and visible geometry contract."""

import importlib.util
from pathlib import Path

import pytest

pytest.importorskip("pxr")
from pxr import Usd, UsdGeom  # noqa: E402


def test_usd_export_preserves_units_geometry_and_authored_pose(tmp_path):
    """Exercise the real USD boundary with centimetres, Y up, nested scale and a hole."""
    module_path = Path(__file__).resolve().parents[3] / "scripts/deploy/export_usd_scene.py"
    spec = importlib.util.spec_from_file_location("usd_export", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    path = tmp_path / "scene.usda"
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageMetersPerUnit(stage, 0.01)
    UsdGeom.SetStageUpAxis(stage, "Y")
    root = UsdGeom.Xform.Define(stage, "/Object")
    root.AddTranslateOp().Set((100, 200, 300))
    root.AddScaleOp().Set((2, 2, 2))
    mesh = UsdGeom.Mesh.Define(stage, "/Object/Mesh")
    # Two quads: the first is visible, the second is a USD hole face.
    mesh.CreatePointsAttr([(0, 0, 0), (10, 0, 0), (10, 10, 0), (0, 10, 0), (20, 0, 0)])
    mesh.CreateFaceVertexCountsAttr([4, 3])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 3, 1, 4, 2])
    mesh.CreateHoleIndicesAttr([1])
    collision = UsdGeom.Sphere.Define(stage, "/Object/Collision")
    collision.CreateVisibilityAttr("invisible")
    stage.GetRootLayer().Save()
    scene = module.export_scene(path, {"hole": "/Object"})
    obj = scene["objects"]["hole"]
    assert obj["pose"]["position"] == pytest.approx([1, -3, 2])
    assert len(obj["shapes"]) == 1
    shape = obj["shapes"][0]
    assert len(shape["indices"]) == 6  # Excludes the hole; triangulates the visible quad.
    assert shape["positions"][3:6] == pytest.approx([0.2, 0, 0])
    assert shape["positions"][6:9] == pytest.approx([0.2, 0.2, 0])
    # A rotated/scaled object remains local: rotation belongs to the authored pose.
    assert obj["pose"]["quaternion"] == pytest.approx([2**-0.5, 0, 0, 2**-0.5])
    base = UsdGeom.Xform.Define(stage, "/Base")
    base.AddTranslateOp().Set((100, 0, 0))
    stage.GetRootLayer().Save()
    calibrated = module.export_scene(path, {"hole": "/Object"}, "/Base", "robot_base")
    assert calibrated["frame"] == "robot_base"
    assert calibrated["objects"]["hole"]["pose"]["position"] == pytest.approx([0, 2, 3])
