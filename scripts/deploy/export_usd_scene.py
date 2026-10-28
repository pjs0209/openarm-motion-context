#!/usr/bin/env python3
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Extract task USD geometry and authored poses for the deployment browser (requires pxr)."""

import argparse
import json
import math
from pathlib import Path

from pxr import Gf, Usd, UsdGeom


def export_scene(source: Path, objects: dict, base_prim=None, frame="usd_world") -> dict:
    """Export visible triangle/convex-quad meshes and spheres, in metres and Z up.

    Geometry is expressed relative to each object prim's rigid pose; authored scale
    and child transforms are baked into vertices. Poses refer to the USD world or
    an explicitly selected rigid base prim. This reads authored data at default
    time, not runtime simulation state or task reset randomization.
    """
    stage = Usd.Stage.Open(str(source.resolve()))
    if stage is None:
        raise ValueError(f"Cannot open USD stage: {source}")
    units = UsdGeom.GetStageMetersPerUnit(stage)
    if not math.isfinite(units) or units <= 0:
        raise ValueError("USD metres per unit must be positive and finite")
    cache = UsdGeom.XformCache()
    basis = Gf.Matrix4d(1)
    up = UsdGeom.GetStageUpAxis(stage)
    if up == "Y":
        basis.SetRotate(Gf.Rotation(Gf.Vec3d(1, 0, 0), 90))
    elif up != "Z":
        raise ValueError(f"Unsupported USD up axis: {up}")
    origin = Gf.Matrix4d(1)
    if base_prim:
        prim = stage.GetPrimAtPath(base_prim)
        if not prim:
            raise ValueError(f"Base prim not found: {base_prim}")
        origin = cache.GetLocalToWorldTransform(prim)
        transform = Gf.Transform(origin)
        if any(abs(v - 1) > 1e-6 for v in transform.GetScale()):
            raise ValueError("Base prim must have unit scale")
        # A named robot frame already defines its own axes. Up-axis conversion
        # applies only when exporting into the stage's world coordinate system.
        basis = Gf.Matrix4d(1)
    result = {"schema": 1, "source": source.name, "frame": frame, "units": "metres", "up_axis": "Z", "objects": {}}
    for name, path in objects.items():
        root = stage.GetPrimAtPath(path)
        if not root:
            raise ValueError(f"Object prim not found: {path}")
        world = cache.GetLocalToWorldTransform(root) * origin.GetInverse() * basis
        rotation = Gf.Transform(world).GetRotation().GetQuat()
        position = world.ExtractTranslation() * units
        rigid = Gf.Matrix4d(1)
        rigid.SetRotate(rotation)
        rigid.SetTranslateOnly(world.ExtractTranslation())
        inverse = rigid.GetInverse()
        shapes = []
        for prim in Usd.PrimRange(root):
            if not prim.IsA(UsdGeom.Gprim):
                continue
            image = UsdGeom.Imageable(prim)
            if image.ComputeVisibility() == "invisible" or image.ComputePurpose() in ("guide", "proxy"):
                continue
            matrix = cache.GetLocalToWorldTransform(prim) * origin.GetInverse() * basis * inverse
            if prim.IsA(UsdGeom.Mesh):
                mesh = UsdGeom.Mesh(prim)
                points = mesh.GetPointsAttr().Get()
                counts = mesh.GetFaceVertexCountsAttr().Get()
                indices = mesh.GetFaceVertexIndicesAttr().Get()
                holes = set(mesh.GetHoleIndicesAttr().Get() or [])
                if points is None or counts is None or indices is None or sum(counts) != len(indices):
                    raise ValueError(f"Invalid mesh topology: {prim.GetPath()}")
                vertices = [matrix.Transform(Gf.Vec3d(p)) * units for p in points]
                triangles, offset = [], 0
                for face, count in enumerate(counts):
                    polygon = list(indices[offset : offset + count])
                    offset += count
                    if face in holes:
                        continue
                    if count not in (3, 4) or any(i < 0 or i >= len(points) for i in polygon):
                        raise ValueError(f"Only triangles and convex quads are supported: {prim.GetPath()}")
                    if count == 4:
                        corners = [vertices[i] for i in polygon]
                        normal = Gf.Cross(corners[1] - corners[0], corners[2] - corners[1])
                        if any(
                            Gf.Dot(
                                Gf.Cross(
                                    corners[(i + 1) % 4] - corners[i], corners[(i + 2) % 4] - corners[(i + 1) % 4]
                                ),
                                normal,
                            )
                            < -1e-12
                            for i in range(4)
                        ):
                            raise ValueError(f"Concave quad needs triangulation: {prim.GetPath()}")
                    for i in range(1, count - 1):
                        triangles.extend((polygon[0], polygon[i], polygon[i + 1]))
                if mesh.GetOrientationAttr().Get() == "leftHanded":
                    triangles = [
                        index
                        for i in range(0, len(triangles), 3)
                        for index in (triangles[i], triangles[i + 2], triangles[i + 1])
                    ]
                shapes.append(
                    {"type": "mesh", "positions": [round(v, 8) for p in vertices for v in p], "indices": triangles}
                )
            elif prim.IsA(UsdGeom.Sphere):
                # Keep the complete affine transform for non-uniformly scaled spheres.
                values = [matrix[row][col] for row in range(4) for col in range(4)]
                values[12:15] = [v * units for v in values[12:15]]
                shapes.append(
                    {"type": "sphere", "radius": UsdGeom.Sphere(prim).GetRadiusAttr().Get() * units, "matrix": values}
                )
            else:
                raise ValueError(f"Unsupported visible geometry {prim.GetTypeName()}: {prim.GetPath()}")
        if not shapes:
            raise ValueError(f"No visible geometry at {path}; check USD references/payloads")
        result["objects"][name] = {
            "prim": path,
            "pose": {"position": list(position), "quaternion": [*rotation.GetImaginary(), rotation.GetReal()]},
            "shapes": shapes,
        }
    # Fail on malformed numeric assets before writing a partial bundle.
    json.dumps(result, allow_nan=False)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--usd", type=Path, required=True)
    parser.add_argument(
        "--object", action="append", required=True, metavar="NAME=/Prim/Path", help="Repeat for box, ball, peg or hole"
    )
    parser.add_argument("--base-prim", help="USD prim defining the calibrated robot base; omit for USD world")
    parser.add_argument("--frame", default="usd_world", help="Coordinate frame of the exported poses")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    objects = {}
    for item in args.object:
        name, separator, path = item.partition("=")
        if separator != "=" or name not in ("box", "ball", "peg", "hole") or name in objects:
            parser.error("Use unique box, ball, peg or hole names followed by =/Prim/Path")
        objects[name] = path
    if bool(args.base_prim) != (args.frame != "usd_world"):
        parser.error("Set --base-prim and a calibrated --frame together, or export in usd_world")
    result = export_scene(args.usd, objects, args.base_prim, args.frame)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, separators=(",", ":"), allow_nan=False) + "\n")
    print(f"Exported {', '.join(objects)} from {args.usd.name} to {args.output} ({args.frame})")


if __name__ == "__main__":
    main()
