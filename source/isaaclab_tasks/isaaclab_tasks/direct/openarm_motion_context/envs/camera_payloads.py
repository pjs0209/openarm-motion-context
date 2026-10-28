# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Explicit approximate camera loads for the legacy OpenArm camera assembly.

D435i nominal mass: Intel D400 datasheet, October 2024, table 3-52 (75 g).
Its COM/inertia use a uniform CAD enclosure approximation, not measured data.
Mounts use the user-approved PLA / 20% infill / 2 mm wall approximation:
1 mm voxels, fully dense boundary, homogeneous 20% interior, 1240 kg/m^3.
PLA density: Prusament PLA TDS 2021-10, ISO 1183. Screws/cables are excluded.
"""

import numpy as np


def _write_transform(prim, matrix):
    from pxr import UsdGeom

    xform = UsdGeom.Xformable(prim)
    xform.ClearXformOpOrder()
    xform.AddTransformOp(opSuffix="newtonPayload").Set(matrix)


def _mesh_in_body(mesh, body):
    from pxr import Usd, UsdGeom

    matrix = UsdGeom.Xformable(mesh.GetPrim()).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    matrix *= UsdGeom.Xformable(body).ComputeLocalToWorldTransform(Usd.TimeCode.Default()).GetInverse()
    points = np.asarray(mesh.GetPointsAttr().Get(), dtype=float)
    return (np.c_[points, np.ones(len(points))] @ np.asarray(matrix))[:, :3]


def bake_reflected_mesh(mesh, body):
    """Keep world vertices/UVs/normals while making the mesh frame proper."""
    from pxr import Usd, UsdGeom, Vt

    prim = mesh.GetPrim()
    world = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    if world.GetDeterminant() >= 0:
        return
    body_world = UsdGeom.Xformable(body).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    if body_world.GetDeterminant() <= 0:
        raise ValueError(f"Mesh requires a proper owning body frame: {body.GetPath()}")
    relative = world * body_world.GetInverse()
    points = _mesh_in_body(mesh, body)
    normal_values = mesh.GetNormalsAttr().Get()
    if normal_values:
        normals = np.asarray(normal_values) @ np.linalg.inv(np.asarray(relative)[:3, :3]).T
        normals /= np.linalg.norm(normals, axis=1, keepdims=True).clip(1e-12)
        mesh.GetNormalsAttr().Set(Vt.Vec3fArray.FromNumpy(normals.astype(np.float32)))
    mesh.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(points.astype(np.float32)))
    mesh.GetExtentAttr().Set(Vt.Vec3fArray.FromNumpy(np.array([points.min(0), points.max(0)], dtype=np.float32)))
    orientation = mesh.GetOrientationAttr().Get()
    mesh.GetOrientationAttr().Set("rightHanded" if orientation == "leftHanded" else "leftHanded")
    parent_world = UsdGeom.Xformable(prim.GetParent()).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    _write_transform(prim, body_world * parent_world.GetInverse())


def _printed_mount_properties(mesh, points):
    import trimesh
    from scipy.ndimage import distance_transform_edt

    faces, offset = [], 0
    indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get())
    for count in mesh.GetFaceVertexCountsAttr().Get():
        faces.extend((indices[offset], indices[offset + i], indices[offset + i + 1]) for i in range(1, count - 1))
        offset += count
    shape = trimesh.Trimesh(vertices=points, faces=faces, process=True)
    if not shape.is_watertight:
        raise ValueError(f"Printed camera mount must be a closed CAD mesh: {mesh.GetPath()}")
    pitch = 0.001
    voxels = shape.voxelized(pitch).fill()
    filled = voxels.matrix
    thickness = distance_transform_edt(np.pad(filled, 1), sampling=pitch)[1:-1, 1:-1, 1:-1]
    indices = np.argwhere(filled)
    positions = voxels.indices_to_points(indices)
    masses = np.where(thickness[filled] <= 0.002, 1.0, 0.2) * 1240 * pitch**3
    mass = masses.sum()
    com = (positions * masses[:, None]).sum(axis=0) / mass
    centered = positions - com
    second = (centered * masses[:, None]).T @ centered
    inertia = np.eye(3) * np.trace(second) - second + np.eye(3) * mass * pitch**2 / 6
    return mass, com, inertia


def prepare_camera_payloads(root):
    """Author intentional loads and bake reflected mount geometry on the live stage.

    Preserve mesh world vertices and optical-frame world transforms. A reflected
    rigid-body frame becomes proper; the reflection is baked into its mesh and
    mesh handedness. Fixed attachment frames are then rebuilt from those proper
    body poses, not from an invalid reflected quaternion decomposition.
    """
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    stage = root.GetStage()
    names = (
        "chest_camera",
        "handneck_camera_left",
        "handneck_camera_right",
        "realsense_d435i",
        "realsense_d435i_left",
        "realsense_d435i_right",
    )
    bodies = {}
    for name in names:
        body = stage.GetPrimAtPath(root.GetPath().AppendChild(name))
        if not body:
            continue
        meshes = [UsdGeom.Mesh(p) for p in Usd.PrimRange(body) if p.IsA(UsdGeom.Mesh)]
        if not meshes:
            continue  # Generic/custom models remain the mechanical preflight's responsibility.
        props = UsdPhysics.MassAPI(body)
        if props and props.GetMassAttr().HasAuthoredValueOpinion() and props.GetMassAttr().Get() > 0:
            continue  # Never replace an intentional physical payload model.
        bodies[name] = body
        old_world = UsdGeom.Xformable(body).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        if old_world.GetDeterminant() < 0:
            children = [
                (child, UsdGeom.Xformable(child).GetLocalTransformation())
                for child in body.GetChildren()
                if child.IsA(UsdGeom.Xformable)
            ]
            reflection = Gf.Matrix4d().SetScale(Gf.Vec3d(-1, 1, 1))
            old_local = UsdGeom.Xformable(body).GetLocalTransformation()
            _write_transform(body, reflection * old_local)
            for child, local in children:
                _write_transform(child, local * reflection)
            for mesh in meshes:
                bake_reflected_mesh(mesh, body)
        if len(meshes) != 1:
            raise ValueError(f"Expected one CAD enclosure per camera payload: {body.GetPath()}")
        points = _mesh_in_body(meshes[0], body)
        if name.startswith("realsense_d435i"):
            mass = 0.075
            com = (points.min(0) + points.max(0)) / 2
            dimensions = points.max(0) - points.min(0)
            inertia = np.diag(mass * (np.square(dimensions).sum() - np.square(dimensions)) / 12)
            model = "D435i_75g_uniform_CAD_enclosure_not_measured_COM_or_inertia"
        else:
            mass, com, inertia = _printed_mount_properties(meshes[0], points)
            model = "PLA1240_20pct_infill_2mm_wall_1mm_voxels_not_measured_screws_cables_excluded"
        moments, rotation = np.linalg.eigh(inertia)
        if rotation.size and np.linalg.det(rotation) < 0:
            rotation[:, 0] *= -1
        if mass <= 0 or np.any(moments <= 0) or 2 * moments.max() > moments.sum() + 1e-12:
            raise ValueError(f"Invalid camera payload mass properties: {body.GetPath()}")
        props = UsdPhysics.MassAPI.Apply(body)
        props.CreateMassAttr(float(mass))
        props.CreateCenterOfMassAttr(Gf.Vec3f(*com))
        props.CreateDiagonalInertiaAttr(Gf.Vec3f(*moments))
        props.CreatePrincipalAxesAttr(Gf.Quatf(Gf.Matrix3d(*rotation.T.ravel()).ExtractRotation().GetQuat()))
        body.SetCustomDataByKey("openarm:payloadApproximation", model)

    # Only attachments involving a converted payload are touched.
    for prim in Usd.PrimRange(root):
        if not prim.IsA(UsdPhysics.FixedJoint):
            continue
        joint = UsdPhysics.FixedJoint(prim)
        targets = [joint.GetBody0Rel().GetTargets(), joint.GetBody1Rel().GetTargets()]
        if not all(len(t) == 1 for t in targets):
            continue
        endpoints = []
        for target in targets:
            body = stage.GetPrimAtPath(target[0])
            while body and not body.HasAPI(UsdPhysics.RigidBodyAPI):
                body = body.GetParent()
            endpoints.append(body)
        if not all(endpoints) or not any(body.GetName() in bodies for body in endpoints):
            continue
        parent, child = endpoints
        parent_world = UsdGeom.Xformable(parent).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        child_world = UsdGeom.Xformable(child).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        frame = parent_world * child_world.GetInverse()
        joint.GetBody0Rel().SetTargets([parent.GetPath()])
        joint.GetBody1Rel().SetTargets([child.GetPath()])
        joint.GetLocalPos0Attr().Set(Gf.Vec3f(0))
        joint.GetLocalRot0Attr().Set(Gf.Quatf(1))
        joint.GetLocalPos1Attr().Set(Gf.Vec3f(frame.ExtractTranslation()))
        joint.GetLocalRot1Attr().Set(Gf.Quatf(frame.ExtractRotation().GetQuat()))
