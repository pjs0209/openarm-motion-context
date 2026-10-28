# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Opt-in mesh/SDF ball contact, preserving the authored sphere's mechanics.

The invisible collider selects Newton's mesh/SDF contact route. The rendered
sphere, camera semantics, box, materials, reward and success conditions remain
unchanged. The proxy has a radial approximation budget of 50 micrometers; it
is not exactly the original analytic surface. Existing task defaults and
checkpoints retain their original spawner unless this function is selected.
"""

import math


def prepare_mesh_ball_collision(root):
    """Replace the centered ball collider only in the spawned USD stage.

    Args:
        root: Spawned environment prim containing one Ball/CollisionSphere.

    Returns:
        Recorded mass, radius, analytic inertia and proxy approximation bound.
    """
    import numpy as np
    import trimesh
    from isaaclab_newton.sim.schemas import NewtonSDFCollisionCfg

    from pxr import Gf, Usd, UsdGeom, UsdPhysics, Vt

    from isaaclab.sim.schemas import UsdPhysicsMeshCollisionCfg, apply_mesh_collision_properties

    candidates = [
        p for p in Usd.PrimRange(root) if p.GetName() == "CollisionSphere" and p.GetParent().GetName() == "Ball"
    ]
    if len(candidates) != 1 or not candidates[0].IsA(UsdGeom.Sphere):
        raise ValueError("Mesh ball contact requires one authored Ball/CollisionSphere")
    collider = candidates[0]
    if UsdGeom.Xformable(collider).GetLocalTransformation() != Gf.Matrix4d(1):
        raise ValueError("Mesh ball contact requires a centered collider without local scale or rotation")
    radius = float(UsdGeom.Sphere(collider).GetRadiusAttr().Get())
    mass = UsdPhysics.MassAPI(collider.GetParent())
    value = float(mass.GetMassAttr().Get())
    if not all(math.isfinite(x) and x > 0 for x in (radius, value)):
        raise ValueError("Mesh ball contact requires a finite positive radius and authored mass")
    inertia = 0.4 * value * radius**2
    center = mass.GetCenterOfMassAttr().Get()
    diagonal = mass.GetDiagonalInertiaAttr().Get()
    if tuple(center) not in ((-math.inf,) * 3, (0.0,) * 3) or not (
        tuple(diagonal) == (0.0,) * 3 or all(math.isclose(x, inertia, rel_tol=1e-6) for x in diagonal)
    ):
        raise ValueError("Mesh ball contact requires the centered analytic sphere mass model")
    proxy = trimesh.creation.icosphere(subdivisions=4, radius=radius)
    triangles = proxy.vertices[proxy.faces]
    cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    distance = np.abs((cross * triangles[:, 0]).sum(-1)) / np.linalg.norm(cross, axis=-1)
    error = float(radius - distance.min())
    if not proxy.is_watertight or not proxy.is_winding_consistent or error > 0.00005:
        raise ValueError("Mesh ball collider must be closed with radial error at most 50 micrometers")
    collider.SetTypeName("Mesh")
    mesh = UsdGeom.Mesh(collider)
    mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(proxy.vertices.astype(np.float32)))
    mesh.CreateFaceVertexCountsAttr([3] * len(proxy.faces))
    mesh.CreateFaceVertexIndicesAttr(proxy.faces.astype(np.int32).ravel().tolist())
    mesh.CreateSubdivisionSchemeAttr("none")
    mass.CreateDiagonalInertiaAttr(Gf.Vec3f(inertia))
    mass.CreateCenterOfMassAttr(Gf.Vec3f(0))
    mass.CreatePrincipalAxesAttr(Gf.Quatf(1))
    apply_mesh_collision_properties(
        str(collider.GetPath()),
        [UsdPhysicsMeshCollisionCfg(mesh_approximation_name="none"), NewtonSDFCollisionCfg(sdf_max_resolution=256)],
        stage=collider.GetStage(),
    )
    UsdPhysics.MeshCollisionAPI(collider).GetApproximationAttr().Set("none")
    metadata = {
        "mass_kg": value,
        "radius_m": radius,
        "inertia_kg_m2": inertia,
        "radial_error_bound_m": error,
        "faces": len(proxy.faces),
        "sdf_resolution": 256,
    }
    collider.GetParent().SetCustomDataByKey("openarmMeshBallCollision", metadata)
    return metadata


def spawn_openarm_mesh_ball_scene(prim_path, cfg, translation=None, orientation=None, **kwargs):
    """Spawn the optional WithBall contact profile on Newton MJWarp."""
    from isaaclab_newton.physics import NewtonCfg

    from isaaclab.sim import SimulationContext

    from .newton_assets import spawn_openarm_scene

    if not isinstance(SimulationContext.instance().cfg.physics, NewtonCfg):
        raise ValueError("The optional mesh/SDF ball contact profile requires Newton MJWarp")
    root = spawn_openarm_scene(prim_path, cfg, translation=translation, orientation=orientation, **kwargs)
    prepare_mesh_ball_collision(root)
    return root
