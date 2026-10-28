# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Newton-only USD authoring at the robot import boundary, before model build."""

import math

from isaaclab_newton.physics import NewtonCfg


def _correct_legacy_v1_finger_inertia(prim):
    """Apply upstream V1 #87 only to the matching legacy mechanical model.

    Source: enactic/openarm_description commit
    62ec8259b1513d9212d0dacc7c496f52b62be30f,
    assets/end_effector/parallel_link/config/inertials.yaml.
    The correction zeros products of inertia in the link frame. USD instead
    stores principal moments and principalAxes, so both must be replaced.
    Mass, COM, link transforms, geometry and the newer V1 mass allocation stay
    untouched. Unknown models are left for the normal preflight to validate.
    """
    from pxr import Gf, UsdPhysics

    if prim.GetName() not in (
        "openarm_left_left_finger",
        "openarm_left_right_finger",
        "openarm_right_left_finger",
        "openarm_right_right_finger",
    ):
        return
    props = UsdPhysics.MassAPI(prim)
    mass, com = props.GetMassAttr().Get(), props.GetCenterOfMassAttr().Get()
    diagonal, axes = props.GetDiagonalInertiaAttr().Get(), props.GetPrincipalAxesAttr().Get()
    if mass is None or com is None or diagonal is None or axes is None:
        return
    if not math.isclose(mass, 0.03602545343277134, rel_tol=1e-6):
        return
    if not all(
        math.isclose(abs(a), b, rel_tol=1e-6, abs_tol=1e-9) for a, b in zip(com, (0.0064528, 0.01702, 0.0219685))
    ):
        return
    rotation = Gf.Matrix3d(Gf.Quatd(axes))
    principal = Gf.Matrix3d(0)
    principal.SetDiagonal(Gf.Vec3d(diagonal))
    tensor = rotation.GetTranspose() * principal * rotation
    expected_diagonal = (2.375e-6, 2.375e-6, 0.75e-6)
    if not all(
        math.isclose(tensor[i][j], expected_diagonal[i] if i == j else 1e-6, rel_tol=2e-5, abs_tol=1e-12)
        for i in range(3)
        for j in range(3)
    ):
        return
    props.GetDiagonalInertiaAttr().Set(Gf.Vec3f(*expected_diagonal))
    props.GetPrincipalAxesAttr().Set(Gf.Quatf(1))


def orient_camera_fixed_joints(root):
    """Reverse the two child-first fixed joints without changing their frames.

    Newton requires parent=body0. Swapping both relationships and local frames
    preserves the physical constraint; swapping only body names does not.
    Source USD layers and PhysX instances are never modified by this function.
    """
    from pxr import Usd, UsdPhysics

    pairs = {
        "FixedJoint_left_handneck": ("handneck_camera_left", "openarm_left_link7"),
        "openarm_to_chestmount": ("chest_camera", "openarm_body_link"),
    }
    for prim in Usd.PrimRange(root):
        pair = pairs.get(prim.GetName())
        if pair is None or not prim.IsA(UsdPhysics.FixedJoint):
            continue
        joint = UsdPhysics.FixedJoint(prim)
        body0, body1 = joint.GetBody0Rel().GetTargets(), joint.GetBody1Rel().GetTargets()
        if len(body0) != 1 or len(body1) != 1:
            raise ValueError(f"Camera fixed joint must connect two bodies: {prim.GetPath()}")
        names = (body0[0].name, body1[0].name)
        if names == pair[::-1]:
            continue
        if names != pair:
            raise ValueError(f"Unexpected camera attachment bodies: {prim.GetPath()} {names}")
        pos0, pos1 = joint.GetLocalPos0Attr().Get(), joint.GetLocalPos1Attr().Get()
        rot0, rot1 = joint.GetLocalRot0Attr().Get(), joint.GetLocalRot1Attr().Get()
        joint.GetLocalPos0Attr().Set(pos1)
        joint.GetLocalPos1Attr().Set(pos0)
        joint.GetLocalRot0Attr().Set(rot1)
        joint.GetLocalRot1Attr().Set(rot0)
        joint.GetBody0Rel().SetTargets(body1)
        joint.GetBody1Rel().SetTargets(body0)


def prepare_openarm_newton_asset(root, gravity_compensation=0.0):
    """Repair unambiguous topology, then reject an unvalidated mechanical model.

    ``gravity_compensation`` is recorded by the robot spawner. Zero reproduces
    early Newton runs; one restores the high-PD robot's PhysX gravity exclusion.

    The legacy robot applies collision APIs to instance-root Xforms. PhysX
    accepts this layout, but Newton requires the API on the actual geometry.
    Local overrides preserve the descendant meshes, authored approximation,
    materials and transforms. Only collision instances are de-instanced.

    Recognized legacy V1 finger inertias use the upstream #87 correction.
    Other invalid inertias and mirrored camera bodies require mechanical
    re-authoring, not solver tuning. Do not silently replace them with invented
    masses, spherical inertias, or absolute-valued scales. All edits remain on
    the live stage; the original USD and the PhysX path are unchanged.
    """
    from isaaclab_newton.sim.schemas import MujocoRigidBodyCfg, NewtonSDFCollisionCfg

    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    from isaaclab.sim.schemas import (
        UsdPhysicsMeshCollisionCfg,
        apply_mesh_collision_properties,
        apply_rigid_body_properties,
    )

    from .camera_payloads import bake_reflected_mesh, prepare_camera_payloads

    if not 0.0 <= gravity_compensation <= 1.0:
        raise ValueError("OpenArm Newton gravity compensation must lie in [0, 1]")

    for prim in Usd.PrimRange(root):
        if not prim.IsA(UsdPhysics.RevoluteJoint) or prim.GetName() not in (
            "openarm_left_hand",
            "openarm_right_hand",
            "openarm_left_ee_tcp_joint",
            "openarm_right_ee_tcp_joint",
        ):
            continue
        joint = UsdPhysics.RevoluteJoint(prim)
        if abs(joint.GetLowerLimitAttr().Get()) > 1e-9 or abs(joint.GetUpperLimitAttr().Get()) > 1e-9:
            raise ValueError(f"Expected a locked zero-mass marker joint, not a moving joint: {prim.GetPath()}")
        # Legacy export used [0, ~5.7e-13] degree revolute limits to represent
        # fixed reference frames. A real fixed joint preserves those frames
        # without inventing inertia for an otherwise massless moving body.
        prim.SetTypeName("PhysicsFixedJoint")
        prim.RemoveAPI(UsdPhysics.DriveAPI, "angular")

    orient_camera_fixed_joints(root)
    prepare_camera_payloads(root)
    containers = [
        prim for prim in Usd.PrimRange(root) if prim.HasAPI(UsdPhysics.CollisionAPI) and not prim.IsA(UsdGeom.Gprim)
    ]
    for container in containers:
        enabled = UsdPhysics.CollisionAPI(container).GetCollisionEnabledAttr().Get()
        approximation = UsdPhysics.MeshCollisionAPI(container).GetApproximationAttr().Get()
        # Editing an instance proxy is invalid. Keep the source reference but
        # allow its collision geometry to receive stage-local schema opinions.
        container.SetInstanceable(False)
        geometry = [prim for prim in Usd.PrimRange(container) if prim.IsA(UsdGeom.Gprim)]
        if not geometry:
            raise ValueError(f"Collision container has no geometry: {container.GetPath()}")
        for prim in geometry:
            if prim.HasAPI(UsdPhysics.CollisionAPI):
                raise ValueError(f"Ambiguous nested collision APIs: {prim.GetPath()}")
            UsdPhysics.CollisionAPI.Apply(prim).CreateCollisionEnabledAttr(enabled)
            if prim.IsA(UsdGeom.Mesh) and approximation is not None:
                UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr(approximation)
        container.RemoveAPI(UsdPhysics.CollisionAPI)
        container.RemoveAPI(UsdPhysics.MeshCollisionAPI)

    for prim in Usd.PrimRange(root):
        if not prim.IsA(UsdGeom.Mesh) or not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        body = prim.GetParent()
        while body and not body.HasAPI(UsdPhysics.RigidBodyAPI):
            body = body.GetParent()
        if body:
            bake_reflected_mesh(UsdGeom.Mesh(prim), body)
        if UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get() == "convexDecomposition":
            # Legacy finger decomposition produced thousands of microscopic
            # hulls that MuJoCo cannot compile. Preserve the CAD surface with
            # Newton SDF contacts instead of discarding tiny hulls or filling
            # the finger's concavities with a single convex approximation.
            apply_mesh_collision_properties(
                str(prim.GetPath()),
                [
                    UsdPhysicsMeshCollisionCfg(mesh_approximation_name="none"),
                    NewtonSDFCollisionCfg(sdf_max_resolution=128),
                ],
                stage=prim.GetStage(),
            )
            # The fragment writer treats its default "none" as no opinion;
            # explicitly clear the legacy approximation on this existing USD.
            UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Set("none")

    issues = []
    for prim in Usd.PrimRange(root):
        if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
            continue
        _correct_legacy_v1_finger_inertia(prim)
        path = str(prim.GetPath())
        world = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        if world.GetDeterminant() < 0:
            issues.append(
                f"{path}: mirrored rigid-body frame; bake reflection into geometry and re-author joint frames"
            )
        mass_api = UsdPhysics.MassAPI(prim)
        if (
            prim.GetName() in ("openarm_left_hand", "openarm_right_hand", "openarm_left_ee_tcp", "openarm_right_ee_tcp")
            and mass_api.GetMassAttr().Get() == 0
        ):
            # These fixed, geometry-free marker links have no physical mass.
            # Their legacy 1e-4 placeholder inertia contradicts that contract.
            if any(child.HasAPI(UsdPhysics.CollisionAPI) for child in Usd.PrimRange(prim)):
                issues.append(f"{path}: massless marker unexpectedly owns collision geometry")
            else:
                mass_api = UsdPhysics.MassAPI.Apply(prim)
                mass_api.CreateDiagonalInertiaAttr(Gf.Vec3f(0))
            continue
        if not mass_api or not mass_api.GetMassAttr().HasAuthoredValueOpinion():
            issues.append(f"{path}: camera/payload mass is unspecified; author an intentional mechanical model")
            continue
        mass = mass_api.GetMassAttr().Get()
        inertia = mass_api.GetDiagonalInertiaAttr().Get()
        if mass is None or not math.isfinite(mass) or mass < 0:
            issues.append(f"{path}: invalid mass {mass}")
        elif mass == 0:
            issues.append(f"{path}: zero mass is an unspecified USD mass, not an intentional payload mass")
        elif inertia is not None and any(inertia):
            if not all(math.isfinite(value) and value > 0 for value in inertia):
                issues.append(f"{path}: non-positive or non-finite principal inertia {inertia}")
            elif 2 * max(inertia) > sum(inertia) + 1e-7 * max(inertia):
                issues.append(f"{path}: principal inertia violates the triangle inequality: {inertia}")
    if issues:
        raise ValueError(
            "OpenArm Newton asset preflight failed; training is not validated. "
            "The original USD is unchanged and physics=isaacsim_physx remains available.\n  " + "\n  ".join(issues)
        )
    # MJWarp does not implement PhysX's per-body disableGravity flag. Restore
    # the high-PD robot's intended gravity exclusion through its native schema;
    # the box, ball and peg/hole scene are outside this robot subtree.
    for prim in Usd.PrimRange(root):
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            apply_rigid_body_properties(
                str(prim.GetPath()), [MujocoRigidBodyCfg(gravcomp=gravity_compensation)], stage=prim.GetStage()
            )


def spawn_openarm_robot(prim_path, cfg, translation=None, orientation=None, **kwargs):
    """Spawn the robot with a fail-closed Newton mechanical-model preflight."""
    # Loading USD/spawner implementations while resolving cfgs would preload
    # usd-core before Kit and mix incompatible USD native runtimes under PhysX.
    from isaaclab.sim import SimulationContext
    from isaaclab.sim.spawners.from_files import spawn_from_usd

    root = spawn_from_usd(prim_path, cfg, translation=translation, orientation=orientation, **kwargs)
    if isinstance(SimulationContext.instance().cfg.physics, NewtonCfg):
        prepare_openarm_newton_asset(root, gravity_compensation=cfg.newton_gravity_compensation)
    return root


def spawn_openarm_scene(prim_path, cfg, translation=None, orientation=None, **kwargs):
    """Keep cavities/bore surfaces in Newton and repair missing binding APIs."""
    from isaaclab.sim import SimulationContext
    from isaaclab.sim.spawners.from_files import spawn_from_usd

    root = spawn_from_usd(prim_path, cfg, translation=translation, orientation=orientation, **kwargs)
    if not isinstance(SimulationContext.instance().cfg.physics, NewtonCfg):
        return root
    prepare_openarm_newton_scene(root)
    return root


def prepare_openarm_newton_scene(root):
    """Author scene-local Newton collision representations without editing source layers."""
    from isaaclab_newton.sim.schemas import NewtonSDFCollisionCfg

    from pxr import Usd, UsdGeom, UsdPhysics, UsdShade

    from isaaclab.sim.schemas import UsdPhysicsMeshCollisionCfg, apply_mesh_collision_properties

    for prim in Usd.PrimRange(root):
        if prim.HasRelationship("material:binding") and not prim.HasAPI(UsdShade.MaterialBindingAPI):
            UsdShade.MaterialBindingAPI.Apply(prim)
        if not prim.IsA(UsdGeom.Mesh) or not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        body = prim.GetParent()
        while body and not body.HasAPI(UsdPhysics.RigidBodyAPI):
            body = body.GetParent()
        # The beveled static rear rail produced mesh/SDF contacts up to 18 cm
        # inside a box sitting above it. Giving this rail its own SDF preserves
        # its CAD surface and removes that first-step impulse; do not disable
        # the collider or turn all static geometry into convex approximations.
        rear_rail = str(prim.GetPath()).endswith("/OpenArm_table/Frame_Back")
        if body or rear_rail:
            apply_mesh_collision_properties(
                str(prim.GetPath()),
                [
                    UsdPhysicsMeshCollisionCfg(mesh_approximation_name="none"),
                    NewtonSDFCollisionCfg(sdf_max_resolution=256),
                ],
                stage=prim.GetStage(),
            )
            UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Set("none")
