# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Newton import must preserve geometry/constraints or reject the asset."""

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from pxr import Gf, Usd, UsdGeom, UsdPhysics

from isaaclab_tasks.direct.openarm_motion_context.envs.backend_cfg import configure_openarm_contact_capacity
from isaaclab_tasks.direct.openarm_motion_context.envs.newton_assets import (
    prepare_openarm_newton_asset,
    prepare_openarm_newton_scene,
)
from isaaclab_tasks.direct.openarm_motion_context.envs.newton_peg import add_peg_welds, robot_reset_fk
from isaaclab_tasks.utils import parse_env_cfg


def _robot_stage():
    """Reference an immutable source layer with a genuine USD instance collider."""
    source = Usd.Stage.CreateInMemory()
    root = UsdGeom.Xform.Define(source, "/OpenArm").GetPrim()
    source.SetDefaultPrim(root)
    mesh = UsdGeom.Mesh.Define(source, "/Collider/mesh")
    mesh.CreatePointsAttr([(0, 0, 0), (0.01, 0, 0), (0, 0.02, 0), (0, 0, 0.03)])
    mesh.CreateFaceVertexCountsAttr([3] * 4)
    mesh.CreateFaceVertexIndicesAttr([0, 2, 1, 0, 1, 3, 0, 3, 2, 1, 2, 3])
    for name in ("openarm_left_link7", "handneck_camera_left"):
        body = UsdGeom.Xform.Define(source, f"/OpenArm/{name}").GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(body)
        mass = UsdPhysics.MassAPI.Apply(body)
        mass.CreateMassAttr(1)
        mass.CreateDiagonalInertiaAttr(Gf.Vec3f(0.1))
    collider = source.DefinePrim("/OpenArm/openarm_left_link7/collisions", "Xform")
    collider.GetReferences().AddInternalReference("/Collider")
    collider.SetInstanceable(True)
    UsdPhysics.CollisionAPI.Apply(collider)
    UsdPhysics.MeshCollisionAPI.Apply(collider).CreateApproximationAttr("convexHull")
    joint = UsdPhysics.FixedJoint.Define(source, "/OpenArm/FixedJoint_left_handneck")
    joint.CreateBody0Rel().SetTargets(["/OpenArm/handneck_camera_left"])
    joint.CreateBody1Rel().SetTargets(["/OpenArm/openarm_left_link7"])
    joint.CreateLocalPos0Attr(Gf.Vec3f(0.02, 0.03, 0.04))
    joint.CreateLocalPos1Attr(Gf.Vec3f(-0.01, 0, 0.05))
    joint.CreateLocalRot0Attr(Gf.Quatf(0.70710678, Gf.Vec3f(0, 0, 0.70710678)))
    joint.CreateLocalRot1Attr(Gf.Quatf(1))
    marker = UsdGeom.Xform.Define(source, "/OpenArm/openarm_left_ee_tcp").GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(marker)
    marker_mass = UsdPhysics.MassAPI.Apply(marker)
    marker_mass.CreateMassAttr(0)
    marker_mass.CreateDiagonalInertiaAttr(Gf.Vec3f(1e-4))
    marker.RemoveAPI(UsdPhysics.MassAPI)
    stage = Usd.Stage.CreateInMemory()
    stage.DefinePrim("/Robot").GetReferences().AddReference(source.GetRootLayer().identifier)
    return source, stage


@pytest.mark.parametrize("joint_type", ["revolute", "prismatic"])
def test_reset_fk_composes_both_joint_frames_and_configured_joint_motion(joint_type):
    """A nonzero child offset catches joint-frame omission/inversion mistakes."""
    stage = Usd.Stage.CreateInMemory()
    root = UsdGeom.Xform.Define(stage, "/Robot").GetPrim()
    for name in ("base", "tip"):
        body = UsdGeom.Xform.Define(stage, f"/Robot/{name}").GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(body)
    UsdGeom.Xformable(stage.GetPrimAtPath("/Robot/base")).AddTranslateOp().Set((10, 20, 30))
    joint = (
        UsdPhysics.RevoluteJoint.Define(stage, "/Robot/hinge")
        if joint_type == "revolute"
        else UsdPhysics.PrismaticJoint.Define(stage, "/Robot/slider")
    )
    joint.CreateBody0Rel().SetTargets(["/Robot/base"])
    joint.CreateBody1Rel().SetTargets(["/Robot/tip"])
    joint.CreateLocalPos0Attr(Gf.Vec3f(1, 0, 0))
    joint.CreateLocalPos1Attr(Gf.Vec3f(0, 2, 0))
    joint.CreateAxisAttr("Z" if joint_type == "revolute" else "X")
    pose = robot_reset_fk(root, {"hinge": np.pi / 2, "slider": 0.5})["tip"]
    expected = (13, 20, 30) if joint_type == "revolute" else (11.5, 18, 30)
    np.testing.assert_allclose(pose.ExtractTranslation(), expected, atol=1e-6)
    np.testing.assert_allclose(
        pose.TransformDir(Gf.Vec3d(1, 0, 0)),
        (0, 1, 0) if joint_type == "revolute" else (1, 0, 0),
        atol=1e-6,
    )


@pytest.mark.parametrize("backend", ["newton_mjwarp", "isaacsim_physx"])
def test_only_native_newton_clones_read_semantic_constants_from_the_source_usd(backend):
    from isaaclab.sim.utils import use_stage

    from isaaclab_tasks.direct.openarm_motion_context.envs.peg_in_hole_task_logic import (
        refresh_peg_hole_frames_from_usd,
    )

    cfg = parse_env_cfg("Isaac-OpenArm-PegInHole-Circle-v0", num_envs=2, overrides=[f"physics={backend}"])
    stage = Usd.Stage.CreateInMemory()
    for name in ("circle_peg", "circle_hole"):
        root = UsdGeom.Xform.Define(stage, "/World/envs/env_0/Scene/Environment/" + name)
        root.AddTranslateOp().Set((10, 20, 30))
        frames = {"grap_point": (0, 0, 0)}
        if name == "circle_peg":
            frames["peg_tip"] = (0, 0, 0.08)
        else:
            frames.update(hole_entrance=(0, 0, 0.04), hole_bottom=(0, 0, 0.015))
        for frame_name, position in frames.items():
            UsdGeom.Xform.Define(stage, str(root.GetPath()) + "/" + frame_name).AddTranslateOp().Set(position)
    ctx = {"_printed_target_debug": True}
    for name in ("peg_tip", "peg_grip", "hole_grip", "hole_entrance", "hole_bottom"):
        ctx[name + "_local_pos"] = torch.zeros(2, 3)
        ctx[name + "_local_quat"] = torch.zeros(2, 4)
    ctx.update(hole_depth=torch.zeros(2), peg_marker_length=torch.zeros(2))
    env = SimpleNamespace(
        cfg=cfg,
        device="cpu",
        num_envs=2,
        peg=SimpleNamespace(cfg=cfg.scene.peg),
        hole=SimpleNamespace(cfg=cfg.scene.hole),
    )
    with use_stage(stage):
        if backend == "isaacsim_physx":
            with pytest.raises(RuntimeError, match="env_1.*peg_tip"):
                refresh_peg_hole_frames_from_usd(env, ctx)
        else:
            refresh_peg_hole_frames_from_usd(env, ctx)
            torch.testing.assert_close(ctx["peg_tip_local_pos"], torch.tensor([[0, 0, 0.08]]).repeat(2, 1))
            torch.testing.assert_close(ctx["hole_depth"], torch.full((2,), 0.025))


def test_physical_peg_welds_compile_at_the_grasp_not_the_authored_free_object_pose():
    import newton
    import warp as wp

    builder = newton.ModelBuilder()
    cache, bodies = {}, {}
    for side, role in (("left", "peg"), ("right", "hole")):
        parent = builder.add_link(
            xform=wp.transform((1, 2, 3), wp.quat_identity()),
            mass=1,
            label=f"/World/envs/env_0/Robot/openarm_{side}_ee_tcp",
        )
        builder.add_articulation([builder.add_joint_free(parent)])
        child = builder.add_link(mass=1, label=f"/World/envs/env_0/Scene/Environment/{role}")
        free_root = builder.add_joint_free(child)
        builder.add_articulation([free_root])
        parent_frame = Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(0, 0, 1), 90)).SetTranslateOnly((0.1, 0, 0))
        grip = Gf.Matrix4d().SetTranslate((0, 0.02, 0))
        cache[side] = (parent_frame, grip, role)
        bodies[side] = (parent, child, free_root)
    add_peg_welds(builder, 0, None, None, cfg=None, cache=cache)
    for parent, child, free_root in bodies.values():
        np.testing.assert_allclose(builder.body_q[child].p, (1.12, 2, 3), atol=1e-6)
        np.testing.assert_allclose(wp.quat_rotate(builder.body_q[child].q, wp.vec3(1, 0, 0)), (0, 1, 0), atol=1e-6)
        start = builder.joint_q_start[free_root]
        np.testing.assert_allclose(
            builder.joint_q[start : start + 7], [*builder.body_q[child].p, *builder.body_q[child].q]
        )
    welds = [i for i, name in enumerate(builder.joint_label) if "physical_grasp" in name]
    assert len(welds) == 2
    assert all(builder.joint_articulation[i] == -1 for i in welds)


@pytest.mark.parametrize("approximation", ["convexHull", "convexDecomposition"])
def test_newton_import_preserves_constraint_frames_and_instance_collision_geometry(approximation):
    source, stage = _robot_stage()
    original_layer = source.GetRootLayer().ExportToString()
    root = stage.GetPrimAtPath("/Robot")
    UsdPhysics.MeshCollisionAPI(stage.GetPrimAtPath("/Robot/openarm_left_link7/collisions")).GetApproximationAttr().Set(
        approximation
    )
    mesh_path = "/Robot/openarm_left_link7/collisions/mesh"
    original_points = UsdGeom.Mesh(stage.GetPrimAtPath(mesh_path)).GetPointsAttr().Get()
    outside = UsdGeom.Xform.Define(stage, "/Box").GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(outside)

    prepare_openarm_newton_asset(root, gravity_compensation=1.0)

    joint = UsdPhysics.FixedJoint(stage.GetPrimAtPath("/Robot/FixedJoint_left_handneck"))
    assert joint.GetBody0Rel().GetTargets() == [stage.GetPrimAtPath("/Robot/openarm_left_link7").GetPath()]
    assert joint.GetBody1Rel().GetTargets() == [stage.GetPrimAtPath("/Robot/handneck_camera_left").GetPath()]
    assert joint.GetLocalPos0Attr().Get() == Gf.Vec3f(-0.01, 0, 0.05)
    assert joint.GetLocalPos1Attr().Get() == Gf.Vec3f(0.02, 0.03, 0.04)
    assert joint.GetLocalRot0Attr().Get() == Gf.Quatf(1)
    assert joint.GetLocalRot1Attr().Get() == Gf.Quatf(0.70710678, Gf.Vec3f(0, 0, 0.70710678))
    collider = stage.GetPrimAtPath(mesh_path)
    assert collider.HasAPI(UsdPhysics.CollisionAPI)
    assert UsdPhysics.MeshCollisionAPI(collider).GetApproximationAttr().Get() == (
        "none" if approximation == "convexDecomposition" else "convexHull"
    )
    if approximation == "convexDecomposition":
        assert collider.GetAttribute("newton:sdfMaxResolution").Get() == 128
    assert UsdGeom.Mesh(collider).GetPointsAttr().Get() == original_points
    assert not stage.GetPrimAtPath("/Robot/openarm_left_link7/collisions").HasAPI(UsdPhysics.CollisionAPI)
    marker_mass = UsdPhysics.MassAPI(stage.GetPrimAtPath("/Robot/openarm_left_ee_tcp"))
    assert marker_mass.GetMassAttr().Get() == 0
    assert marker_mass.GetDiagonalInertiaAttr().Get() == Gf.Vec3f(0)
    assert stage.GetPrimAtPath("/Robot/openarm_left_link7").GetAttribute("mjc:gravcomp").Get() == 1.0
    assert not outside.GetAttribute("mjc:gravcomp").HasAuthoredValueOpinion()
    assert source.GetRootLayer().ExportToString() == original_layer
    # Repeated preparation must not swap attachment frames back or duplicate colliders.
    prepared = stage.GetRootLayer().ExportToString()
    prepare_openarm_newton_asset(root, gravity_compensation=1.0)
    assert stage.GetRootLayer().ExportToString() == prepared


@pytest.mark.parametrize("problem", ["inertia", "reflection", "mass", "missing_mass"])
def test_newton_preflight_rejects_invalid_models_without_changing_source_mass(problem):
    source, stage = _robot_stage()
    root = stage.GetPrimAtPath("/Robot")
    body = stage.GetPrimAtPath("/Robot/handneck_camera_left")
    if problem == "inertia":
        UsdPhysics.MassAPI(body).GetDiagonalInertiaAttr().Set(Gf.Vec3f(4, 1, 1))
        message = "triangle inequality"
    elif problem == "reflection":
        UsdGeom.Xformable(body).AddScaleOp().Set(Gf.Vec3f(-1, 1, 1))
        message = "mirrored rigid-body frame"
    elif problem == "mass":
        body.GetAttribute("physics:mass").Set(float("nan"))
        message = "invalid mass"
    else:
        body.RemoveAPI(UsdPhysics.MassAPI)
        message = "mass is unspecified"
    original_source = source.GetRootLayer().ExportToString()
    with pytest.raises(ValueError, match=message):
        prepare_openarm_newton_asset(root)
    assert source.GetRootLayer().ExportToString() == original_source


@pytest.mark.parametrize("variant", ["legacy_v1", "different_mass", "different_axes"])
def test_v1_finger_correction_preserves_mass_com_and_does_not_overwrite_other_models(variant):
    source, stage = _robot_stage()
    names = [f"openarm_{side}_{finger}_finger" for side in ("left", "right") for finger in ("left", "right")]
    for name in names:
        body = UsdGeom.Xform.Define(source, f"/OpenArm/{name}").GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(body)
        mass = UsdPhysics.MassAPI.Apply(body)
        mass.CreateMassAttr(0.07445 if variant == "different_mass" else 0.03602545343277134)
        mass.CreateCenterOfMassAttr(Gf.Vec3f(0.0064528, 0.01702 if "_left_finger" in name else -0.01702, 0.0219685))
        mass.CreateDiagonalInertiaAttr(Gf.Vec3f(3.991918e-6, 1.3750002e-6, 1.3308144e-7))
        mass.CreatePrincipalAxesAttr(
            Gf.Quatf(1)
            if variant == "different_axes"
            else Gf.Quatf(0.9044081, Gf.Vec3f(0.078153096, -0.18867828, 0.37461808))
        )
    original_source = source.GetRootLayer().ExportToString()
    original_mass = UsdPhysics.MassAPI(stage.GetPrimAtPath(f"/Robot/{names[0]}")).GetMassAttr().Get()
    root = stage.GetPrimAtPath("/Robot")
    if variant != "legacy_v1":
        with pytest.raises(ValueError, match="triangle inequality"):
            prepare_openarm_newton_asset(root)
        for name in names:
            mass = UsdPhysics.MassAPI(stage.GetPrimAtPath(f"/Robot/{name}"))
            assert mass.GetDiagonalInertiaAttr().Get() == Gf.Vec3f(3.991918e-6, 1.3750002e-6, 1.3308144e-7)
    else:
        prepare_openarm_newton_asset(root)
        for name in names:
            mass = UsdPhysics.MassAPI(stage.GetPrimAtPath(f"/Robot/{name}"))
            assert mass.GetMassAttr().Get() == original_mass
            assert mass.GetCenterOfMassAttr().Get() == Gf.Vec3f(
                0.0064528, 0.01702 if "_left_finger" in name else -0.01702, 0.0219685
            )
            # Upstream #87 is diagonal in the LINK frame, not the old principal frame.
            assert mass.GetDiagonalInertiaAttr().Get() == Gf.Vec3f(2.375e-6, 2.375e-6, 0.75e-6)
            assert mass.GetPrincipalAxesAttr().Get() == Gf.Quatf(1)
        prepared = stage.GetRootLayer().ExportToString()
        prepare_openarm_newton_asset(root)
        assert stage.GetRootLayer().ExportToString() == prepared
    assert source.GetRootLayer().ExportToString() == original_source


def test_backend_selection_resolves_all_three_task_configurations_without_physx_fields_on_newton():
    for task in ("Isaac-OpenArm-Lift-v0", "Isaac-OpenArm-LiftWithBall-v0", "Isaac-OpenArm-PegInHole-Circle-v0"):
        default = parse_env_cfg(task, num_envs=1)
        selected = parse_env_cfg(task, num_envs=1, overrides=["physics=newton_mjwarp"])
        historical = parse_env_cfg(task, num_envs=1, overrides=["physics=isaacsim_physx"])
        # Resolution runs each task's post-init; the former assignment leaked
        # this PhysX-only field into the Newton cfg or failed before selection.
        assert historical.sim.physics.friction_correlation_distance == 0.00625
        assert type(default.sim.physics) is type(selected.sim.physics)
        assert default.sim.physics.solver_cfg.use_mujoco_contacts is False
        assert default.sim.physics.solver_cfg.nconmax >= 512
        assert default.sim.physics.bvh_constructor_geometry == "sah"
        assert not hasattr(selected.sim.physics, "friction_correlation_distance")
        assert selected.scene.num_envs == 1


def test_newton_contact_workspaces_cover_observed_training_demands_and_preserve_larger_overrides():
    cfg = parse_env_cfg("Isaac-OpenArm-PegInHole-Circle-V13-v0", num_envs=64)
    cfg.__post_init__()  # Same finalization boundary as direct gym and train/play.
    assert cfg.sim.physics.collision_cfg.max_triangle_pairs >= 64 * 65536
    cfg.sim.physics.collision_cfg.max_triangle_pairs = 128 * 65536
    configure_openarm_contact_capacity(cfg.sim.physics, 1)
    assert cfg.sim.physics.collision_cfg.max_triangle_pairs == 128 * 65536

    # PPO contact poses requested up to 4156 rows/world in the 64-world
    # preflight. A zero-action smoke test never reached those poses.
    assert cfg.sim.physics.solver_cfg.njmax >= 4156


def test_newton_scene_converts_rear_frame_contacts_without_changing_cad_or_unrelated_static_meshes():
    source = Usd.Stage.CreateInMemory()
    root = UsdGeom.Xform.Define(source, "/Scene").GetPrim()
    source.SetDefaultPrim(root)
    for name in ("Frame_Back", "Other"):
        mesh = UsdGeom.Mesh.Define(source, f"/Scene/OpenArm_table/{name}")
        mesh.CreatePointsAttr([(0, 0, 0), (0.7, 0, 0), (0, 0.03, 0), (0, 0, 0.03)])
        mesh.CreateFaceVertexCountsAttr([3] * 4)
        mesh.CreateFaceVertexIndicesAttr([0, 2, 1, 0, 1, 3, 0, 3, 2, 1, 2, 3])
        UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
        UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr("none")
    original = source.GetRootLayer().ExportToString()
    stage = Usd.Stage.CreateInMemory()
    stage.DefinePrim("/Environment").GetReferences().AddReference(source.GetRootLayer().identifier)
    prepare_openarm_newton_scene(stage.GetPrimAtPath("/Environment"))
    frame = stage.GetPrimAtPath("/Environment/OpenArm_table/Frame_Back")
    other = stage.GetPrimAtPath("/Environment/OpenArm_table/Other")
    assert frame.GetAttribute("newton:sdfMaxResolution").Get() == 256
    assert not other.GetAttribute("newton:sdfMaxResolution").HasAuthoredValueOpinion()
    assert (
        UsdGeom.Mesh(frame).GetPointsAttr().Get()
        == UsdGeom.Mesh(source.GetPrimAtPath("/Scene/OpenArm_table/Frame_Back")).GetPointsAttr().Get()
    )
    assert UsdPhysics.CollisionAPI(frame).GetCollisionEnabledAttr().Get()
    assert source.GetRootLayer().ExportToString() == original


def test_openarm_config_resolution_does_not_preload_usd_before_kit():
    # A fresh process is essential: this test file already uses pxr to build
    # in-memory USD fixtures, whereas production resolves cfg before Kit starts.
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import isaaclab_tasks; from isaaclab_tasks.utils import parse_env_cfg; "
                "parse_env_cfg('Isaac-OpenArm-Lift-v0', num_envs=1); "
                "from isaaclab_tasks.utils.hydra import resolve_task_config; "
                "cfg, _ = resolve_task_config('Isaac-OpenArm-LiftWithBall-v0', 'skrl_mappo_cfg_entry_point', "
                "overrides=['env.scene.environment.spawn.func="
                "isaaclab_tasks.direct.openarm_motion_context.envs.ball_contact_mesh:spawn_openarm_mesh_ball_scene']); "
                "from isaaclab.utils import clone; "
                "assert callable(clone(cfg).scene.environment.spawn.func); "
                "lift, agent = resolve_task_config('Isaac-OpenArm-Lift-RotationObs-v0', 'skrl_mappo_cfg_entry_point'); "
                "assert (lift.own_observation_dim, lift.actor_input_dim, lift.state_space) == (33,66,79); "
                "ball, _ = resolve_task_config('Isaac-OpenArm-LiftWithBall-RotationObs-v0', "
                "'skrl_mappo_cfg_entry_point'); "
                "assert (ball.own_observation_dim, ball.actor_input_dim, ball.state_space) == (37,74,87); "
                "assert agent['runtime']['trainer'].endswith(':CameraPoseMotionContextTrainer'); "
                "assert 'pxr' not in sys.modules, 'USD loaded before Kit launcher'"
            ),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_mesh_ball_contact_preserves_visuals_source_layers_and_analytic_mass():
    """A collider approximation must not change rendering, mass or stored USD assets."""
    from isaaclab_tasks.direct.openarm_motion_context.envs.ball_contact_mesh import prepare_mesh_ball_collision
    from isaaclab_tasks.direct.openarm_motion_context.envs.robot_cfg import OPENARM_ENVIRONMENT_ASSET_DIR

    asset = Path(OPENARM_ENVIRONMENT_ASSET_DIR) / "lift/openarm_env_box_ball_matched.usda"
    source = Usd.Stage.Open(str(asset))
    original = source.GetRootLayer().ExportToString()
    stage = Usd.Stage.CreateInMemory()
    root = stage.DefinePrim("/Scene", "Xform")
    root.GetReferences().AddReference(str(asset), "/World/Environment")
    ball = stage.GetPrimAtPath("/Scene/Ball")
    pose = UsdGeom.Xformable(ball).GetLocalTransformation()
    visual = stage.GetPrimAtPath("/Scene/Ball/VisualSphere__")
    visual_attributes = {a.GetName(): a.Get() for a in visual.GetAttributes()}
    metadata = prepare_mesh_ball_collision(root)
    collider = stage.GetPrimAtPath("/Scene/Ball/CollisionSphere")
    assert collider.IsA(UsdGeom.Mesh)
    assert UsdPhysics.CollisionAPI(collider).GetCollisionEnabledAttr().Get()
    assert UsdPhysics.MeshCollisionAPI(collider).GetApproximationAttr().Get() == "none"
    assert collider.GetAttribute("newton:sdfMaxResolution").Get() == 256
    np.testing.assert_allclose(
        np.linalg.norm(np.asarray(UsdGeom.Mesh(collider).GetPointsAttr().Get()), axis=1), 0.039, atol=1e-8, rtol=0
    )
    mass = UsdPhysics.MassAPI(ball)
    assert mass.GetMassAttr().Get() == pytest.approx(2.102)
    # Independent analytic-sphere baseline read back from the native model.
    np.testing.assert_allclose(mass.GetDiagonalInertiaAttr().Get(), [0.0012788567692041397] * 3, rtol=1e-6)
    assert mass.GetCenterOfMassAttr().Get() == Gf.Vec3f(0)
    assert mass.GetPrincipalAxesAttr().Get() == Gf.Quatf(1)
    assert UsdGeom.Xformable(ball).GetLocalTransformation() == pose
    assert {a.GetName(): a.Get() for a in visual.GetAttributes()} == visual_attributes
    assert collider.GetAttribute("visibility").Get() == "invisible"
    assert collider.GetAttribute("physxCollision:contactOffset").Get() == pytest.approx(0.002)
    assert metadata["radial_error_bound_m"] < 0.00005
    assert source.GetRootLayer().ExportToString() == original
    assert source.GetPrimAtPath("/World/Environment/Ball/CollisionSphere").IsA(UsdGeom.Sphere)


def test_reflected_collision_mesh_preserves_world_geometry_and_winding():
    source, stage = _robot_stage()
    container = UsdGeom.Xformable(stage.GetPrimAtPath("/Robot/openarm_left_link7/collisions"))
    container.AddScaleOp().Set(Gf.Vec3f(-2, 1, 1))
    prepare_openarm_newton_asset(stage.GetPrimAtPath("/Robot"))
    mesh = UsdGeom.Mesh(stage.GetPrimAtPath("/Robot/openarm_left_link7/collisions/mesh"))
    world = mesh.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    assert world.GetDeterminant() > 0
    points = np.asarray(mesh.GetPointsAttr().Get())
    actual = (np.c_[points, np.ones(len(points))] @ np.asarray(world))[:, :3]
    np.testing.assert_allclose(actual, [(0, 0, 0), (-0.02, 0, 0), (0, 0.02, 0), (0, 0, 0.03)], atol=1e-8)
    assert mesh.GetOrientationAttr().Get() == "leftHanded"


def test_locked_massless_tcp_export_becomes_fixed_without_changing_attachment_frames():
    source, stage = _robot_stage()
    joint = UsdPhysics.RevoluteJoint.Define(stage, "/Robot/openarm_left_ee_tcp_joint")
    joint.CreateBody0Rel().SetTargets(["/Robot/openarm_left_link7"])
    joint.CreateBody1Rel().SetTargets(["/Robot/openarm_left_ee_tcp"])
    joint.CreateLowerLimitAttr(0)
    joint.CreateUpperLimitAttr(5.7295775e-13)
    joint.CreateLocalPos0Attr(Gf.Vec3f(0, 0, 0.093))
    joint.CreateLocalPos1Attr(Gf.Vec3f(0))
    prepare_openarm_newton_asset(stage.GetPrimAtPath("/Robot"))
    actual = stage.GetPrimAtPath("/Robot/openarm_left_ee_tcp_joint")
    assert actual.IsA(UsdPhysics.FixedJoint)
    assert UsdPhysics.FixedJoint(actual).GetLocalPos0Attr().Get() == Gf.Vec3f(0, 0, 0.093)
    assert UsdPhysics.FixedJoint(actual).GetBody1Rel().GetTargets()[0].name == "openarm_left_ee_tcp"


@pytest.mark.parametrize("payload_name", ["handneck_camera_left", "realsense_d435i"])
def test_payload_model_preserves_optical_frame_and_authors_positive_inertia(payload_name):
    source, stage = _robot_stage()
    if payload_name == "handneck_camera_left":
        body = stage.GetPrimAtPath("/Robot/handneck_camera_left")
        body.RemoveAPI(UsdPhysics.MassAPI)
        body.GetAttribute("physics:mass").Block()
    else:
        body = UsdGeom.Xform.Define(stage, "/Robot/" + payload_name).GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(body)
    UsdGeom.Xformable(body).AddScaleOp().Set(Gf.Vec3f(-1, 1, 1))
    mesh = UsdGeom.Mesh.Define(stage, str(body.GetPath()) + "/mount_mesh")
    prototype = UsdGeom.Mesh(source.GetPrimAtPath("/Collider/mesh"))
    mesh.CreatePointsAttr(prototype.GetPointsAttr().Get())
    mesh.CreateFaceVertexCountsAttr(prototype.GetFaceVertexCountsAttr().Get())
    mesh.CreateFaceVertexIndicesAttr(prototype.GetFaceVertexIndicesAttr().Get())
    optical = UsdGeom.Xform.Define(stage, str(body.GetPath()) + "/optical")
    optical.AddTranslateOp().Set(Gf.Vec3d(0.01, 0.02, 0.03))
    before = optical.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    prepare_openarm_newton_asset(stage.GetPrimAtPath("/Robot"))
    np.testing.assert_allclose(
        np.asarray(optical.ComputeLocalToWorldTransform(Usd.TimeCode.Default())), np.asarray(before)
    )
    assert UsdGeom.Xformable(body).ComputeLocalToWorldTransform(Usd.TimeCode.Default()).GetDeterminant() > 0
    props = UsdPhysics.MassAPI(body)
    if payload_name == "handneck_camera_left":
        assert 0 < props.GetMassAttr().Get() < 0.01
    else:
        assert props.GetMassAttr().Get() == pytest.approx(0.075)
    moments = np.asarray(props.GetDiagonalInertiaAttr().Get())
    assert np.all(moments > 0) and 2 * moments.max() <= moments.sum()
    com = props.GetCenterOfMassAttr().Get()
    assert -0.01 <= com[0] <= 0 and 0 <= com[1] <= 0.02 and 0 <= com[2] <= 0.03


def test_circle_seating_protocol_fits_the_bundled_shaft_and_bore():
    """A changed shaft length or bore must invalidate stale task tolerances."""
    from isaaclab_tasks.direct.openarm_motion_context.envs.robot_cfg import OPENARM_OBJECT_ASSET_DIR

    cfg = parse_env_cfg("Isaac-OpenArm-PegInHole-Circle-Seated-v0", num_envs=1)

    def geometry(filename, first_marker, second_marker):
        stage = Usd.Stage.Open(str(Path(OPENARM_OBJECT_ASSET_DIR) / "peg_in_hole" / filename))
        root = stage.GetDefaultPrim()
        inverse = UsdGeom.Xformable(root).ComputeLocalToWorldTransform(Usd.TimeCode.Default()).GetInverse()

        def position(name):
            pose = UsdGeom.Xformable(stage.GetPrimAtPath(root.GetPath().AppendChild(name)))
            return np.asarray(
                (pose.ComputeLocalToWorldTransform(Usd.TimeCode.Default()) * inverse).ExtractTranslation()
            )

        mesh = UsdGeom.Mesh(next(prim for prim in stage.Traverse() if prim.IsA(UsdGeom.Mesh)))
        transform = mesh.ComputeLocalToWorldTransform(Usd.TimeCode.Default()) * inverse
        points = np.array([transform.Transform(point) for point in mesh.GetPointsAttr().Get()])
        faces, offset = [], 0
        indices = mesh.GetFaceVertexIndicesAttr().Get()
        for count in mesh.GetFaceVertexCountsAttr().Get():
            faces.extend((indices[offset], indices[offset + j], indices[offset + j + 1]) for j in range(1, count - 1))
            offset += count
        return points, points[np.asarray(faces)], position(first_marker), position(second_marker)

    points, _, grip, tip = geometry("circle_peg.usd", "grap_point", "peg_tip")
    axis = (tip - grip) / np.linalg.norm(tip - grip)
    projection = (points - grip) @ axis
    radius = np.linalg.norm(points - grip - projection[:, None] * axis, axis=1)
    shaft_radius = radius[projection > projection.max() - 1e-6].max()
    shoulder = projection[radius > shaft_radius + 1e-5].max()
    protrusion = projection.max() - shoulder
    _, triangles, entrance, bottom = geometry("circle_hole.usd", "hole_entrance", "hole_bottom")
    bore_axis = (bottom - entrance) / np.linalg.norm(bottom - entrance)
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    lengths = np.linalg.norm(normals, axis=1)
    side = (lengths > 1e-9) & (np.abs(normals @ bore_axis) < 1e-6 * lengths)
    distances = np.abs(np.einsum("ij,ij->i", normals, triangles[:, 0] - entrance)) / lengths.clip(1e-12)
    bore_radius = distances[side].min()
    usable_depth = min(protrusion, np.linalg.norm(bottom - entrance))
    assert cfg.insertion_depth_limit == pytest.approx(usable_depth, abs=1e-6)
    assert cfg.peg_collision_radius + 1e-6 >= shaft_radius
    assert 0 < cfg.success_radial_clearance <= bore_radius - cfg.peg_collision_radius
    assert cfg.success_depth_threshold == pytest.approx(0.9 * usable_depth, abs=1e-6)
