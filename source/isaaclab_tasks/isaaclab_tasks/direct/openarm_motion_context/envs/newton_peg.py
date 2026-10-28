# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Build physical Peg welds before Newton model finalization, not after reset."""

import math
import re


def robot_reset_fk(root, joint_positions):
    """Resolve USD joint frames at configured reset positions (radians/meters)."""
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    bodies = {p.GetPath(): p for p in Usd.PrimRange(root) if p.HasAPI(UsdPhysics.RigidBodyAPI)}
    joints = [UsdPhysics.Joint(p) for p in Usd.PrimRange(root) if p.IsA(UsdPhysics.Joint)]
    children = {t for j in joints for t in j.GetBody1Rel().GetTargets() if j.GetBody0Rel().GetTargets()}
    poses = {
        path: UsdGeom.Xformable(p).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        for path, p in bodies.items()
        if path not in children
    }

    def frame(position, rotation):
        return Gf.Matrix4d().SetRotate(Gf.Quatd(rotation)).SetTranslateOnly(Gf.Vec3d(position))

    pending = joints[:]
    while pending:
        progressed = False
        for joint in pending[:]:
            parent, child = joint.GetBody0Rel().GetTargets(), joint.GetBody1Rel().GetTargets()
            if not parent or not child or child[0] in poses:
                pending.remove(joint)
                continue
            if parent[0] not in poses:
                continue
            prim = joint.GetPrim()
            value = 0.0
            for expression, configured in joint_positions.items():
                if re.fullmatch(expression, prim.GetName()):
                    value = float(configured)
            motion = Gf.Matrix4d(1)
            if prim.IsA(UsdPhysics.RevoluteJoint) or prim.IsA(UsdPhysics.PrismaticJoint):
                token = prim.GetAttribute("physics:axis").Get()
                axis = {"X": Gf.Vec3d(1, 0, 0), "Y": Gf.Vec3d(0, 1, 0), "Z": Gf.Vec3d(0, 0, 1)}[token]
                if prim.IsA(UsdPhysics.RevoluteJoint):
                    motion.SetRotate(Gf.Rotation(axis, math.degrees(value)))
                else:
                    motion.SetTranslate(axis * value)
            elif not prim.IsA(UsdPhysics.FixedJoint):
                raise ValueError(f"Unsupported joint in OpenArm reset FK: {prim.GetPath()}")
            local0 = frame(joint.GetLocalPos0Attr().Get(), joint.GetLocalRot0Attr().Get())
            local1 = frame(joint.GetLocalPos1Attr().Get(), joint.GetLocalRot1Attr().Get())
            poses[child[0]] = local1.GetInverse() * motion * local0 * poses[parent[0]]
            pending.remove(joint)
            progressed = True
        if pending and not progressed:
            raise ValueError(f"Unresolved robot reset FK joints: {[str(j.GetPath()) for j in pending]}")
    return {path.name: matrix for path, matrix in poses.items()}


def peg_grasp_frames(cfg):
    """Compute immutable grasp frames using the existing task's axis convention."""
    from pxr import Gf, Usd, UsdGeom

    from isaaclab.sim.utils import get_current_stage

    stage = get_current_stage()
    namespace = "/World/envs/env_0"
    robot = stage.GetPrimAtPath(namespace + "/Robot")
    poses = robot_reset_fk(robot, cfg.scene.robot.init_state.joint_pos)

    def resolve(path):
        return (
            path.replace("{ENV_REGEX_NS}", namespace)
            .replace("/World/envs/env_.*/", namespace + "/")
            .replace("/World/envs/env_[^/]+/", namespace + "/")
        )

    def local(path, root):
        prim = stage.GetPrimAtPath(resolve(path))
        if not prim or not root:
            raise ValueError(f"Missing Peg semantic frame or rigid root: {path}, {root}")
        return UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default()) * (
            UsdGeom.Xformable(root).ComputeLocalToWorldTransform(Usd.TimeCode.Default()).GetInverse()
        )

    def outward(side):
        delta = (
            poses[f"openarm_{side}_ee_tcp"].ExtractTranslation() - poses[f"openarm_{side}_link7"].ExtractTranslation()
        )
        if delta.GetLength() < 1e-5:
            raise ValueError("OpenArm TCP must differ from its wrist when constructing a physical grasp")
        return delta.GetNormalized()

    result = {}
    for side, role in (("left", "peg"), ("right", "hole")):
        object_cfg = cfg.scene.peg if role == "peg" else cfg.scene.hole
        root_path = resolve(object_cfg.prim_path)
        body = stage.GetPrimAtPath(root_path)
        if not body:
            raise ValueError(
                f"Peg rigid root was not spawned: configured={object_cfg.prim_path!r}, resolved={root_path!r}"
            )
        grip = local(cfg.peg_grip_prim if role == "peg" else cfg.hole_grip_prim, body)
        if role == "peg":
            axis = local(cfg.peg_tip_prim, body).ExtractTranslation() - grip.ExtractTranslation()
        else:
            axis = (
                local(cfg.hole_bottom_prim, body).ExtractTranslation()
                - local(cfg.hole_entrance_prim, body).ExtractTranslation()
            )
        axis.Normalize()
        tcp = poses[f"openarm_{side}_ee_tcp"]
        nominal = grip.ExtractRotationMatrix().GetInverse() * tcp.ExtractRotationMatrix()
        source = (axis * nominal).GetNormalized()
        target = outward("left") if role == "peg" or cfg.peg_geometry == "circle" else -outward("right")
        dot = max(-1.0, min(1.0, Gf.Dot(source, target)))
        if dot < -0.999999:
            cross = Gf.Cross(source, Gf.Vec3d(1, 0, 0))
            if cross.GetLength() < 1e-6:
                cross = Gf.Cross(source, Gf.Vec3d(0, 1, 0))
            delta = Gf.Quatd(0, cross.GetNormalized())
        else:
            delta = Gf.Quatd(1 + dot, Gf.Cross(source, target)).GetNormalized()
        rotation = nominal * Gf.Matrix3d(delta)
        position = tcp.ExtractTranslation() - grip.ExtractTranslation() * rotation
        object_world = Gf.Matrix4d().SetRotate(rotation).SetTranslateOnly(position)
        parent_frame = grip * object_world * tcp.GetInverse()
        result[side] = (parent_frame, grip, body.GetName())
    return result


def add_peg_welds(builder, world, position, rotation, *, cfg, cache):
    """Use Newton's public world-builder hook to add physical loop welds."""
    import warp as wp
    from newton import JointType

    del world, position, rotation
    if not cache:
        cache.update(peg_grasp_frames(cfg))

    def transform(matrix):
        quat = matrix.ExtractRotation().GetQuat()
        return wp.transform(wp.vec3(*matrix.ExtractTranslation()), wp.quat(*quat.GetImaginary(), quat.GetReal()))

    def body_index(suffix):
        matches = [
            i
            for i, label in enumerate(builder.body_label)
            if label.endswith(suffix) and builder.body_world[i] == builder.current_world
        ]
        if len(matches) != 1:
            raise ValueError(f"Expected one Newton body ending in {suffix!r}, found {matches}")
        return matches[0]

    for side, (parent_frame, grip, object_name) in cache.items():
        parent = body_index(f"/Robot/openarm_{side}_ee_tcp")
        child = body_index(f"/Scene/Environment/{object_name}")
        parent_xform, child_xform = transform(parent_frame), transform(grip)
        # Newton 1.6's MJWarp exporter derives a loop-weld's relative pose
        # from the compile-time body poses, not the supplied child anchor.
        # Put the free object at the intended grasp in the builder's reference
        # configuration. Runtime resets then use that same immutable grasp.
        pose = builder.body_q[parent] * parent_xform * wp.transform_inverse(child_xform)
        builder.body_q[child] = pose
        free_joints = [
            i for i, body in enumerate(builder.joint_child) if body == child and builder.joint_type[i] == JointType.FREE
        ]
        if len(free_joints) != 1:
            raise ValueError(f"Expected one free root for the physical Peg object: {object_name}")
        start = builder.joint_q_start[free_joints[0]]
        builder.joint_q[start : start + 7] = [*pose.p, *pose.q]
        builder.add_joint_fixed(
            parent=parent,
            child=child,
            parent_xform=parent_xform,
            child_xform=child_xform,
            label=f"OpenArm_{side}_physical_grasp",
            collision_filter_parent=True,
        )
