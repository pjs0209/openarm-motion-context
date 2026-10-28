# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Measure OpenArm gripper response or oracle pose tracking without a learned policy.

The oracle uses simulated target poses for diagnosis only. Its results are not
camera-policy evaluation results. Every run saves actions, measured joints,
target tracking and task gates, including resets that invalidate a continuous
trajectory. Use separate output files for backend and controller comparisons.
"""

import argparse
import hashlib
import json
import sys
from functools import partial
from pathlib import Path

import torch
import warp as wp

wp.config.enable_backward = False

import gymnasium as gym

from isaaclab.app import add_launcher_args, launch_simulation

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.direct.openarm_motion_context.envs.lift_task_logic import compute_collision_target_poses_w
from isaaclab_tasks.direct.openarm_motion_context.envs.peg_in_hole_task_logic import compute_peg_in_hole_metrics
from isaaclab_tasks.direct.openarm_motion_context.legacy_api import read_data, tensor
from isaaclab_tasks.direct.openarm_motion_context.legacy_math import quat_apply, quat_inv, quat_mul
from isaaclab_tasks.utils.hydra import resolve_task_config


def pose_error(pose, target_pos, target_quat):
    """Return world-frame position and shortest rotation-vector error."""
    error_quat = quat_mul(target_quat, quat_inv(pose[:, 3:7]))
    error_quat = torch.where(error_quat[:, :1] < 0, -error_quat, error_quat)
    xyz = error_quat[:, 1:]
    norm = xyz.norm(dim=-1, keepdim=True)
    angle = 2 * torch.atan2(norm, error_quat[:, :1].clamp_min(0))
    rot_error = xyz * angle / norm.clamp_min(1e-8)
    return torch.cat((target_pos - pose[:, :3], rot_error), dim=-1)


def pose_action(env, side, target_pos, target_quat, integral, integral_gain):
    """Track a world-frame TCP pose through the task's real incremental actions."""
    joint_ids = getattr(env, f"{side}_arm_joint_ids")
    body_id = env._ctx[f"{side}_ee_body_id"]
    pose = read_data(env.robot.data, "body_state_w")[:, body_id]
    error = pose_error(pose, target_pos, target_quat)
    if integral_gain > 0.0:
        near = torch.linalg.vector_norm(error[:, :3], dim=-1, keepdim=True) < 0.01
        integral.mul_(near)
        integral.add_(torch.where(near, error[:, :3] * env.step_dt, 0.0)).clamp_(-0.003, 0.003)
        error[:, :3] += integral_gain * integral
    jacobian = tensor(env.robot.data.body_link_jacobian_w)[:, body_id - int(env.robot.is_fixed_base)]
    jacobian = jacobian[:, :, joint_ids]
    current = read_data(env.robot.data, "joint_pos")[:, joint_ids]
    limits = read_data(env.robot.data, "joint_pos_limits")[:, joint_ids]
    default = getattr(env, f"{side}_arm_default_joint_pos")
    lower = torch.maximum(limits[:, :, 0], default - env.cfg.action_scale)
    upper = torch.minimum(limits[:, :, 1], default + env.cfg.action_scale)
    # Remove outward directions at active joint limits before recomputing IK.
    # Merely clipping the result can leave this diagnostic stuck at wrist stops.
    active_jacobian = jacobian.clone()
    for _ in range(len(joint_ids)):
        normal = active_jacobian @ active_jacobian.transpose(-1, -2)
        normal += 0.03**2 * torch.eye(6, device=env.device)
        delta = (active_jacobian.transpose(-1, -2) @ torch.linalg.solve(normal, error.unsqueeze(-1))).squeeze(-1)
        outward = ((current <= lower + 0.015) & (delta < 0)) | ((current >= upper - 0.015) & (delta > 0))
        if not outward.any():
            break
        active_jacobian *= (~outward).unsqueeze(1)
    desired = (current + delta.clamp(-0.08, 0.08)).clamp(lower, upper)
    previous = getattr(env, f"_{side}_arm_incremental_target")
    return ((desired - previous) / float(env.cfg.action_step_scale)).clamp(-1, 1)


def kinematic_reachability(env, args, action_scale):
    """Find bounded joint candidates with FK only; this does not validate contact."""
    import numpy as np
    from scipy.optimize import least_squares

    original = read_data(env.robot.data, "joint_pos").clone()
    limits = read_data(env.robot.data, "joint_pos_limits")
    targets = tuple(value.clone() for value in compute_collision_target_poses_w(env, env._ctx))
    for position in targets[:2]:
        position[:, 2] += args.ik_target_height_offset
    rng = np.random.default_rng(args.seed)
    position_scale = args.ik_position_scale or float(env.cfg.success_grasp_distance)
    rotation_scale = float(np.deg2rad(args.ik_rotation_scale_deg or env.cfg.success_grasp_orientation_deg))
    scales = np.array([position_scale] * 3 + [rotation_scale] * 3)
    result = {
        "kinematics_only": True,
        "contact_validated": False,
        "candidates": {},
        "objective_position_scale_m": position_scale,
        "objective_rotation_scale_rad": rotation_scale,
        "action_scale": action_scale,
        "requested_starts_per_arm": args.ik_starts,
        "task_pose_gate_position_m": float(env.cfg.success_grasp_distance),
        "task_pose_gate_rotation_deg": float(env.cfg.success_grasp_orientation_deg),
        "target_height_offset_m": args.ik_target_height_offset,
    }
    try:
        for index, side in enumerate(("left", "right")):
            ids = getattr(env, f"{side}_arm_joint_ids")
            default = getattr(env, f"{side}_arm_default_joint_pos")[0]
            lower = torch.maximum(default - action_scale, limits[0, ids, 0]).cpu().numpy()
            upper = torch.minimum(default + action_scale, limits[0, ids, 1]).cpu().numpy()

            def residual(values):
                candidate = torch.as_tensor(values, dtype=torch.float32, device=env.device).unsqueeze(0)
                env.robot.write_joint_state_to_sim_index(
                    position=candidate, velocity=torch.zeros_like(candidate), joint_ids=ids
                )
                pose = read_data(env.robot.data, "body_state_w")[:, env._ctx[f"{side}_ee_body_id"]]
                error = pose_error(pose, targets[index], targets[index + 2])[0].cpu().numpy().astype(float)
                return error / scales

            attempts = []
            for start in (default.cpu().numpy(), *(rng.uniform(lower, upper) for _ in range(args.ik_starts - 1))):
                fit = least_squares(
                    residual,
                    np.clip(start, lower + 1e-6, upper - 1e-6),
                    bounds=(lower, upper),
                    diff_step=1e-3,
                    max_nfev=150,
                )
                error = residual(fit.x) * scales
                distance, angle = float(np.linalg.norm(error[:3])), float(np.linalg.norm(error[3:]))
                attempts.append(
                    {
                        "distance_m": distance,
                        "rotation_deg": float(np.rad2deg(angle)),
                        "satisfies_pose_gate": bool(
                            distance < env.cfg.success_grasp_distance
                            and angle < np.deg2rad(env.cfg.success_grasp_orientation_deg)
                        ),
                        "satisfies_requested_accuracy": distance < position_scale and angle < rotation_scale,
                        "optimizer_converged": bool(fit.success),
                        "joint_positions": fit.x.tolist(),
                        "cost": float(np.sum((error / scales) ** 2)),
                    }
                )
                if attempts[-1]["satisfies_requested_accuracy"]:
                    break
            result["candidates"][side] = min(attempts, key=lambda item: item["cost"])
            result.setdefault("attempts", {})[side] = attempts
    finally:
        env.robot.write_joint_state_to_sim_index(position=original, velocity=torch.zeros_like(original))
    return result


def initialize_diagnostic_grasp(env, args):
    """Isolate grasp mechanics from approach using an explicitly recorded IK pose.

    This changes only the robot's initial joint state in the diagnostic. It is
    not a pickup policy, and does not validate a collision-free approach path.
    Ordinary physics and position commands control every subsequent step.
    """
    if args.initial_ik_candidate is None:
        return None
    if args.probe != "lift":
        raise ValueError("An initial IK candidate is supported only for the Lift mechanics diagnostic")
    path = Path(args.initial_ik_candidate)
    reference = json.loads(path.read_text())
    if reference.get("task") != args.task:
        raise ValueError("Initial IK candidates must come from the same task")
    initial = torch.tensor(reference["initial_state"]["object_pose"], device=env.device)
    current = read_data(env.object.data, "root_state_w")[0, :7]
    if (
        not torch.allclose(initial[:3], current[:3], atol=1e-5, rtol=0)
        or min(float((initial[3:] - current[3:]).abs().max()), float((initial[3:] + current[3:]).abs().max())) > 1e-5
    ):
        raise ValueError("Initial IK candidates require the same object pose; do not reuse across reset conditions")
    source_scale = args.initial_ik_scale if args.initial_ik_scale is not None else float(env.cfg.action_scale)
    candidates = reference["kinematic_scale_sweep"][str(source_scale)]["candidates"]
    joints = read_data(env.robot.data, "joint_pos").clone()
    limits = read_data(env.robot.data, "joint_pos_limits")
    for side in ("left", "right"):
        ids = getattr(env, f"{side}_arm_joint_ids")
        target = torch.tensor(candidates[side]["joint_positions"], device=env.device).unsqueeze(0)
        default = getattr(env, f"{side}_arm_default_joint_pos")
        lower = torch.maximum(limits[:, ids, 0], default - env.cfg.action_scale)
        upper = torch.minimum(limits[:, ids, 1], default + env.cfg.action_scale)
        if not torch.isfinite(target).all() or ((target < lower - 1e-6) | (target > upper + 1e-6)).any():
            raise ValueError(f"Initial {side} IK candidate exceeds the actual joint/action bounds")
        joints[:, ids] = target
        getattr(env, f"_{side}_arm_incremental_target").copy_(target)
    env.robot.write_joint_state_to_sim_index(position=joints, velocity=torch.zeros_like(joints))
    env.sim.forward()
    env.scene.update(dt=env.physics_dt)
    return {
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "action_scale": float(env.cfg.action_scale),
        "candidate_source_action_scale": source_scale,
        "initial_candidates": candidates,
        "approach_path_validated": False,
        "object_pose_modified": False,
    }


def load_joint_waypoints(env, args):
    """Load FK waypoints for a full-start diagnostic using ordinary task actions."""
    if args.joint_waypoints is None:
        return None
    if args.probe != "lift" or args.initial_ik_candidate is not None or args.action_trace is not None:
        raise ValueError("Joint waypoints require a full-start Lift probe without initial IK writes or action replay")
    path = Path(args.joint_waypoints)
    specification = json.loads(path.read_text())
    if specification["task"] != args.task:
        raise ValueError("Joint waypoint task identity differs")
    pose = read_data(env.object.data, "root_state_w")[0, :7]
    joints = read_data(env.robot.data, "joint_pos")
    limits = read_data(env.robot.data, "joint_pos_limits")
    result = {"source": str(path), "phases": {}, "initial_joints": {}}
    for side in ("left", "right"):
        result["initial_joints"][side] = joints[0, getattr(env, f"{side}_arm_joint_ids")].tolist()
    for phase in ("pregrasp", "grasp", "lift"):
        source = Path(specification["sources"][phase])
        reference = json.loads(source.read_text())
        initial = torch.tensor(reference["initial_state"]["object_pose"], device=env.device)
        if reference["task"] != args.task or not torch.allclose(initial, pose, atol=1e-5, rtol=0):
            raise ValueError("Joint waypoints require the same task and initial object pose")
        entry = reference["kinematic_scale_sweep"][str(float(env.cfg.action_scale))]
        height = float(entry.get("target_height_offset_m", 0.0))
        if (phase == "grasp" and abs(height) > 1e-6) or (phase == "lift" and abs(height - args.lift_height) > 1e-6):
            raise ValueError("Waypoint heights disagree with the grasp/lift schedule")
        for side, candidate in entry["candidates"].items():
            ids = getattr(env, f"{side}_arm_joint_ids")
            target = torch.tensor(candidate["joint_positions"], device=env.device)
            default = getattr(env, f"{side}_arm_default_joint_pos")[0]
            lower = torch.maximum(limits[0, ids, 0], default - env.cfg.action_scale)
            upper = torch.minimum(limits[0, ids, 1], default + env.cfg.action_scale)
            if not candidate["satisfies_requested_accuracy"] or not torch.isfinite(target).all():
                raise ValueError("Each waypoint needs a finite pose candidate at the requested accuracy")
            if ((target < lower - 1e-6) | (target > upper + 1e-6)).any():
                raise ValueError("Joint waypoint exceeds the current physical/action limits")
        result["phases"][phase] = {
            "source": str(source),
            "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "height_offset_m": height,
            "candidates": entry["candidates"],
        }
    return result


def joint_waypoint_action(env, side, waypoints, step, args):
    """Interpolate joint targets, retaining the task's normal command-rate limit."""
    phases = waypoints["phases"]
    pregrasp, grasp, elevated = (
        torch.tensor(phases[phase]["candidates"][side]["joint_positions"], device=env.device).unsqueeze(0)
        for phase in ("pregrasp", "grasp", "lift")
    )
    initial = torch.tensor(waypoints["initial_joints"][side], device=env.device).unsqueeze(0)
    if step < 210:
        start, end, fraction = initial, pregrasp, (step + 1) / 210
    elif step < 420:
        start, end, fraction = pregrasp, grasp, (step - 209) / 210
    elif step < lift_start_step(args):
        start, end, fraction = grasp, grasp, 0.0
    else:
        start, end, fraction = grasp, elevated, min((step - lift_start_step(args)) / args.lift_duration_steps, 1.0)
    desired = torch.lerp(start, end, fraction)
    current_target = getattr(env, f"_{side}_arm_incremental_target")
    return ((desired - current_target) / env.cfg.action_step_scale).clamp(-1, 1)


def physics_snapshot(env):
    """Record resolved Newton gravity, rather than assuming a USD flag is honored."""
    from isaaclab_newton.physics import NewtonCfg

    ordering = {
        "robot_joint_names": list(env.robot.joint_names),
        "robot_backend_joint_names": list(env.robot.backend_joint_names),
        "robot_body_names": list(env.robot.body_names),
        "robot_backend_body_names": list(env.robot.backend_body_names),
        "robot_body_mass_kg": read_data(env.robot.data, "body_mass")[0].float().cpu().tolist(),
        "robot_body_inertia": read_data(env.robot.data, "body_inertia")[0].float().cpu().tolist(),
        "robot_joint_velocity_limits": read_data(env.robot.data, "joint_vel_limits")[0].float().cpu().tolist(),
    }
    if not isinstance(env.cfg.sim.physics, NewtonCfg):
        return {"backend": type(env.cfg.sim.physics).__name__, **ordering}
    manager = env.sim.physics_manager
    model = manager.get_model()
    solver = manager._solver
    mj_model = solver.mj_model
    sdf_sampling = []
    scales = model.shape_scale.numpy()
    sdf_indices = model._shape_sdf_index.numpy() if model._shape_sdf_index is not None else []
    sdf_data = model._texture_sdf_data.numpy() if model._texture_sdf_data is not None else []
    for index, sdf_index in enumerate(sdf_indices):
        label = model.shape_label[index]
        if sdf_index < 0 or not ("finger" in label or "/box" in label or "/Ball" in label):
            continue
        data = sdf_data[sdf_index]
        voxel = data["voxel_size"]
        scale = scales[index]
        sdf_sampling.append(
            {
                "shape": label,
                "voxel_size_world_m": [
                    float(voxel[i]) * (1.0 if data["scale_baked"] else float(scale[i])) for i in range(3)
                ],
            }
        )
    return {
        **ordering,
        "backend": "newton_mjwarp",
        "deterministic_requested": env.cfg.sim.physics.deterministic,
        "deterministic_mode": manager._deterministic_mode.name,
        "deterministic_contact_sorting": manager._collision_pipeline.deterministic,
        "mjwarp_internal_sensors_disabled": env.cfg.sim.physics.solver_cfg.disable_sensors,
        "solver_dt": float(manager._solver_dt),
        "world_gravity": model.gravity.numpy().tolist(),
        "body_gravity_compensation": mj_model.body_gravcomp.tolist(),
        "body_masses_kg": mj_model.body_mass.tolist(),
        "body_inertia_kg_m2": mj_model.body_inertia.tolist(),
        "body_names": [mj_model.body(index).name for index in range(mj_model.nbody)],
        "free_roots": [
            {
                "joint": mj_model.joint(index).name,
                "body": mj_model.body(int(mj_model.jnt_bodyid[index])).name,
                "mass_kg": float(mj_model.body_mass[int(mj_model.jnt_bodyid[index])]),
                "gravity_compensation": float(mj_model.body_gravcomp[int(mj_model.jnt_bodyid[index])]),
            }
            for index in range(mj_model.njnt)
            if int(mj_model.jnt_type[index]) == 0  # MuJoCo free root, not a weld
        ],
        "cone": int(mj_model.opt.cone),
        "impratio": float(mj_model.opt.impratio),
        "mesh_sdf_sampling": sdf_sampling,
        "equality_constraints": [
            {
                "type": int(mj_model.eq_type[index]),
                "object1": int(mj_model.eq_obj1id[index]),
                "object2": int(mj_model.eq_obj2id[index]),
                "data": mj_model.eq_data[index].tolist(),
                "solref": solver.mjw_model.eq_solref.numpy()[0, index].tolist(),
                "solimp": solver.mjw_model.eq_solimp.numpy()[0, index].tolist(),
            }
            for index in range(mj_model.neq)
        ],
        "finger_joint_constraints": [
            {
                "joint": mj_model.joint(index).name,
                "position_limited": bool(mj_model.jnt_limited[index]),
                "position_range_m": mj_model.jnt_range[index].tolist(),
                "limit_solref": solver.mjw_model.jnt_solref.numpy()[0, index].tolist(),
                "limit_solimp": mj_model.jnt_solimp[index].tolist(),
                # Separate P/D actuator rows are unlimited, while their SUM
                # is clamped at the joint. Record both to avoid a false claim
                # of missing motor limits from actuator_forcelimited alone.
                "total_actuator_force_limited": bool(mj_model.jnt_actfrclimited[index]),
                "total_actuator_force_range_n": solver.mjw_model.jnt_actfrcrange.numpy()[0, index].tolist(),
            }
            for index in range(mj_model.njnt)
            if "finger" in mj_model.joint(index).name
        ],
        "finger_drives": [
            {
                "joint": mj_model.joint(int(mj_model.actuator_trnid[index, 0])).name,
                "gain": mj_model.actuator_gainprm[index].tolist(),
                "bias": mj_model.actuator_biasprm[index].tolist(),
                "force_range": mj_model.actuator_forcerange[index].tolist(),
                "force_limited": bool(mj_model.actuator_forcelimited[index]),
            }
            for index in range(mj_model.nu)
            if "finger" in mj_model.joint(int(mj_model.actuator_trnid[index, 0])).name
        ],
        "contact_materials": [
            {
                "body": mj_model.body(int(mj_model.geom_bodyid[index])).name,
                "friction": mj_model.geom_friction[index].tolist(),
                "condim": int(mj_model.geom_condim[index]),
                "priority": int(mj_model.geom_priority[index]),
                "margin_m": float(mj_model.geom_margin[index]),
                "gap_m": float(mj_model.geom_gap[index]),
                "solref": mj_model.geom_solref[index].tolist(),
                "solimp": mj_model.geom_solimp[index].tolist(),
            }
            for index in range(mj_model.ngeom)
            if any(
                token in mj_model.body(int(mj_model.geom_bodyid[index])).name
                for token in ("finger", "Environment_box", "Environment_Ball")
            )
        ],
        "robot_disable_gravity_requested": env.cfg.scene.robot.spawn.rigid_props.disable_gravity,
        "robot_gravity_compensation_requested": env.cfg.scene.robot.spawn.newton_gravity_compensation,
    }


def change_diagnostic_mimic_compliance(env, time_constant, impedance):
    """Vary only finger equality compliance in a disposable Newton diagnostic.

    This is not a production task setting or a change to ongoing training.
    Preserve unselected parameters and the damping ratio. Respect the
    two-timestep lower bound used by MuJoCo's reference-safety mechanism.
    """
    from isaaclab_newton.physics import NewtonCfg

    if not isinstance(env.cfg.sim.physics, NewtonCfg):
        raise ValueError("Mimic compliance diagnostics require Newton MJWarp")
    solver = env.sim.physics_manager._solver
    model = solver.mj_model
    solver_dt = env.physics_dt / env.cfg.sim.physics.num_substeps
    if 0.0 < time_constant < 2 * solver_dt:
        raise ValueError("Mimic time constant must be at least twice the solver timestep")
    references = solver.mjw_model.eq_solref.numpy()
    original = references.copy()
    impedances = solver.mjw_model.eq_solimp.numpy()
    original_impedances = impedances.copy()
    indices = [
        index
        for index in range(model.neq)
        if int(model.eq_type[index]) == 2
        and all("finger" in model.joint(int(ids[index])).name for ids in (model.eq_obj1id, model.eq_obj2id))
    ]
    if len(indices) != 2:
        raise ValueError(f"Expected the two gripper mimic equalities, got {indices}")
    if time_constant > 0.0:
        references[:, indices, 0] = time_constant
    if impedance > 0.0:
        impedances[:, indices, :2] = impedance
    solver.mjw_model.eq_solref.assign(references)
    solver.mjw_model.eq_solimp.assign(impedances)
    model.eq_solref[indices] = references[0, indices]
    model.eq_solimp[indices] = impedances[0, indices]
    return {
        "indices": indices,
        "original": original.tolist(),
        "diagnostic": references.tolist(),
        "original_impedance": original_impedances.tolist(),
        "diagnostic_impedance": impedances.tolist(),
    }


def contact_snapshot(env):
    """Measure finger/box pairs and surface gaps; Newton's force buffer is unused."""
    from collections import Counter

    import numpy as np
    from isaaclab_newton.physics import NewtonCfg
    from scipy.spatial.transform import Rotation

    if not isinstance(env.cfg.sim.physics, NewtonCfg):
        return None
    manager = env.sim.physics_manager
    contacts = manager.get_contacts()
    if contacts is None:
        return None
    model = manager.get_model()
    count = int(contacts.rigid_contact_count.numpy()[0])
    shapes0, shapes1 = contacts.rigid_contact_shape0.numpy(), contacts.rigid_contact_shape1.numpy()
    shape_bodies = model.shape_body.numpy()
    body_poses = manager.get_state_0().body_q.numpy()
    points0, points1 = contacts.rigid_contact_point0.numpy(), contacts.rigid_contact_point1.numpy()
    normals = contacts.rigid_contact_normal.numpy()
    margins0, margins1 = contacts.rigid_contact_margin0.numpy(), contacts.rigid_contact_margin1.numpy()
    margins = margins0 + margins1
    friction = model.shape_material_mu.numpy()
    pairs, gaps, pair_normals, pair_friction, pair_points = Counter(), [], {}, {}, {}
    robot_box_pairs = Counter()
    object_robot_pairs = Counter()
    ball_box_contacts = []
    object_names = {
        getattr(env, role).cfg.prim_path.rsplit("/", 1)[-1] for role in ("object", "peg", "hole") if hasattr(env, role)
    }
    for index in range(min(count, len(shapes0))):
        shape0, shape1 = int(shapes0[index]), int(shapes1[index])
        if shape0 < 0 or shape1 < 0:
            continue
        body0, body1 = int(shape_bodies[shape0]), int(shape_bodies[shape1])
        if body0 < 0 or body1 < 0:
            continue
        name0, name1 = model.body_label[body0], model.body_label[body1]
        part0 = name0.rsplit("/", 1)[-1] in object_names
        part1 = name1.rsplit("/", 1)[-1] in object_names
        if (part0 and "/Robot/" in name1) or (part1 and "/Robot/" in name0):
            object_robot_pairs[name0 + " <-> " + name1] += 1
        if ("/box" in name0 and "/Robot/" in name1) or ("/box" in name1 and "/Robot/" in name0):
            robot_box_pairs[name0 + " <-> " + name1] += 1
        if (name0.endswith("/Ball") and name1.endswith("/box")) or (name1.endswith("/Ball") and name0.endswith("/box")):
            pose0, pose1 = body_poses[body0], body_poses[body1]
            point0 = Rotation.from_quat(pose0[3:]).apply(points0[index]) + pose0[:3]
            point1 = Rotation.from_quat(pose1[3:]).apply(points1[index]) + pose1[:3]
            ball_box_contacts.append(
                {
                    "shape0": model.shape_label[shape0],
                    "shape1": model.shape_label[shape1],
                    "normal_0_to_1_w": normals[index].tolist(),
                    "witness_point0_w": point0.tolist(),
                    "witness_point1_w": point1.tolist(),
                    "surface_point0_w": (point0 + margins0[index] * normals[index]).tolist(),
                    "surface_point1_w": (point1 - margins1[index] * normals[index]).tolist(),
                    "margin0_m": float(margins0[index]),
                    "margin1_m": float(margins1[index]),
                    "surface_gap_m": float(np.dot(point1 - point0, normals[index]) - margins[index]),
                    "surface_friction": [float(friction[shape0]), float(friction[shape1])],
                }
            )
        if not ((part0 and "finger" in name1) or (part1 and "finger" in name0)):
            continue
        key = name0 + " <-> " + name1
        pairs[key] += 1
        pair_normals.setdefault(key, []).append(abs(float(normals[index, 2])))
        pair_friction[key] = [float(friction[shape0]), float(friction[shape1])]
        pose0, pose1 = body_poses[body0], body_poses[body1]
        point0 = Rotation.from_quat(pose0[3:]).apply(points0[index]) + pose0[:3]
        point1 = Rotation.from_quat(pose1[3:]).apply(points1[index]) + pose1[:3]
        pair_points.setdefault(key, []).append(0.5 * (point0 + point1))
        gaps.append(float(np.dot(point1 - point0, normals[index]) - margins[index]))
    return {
        "total_contacts": count,
        "solver_iterations": manager._solver.mjw_data.solver_niter.numpy().tolist(),
        "constraint_rows": manager._solver.mjw_data.nefc.numpy().tolist(),
        "mujoco_contact_count": int(manager._solver.mjw_data.nacon.numpy()[0]),
        "constraint_capacity": env.cfg.sim.physics.solver_cfg.njmax,
        "finger_box_pairs": {key: value for key, value in pairs.items() if "/box" in key},
        "finger_object_pairs": dict(pairs),
        "robot_box_pairs": dict(robot_box_pairs),
        "object_robot_pairs": dict(object_robot_pairs),
        "ball_box_contacts": ball_box_contacts,
        "finger_box_material_friction": pair_friction,
        "finger_box_mean_abs_normal_z": {key: float(np.mean(value)) for key, value in pair_normals.items()},
        "finger_box_mean_contact_position_w": {
            key: np.mean(value, axis=0).tolist() for key, value in pair_points.items()
        },
        "minimum_finger_box_gap_m": min(gaps) if gaps else None,
    }


def motor_effort_layout(env):
    """Map named scalar joints to the solver's last-substep actuator forces.

    These are generalized motor efforts, not fingertip contact forces or the
    sum of actuator, constraint and gravity forces. Reading the native solver
    buffer avoids treating an implicit actuator's PD estimate as a measurement.
    """
    import mujoco
    from isaaclab_newton.physics import NewtonCfg

    if not isinstance(env.cfg.sim.physics, NewtonCfg):
        raise ValueError("Motor-effort recording currently requires Newton MJWarp")
    model = env.sim.physics_manager._solver.mj_model
    entries = []
    for name in env.robot.joint_names:
        matches = [i for i in range(model.njnt) if model.joint(i).name.endswith("_" + name)]
        if len(matches) != 1:
            raise ValueError(f"Expected one native scalar joint for {name}, got {matches}")
        index = matches[0]
        kind = int(model.jnt_type[index])
        if kind not in (int(mujoco.mjtJoint.mjJNT_SLIDE), int(mujoco.mjtJoint.mjJNT_HINGE)):
            raise ValueError(f"Motor-effort recording requires a scalar joint: {name}, native type {kind}")
        entries.append(
            {
                "joint": name,
                "dof_index": int(model.jnt_dofadr[index]),
                "unit": "N" if kind == int(mujoco.mjtJoint.mjJNT_SLIDE) else "N*m",
                "force_limited": bool(model.jnt_actfrclimited[index]),
                "force_range": model.jnt_actfrcrange[index].tolist(),
            }
        )
    return {
        "source": "mjwarp.qfrc_actuator",
        "sample_timing": "Last solver substep, before terminal auto-reset; not averaged over the control tick",
        "contact_force_measurement": False,
        "joints": entries,
    }


def check_closure_reward(env):
    """Compare coefficients on the same measured closed-gripper state."""
    original = env.cfg.premature_closure_penalty_scale
    try:
        before = env._get_rewards()["left_arm"].clone()
        signals = env._ctx["trace_signals"]
        premature = signals["left_premature_closure"] + signals["right_premature_closure"]
        env.cfg.premature_closure_penalty_scale = 0.25
        after = env._get_rewards()["left_arm"].clone()
        expected = 0.5 * env.cfg.reward_grasp_scale * (original - 0.25) * premature
        torch.testing.assert_close(after - before, expected, rtol=1e-5, atol=1e-5)
        return {
            "original_coefficient": original,
            "counterfactual_coefficient": 0.25,
            "measured_reward_change": float((after - before)[0]),
            "expected_reward_change": float(expected[0]),
        }
    finally:
        env.cfg.premature_closure_penalty_scale = original


def check_post_close_reset(env):
    """Check the first actual actuator commands after resetting closed fingers."""
    closing = {agent: torch.zeros((1, 8), device=env.device) for agent in env.possible_agents}
    for action in closing.values():
        action[:, 7] = -1
    for _ in range(60):
        env.step(closing)
    env.reset()
    env.step({agent: torch.zeros_like(action) for agent, action in closing.items()})
    commands = {}
    for side in ("left", "right"):
        ids = getattr(env, f"{side}_gripper_joint_ids")
        command = tensor(env.robot.actuators.target_command.position)[0, ids]
        torch.testing.assert_close(command, torch.full_like(command, env.cfg.gripper_open_target))
        commands[side] = command.tolist()
    return commands


def capture_probe_frames(env, output, step):
    """Render two diagnostic views without feeding either image to the policy."""
    from PIL import Image

    paths = []
    for view in ("front", "side"):
        camera = env.scene.sensors[f"probe_camera_{view}"]
        env.sim.render()
        camera.update(0.0, force_recompute=True)
        pixels = tensor(camera.data.output["rgb"])[0, :, :, :3].cpu().numpy()
        path = output.with_name(f"{output.stem}_{step:04d}_{view}.png")
        Image.fromarray(pixels).save(path)
        paths.append(str(path))
    return paths


def add_probe_cameras(scene_cfg, *, peg_task=False):
    """Spawn fixed diagnostic cameras before the renderer scene is built."""
    import numpy as np
    from isaaclab_physx.renderers import IsaacRtxRendererCfg
    from scipy.spatial.transform import Rotation

    from isaaclab.sensors import TiledCameraCfg
    from isaaclab.sim import PinholeCameraCfg

    height = 0.85 if peg_task else 0.6
    look_at = np.array([0.35, 0.0, 0.61 if peg_task else 0.35])
    for view, eye in {"front": (0.85, 0.0, height), "side": (0.65, 0.7, height)}.items():
        forward = look_at - eye
        forward /= np.linalg.norm(forward)
        left = np.cross([0, 0, 1], forward)
        left /= np.linalg.norm(left)
        rotation = Rotation.from_matrix(np.stack([forward, left, np.cross(forward, left)], axis=1)).as_quat()
        camera = TiledCameraCfg(
            prim_path=f"{{ENV_REGEX_NS}}/ProbeCamera_{view}",
            update_period=0.1,
            height=480,
            width=640,
            data_types=["rgb"],
            renderer_cfg=IsaacRtxRendererCfg(),
            spawn=PinholeCameraCfg(focal_length=24.0, horizontal_aperture=30.0, clipping_range=(0.05, 3.0)),
            offset=TiledCameraCfg.OffsetCfg(pos=eye, rot=tuple(rotation), convention="world"),
        )
        setattr(scene_cfg, f"probe_camera_{view}", camera)


def spawn_diagnostic_contacts(
    prim_path, cfg, translation=None, orientation=None, *, original_spawn, condim, sdf_resolution, **kwargs
):
    """Apply contact dimensions and mesh-SDF resolution only in this diagnostic."""
    from isaaclab_newton.sim.schemas import MujocoCollisionCfg, NewtonSDFCollisionCfg

    from pxr import Usd, UsdGeom, UsdPhysics

    from isaaclab.sim.schemas import apply_collision_properties, apply_mesh_collision_properties

    root = original_spawn(prim_path, cfg, translation=translation, orientation=orientation, **kwargs)
    for prim in Usd.PrimRange(root):
        if not prim.IsA(UsdGeom.Gprim) or not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        path = str(prim.GetPath())
        if condim and "finger" in path:
            apply_collision_properties(str(prim.GetPath()), [MujocoCollisionCfg(condim=condim)], stage=prim.GetStage())
        if sdf_resolution and prim.IsA(UsdGeom.Mesh) and ("finger" in path or "/box" in path):
            apply_mesh_collision_properties(
                path, [NewtonSDFCollisionCfg(sdf_max_resolution=sdf_resolution)], stage=prim.GetStage()
            )
    return root


def configure_diagnostic_contacts(env_cfg, args):
    """Keep numerical contact experiments explicit and local to the probe."""
    if args.finger_condim or args.fine_contact_sdf:
        from isaaclab_newton.physics import NewtonCfg

        if not isinstance(env_cfg.sim.physics, NewtonCfg):
            raise ValueError("Finger contact-model diagnostics require Newton MJWarp")
        env_cfg.scene.robot.spawn.func = partial(
            spawn_diagnostic_contacts,
            original_spawn=env_cfg.scene.robot.spawn.func,
            condim=args.finger_condim,
            sdf_resolution=256 if args.fine_contact_sdf else 0,
        )
        if args.fine_contact_sdf:
            env_cfg.scene.environment.spawn.func = partial(
                spawn_diagnostic_contacts,
                original_spawn=env_cfg.scene.environment.spawn.func,
                condim=0,
                sdf_resolution=512,
            )
    if args.mesh_ball_collision:
        from isaaclab_newton.physics import NewtonCfg

        if not isinstance(env_cfg.sim.physics, NewtonCfg) or args.task != "Isaac-OpenArm-LiftWithBall-v0":
            raise ValueError("The mesh-ball collision diagnostic requires the Newton WithBall task")
        env_cfg.scene.environment.spawn.func = partial(
            spawn_diagnostic_mesh_ball, original_spawn=env_cfg.scene.environment.spawn.func
        )


def spawn_diagnostic_mesh_ball(prim_path, cfg, translation=None, orientation=None, *, original_spawn, **kwargs):
    """Compare a close sphere mesh/SDF proxy without editing the source USD.

    Preserve the analytic sphere's mass, inertia, center and collision radius.
    The visual sphere, camera, material and every box collider remain unchanged.
    This explicitly tests a different Newton narrow-phase route; it is not a
    claim that a polygonal proxy is exactly the analytic collision surface.
    """
    import numpy as np
    import trimesh
    from isaaclab_newton.sim.schemas import NewtonSDFCollisionCfg

    from pxr import Gf, Usd, UsdGeom, UsdPhysics, Vt

    from isaaclab.sim.schemas import UsdPhysicsMeshCollisionCfg, apply_mesh_collision_properties

    root = original_spawn(prim_path, cfg, translation=translation, orientation=orientation, **kwargs)
    spheres = [p for p in Usd.PrimRange(root) if p.GetName() == "CollisionSphere" and p.GetParent().GetName() == "Ball"]
    if len(spheres) != 1 or not spheres[0].IsA(UsdGeom.Sphere):
        raise ValueError("Expected the authored analytic ball collision sphere")
    prim = spheres[0]
    radius = float(UsdGeom.Sphere(prim).GetRadiusAttr().Get())
    sphere = trimesh.creation.icosphere(subdivisions=4, radius=radius)
    if not sphere.is_watertight or not sphere.is_winding_consistent:
        raise ValueError("Sphere collision proxy must be a consistently oriented closed mesh")
    triangles = sphere.vertices[sphere.faces]
    cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    face_distance = np.abs((cross * triangles[:, 0]).sum(-1)) / np.linalg.norm(cross, axis=-1)
    error_bound = float(radius - face_distance.min())
    if error_bound > 0.00005:
        raise ValueError("Sphere collision proxy exceeds the 50 micrometer radial error budget")
    prim.SetTypeName("Mesh")
    mesh = UsdGeom.Mesh(prim)
    mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(sphere.vertices.astype(np.float32)))
    mesh.CreateFaceVertexCountsAttr([3] * len(sphere.faces))
    mesh.CreateFaceVertexIndicesAttr(sphere.faces.astype(np.int32).ravel().tolist())
    mesh.CreateSubdivisionSchemeAttr("none")
    mass = UsdPhysics.MassAPI(prim.GetParent())
    value = float(mass.GetMassAttr().Get())
    inertia = 0.4 * value * radius**2
    mass.CreateDiagonalInertiaAttr(Gf.Vec3f(inertia))
    mass.CreateCenterOfMassAttr(Gf.Vec3f(0))
    mass.CreatePrincipalAxesAttr(Gf.Quatf(1))
    apply_mesh_collision_properties(
        str(prim.GetPath()),
        [UsdPhysicsMeshCollisionCfg(mesh_approximation_name="none"), NewtonSDFCollisionCfg(sdf_max_resolution=256)],
        stage=prim.GetStage(),
    )
    UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Set("none")
    print(
        "BALL_MESH_COLLISION "
        + json.dumps(
            {
                "mass_kg": value,
                "radius_m": radius,
                "radial_error_bound_m": error_bound,
                "faces": len(sphere.faces),
                "inertia_kg_m2": inertia,
            }
        ),
        flush=True,
    )
    return root


def spawn_diagnostic_physx_mechanics(prim_path, cfg, translation=None, orientation=None, *, original_spawn, **kwargs):
    """Match corrected payloads using negligible positive PhysX marker proxies.

    USD zero mass requests automatic mass computation in PhysX. The four
    geometry-free coordinate frames therefore use 1 milligram each, with
    1e-10 kg m^2 numerical inertia, rather than the fallback 1 kg. These are
    diagnostic proxies for massless fixed frames, not physical payload estimates.
    Original files and the production PhysX/Newton spawners are unchanged.
    """
    from pxr import Gf, Usd, UsdPhysics

    from isaaclab_tasks.direct.openarm_motion_context.envs.camera_payloads import prepare_camera_payloads
    from isaaclab_tasks.direct.openarm_motion_context.envs.newton_assets import (
        _correct_legacy_v1_finger_inertia,
        orient_camera_fixed_joints,
    )

    root = original_spawn(prim_path, cfg, translation=translation, orientation=orientation, **kwargs)
    markers = {"openarm_left_hand", "openarm_right_hand", "openarm_left_ee_tcp", "openarm_right_ee_tcp"}
    joints = {"openarm_left_hand", "openarm_right_hand", "openarm_left_ee_tcp_joint", "openarm_right_ee_tcp_joint"}
    for prim in Usd.PrimRange(root):
        if prim.GetName() in joints and prim.IsA(UsdPhysics.RevoluteJoint):
            joint = UsdPhysics.RevoluteJoint(prim)
            if abs(joint.GetLowerLimitAttr().Get()) > 1e-9 or abs(joint.GetUpperLimitAttr().Get()) > 1e-9:
                raise ValueError(f"Expected a locked marker joint: {prim.GetPath()}")
            prim.SetTypeName("PhysicsFixedJoint")
            prim.RemoveAPI(UsdPhysics.DriveAPI, "angular")
    orient_camera_fixed_joints(root)
    prepare_camera_payloads(root)
    for prim in Usd.PrimRange(root):
        if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
            continue
        _correct_legacy_v1_finger_inertia(prim)
        if prim.GetName() in markers:
            mass = UsdPhysics.MassAPI.Apply(prim)
            mass.CreateMassAttr(1e-6)
            mass.CreateDiagonalInertiaAttr(Gf.Vec3f(1e-10))
    return root


def configure_probe_scene(env_cfg, args):
    """Prepare a single reproducible world without changing policy evaluations."""
    env_cfg.scene.num_envs = 1
    env_cfg.seed = args.seed
    env_cfg.episode_length_s = 30.0
    # A physical response comparison needs identical poses even if renderer
    # initialization consumes RNG. Training and policy evaluation retain DR.
    if args.probe == "insert" or "peg_in_hole" in str(env_cfg.intent_task_label):
        env_cfg.reset_arm_joint_noise_rad = 0.0
    else:
        env_cfg.enable_reset_randomization = False
    if args.capture_frames:
        add_probe_cameras(env_cfg.scene, peg_task="PegInHole" in args.task)
    configure_diagnostic_contacts(env_cfg, args)
    if args.match_robot_mechanics:
        from isaaclab_newton.physics import NewtonCfg

        if isinstance(env_cfg.sim.physics, NewtonCfg):
            raise ValueError("The mechanical matching override is for the supplemental PhysX comparison only")
        env_cfg.scene.robot.spawn.func = partial(
            spawn_diagnostic_physx_mechanics, original_spawn=env_cfg.scene.robot.spawn.func
        )


def control_contract(env):
    """Record command semantics needed for a same-action backend comparison."""
    contract = {
        name: float(getattr(env.cfg, name))
        for name in (
            "action_scale",
            "action_step_scale",
            "gripper_open_target",
            "gripper_close_target",
            "gripper_switch_epsilon",
            "gripper_target_speed_limit",
            "reset_grace_steps",
        )
    }
    contract["control_dt"] = float(env.step_dt)
    if hasattr(env, "_arm_reference_rate_scale"):
        contract["arm_reference_rate_scale"] = env._arm_reference_rate_scale
    if hasattr(env, "_arm_target_bounds_error_rad"):
        contract["arm_target_error_rad"] = env._arm_target_bounds_error_rad
    for side in ("left", "right"):
        for group in ("arm", "gripper"):
            ids = getattr(env, f"{side}_{group}_joint_ids")
            contract[f"{side}_{group}_joint_names"] = [env.robot.joint_names[index] for index in ids]
    return contract


def load_action_replay(env, args, initial_state):
    """Require equal named initial states and command semantics, not equal dynamics."""
    if args.action_trace is None:
        return None
    reference = json.loads(Path(args.action_trace).read_text())
    if reference.get("task") != args.task or reference.get("control_contract") != control_contract(env):
        raise ValueError("Replay requires the same task, ordered joints, control period and command semantics")
    before, now = reference["initial_state"], initial_state
    left, right = before["joint_positions_by_name"], now["joint_positions_by_name"]
    if set(left) != set(right):
        raise ValueError("Replay initial joint-name sets differ")
    joint_error = max(abs(left[name] - right[name]) for name in left)
    a, b = torch.tensor(before["object_pose"]), torch.tensor(now["object_pose"])
    position_error = float((a[:3] - b[:3]).abs().max())
    quaternion_error = float(torch.minimum((a[3:] - b[3:]).abs().max(), (a[3:] + b[3:]).abs().max()))
    if max(joint_error, position_error, quaternion_error) > 1e-5:
        raise ValueError("Replay requires equal recorded initial robot joints and object pose")
    reference["initial_state_errors"] = {
        "joint_max_abs": joint_error,
        "object_position_max_abs_m": position_error,
        "quaternion_max_abs": quaternion_error,
    }
    return reference


def verify_replayed_commands(row, reference):
    """Fail if the two simulations did not execute the same target sequence."""
    for side in ("left", "right"):
        for key in (f"{side}_joint_targets", f"{side}_gripper_joint_targets"):
            torch.testing.assert_close(torch.tensor(row[key]), torch.tensor(reference[key]), rtol=0, atol=1e-6)


def probe_actions(env, args, targets, integral, step, phase, replay, waypoints):
    """Build diagnostic commands or replay a previously recorded command sequence."""
    actions = {
        agent: torch.zeros((1, env.cfg.action_spaces[agent]), device=env.device) for agent in env.possible_agents
    }
    for index, side in enumerate(("left", "right")):
        if args.probe == "arm_step":
            joint = args.arm_joint_index
            offset = 0.0 if phase == "arm_hold" else 0.7 if phase == "arm_positive" else -0.7
            default = getattr(env, f"{side}_arm_default_joint_pos")[:, joint]
            previous = getattr(env, f"_{side}_arm_incremental_target")[:, joint]
            actions[f"{side}_arm"][:, joint] = ((default + offset - previous) / env.cfg.action_step_scale).clamp(-1, 1)
        elif waypoints is not None and not (args.cartesian_lift and phase == "lift"):
            actions[f"{side}_arm"][:, :7] = joint_waypoint_action(env, side, waypoints, step, args)
        elif targets is not None and replay is None:
            position = targets[index].clone()
            rotation = targets[index + 2]
            if args.probe == "insert" and side == "left" and phase in ("align", "insert"):
                metrics = compute_peg_in_hole_metrics(env, env._ctx)
                peg_axis, hole_axis = metrics["peg_axis_w"], metrics["hole_axis_w"]
                cross = torch.linalg.cross(peg_axis, hole_axis, dim=-1)
                dot = (peg_axis * hole_axis).sum(dim=-1, keepdim=True)
                delta_quat = torch.cat((1 + dot, cross), dim=-1)
                if delta_quat.norm().item() < 1e-5:
                    raise RuntimeError("Antiparallel insertion axes require a separate orientation probe")
                delta_quat = torch.nn.functional.normalize(delta_quat, dim=-1)
                pose = read_data(env.robot.data, "body_state_w")[:, env._ctx["left_ee_body_id"]]
                rotation = quat_mul(delta_quat, pose[:, 3:7])
                offset = quat_apply(delta_quat, metrics["peg_tip_w"] - pose[:, :3])
                travel = float(env.target_insertion_depth) + 0.015
                elapsed = step - args.grasp_settle_steps - args.insert_align_steps
                depth = -0.015 if phase == "align" else -0.015 + min(elapsed / args.insert_duration_steps, 1) * travel
                position = metrics["hole_entrance_w"] + depth * hole_axis - offset
            elif phase == "lift":
                position[:, 2] += min((step - lift_start_step(args)) / args.lift_duration_steps, 1) * args.lift_height
            elif args.probe == "lift" and phase == "approach":
                # A diagnostic above-rim approach separates contact feasibility
                # from the straight-line controller's early rim impacts.
                position[:, 2] += args.approach_clearance * min(max((420 - step) / 210, 0), 1)
            elif args.probe == "grasp_release":
                position[:, 2] += args.grasp_transport_height * min(step / 180, 1)
            actions[f"{side}_arm"][:, :7] = pose_action(
                env, side, position, rotation, integral[side], args.integral_gain
            )
        if actions[f"{side}_arm"].shape[-1] == 8:
            actions[f"{side}_arm"][:, 7] = -1.0 if phase in ("close", "lift", "grasp_hold", "align", "insert") else 1.0
        if replay is not None:
            previous = replay["rows"][step]
            phase = previous["phase"]
            actions[f"{side}_arm"][:, :7] = torch.tensor(previous[f"{side}_arm_action"], device=env.device)
            if actions[f"{side}_arm"].shape[-1] == 8:
                actions[f"{side}_arm"][:, 7] = previous[f"{side}_gripper_action"]
    return actions, phase


def lift_start_step(args):
    """Allow pose settling with open fingers before closing in Lift probes."""
    return 540 + (args.grasp_settle_steps if args.probe == "lift" else 0)


def probe_phase(args, step):
    """Name each scripted phase independently of simulator recording."""
    if args.probe == "ball_passive":
        return "passive"
    if args.probe == "box_ballistic":
        return "ballistic"
    if args.probe == "gripper":
        return "open" if step < 60 else "close" if step < 180 else "reopen"
    if args.probe == "arm_step":
        return "arm_hold" if step < 60 else "arm_positive" if step < 180 else "arm_negative"
    if args.probe == "insert":
        if step < args.grasp_settle_steps:
            return "grasp_hold"
        return "align" if step < args.grasp_settle_steps + args.insert_align_steps else "insert"
    if args.probe == "grasp_release":
        return "grasp_hold" if step < 180 else "release"
    if step < 420:
        return "approach"
    if step < 420 + args.grasp_settle_steps:
        return "pregrasp_settle"
    return "close" if step < lift_start_step(args) else "lift"


def probe_duration(args):
    """Keep historical default schedules while supporting slower free-part motion."""
    if args.probe == "ball_passive":
        return args.num_steps if args.num_steps is not None else 300
    if args.probe == "box_ballistic":
        return 90
    if args.probe in ("gripper", "grasp_release", "arm_step"):
        return 300
    if args.probe == "lift":
        return lift_start_step(args) + args.lift_duration_steps + args.hold_steps
    return args.grasp_settle_steps + args.insert_align_steps + args.insert_duration_steps + args.insert_hold_steps


def run(env_cfg, args):
    """Collect the requested physical probe, preserving the original success gates."""
    configure_probe_scene(env_cfg, args)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with launch_simulation(env_cfg, args), torch.inference_mode():
        wrapper = gym.make(args.task, cfg=env_cfg)
        env = wrapper.unwrapped
        env.reset(seed=args.seed)
        if args.arm_reference_rate_scale > 0.0:
            from isaaclab_tasks.direct.openarm_motion_context.envs.arm_reference_rate import (
                install_arm_reference_rate_limit,
            )

            install_arm_reference_rate_limit(env, args.arm_reference_rate_scale)
        if args.arm_target_error_rad is not None:
            from isaaclab_tasks.direct.openarm_motion_context.envs.arm_target_bounds import install_arm_target_bounds

            install_arm_target_bounds(env, args.arm_target_error_rad)
        initial_ik = initialize_diagnostic_grasp(env, args)
        waypoints = load_joint_waypoints(env, args)
        if args.probe not in ("insert", "model") and any(width != 8 for width in env.cfg.action_spaces.values()):
            raise ValueError("Gripper/Lift/reachability probes require a Lift task with eight actions per arm")
        if args.probe == "insert" and not (
            env.cfg.use_runtime_fixed_joints or bool(getattr(env.cfg, "require_contact_grasp", False))
        ):
            raise ValueError("Insertion diagnostics require physical welds or the explicit free-contact grasp task")
        mimic_change = (
            change_diagnostic_mimic_compliance(env, args.mimic_time_constant, args.mimic_impedance)
            if args.mimic_time_constant > 0.0 or args.mimic_impedance > 0.0
            else None
        )
        resolved_physics = physics_snapshot(env)
        effort_layout = motor_effort_layout(env) if args.record_motor_efforts else None
        effort_indices = [entry["dof_index"] for entry in effort_layout["joints"]] if effort_layout else []
        tracked_object = env.peg if hasattr(env, "peg") else env.object
        initial_state = {
            "object_pose": read_data(tracked_object.data, "root_state_w")[0, :7].tolist(),
            "joint_positions": read_data(env.robot.data, "joint_pos")[0].tolist(),
            "joint_positions_by_name": dict(
                zip(env.robot.joint_names, read_data(env.robot.data, "joint_pos")[0].tolist())
            ),
        }
        replay = load_action_replay(env, args, initial_state)
        initial_object_z = None
        if args.probe == "box_ballistic":
            initial_object_z = float(read_data(env.object.data, "root_pos_w")[0, 2])
            velocity = torch.zeros((1, 6), device=env.device)
            velocity[:, 2] = 1.5
            env.object.write_root_velocity_to_sim_index(root_velocity=velocity)
        if args.probe in ("reachability", "model"):
            if args.probe == "reachability":
                scales = (
                    [float(x) for x in args.ik_action_scales.split(",")]
                    if args.ik_action_scales
                    else [float(env.cfg.action_scale)]
                )
                result = {"kinematic_scale_sweep": {str(s): kinematic_reachability(env, args, s) for s in scales}}
            else:
                result = {"mechanics_only": True}
            result["task"] = args.task
            result["physics"] = resolved_physics
            result["initial_state"] = initial_state
            result["diagnostic_physx_mechanics"] = args.match_robot_mechanics
            output.write_text(json.dumps(result, indent=2) + "\n")
            print("PROBE_RESULT " + json.dumps(result), flush=True)
            wrapper.close()
            return
        targets = None
        if args.probe == "lift":
            targets = tuple(value.clone() for value in compute_collision_target_poses_w(env, env._ctx))
        elif args.probe == "insert" or (args.probe == "grasp_release" and args.grasp_transport_height > 0.0):
            poses = read_data(env.robot.data, "body_state_w")
            targets = tuple(
                poses[:, env._ctx[f"{side}_ee_body_id"], component].clone()
                for component in (slice(0, 3), slice(3, 7))
                for side in ("left", "right")
            )
        rows = []
        integral = {side: torch.zeros((1, 3), device=env.device) for side in ("left", "right")}
        frame_paths = []
        reward_check = None
        steps = probe_duration(args)
        if replay is not None:
            steps = len(replay["rows"])
        if args.num_steps is not None:
            steps = min(steps, args.num_steps)
        for step in range(steps):
            phase = probe_phase(args, step)
            actions, phase = probe_actions(env, args, targets, integral, step, phase, replay, waypoints)
            _, _, terminated, truncated, _ = env.step(actions)
            joints = read_data(env.robot.data, "joint_pos")
            velocities = read_data(env.robot.data, "joint_vel")
            row = {"step": step, "phase": phase, "episode_step": int(env.episode_length_buf[0])}
            tracked_object = env.peg if args.probe == "insert" else env.object
            row["object_pose"] = read_data(tracked_object.data, "root_state_w")[0, :7].tolist()
            if hasattr(env, "hole"):
                row["hole_pose"] = read_data(env.hole.data, "root_state_w")[0, :7].tolist()
            for side in ("left", "right"):
                ids = getattr(env, f"{side}_gripper_joint_ids")
                row[f"{side}_finger_positions"] = joints[0, ids].tolist()
                arm_ids = getattr(env, f"{side}_arm_joint_ids")
                row[f"{side}_joint_positions"] = joints[0, arm_ids].tolist()
                row[f"{side}_joint_velocities"] = velocities[0, arm_ids].tolist()
                row[f"{side}_joint_targets"] = getattr(env, f"_{side}_arm_incremental_target")[0].tolist()
                row[f"{side}_gripper_joint_targets"] = tensor(env.robot.actuators.target_command.position)[
                    0, getattr(env, f"{side}_gripper_joint_ids")
                ].tolist()
                row[f"{side}_default_joints"] = getattr(env, f"{side}_arm_default_joint_pos")[0].tolist()
                row[f"{side}_tcp_pose"] = read_data(env.robot.data, "body_state_w")[
                    0, env._ctx[f"{side}_ee_body_id"], :7
                ].tolist()
                row[f"{side}_arm_action"] = actions[f"{side}_arm"][0, :7].tolist()
                row[f"{side}_gripper_action"] = (
                    float(actions[f"{side}_arm"][0, 7]) if actions[f"{side}_arm"].shape[-1] == 8 else None
                )
            row["terminated"] = any(bool(value[0]) for value in terminated.values())
            row["truncated"] = any(bool(value[0]) for value in truncated.values())
            if effort_layout:
                efforts = env.sim.physics_manager._solver.mjw_data.qfrc_actuator.numpy()[0, effort_indices]
                row["joint_motor_efforts"] = efforts.tolist()
            row["metrics"] = {
                name: float(value)
                for name, value in env.extras["log"].items()
                if isinstance(value, (int, float))
                and any(
                    key in name
                    for key in (
                        "closure",
                        "grasp",
                        "success",
                        "distance",
                        "height",
                        "readiness",
                        "target_error",
                        "axis",
                        "depth",
                        "lateral",
                        "radial",
                        "tilt",
                        "hold",
                        "reward",
                        "offset",
                        "camera",
                        "ball",
                    )
                )
            }
            if hasattr(env, "ball"):
                signals = env.get_paper_motion_context_trace_signals(0)
                row["ball_camera_observation"] = signals["ball_camera_observation"]
                ball_state = read_data(env.ball.data, "root_state_w")[0]
                row["ball_pose"] = ball_state[:7].tolist()
                row["ball_linear_velocity_w"] = ball_state[7:10].tolist()
                row["ball_angular_velocity_w"] = ball_state[10:13].tolist()
            rows.append(row)
            if step % 60 == 0:
                print(
                    "PROBE_PROGRESS "
                    + json.dumps(
                        {
                            "step": step,
                            "phase": phase,
                            "object_z": row["object_pose"][2],
                            "hold_count": row["metrics"].get("hold_count_max", 0.0),
                            "grasp_ok": [
                                row["metrics"].get("left_grasp_ok_ratio"),
                                row["metrics"].get("right_grasp_ok_ratio"),
                            ],
                            "terminated": row["terminated"],
                        }
                    ),
                    flush=True,
                )
            if replay is not None:
                verify_replayed_commands(row, replay["rows"][step])
            if step % 30 == 0:
                row["contacts"] = contact_snapshot(env)
            if args.probe == "gripper" and step == 179:
                reward_check = check_closure_reward(env)
            capture_steps = (0, 179, 199) if args.probe == "grasp_release" else (0, 419, 539, 659, 809)
            if args.capture_frames and step in capture_steps:
                frame_paths.extend(capture_probe_frames(env, output, step))
            if (
                replay is not None
                or args.initial_ik_candidate is not None
                or waypoints is not None
                or args.probe in ("grasp_release", "insert", "ball_passive")
            ) and (row["terminated"] or row["truncated"]):
                break
        reset_check = None
        if args.reset_after_close:
            if args.probe != "gripper":
                raise ValueError("--reset_after_close is only supported by the gripper probe")
            reset_check = check_post_close_reset(env)
        payload = {
            "task": args.task,
            "probe": args.probe,
            "seed": args.seed,
            "oracle_diagnostic": True,
            "episode_length_s": 30.0,
            "physics": type(env.cfg.sim.physics).__name__,
            "resolved_physics": resolved_physics,
            "resolved_physics_final": physics_snapshot(env),
            "motor_effort_layout": effort_layout,
            "action_step_scale": env.cfg.action_step_scale,
            "reset_grace_steps": env.cfg.reset_grace_steps,
            "gripper_target_speed_limit": env.cfg.gripper_target_speed_limit,
            "reward_coefficient_check": reward_check,
            "post_close_reset_commands": reset_check,
            "initial_object_z": initial_object_z,
            "initial_state": initial_state,
            "diagnostic_initial_ik": initial_ik,
            "diagnostic_joint_waypoints": waypoints,
            "diagnostic_pregrasp_settle_steps": args.grasp_settle_steps if args.probe == "lift" else None,
            "diagnostic_arm_joint_index": args.arm_joint_index if args.probe == "arm_step" else None,
            "control_contract": control_contract(env),
            "replay_source": args.action_trace,
            "replay_initial_state_errors": replay["initial_state_errors"] if replay is not None else None,
            "same_command_sequence_prefix_verified": replay is not None,
            "reset_randomization_disabled": True,
            "diagnostic_position_integral_gain": args.integral_gain,
            "diagnostic_mimic_change": mimic_change,
            "diagnostic_finger_condim": args.finger_condim,
            "diagnostic_fine_contact_sdf": args.fine_contact_sdf,
            "diagnostic_mesh_ball_collision": args.mesh_ball_collision,
            "diagnostic_physx_mechanics": args.match_robot_mechanics,
            "diagnostic_lift_height_m": args.lift_height,
            "diagnostic_approach_clearance_m": args.approach_clearance,
            "diagnostic_grasp_transport_height_m": args.grasp_transport_height,
            "diagnostic_insert_schedule": {
                "settle_steps": args.grasp_settle_steps,
                "align_steps": args.insert_align_steps,
                "insert_steps": args.insert_duration_steps,
                "hold_steps": args.insert_hold_steps,
            },
            "diagnostic_lift_duration_steps": args.lift_duration_steps,
            "diagnostic_lift_space": "joint" if waypoints is not None and not args.cartesian_lift else "cartesian",
            "diagnostic_hold_steps": args.hold_steps,
            "success_protocol": getattr(env.cfg, "success_protocol", "legacy"),
            "use_preinsert_deficit_reward": getattr(env.cfg, "use_preinsert_deficit_reward", False),
            "rows": rows,
            "frame_paths": frame_paths,
        }
        output.write_text(json.dumps(payload, indent=2) + "\n")
        summary = {"output": str(output), "resets": sum(row["terminated"] or row["truncated"] for row in rows)}
        for phase in dict.fromkeys(row["phase"] for row in rows):
            tail = [row for row in rows if row["phase"] == phase][-20:]
            summary[phase] = {
                side: sum(sum(row[f"{side}_finger_positions"]) / len(row[f"{side}_finger_positions"]) for row in tail)
                / len(tail)
                for side in ("left", "right")
            }
        print("PROBE_RESULT " + json.dumps(summary), flush=True)
        if args.probe == "box_ballistic":
            rise = max(row["object_pose"][2] for row in rows) - initial_object_z
            print(f"BALLISTIC_RISE_M={rise:.6f}", flush=True)
            if rise < 0.05:
                raise RuntimeError("The box did not clear the support during the ballistic geometry probe")
        wrapper.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="Isaac-OpenArm-Lift-v0")
    parser.add_argument(
        "--probe",
        choices=(
            "gripper",
            "arm_step",
            "lift",
            "insert",
            "reachability",
            "box_ballistic",
            "model",
            "grasp_release",
            "ball_passive",
        ),
        default="gripper",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--arm_joint_index", type=int, choices=range(7), default=4)
    parser.add_argument("--arm_reference_rate_scale", type=float, default=0.0)
    parser.add_argument("--arm_target_error_rad", type=float, default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--reset_after_close", action="store_true")
    parser.add_argument("--capture_frames", action="store_true")
    parser.add_argument(
        "--record_motor_efforts", action="store_true", help="Record Newton MJWarp generalized motor forces/torques"
    )
    parser.add_argument("--integral_gain", type=float, default=0.0)
    parser.add_argument("--mimic_time_constant", type=float, default=0.0)
    parser.add_argument("--mimic_impedance", type=float, default=0.0)
    parser.add_argument("--finger_condim", type=int, choices=(0, 3, 4, 6), default=0)
    parser.add_argument("--fine_contact_sdf", action="store_true")
    parser.add_argument("--mesh_ball_collision", action="store_true")
    parser.add_argument("--match_robot_mechanics", action="store_true")
    parser.add_argument("--action_trace", help="Replay a probe's commands and verify its initial state/target sequence")
    parser.add_argument("--num_steps", type=int, default=None)
    parser.add_argument("--lift_height", type=float, default=0.15)
    parser.add_argument("--approach_clearance", type=float, default=0.0)
    parser.add_argument("--grasp_transport_height", type=float, default=0.0)
    parser.add_argument("--grasp_settle_steps", type=int, default=0)
    parser.add_argument("--insert_align_steps", type=int, default=180)
    parser.add_argument("--insert_duration_steps", type=int, default=180)
    parser.add_argument("--insert_hold_steps", type=int, default=210)
    parser.add_argument("--ik_position_scale", type=float, default=None)
    parser.add_argument("--ik_rotation_scale_deg", type=float, default=None)
    parser.add_argument("--ik_target_height_offset", type=float, default=0.0)
    parser.add_argument(
        "--joint_waypoints", help="Lift-only: full-start joint path through recorded IK pose candidates"
    )
    parser.add_argument(
        "--cartesian_lift",
        action="store_true",
        help="Keep the joint-waypoint approach; track a level TCP lift in Cartesian space",
    )
    parser.add_argument("--ik_starts", type=int, default=3)
    parser.add_argument(
        "--initial_ik_candidate", help="Lift-only: start at a same-scene FK candidate to isolate grasp mechanics"
    )
    parser.add_argument(
        "--initial_ik_scale", type=float, help="Select a stored candidate while varying the action bound"
    )
    parser.add_argument(
        "--ik_action_scales", default=None, help="Comma-separated joint-range bounds for FK-only search"
    )
    parser.add_argument("--lift_duration_steps", type=int, default=120)
    parser.add_argument("--hold_steps", type=int, default=150)
    add_launcher_args(parser)
    args, overrides = parser.parse_known_args()
    if args.num_steps is not None and args.num_steps <= 0:
        parser.error("Number of probe steps must be positive")
    if args.probe == "ball_passive" and (
        args.task != "Isaac-OpenArm-LiftWithBall-v0"
        or args.action_trace is not None
        or args.joint_waypoints is not None
        or args.initial_ik_candidate is not None
    ):
        parser.error("Passive ball diagnostics require WithBall without replay, waypoints or initial IK")
    if args.cartesian_lift and (args.probe != "lift" or args.joint_waypoints is None or args.action_trace is not None):
        parser.error("Cartesian lift requires a full-start Lift joint-waypoint probe without action replay")
    if not 0.0 <= args.arm_reference_rate_scale <= 1.0:
        parser.error("Arm reference-rate scale must lie in [0, 1]; zero disables the experiment")
    if args.arm_target_error_rad is not None and (
        not 0.0 <= args.arm_target_error_rad < float("inf") or args.arm_reference_rate_scale > 0.0
    ):
        parser.error("Arm target error must be finite/nonnegative and cannot be combined with reference-rate control")
    if not 0.0 <= args.mimic_time_constant < float("inf"):
        parser.error("Mimic time constant must be finite and nonnegative")
    if not 0.0 <= args.mimic_impedance < 1.0:
        parser.error("Mimic impedance must lie in [0, 1)")
    if args.lift_duration_steps <= 0 or args.hold_steps < 0 or not 0.0 < args.lift_height < float("inf"):
        parser.error("Lift duration and height must be positive, and hold steps nonnegative")
    if not 0.0 <= args.approach_clearance < float("inf"):
        parser.error("Approach clearance must be finite and nonnegative")
    if not 0.0 <= args.grasp_transport_height < float("inf"):
        parser.error("Grasp transport height must be finite and nonnegative")
    if (
        min(args.grasp_settle_steps, args.insert_hold_steps) < 0
        or min(args.insert_align_steps, args.insert_duration_steps) <= 0
    ):
        parser.error("Grasp settling/holding must be nonnegative and insertion phase durations positive")
    if args.ik_starts < 1 or any(
        x is not None and not 0.0 < x < float("inf") for x in (args.ik_position_scale, args.ik_rotation_scale_deg)
    ):
        parser.error("IK accuracy scales and start count must be positive")
    if not -float("inf") < args.ik_target_height_offset < float("inf"):
        parser.error("IK target height offset must be finite")
    if args.initial_ik_scale is not None and (
        args.initial_ik_candidate is None or not 0.0 < args.initial_ik_scale < float("inf")
    ):
        parser.error("A candidate scale requires an IK source and must be finite and positive")
    sys.argv = [sys.argv[0], *overrides]
    cfg, _ = resolve_task_config(args.task, "skrl_mappo_cfg_entry_point")
    run(cfg, args)
