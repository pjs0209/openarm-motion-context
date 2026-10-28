# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Post-physics, pre-reset contact and independent CAD-frame observations."""

import hashlib
import json
from pathlib import Path
from types import MethodType

import numpy as np
import warp as wp
from scipy.spatial.transform import Rotation

from ..legacy_api import read_data


def _circle_cad():
    from pxr import Gf, Usd, UsdGeom

    assets = Path(__file__).resolve().parents[6] / "source/isaaclab_assets/data/Objects/OpenArm/peg_in_hole"
    frames, hashes = {}, {}
    radius = None
    for role, children in (("peg", ("peg_tip", "grap_point")), ("hole", ("hole_entrance", "hole_bottom"))):
        file = assets / f"circle_{role}.usd"
        hashes[str(file)] = hashlib.sha256(file.read_bytes()).hexdigest()
        stage = Usd.Stage.Open(str(file))
        if UsdGeom.GetStageMetersPerUnit(stage) != 1:
            raise ValueError("Circle CAD audit requires metre-authored assets")
        root = stage.GetDefaultPrim()
        cache = UsdGeom.XformCache()
        inverse = cache.GetLocalToWorldTransform(root).GetInverse()
        for name in children:
            prim = stage.GetPrimAtPath(root.GetPath().AppendChild(name))
            frames[f"{role}/{name}"] = np.array((cache.GetLocalToWorldTransform(prim) * inverse).ExtractTranslation())
        if role == "peg":
            points = []
            for prim in Usd.PrimRange(root):
                if prim.IsA(UsdGeom.Mesh):
                    matrix = cache.GetLocalToWorldTransform(prim) * inverse
                    points.extend(
                        matrix.Transform(Gf.Vec3d(point)) for point in UsdGeom.Mesh(prim).GetPointsAttr().Get()
                    )
            points = np.asarray(points)
            tip = frames["peg/peg_tip"]
            axis = tip - frames["peg/grap_point"]
            axis /= np.linalg.norm(axis)
            depth = (points - tip) @ axis
            cap = points[depth > depth.max() - 1e-6] - tip
            radius = float(np.linalg.norm(cap - np.outer(cap @ axis, axis), axis=1).max())
    return frames, hashes, radius


def _cad_geometry(task, frames, radius):
    poses, rotations = {}, {}
    for role in ("peg", "hole"):
        state = read_data(getattr(task, role).data, "root_state_w")[0].detach().cpu().numpy()
        poses[role] = state[:3]
        rotations[role] = Rotation.from_quat([*state[4:7], state[3]])
    points = {
        key: poses[key.split("/")[0]] + rotations[key.split("/")[0]].apply(value) for key, value in frames.items()
    }
    peg_axis = points["peg/peg_tip"] - points["peg/grap_point"]
    hole_axis = points["hole/hole_bottom"] - points["hole/hole_entrance"]
    peg_axis /= np.linalg.norm(peg_axis)
    hole_axis /= np.linalg.norm(hole_axis)
    delta = points["peg/peg_tip"] - points["hole/hole_entrance"]
    depth = float(delta @ hole_axis)
    lateral = float(np.linalg.norm(delta - depth * hole_axis))
    cosine = float(np.clip(peg_axis @ hole_axis, -1, 1))
    entrance = points["peg/peg_tip"] - max(depth, 0) / max(cosine, 1e-6) * peg_axis
    radial = entrance - points["hole/hole_entrance"]
    radial = float(np.linalg.norm(radial - (radial @ hole_axis) * hole_axis))
    envelope = max(lateral, radial) + radius * (1 / max(cosine, 1e-6) - 1)
    passed = (
        task.cfg.success_depth_ratio * task.cfg.insertion_depth_limit
        <= depth
        <= task.cfg.insertion_depth_limit + task.cfg.success_depth_overrun_tolerance
        and envelope <= task.cfg.success_radial_clearance
        and cosine >= task.cfg.success_axis_threshold
    )
    return {
        "depth_m": depth,
        "lateral_m": lateral,
        "axis_alignment": cosine,
        "shaft_radial_envelope_m": envelope,
        "geometry_pass": bool(passed),
    }


class PhysicalAudit:
    """Read solved last-substep contacts before the task automatically resets."""

    def __init__(self, task, directory: Path, pulse):
        self.task, self.pulse = task, pulse
        self.directory = directory
        self.original_dones = task._get_dones
        self.frames, self.cad_hashes, self.radius = None, {}, None
        if getattr(task.cfg, "success_protocol", "") == "circle_contact_grasp_seating_v1":
            self.frames, self.cad_hashes, self.radius = _circle_cad()
            if abs(self.radius - task.cfg.peg_collision_radius) > 1e-6:
                raise ValueError("Authored shaft radius differs from configured clearance geometry")
        solver = task.sim.physics_manager._solver
        model = solver.mj_model
        roles = [
            getattr(task, name).cfg.prim_path.rsplit("/", 1)[-1]
            for name in ("object", "peg", "hole")
            if hasattr(task, name)
        ]
        self.parts = {
            i: model.body(i).name
            for i in range(model.nbody)
            if any(model.body(i).name.endswith("_" + role) for role in roles)
        }
        if not self.parts:
            raise ValueError("No native task body found for contact audit")
        self.native_parts = {}
        for index, name in self.parts.items():
            joint = model.body_jntadr[index]
            self.native_parts[name] = {
                "mass_kg": float(model.body_mass[index]),
                "gravity_compensation": float(model.body_gravcomp[index]),
                "free_body": bool(model.body_jntnum[index] == 1 and joint >= 0 and model.jnt_type[joint] == 0),
            }
        self.stream = (directory / "physical_audit.jsonl").open("x")
        self.rows = 0
        self.max_overlap = {}
        self.geometry_pass_ticks = {}

        def dones(env):
            result = self.original_dones()
            self.record()
            return result

        task._get_dones = MethodType(dones, task)

    def record(self):
        import mujoco_warp as mjw

        task = self.task
        solver = task.sim.physics_manager._solver
        model, data = solver.mj_model, solver.mjw_data
        count = int(data.nacon.numpy()[0])
        if count > data.naconmax:
            raise RuntimeError("Contact buffer overflow invalidates physical audit")
        ids = wp.array(np.arange(count, dtype=np.int32), dtype=wp.int32, device=task.device)
        output = wp.empty(count, dtype=wp.spatial_vector, device=task.device)
        if count:
            mjw.contact_force(solver.mjw_model, data, ids, True, output)
        forces = output.numpy()
        geoms = data.contact.geom.numpy()[:count]
        worlds = data.contact.worldid.numpy()[:count]
        distances = data.contact.dist.numpy()[:count]
        frames = data.contact.frame.numpy()[:count]
        addresses = data.contact.efc_address.numpy()[:count, 0]
        pairs = {}
        net = {name: np.zeros(3) for name in self.parts.values()}
        for index, pair in enumerate(geoms):
            if worlds[index] != 0 or np.any(pair < 0):
                continue
            bodies = [int(model.geom_bodyid[int(geometry)]) for geometry in pair]
            if not any(body in self.parts for body in bodies):
                continue
            names = [model.body(body).name for body in bodies]
            key = " <-> ".join(names)
            normal = float(forces[index, :3] @ frames[index, 0])
            if not np.isfinite(forces[index]).all() or not np.isfinite(distances[index]) or normal < -1e-3:
                raise RuntimeError("Invalid solved contact observation")
            entry = pairs.setdefault(
                key,
                {
                    "contacts": 0,
                    "active_constraints": 0,
                    "minimum_solver_distance_m": float(distances[index]),
                    "sum_normal_pressure_n": 0.0,
                },
            )
            entry["contacts"] += 1
            entry["active_constraints"] += int(addresses[index] >= 0)
            entry["minimum_solver_distance_m"] = min(entry["minimum_solver_distance_m"], float(distances[index]))
            entry["sum_normal_pressure_n"] += max(0, normal)
            for body, sign in zip(bodies, (-1, 1)):
                if body in self.parts:
                    net[self.parts[body]] += sign * forces[index, :3]
            self.max_overlap[key] = max(self.max_overlap.get(key, 0.0), -float(distances[index]))
        geometry = _cad_geometry(task, self.frames, self.radius) if self.frames else None
        terminal_success = bool(task._ctx.get("termination_success", [False])[0])
        if terminal_success and geometry is not None and not geometry["geometry_pass"]:
            raise RuntimeError("Native task success disagrees with independently recomputed CAD insertion geometry")
        episode = str(self.pulse.episode)
        if geometry and geometry["geometry_pass"]:
            self.geometry_pass_ticks[episode] = self.geometry_pass_ticks.get(episode, 0) + 1
        row = {
            "episode_index": self.pulse.episode,
            "control_tick": int(task.episode_length_buf[0]),
            "time_s": float(task.episode_length_buf[0]) * task.cfg.sim.dt * task.cfg.decimation,
            "pairs": pairs,
            "net_part_contact_force_w_n": {name: value.tolist() for name, value in net.items()},
            "fresh_cad_geometry": geometry,
            "task_terminal_success": terminal_success,
        }
        self.stream.write(json.dumps(row, allow_nan=False) + "\n")
        self.rows += 1

    def close(self):
        self.stream.close()
        self.task._get_dones = self.original_dones
        (self.directory / "physical_audit.json").write_text(
            json.dumps(
                {
                    "boundary": "after native solver and termination update, before automatic reset",
                    "contact_force_semantics": "last solver substep world-frame force, not control-step average",
                    "solver_distance_semantics": (
                        "signed MuJoCo contact distance from SDF/collision approximation, not exact CAD overlap"
                    ),
                    "normal_pressure_is_not_net_force": True,
                    "native_parts": self.native_parts,
                    "cad_sha256": self.cad_hashes,
                    "shaft_radius_from_usd_tip_cap_m": self.radius,
                    "observed_control_ticks": self.rows,
                    "maximum_solver_overlap_m_by_pair": self.max_overlap,
                    "independent_geometry_pass_ticks_by_episode": self.geometry_pass_ticks,
                },
                indent=2,
                allow_nan=False,
            )
            + "\n"
        )
