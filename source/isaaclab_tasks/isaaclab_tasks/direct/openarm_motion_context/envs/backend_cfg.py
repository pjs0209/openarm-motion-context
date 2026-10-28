# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""OpenArm Newton/MJWarp default and explicit historical PhysX replay profile."""

from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg, NewtonCollisionPipelineCfg
from isaaclab_physx.physics import PhysxCfg

from isaaclab.utils import configclass

from isaaclab_tasks.utils import PresetCfg


@configclass
class OpenArmPhysicsCfg(PresetCfg):
    """Keep engine-specific settings out of the common task logic.

    Short multi-world runtime, reset, camera and learning checks cover the
    Newton path. They do not establish success rates or PhysX equivalence.
    Validate contact capacity again when scaling/changing the scene; solver
    convergence defaults are intentionally not retuned here.
    """

    isaacsim_physx = PhysxCfg(
        bounce_threshold_velocity=0.2,
        enable_ccd=True,
        enable_external_forces_every_iteration=True,
        min_position_iteration_count=16,
        min_velocity_iteration_count=1,
        solve_articulation_contact_last=True,
        gpu_max_rigid_contact_count=2**22,
        gpu_max_rigid_patch_count=2**22,
        gpu_found_lost_aggregate_pairs_capacity=2**25,
        gpu_total_aggregate_pairs_capacity=2**23,
        friction_correlation_distance=0.00625,
    )
    newton_mjwarp = NewtonCfg(
        # The cuBQL mesh-overlap path faulted on this OpenArm CAD scene on
        # Ada. SAH keeps the same collider surfaces and passed the paired
        # 64-world stress run; do not replace cavities with convex hulls.
        bvh_constructor_geometry="sah",
        solver_cfg=MJWarpSolverCfg(
            solver="newton",
            integrator="implicitfast",
            # Full-action 64-world PPO requested up to 4156 constraint rows
            # per world; small-action preflights underestimated this demand.
            # Retain headroom without changing contacts or convergence settings.
            njmax=8192,
            nconmax=512,
            # Keep the box interior and insertion bore non-convex. MuJoCo's
            # internal mesh contacts would replace them with convex hulls.
            use_mujoco_contacts=False,
            update_data_interval=1,
        ),
        collision_cfg=NewtonCollisionPipelineCfg(),
        num_substeps=2,
        debug_mode=False,
        use_cuda_graph=False,
    )
    default = newton_mjwarp


def configure_openarm_contact_capacity(physics, num_envs):
    """Scale the global mesh-pair workspace without changing contact physics.

    A 64-world Circle probe exceeded the default one-million triangle-pair
    buffer (~1.044 million). Four times the observed per-world count leaves
    headroom for different poses; rerun overflow checks for changed geometry.
    User-provided larger capacities are preserved.
    """
    if isinstance(physics, NewtonCfg):
        physics.collision_cfg.max_triangle_pairs = max(physics.collision_cfg.max_triangle_pairs, 65536 * int(num_envs))
