"""Opt-in Circle reward for a robust, low-speed insertion hold region.

This reward leaves the native 45-step success predicate unchanged. It adds a
bounded team bonus only when depth has a margin beyond the success threshold,
without paying for a deeply inserted but tilted or laterally displaced peg.
The simulator geometry is reward/critic-only, never added to actor messages.

The constants were chosen after exploratory eval42 on the seed47 V11
target-depth checkpoint. They require prospective matched Full6D/NoShare
training and fresh evaluation seeds before any performance claim.
"""

import math
from types import MethodType

import torch
from isaaclab_tasks.direct.openarm_motion_context.legacy_api import (
    read_data,
    write_root_pose,
    set_joint_position,
    tensor,
)


SCALE = 48.0
DEPTH_MARGIN_M = 0.003
DEPTH_WIDTH_M = 0.002
AXIS_TARGET = 0.97
AXIS_WIDTH = 0.006
LATERAL_TARGET_M = 0.010
LATERAL_WIDTH_M = 0.003
SPEED_STD_MPS = 0.10
MARKER = "_circle_depth_hold_margin_reward_version"


def depth_hold_margin_bonus(
    axis: torch.Tensor,
    lateral: torch.Tensor,
    depth: torch.Tensor,
    success_depth: torch.Tensor,
    hole_depth: torch.Tensor,
    ee_speed: torch.Tensor,
) -> torch.Tensor:
    """Return a bounded smooth bonus for a robust insertion-hold region."""

    values = (axis, lateral, depth, success_depth, hole_depth, ee_speed)
    if (
        axis.ndim != 1
        or any(value.shape != axis.shape or not torch.isfinite(value).all() for value in values)
        or torch.any(axis < -1)
        or torch.any(axis > 1)
        or torch.any(lateral < 0)
        or torch.any(success_depth <= 0)
        or torch.any(hole_depth <= 0)
        or torch.any(success_depth > hole_depth)
        or torch.any(ee_speed < 0)
        or not all(
            math.isfinite(value) and value > 0
            for value in (SCALE, DEPTH_MARGIN_M, DEPTH_WIDTH_M, AXIS_WIDTH, LATERAL_WIDTH_M, SPEED_STD_MPS)
        )
    ):
        raise ValueError("Finite per-environment Circle pose, target, and speed required")

    # A requested safety margin must not ask the peg tip to pass through the
    # physical bottom of a shallower hole.
    depth_target = torch.minimum(success_depth + DEPTH_MARGIN_M, hole_depth)
    depth_floor = torch.sigmoid((depth - depth_target) / DEPTH_WIDTH_M)
    # The paired lower/upper gates peak inside the actual hole; unlike a
    # delayed ceiling, they never reward insertion past the physical bottom.
    depth_ceiling = torch.sigmoid((hole_depth - depth) / DEPTH_WIDTH_M)
    axis_quality = torch.sigmoid((axis - AXIS_TARGET) / AXIS_WIDTH)
    lateral_quality = torch.sigmoid((LATERAL_TARGET_M - lateral) / LATERAL_WIDTH_M)
    speed_quality = 1.0 / (1.0 + ee_speed / SPEED_STD_MPS)
    return SCALE * depth_floor * depth_ceiling * axis_quality * lateral_quality * speed_quality


def install_depth_hold_margin_reward(base, logic) -> None:
    """Compose the margin bonus after the existing V11 target-depth reward."""

    if getattr(base, "_circle_depth_hold_margin_installed", False):
        raise ValueError("Circle depth-hold margin reward already installed")
    original = base._get_rewards

    def rewards(self):
        values = original()
        metrics = logic.compute_peg_in_hole_metrics(self, self._ctx)
        body_vel_w = read_data(self.robot.data, "body_state_w")[:, :, 7:10]
        left_speed = torch.linalg.vector_norm(body_vel_w[:, self._ctx["left_ee_body_id"]], dim=-1)
        right_speed = torch.linalg.vector_norm(body_vel_w[:, self._ctx["right_ee_body_id"]], dim=-1)
        bonus = depth_hold_margin_bonus(
            metrics["axis_alignment"],
            metrics["lateral_error"],
            metrics["insertion_depth"],
            metrics["success_depth"],
            metrics["hole_depth"],
            0.5 * (left_speed + right_speed),
        )
        if not values or any(value.shape != bonus.shape for value in values.values()):
            raise ValueError("Circle team-reward shape changed")
        adjusted = {uid: value + bonus for uid, value in values.items()}
        stats = self._ctx["debug_stats"]
        amount = float(bonus.mean())
        stats["reward_circle_depth_hold_margin"] = amount
        for key in ("team_reward", "left_reward", "right_reward", "left_reward_raw", "right_reward_raw"):
            if key in stats:
                stats[key] += amount
        self.extras["log"] = dict(stats)
        return adjusted

    base._get_rewards = MethodType(rewards, base)
    base._circle_depth_hold_margin_installed = True


def build_circle_depth_hold_margin_mappo_agent(env, experiment_cfg):
    """Build the matched actor, then add the opt-in robust hold bonus."""

    from isaaclab_tasks.direct.openarm_motion_context.envs import peg_in_hole_task_logic as logic
    from scripts.studies.circle_target_depth_alignment import (
        build_circle_target_depth_alignment_mappo_agent,
    )

    agent = build_circle_target_depth_alignment_mappo_agent(env, experiment_cfg)
    base = env.unwrapped
    install_depth_hold_margin_reward(base, logic)
    for modules in agent.models.values():
        critic = modules["value"]
        critic.register_buffer(MARKER, torch.tensor(1.0, device=critic.device))

        def validate(module, incompatible):
            if MARKER in incompatible.missing_keys or float(getattr(module, MARKER)) != 1.0:
                raise ValueError("Circle depth-hold margin critic identity missing or changed")

        critic.register_load_state_dict_post_hook(validate)
    return agent
