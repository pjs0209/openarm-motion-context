"""Opt-in, matched Lift reward for retaining both grasp orientations aloft.

The original Lift reward and native success predicate remain unchanged. This
adds a small, bounded team bonus only when the box is elevated, both wrists
are near their actor-visible grasp targets, and both grippers are closing.
It is a proposed matched Full6D/NoShare experiment, not an active study.
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


# Archived trace replay puts this below about 2 per elevated step for the
# successful seed-1 policy, versus the native Lift term's scale of 20.
BONUS_SCALE = 8.0
ORIENTATION_WIDTH_DEG = 10.0
MARKER = "_lift_orientation_retention_reward_version"


def orientation_retention_bonus(
    height_fraction: torch.Tensor,
    left_distance_m: torch.Tensor,
    right_distance_m: torch.Tensor,
    left_error_rad: torch.Tensor,
    right_error_rad: torch.Tensor,
    left_closure: torch.Tensor,
    right_closure: torch.Tensor,
    *,
    position_width_m: float,
    success_orientation_deg: float,
) -> torch.Tensor:
    """Return bounded nonnegative reward using only actor-side pose signals."""

    tensors = (
        height_fraction,
        left_distance_m,
        right_distance_m,
        left_error_rad,
        right_error_rad,
        left_closure,
        right_closure,
    )
    shape = height_fraction.shape
    if (
        len(shape) != 1
        or any(t.shape != shape or not torch.isfinite(t).all() for t in tensors)
        or any(torch.any(t < 0) for t in (left_distance_m, right_distance_m, left_error_rad, right_error_rad))
        or not math.isfinite(position_width_m)
        or position_width_m <= 0
        or not math.isfinite(success_orientation_deg)
        or success_orientation_deg <= 0
    ):
        raise ValueError("Finite per-environment Lift pose/closure signals required")
    elevated = height_fraction.clamp(0.0, 1.0)
    near = torch.exp(-torch.maximum(left_distance_m, right_distance_m) / position_width_m)
    closed = torch.minimum(left_closure, right_closure).clamp(0.0, 1.0)
    worst_error_deg = torch.rad2deg(torch.maximum(left_error_rad, right_error_rad))
    alignment = torch.sigmoid((success_orientation_deg - worst_error_deg) / ORIENTATION_WIDTH_DEG)
    return BONUS_SCALE * elevated * near * closed * alignment


def install_orientation_retention_reward(base, logic):
    if getattr(base, "_lift_orientation_retention_installed", False):
        raise ValueError("Lift orientation-retention reward already installed")
    original = base._get_rewards

    def rewards(self):
        values = original()
        ctx = self._ctx
        robot = ctx["robot"]
        left_target, right_target, left_target_q, right_target_q = logic.compute_actor_collision_target_poses_w(
            self, ctx
        )
        left_ee = read_data(robot.data, "body_pos_w")[:, ctx["left_ee_body_id"]]
        right_ee = read_data(robot.data, "body_pos_w")[:, ctx["right_ee_body_id"]]
        left_dist = torch.linalg.vector_norm(left_ee - left_target, dim=-1)
        right_dist = torch.linalg.vector_norm(right_ee - right_target, dim=-1)
        left_error = logic.quat_error_magnitude(
            read_data(robot.data, "body_quat_w")[:, ctx["left_ee_body_id"]], left_target_q
        )
        right_error = logic.quat_error_magnitude(
            read_data(robot.data, "body_quat_w")[:, ctx["right_ee_body_id"]], right_target_q
        )
        left_open = read_data(robot.data, "joint_pos")[:, ctx["left_gripper_joint_ids"]].mean(dim=-1)
        right_open = read_data(robot.data, "joint_pos")[:, ctx["right_gripper_joint_ids"]].mean(dim=-1)
        object_pos, _ = logic.get_actor_object_pose_w(self, ctx)
        height_fraction = (object_pos[:, 2] - ctx["initial_actor_object_pos_w"][:, 2]) / max(
            float(self.cfg.lift_target_height), 1.0e-6
        )
        bonus = orientation_retention_bonus(
            height_fraction,
            left_dist,
            right_dist,
            left_error,
            right_error,
            logic._closure_from_opening(self, left_open),
            logic._closure_from_opening(self, right_open),
            position_width_m=float(self.cfg.grasp_position_std),
            success_orientation_deg=float(self.cfg.success_grasp_orientation_deg),
        )
        if not values or any(value.shape != bonus.shape for value in values.values()):
            raise ValueError("Lift team-reward shape changed")
        adjusted = {uid: value + bonus for uid, value in values.items()}
        stats = ctx["debug_stats"]
        amount = float(bonus.mean())
        stats["reward_lift_orientation_retention"] = amount
        for key in ("team_reward", "left_reward", "right_reward", "left_reward_raw", "right_reward_raw"):
            if key in stats:
                stats[key] += amount
        self.extras["log"] = dict(stats)
        return adjusted

    base._get_rewards = MethodType(rewards, base)
    base._lift_orientation_retention_installed = True


def build_lift_orientation_retention_mappo_agent(env, experiment_cfg):
    from isaaclab_tasks.direct.openarm_motion_context.envs import lift_task_logic as logic
    from scripts.studies.raw_reference_actor import build_raw_reference_mappo_agent

    base = env.unwrapped
    if (
        getattr(type(base.cfg), "__name__", "") != "OpenArmReIncrementalBimanualLiftEnvCfg"
        or getattr(base.cfg, "intent_task_label", "") != "openarm_lift"
        or getattr(getattr(base._get_rewards, "__func__", None), "__qualname__", "")
        != "OpenArmReBimanualLiftEnv._get_rewards"
    ):
        raise ValueError("Exact native Lift task and reward required")
    # Match the completed raw seed-5 reference package exactly except for
    # this reward. The raw builder rejects experimental mean/noise guards.
    agent = build_raw_reference_mappo_agent(env, experiment_cfg)
    install_orientation_retention_reward(base, logic)
    for modules in agent.models.values():
        critic = modules["value"]
        critic.register_buffer(MARKER, torch.tensor(1.0, device=critic.device))

        def validate(module, incompatible):
            if MARKER in incompatible.missing_keys or float(getattr(module, MARKER)) != 1.0:
                raise ValueError("Lift orientation-retention critic identity missing or changed")

        critic.register_load_state_dict_post_hook(validate)
    return agent
