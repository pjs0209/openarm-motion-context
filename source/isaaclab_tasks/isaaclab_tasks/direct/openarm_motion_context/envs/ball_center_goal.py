"""Opt-in centered-hold goal; never changes the standard Lift/Ball classes.

The versioned agent builder installs these instance hooks before training/eval.
Ground-truth ball geometry is used only for reward/termination, not observations.
"""

from types import MethodType
import math

import torch

VARIANT = "centered_hold_v1"


def unwrap_ball_task(env):
    """Find the owner of the bound metrics method, not a delegating wrapper."""
    current, visited = env, set()
    while current is not None and id(current) not in visited:
        visited.add(id(current))
        method = getattr(current, "_ball_metrics", None)
        owner = getattr(method, "__self__", None)
        if owner is not None:
            return owner
        next_env = getattr(current, "_env", None)
        if next_env is None:
            next_env = getattr(current, "env", None)
        if next_env is None:
            next_env = getattr(current, "unwrapped", None)
        current = next_env
    raise ValueError("Could not unwrap the rendered-camera Ball task")


def install_centered_hold_goal(env):
    if getattr(env, "_ball_goal_variant", None) == VARIANT:
        return
    if not hasattr(env, "_ball_metrics") or env.cfg.ball_reward_profile != "center_penalty":
        raise ValueError("Centered-hold variant requires the rendered-camera center-penalty Ball task")
    hold = int(env.cfg.hold_required_steps)
    scale = float(env.cfg.reward_success_scale)
    if hold < 2 or not math.isfinite(scale) or scale < 0:
        raise ValueError("Invalid centered-hold success parameters")
    original_dones, original_rewards, original_reset = env._get_dones, env._get_rewards, env._reset_idx
    env._ctx["ball_box_only_hold_count"] = torch.zeros_like(env._ctx["success_hold_count"])

    def dones(self):
        previous = self._ctx["success_hold_count"].clone()
        terminated, timed_out = original_dones()
        # Parent increments its counter iff all ORIGINAL Lift predicates hold.
        box_now = self._ctx["success_hold_count"] > 0
        _, metrics = self._ball_metrics()
        center = metrics["center_ok"]
        if center.shape != previous.shape or center.dtype != torch.bool:
            raise ValueError("Invalid ball goal predicate")
        self._ctx["ball_box_only_hold_count"] = torch.where(box_now, self._ctx["ball_box_only_hold_count"] + 1, 0)
        self._ctx["success_hold_count"] = torch.where(box_now & center, previous + 1, 0)
        success = self._ctx["success_hold_count"] >= hold
        # Remove only the old box-success reset, preserving every physical failure.
        failed = torch.zeros_like(success)
        for key in ("termination_drop", "termination_far", "termination_tilt_fail", "termination_invalid"):
            failed |= self._ctx[key]
        self._ctx["termination_success"] = success
        self._ctx["ball_goal_box_now"] = box_now
        return {uid: failed | success for uid in terminated}, timed_out

    def rewards(self):
        result = original_rewards()
        # Keep this opt-in goal consistent with the parent's terminal bonus.
        early = self._ctx["trace_signals"]["terminal_success"]
        actual = self._ctx["termination_success"]
        delta = scale * (actual.float() - early.float())
        result = {uid: value + delta for uid, value in result.items()}
        # Reset clears trace tensors in place. Never alias termination_success.
        self._ctx["trace_signals"]["terminal_success"] = actual.detach().clone()
        for key in ("left_reward", "right_reward"):
            if key in self._ctx["trace_signals"]:
                self._ctx["trace_signals"][key] = self._ctx["trace_signals"][key] + delta
        logs = self.extras.setdefault("log", {})
        logs.update(
            {
                "terminal_success_bonus": float((scale * actual.float()).mean()),
                "paper/terminal_success_bonus": float((scale * actual.float()).mean()),
                "reward_success_term": float((scale * actual.float()).mean()),
                "paper/success": float(actual.float().mean()),
                "ball_joint_success_ratio": float(actual.float().mean()),
                "box_only_success_ratio": float((self._ctx["ball_box_only_hold_count"] >= hold).float().mean()),
            }
        )
        for key in (
            "team_reward",
            "left_reward",
            "right_reward",
            "left_reward_raw",
            "right_reward_raw",
            "left_reward_total",
            "right_reward_total",
        ):
            if key in logs:
                logs[key] += float(delta.mean())
        return result

    def reset(self, env_ids):
        original_reset(env_ids)
        if env_ids is None:
            self._ctx["ball_box_only_hold_count"].zero_()
        else:
            self._ctx["ball_box_only_hold_count"][env_ids] = 0

    env._get_dones = MethodType(dones, env)
    env._get_rewards = MethodType(rewards, env)
    env._reset_idx = MethodType(reset, env)
    env._ball_goal_variant = VARIANT
