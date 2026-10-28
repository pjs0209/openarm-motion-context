"""Circle-only reward ablation: retain axis reward away from preinsert target.

Replace the existing alignment component scale*preinsert_stage*axis_quality
with scale*axis_quality. All other reward terms, physics, observations and
strict insertion/hold conditions stay unchanged. This is not potential-based
shaping: a distant aligned pose may become more attractive. Both methods must
receive the change and be evaluated against the unchanged strict goal.
"""

import ast
import inspect
import math
from types import MethodType

import torch

MARKER = "_circle_independent_axis_reward_version"


def correction(axis_quality, preinsert_stage, scale):
    if not math.isfinite(scale) or scale < 0:
        raise ValueError("Finite nonnegative alignment scale required")
    if (
        axis_quality.ndim != 1
        or axis_quality.shape != preinsert_stage.shape
        or any(
            not torch.isfinite(value).all() or (value < 0).any() or (value > 1).any()
            for value in (axis_quality, preinsert_stage)
        )
    ):
        raise ValueError("Finite per-environment qualities in[0,1] required")
    return scale * axis_quality * (1 - preinsert_stage)


def validate_original(function):
    tree = ast.parse(inspect.getsource(function))
    assignments = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "r_axis_alignment" for target in node.targets)
    ]
    if len(assignments) != 1 or ast.unparse(assignments[0].value) != "r_preinsert_stage * axis_quality":
        raise ValueError("Original preinsert-coupled axis reward required")


def install_environment(base, logic):
    if hasattr(base, "_circle_independent_axis_reward_original"):
        raise ValueError("Reward adjustment already installed")
    validate_original(logic.compute_openarm_re_rewards)
    original = base._get_rewards
    base._circle_independent_axis_reward_original = original

    def rewards(self):
        # Preserve original fixed-joint attachment checks and all base terms.
        values = original()
        metrics = logic.compute_peg_in_hole_metrics(self, self._ctx)
        _, keypoints = logic._compute_factory_keypoint_reward(self, metrics)
        _, _, _, gate = logic._smooth_insertion_gate(
            self, metrics["axis_alignment"], metrics["lateral_error"], metrics["twist_alignment"]
        )
        quality = logic.axis_alignment_quality(metrics["axis_alignment"], self.cfg.alignment_shaping_power)
        delta = correction(quality, (1 - gate) * keypoints["r_preinsert"], float(self.cfg.reward_axis_alignment_scale))
        if not values or any(value.shape != delta.shape for value in values.values()):
            raise ValueError("Reward shape changed")
        adjusted = {uid: value + delta for uid, value in values.items()}
        stats = self._ctx["debug_stats"]
        stats.update(
            circle_independent_axis_reward_version=1.0,
            reward_circle_independent_axis_delta=float(delta.mean()),
            reward_circle_independent_axis_component=float(
                (float(self.cfg.reward_axis_alignment_scale) * quality).mean()
            ),
        )
        # Trace reward_mean is sourced from these stats; do not leave base totals.
        for name in ("team_reward", "left_reward", "right_reward", "left_reward_raw", "right_reward_raw"):
            stats[name] = float(adjusted["left_arm"].mean())
        self.extras["log"] = dict(stats)
        return adjusted

    base._get_rewards = MethodType(rewards, base)


def build_circle_independent_axis_bounded_all_mappo_agent(env, experiment_cfg):
    from isaaclab_tasks.direct.openarm_motion_context.envs import peg_in_hole_task_logic as logic
    from scripts.studies.bounded_all_control import build_bounded_all_control_mappo_agent

    base = env.unwrapped
    if (
        getattr(type(base.cfg), "__name__", "") != "OpenArmFixedCirclePegAndHoleV11EnvCfg"
        or int(getattr(base.cfg, "rotational_symmetry_order", -1)) != 0
        or getattr(getattr(base._get_rewards, "__func__", None), "__qualname__", "")
        != "OpenArmFixedPegAndHoleEnv._get_rewards"
        or experiment_cfg["agent"].get("clip_predicted_values") is not True
    ):
        raise ValueError("Exact Circle V11 and original clipping-on optimizer required")
    agent = build_bounded_all_control_mappo_agent(env, experiment_cfg)
    install_environment(base, logic)
    for modules in agent.models.values():
        critic = modules["value"]
        critic.register_buffer(MARKER, torch.tensor(1.0, device=critic.device))

        def validate(module, incompatible):
            if MARKER in incompatible.missing_keys or float(getattr(module, MARKER)) != 1.0:
                raise ValueError("Circle reward critic identity missing or changed")

        critic.register_load_state_dict_post_hook(validate)
    return agent
