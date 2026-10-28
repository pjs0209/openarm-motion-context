"""Unlaunched Circle-only actor ablation; never hot-edit a frozen study.

Replace own observation3:7 with the negative target-semantic Z direction
expressed in the current semantic frame, followed by zero padding. This is
q-sign invariant and drops target-local axial twist for rotationally symmetric
Circle. Physical alignment is empirically the encoded Z component for the
current Circle asset conventions; do not generalize that convention to others.
Actor dimensions, partner messages, critic states, physics and strict success
are unchanged. Both compared methods must receive this preprocessing.
"""

from types import MethodType

import torch

MARKER = "_circle_axis_actor_observation_version"


def encode_axis(quaternion):
    if (
        quaternion.ndim != 2
        or quaternion.shape[1] != 4
        or not quaternion.is_floating_point()
        or not torch.isfinite(quaternion).all()
    ):
        raise ValueError("Finite batch of floating wxyz quaternions required")
    norm = torch.linalg.vector_norm(quaternion, dim=-1, keepdim=True)
    if (norm < 1.0e-6).any():
        raise ValueError("Nonzero quaternion required")
    w, x, y, z = (quaternion / norm).unbind(-1)
    return torch.stack(
        (-2 * (x * z + w * y), -2 * (y * z - w * x), 2 * (x.square() + y.square()) - 1, torch.zeros_like(w)), dim=-1
    )


def transform_actor_observation(observation):
    if observation.ndim != 2 or observation.shape[1] not in (30, 60):
        raise ValueError("Circle own30 with optional unchanged partner30 required")
    result = observation.clone()
    result[:, 3:7] = encode_axis(observation[:, 3:7])
    return result


def install_environment(base):
    if (
        type(base.cfg).__name__ != "OpenArmFixedCirclePegAndHoleV11EnvCfg"
        or int(getattr(base.cfg, "rotational_symmetry_order", -1)) != 0
    ):
        raise ValueError("Exact rotationally symmetric Circle V11 required")
    if hasattr(base, MARKER):
        raise ValueError("Circle axis actor observation already installed")
    original = base._get_observations

    def observations(self):
        values = original()
        if set(values) != {"left_arm", "right_arm"}:
            raise ValueError("Original two-arm actor observation required")
        return {uid: transform_actor_observation(value) for uid, value in values.items()}

    base._get_observations = MethodType(observations, base)
    setattr(base, MARKER, 1)


def build_circle_axis_independent_reward_mappo_agent(env, experiment_cfg):
    from scripts.studies.circle_independent_axis_reward import build_circle_independent_axis_bounded_all_mappo_agent

    # Parent enforces exact Circle V11, original optimizer and reward identity.
    agent = build_circle_independent_axis_bounded_all_mappo_agent(env, experiment_cfg)
    install_environment(env.unwrapped)
    for models in agent.models.values():
        policy = models["policy"]
        policy.register_buffer(MARKER, torch.tensor(1.0, device=policy.device))

        def validate(module, incompatible):
            if MARKER in incompatible.missing_keys or float(getattr(module, MARKER)) != 1.0:
                raise ValueError("Circle axis actor checkpoint identity missing or changed")

        policy.register_load_state_dict_post_hook(validate)
    return agent
