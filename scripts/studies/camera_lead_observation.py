"""Unlaunched WithBall actor-camera augmentation, not a frozen queue builder.

Own37 = original proprioception30 + measured camera4 + causal lead XY/valid3.
Partner37 remains a padded slot with the original six motion/trend channels;
no ball measurement is inserted into communication. The environment config
must declare camera_extra_observation_dim=7 BEFORE construction. Critic width
therefore also changes from81 to87; this is not an actor-only ablation.
No reward, goal, physics, base camera method or frozen source is edited.
An online adapter still needs GPU smoke, source transfer and trace auditing.
"""

from types import MethodType

import torch

from scripts.studies.camera_forecast_runtime import CameraForecastRuntime

MARKER = "_ball_camera_lead_observation_version"


def install_policy_marker(policy):
    if (
        hasattr(policy, MARKER)
        or policy.own_backbone[0].weight.shape[1] != 37
        or policy.partner_intent_encoder[0].weight.shape[1] != 37
    ):
        raise ValueError("Unmarked37D own/partner camera-lead policy required")
    policy.register_buffer(MARKER, policy.own_backbone[0].weight.new_tensor(1.0))

    def validate(module, state, prefix, *args):
        value = state.get(prefix + MARKER)
        if value is None or value.numel() != 1 or not torch.isfinite(value).all() or float(value) != 1.0:
            raise ValueError("Camera-lead checkpoint identity missing or changed")

    policy.register_load_state_dict_pre_hook(validate)


def install_environment(base):
    cfg = base.cfg
    if (
        type(base).__name__ != "OpenArmLiftWithBallEnv"
        or type(cfg).__name__ != "OpenArmLiftWithBallEnvCfg"
        or cfg.camera_extra_observation_dim != 7
        or base.own_observation_dim != 37
        or base.communication_feature_dim != 37
        or base.actor_input_dim != 74
        or cfg.state_space != 87
        or cfg.actor_object_pose_source != "apriltag"
        or cfg.apriltag_measurement_source != "camera_transform"
    ):
        raise ValueError("Exact rendered WithBall37/74/87 layout and camera-transform pose source required")
    if hasattr(base, MARKER):
        raise ValueError("Camera lead observation already installed")
    runtime = CameraForecastRuntime(
        dt=float(cfg.sim.dt) * int(cfg.decimation), position_scale=float(cfg.ball_camera_position_scale)
    )
    original_camera = base.get_additional_actor_camera_observation
    original_trace = base.get_paper_motion_context_trace_signals

    def observation(self, side):
        measured = original_camera(side)
        camera_sensor = self.scene.sensors["ball_camera"]
        # original_camera has already read camera.data, refreshing sensor buffers
        # when needed. Reading these counters must not trigger a second capture.
        lead, ready = runtime.update(
            measured,
            control_step=int(self.common_step_counter),
            episode_lengths=self.episode_length_buf,
            frame_ids=camera_sensor.frame,
            acquisition_times=camera_sensor._timestamp_last_update,
            current_times=camera_sensor._timestamp,
        )
        return torch.cat((measured, lead, ready.to(measured.dtype)[:, None]), dim=1)

    def trace(self, env_index=0):
        signals = original_trace(env_index)
        index = int(env_index)
        delivered = self.get_additional_actor_camera_observation("left")[index].detach().cpu().tolist()
        # Preserve the old field's semantics/width. New fields need an explicit
        # versioned trainer; the legacy trainer intentionally ignores them.
        signals["ball_camera_observation"] = delivered[:4]
        signals["ball_camera_lead_observation"] = delivered
        sensor = self.scene.sensors["ball_camera"]
        signals["ball_camera_frame"] = int(sensor.frame[index])
        signals["ball_camera_acquisition_time"] = float(sensor._timestamp_last_update[index])
        signals["ball_camera_current_time"] = float(sensor._timestamp[index])
        signals["ball_camera_control_step"] = int(self.common_step_counter)
        signals["ball_camera_episode_length"] = int(self.episode_length_buf[index])
        signals["ball_camera_timestamp_dtype"] = str(sensor._timestamp.dtype)
        return signals

    base.get_additional_actor_camera_observation = MethodType(observation, base)
    base.get_paper_motion_context_trace_signals = MethodType(trace, base)
    setattr(base, MARKER, 1)


def build_camera_lead_robust_bounded_all_mappo_agent(env, experiment_cfg):
    """Unqueued builder; GPU smoke/trace audit still required before training."""
    if (
        experiment_cfg["models"]["policy"].get("clip_actions") is not False
        or experiment_cfg["agent"].get("clip_predicted_values") is not True
    ):
        raise ValueError("Declared raw Gaussian samples and robust clipped-value package required")
    from isaaclab_tasks.direct.openarm_motion_context.envs.ball_center_goal import (
        install_centered_hold_goal,
        unwrap_ball_task,
    )
    from scripts.studies.bounded_all_control import build_bounded_all_control_mappo_agent
    from scripts.studies import robust_clipped_value_regression as robust

    base = unwrap_ball_task(env)
    install_environment(base)
    install_centered_hold_goal(base)
    agent = build_bounded_all_control_mappo_agent(env, experiment_cfg)
    agent._robust_clipped_value_identity = robust.install(agent)
    for uid, models in agent.models.items():
        install_policy_marker(models["policy"])
        scaler = agent.checkpoint_modules[uid]["state_preprocessor"]
        if scaler.running_mean.shape != (74,) or scaler.current_count.ndim != 0:
            raise ValueError("Standard74D camera-lead actor scaler required")
        scaler.current_count = torch.ones_like(scaler.running_mean)
    return agent
