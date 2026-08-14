# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""skrl 기반 OpenArm 양팔 lift policy 실행/evaluation 진입점.

학습 script와 같은 config/runtime 흐름을 사용하지만, 여기서는 checkpoint loading,
deterministic/stochastic evaluation, motion-context trace 저장에 집중한다.

motion-context 분석에는 ``--save_motion_context_trace``와 고유한
``--motion_context_trace_dir``를 사용한다. 기존 mode-trace 옵션도 호환된다.
"""

import argparse
import importlib
import sys

from isaaclab.app import AppLauncher


# 학습 script와 마찬가지로 simulator-heavy module import 전에 launcher argument를 parsing한다.
parser = argparse.ArgumentParser(description="Play OpenArm bimanual direct latent-interaction-intent MAPPO.")


def _str_to_bool(value):
    """선택형 boolean flag에서 자주 쓰는 문자열 표현을 bool로 바꾼다."""

    if isinstance(value, bool):
        return value
    return str(value).lower() in ("1", "true", "yes", "on")


parser.add_argument("--mode", type=str, default="intent_share", choices=("intent_share", "no_share"))
parser.add_argument(
    "--task",
    type=str,
    default="Isaac-Lift-OpenArm-Bimanual-Direct-PlusJoint-v0",
    help="Task name registered in gym.",
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments.")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint.")
parser.add_argument("--num_steps", type=int, default=1000, help="Number of evaluation steps.")
parser.add_argument("--debug_interval", type=int, default=25, help="Print debug stats every N steps.")
parser.add_argument("--intent_head_checkpoint", type=str, default=None, help="Path to latent-interaction-intent module weights.")
parser.add_argument(
    "--save_motion_context_trace",
    nargs="?",
    const=True,
    default=False,
    type=_str_to_bool,
    help="Save motion-context communication traces during play/eval.",
)
parser.add_argument("--motion_context_trace_dir", type=str, default=None, help="Directory for motion-context trace JSON files.")
parser.add_argument(
    "--save_mode_trace",
    nargs="?",
    const=True,
    default=False,
    type=_str_to_bool,
    help="Deprecated alias for --save_motion_context_trace.",
)
parser.add_argument("--mode_trace_dir", type=str, default=None, help="Directory for motion-mode trace JSON files.")
parser.add_argument("--mode_trace_env_index", type=int, default=None, help="Environment index to record in motion-mode traces.")
parser.add_argument("--mode_trace_format", type=str, default=None, choices=("json",), help="Trace format for motion-mode traces.")
parser.add_argument("--num_eval_episodes", type=int, default=None, help="Stop trace-saving eval after this many episodes.")
parser.add_argument(
    "--deterministic_eval",
    nargs="?",
    const=True,
    default=False,
    type=_str_to_bool,
    help="Use policy mean actions during eval.",
)
parser.add_argument(
    "--stochastic_eval",
    nargs="?",
    const=True,
    default=False,
    type=_str_to_bool,
    help="Use sampled actions during eval.",
)
parser.add_argument(
    "--save_mode_frames",
    nargs="?",
    const=True,
    default=False,
    type=_str_to_bool,
    help="Enable optional representative mode frame capture.",
)
parser.add_argument("--mode_frame_threshold", type=float, default=None, help="Mode probability threshold for representative frame capture.")
parser.add_argument(
    "--intent_variant",
    type=str,
    default=None,
    choices=("no_intent", "share_intent"),
    help="Final MotionMode variant used by the checkpoint: no_intent or share_intent.",
)
parser.add_argument(
    "--intent_arch",
    type=str,
    default=None,
    choices=("shared_intent_encoder",),
    help="share_intent architecture used by the checkpoint. Only shared_intent_encoder is supported.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym

from isaaclab.utils.assets import retrieve_file_path
from isaaclab_rl.skrl import SkrlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config


def _unwrap_task_env(env):
    """wrapper 아래에 있는 실제 Isaac Lab task env를 찾는다."""

    def _is_task(obj) -> bool:
        return hasattr(obj, "possible_agents") and hasattr(obj, "action_spaces") and hasattr(obj, "observation_spaces")

    try:
        base = getattr(env, "unwrapped", None)
        if base is not None and _is_task(base):
            return base
    except Exception:
        pass

    queue = [env]
    visited = set()
    while queue:
        current = queue.pop(0)
        if current is None or id(current) in visited:
            continue
        visited.add(id(current))
        if _is_task(current):
            return current
        for attr in ("env", "_env", "unwrapped"):
            try:
                nxt = getattr(current, attr, None)
            except Exception:
                nxt = None
            if nxt is not None and nxt is not current:
                queue.append(nxt)
    return env


def _attach_multi_agent_api(wrapped_env, task_env):
    """task metadata와 intent hook을 skrl wrapper에 노출한다."""

    if hasattr(task_env, "intent_horizon"):
        wrapped_env.intent_horizon = int(getattr(task_env, "intent_horizon"))
    if hasattr(task_env, "intent_stride"):
        wrapped_env.intent_stride = int(getattr(task_env, "intent_stride"))
    if hasattr(task_env, "intent_feature_dim"):
        wrapped_env.intent_feature_dim = int(getattr(task_env, "intent_feature_dim"))
    for attr in (
        "motion_intent_dim",
        "latent_mode_dim",
        "motion_intent_horizon",
        "interaction_motion_scale",
        "base_own_obs_dim",
        "gripper_extra_obs_dim",
        "use_gripper_proprio_obs",
        "intent_variant",
        "intent_arch",
        "intent_task_label",
        "actor_partner_intent_dim",
        "actor_input_dim",
    ):
        if hasattr(task_env, attr):
            setattr(wrapped_env, attr, getattr(task_env, attr))
    if hasattr(task_env, "intent_target_dim"):
        wrapped_env.intent_target_dim = int(getattr(task_env, "intent_target_dim"))
    if hasattr(task_env, "own_observation_dim"):
        wrapped_env.own_observation_dim = int(getattr(task_env, "own_observation_dim"))
    wrapped_env.sharing_mode = str(getattr(task_env, "sharing_mode", args_cli.mode))
    if hasattr(task_env, "set_pending_intents"):
        wrapped_env.set_pending_intents = task_env.set_pending_intents
    if hasattr(task_env, "get_current_motion_intents"):
        wrapped_env.get_current_motion_intents = task_env.get_current_motion_intents
    return wrapped_env


def _load_symbol(path: str):
    """YAML의 ``module:symbol`` runtime 참조를 import한다."""

    module_name, symbol_name = path.split(":", 1)
    module = importlib.import_module(module_name)
    return getattr(module, symbol_name)


def _runtime_components(agent_cfg: dict):
    """Load the motion-context agent and trainer declared by the task YAML."""

    runtime_cfg = agent_cfg.get("runtime", {}) if isinstance(agent_cfg, dict) else {}
    try:
        builder_path = runtime_cfg["agent_builder"]
        trainer_path = runtime_cfg["trainer"]
    except KeyError as exc:
        raise ValueError("Motion-context agent YAML requires runtime.agent_builder and runtime.trainer") from exc
    return _load_symbol(builder_path), _load_symbol(trainer_path)


def _canonical_intent_variant(variant: str | None) -> str | None:
    """최종 MotionMode variant 이름을 정규화한다."""

    if variant is None:
        return None
    return str(variant)


def _apply_compact_motion_mode_cfg(agent_cfg: dict, env_cfg=None) -> None:
    """실행 시점 override 전에 압축된 mode YAML block을 적용한다."""

    mode_cfg = agent_cfg.get("mode_cfg", {}) if isinstance(agent_cfg, dict) else {}
    if mode_cfg:
        policy_cfg = agent_cfg.setdefault("models", {}).setdefault("policy", {})
        agent_section = agent_cfg.setdefault("agent", {})
        trainer_section = agent_cfg.setdefault("trainer", {})
        if "temperature" in mode_cfg:
            policy_cfg["mode_temperature"] = mode_cfg["temperature"]
            agent_section["mode_temperature"] = mode_cfg["temperature"]
        if "lambda_pred" in mode_cfg:
            agent_section["lambda_pred"] = mode_cfg["lambda_pred"]
        if "lambda_temporal" in mode_cfg:
            agent_section["lambda_temporal"] = mode_cfg["lambda_temporal"]
        if "low_confidence_threshold" in mode_cfg:
            agent_section["mode_low_confidence_threshold"] = mode_cfg["low_confidence_threshold"]
        if "share_strategy" in mode_cfg:
            trainer_section["mode_share_strategy"] = mode_cfg["share_strategy"]
        if "share_start" in mode_cfg:
            trainer_section["mode_share_start"] = mode_cfg["share_start"]
        if "share_confidence_threshold" in mode_cfg:
            trainer_section["mode_share_confidence_threshold"] = mode_cfg["share_confidence_threshold"]


@hydra_task_config(args_cli.task, "skrl_mappo_cfg_entry_point")
def main(env_cfg, agent_cfg):
    """checkpoint evaluation을 위한 Hydra entry point."""

    checkpoint_path = retrieve_file_path(args_cli.checkpoint)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    if hasattr(env_cfg, "sharing_mode"):
        env_cfg.sharing_mode = args_cli.mode
    _apply_compact_motion_mode_cfg(agent_cfg, env_cfg)
    if args_cli.intent_variant is not None:
        variant = _canonical_intent_variant(args_cli.intent_variant)
        if hasattr(env_cfg, "intent_variant"):
            env_cfg.intent_variant = variant
        agent_cfg.setdefault("models", {}).setdefault("policy", {})["intent_variant"] = variant
        agent_cfg.setdefault("agent", {})["intent_variant"] = variant
        agent_cfg.setdefault("trainer", {})["intent_variant"] = variant
    if args_cli.intent_arch is not None:
        if args_cli.intent_variant == "no_intent":
            raise ValueError("--intent_arch is only meaningful with --intent_variant share_intent.")
        if hasattr(env_cfg, "intent_arch"):
            env_cfg.intent_arch = args_cli.intent_arch
        agent_cfg.setdefault("models", {}).setdefault("policy", {})["intent_arch"] = args_cli.intent_arch
        agent_cfg.setdefault("agent", {})["intent_arch"] = args_cli.intent_arch
        agent_cfg.setdefault("trainer", {})["intent_arch"] = args_cli.intent_arch
    if hasattr(env_cfg, "__post_init__"):
        env_cfg.__post_init__()

    print(f"[INFO] Using sharing mode for play: {args_cli.mode}")
    if args_cli.intent_variant is not None:
        print(f"[INFO] Using intent_variant for play: {_canonical_intent_variant(args_cli.intent_variant)}")
    if args_cli.intent_arch is not None:
        print(f"[INFO] Using intent_arch for play: {args_cli.intent_arch}")

    # 선택적 trace frame 저장을 위해 play는 항상 rgb_array render mode로 env를 만든다.
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array")
    task_env = _unwrap_task_env(env)
    if not hasattr(task_env, "possible_agents"):
        raise RuntimeError("Failed to unwrap multi-agent task env.")
    env = SkrlVecEnvWrapper(env, ml_framework="torch", wrapper="isaaclab-multi-agent")
    env = _attach_multi_agent_api(env, task_env)
    agent_builder, trainer_cls = _runtime_components(agent_cfg)
    agent = agent_builder(env, agent_cfg)
    # 학습 시점 trainer config를 기반으로 play 전용 trace/eval 설정만 덧씌운다.
    trainer_cfg = dict(agent_cfg["trainer"])
    trainer_cfg["timesteps"] = args_cli.num_steps
    trainer_cfg["close_environment_at_exit"] = False
    trainer_cfg["debug_interval"] = args_cli.debug_interval
    trainer_cfg["mode_trace_task"] = args_cli.task
    trainer_cfg["mode_trace_checkpoint"] = checkpoint_path
    if args_cli.save_motion_context_trace or args_cli.save_mode_trace:
        trainer_cfg["save_mode_trace"] = True
    trace_dir = args_cli.motion_context_trace_dir or args_cli.mode_trace_dir
    if trace_dir is not None:
        trainer_cfg["mode_trace_dir"] = trace_dir
    if args_cli.mode_trace_env_index is not None:
        trainer_cfg["mode_trace_env_index"] = args_cli.mode_trace_env_index
    if args_cli.mode_trace_format is not None:
        trainer_cfg["mode_trace_format"] = args_cli.mode_trace_format
    if args_cli.num_eval_episodes is not None:
        trainer_cfg["num_trace_episodes"] = args_cli.num_eval_episodes
    if args_cli.deterministic_eval:
        trainer_cfg["deterministic_eval"] = True
        trainer_cfg["stochastic_evaluation"] = False
    if args_cli.stochastic_eval:
        trainer_cfg["deterministic_eval"] = False
        trainer_cfg["stochastic_evaluation"] = True
    if args_cli.save_mode_frames:
        trainer_cfg["save_mode_frames"] = True
    if args_cli.mode_frame_threshold is not None:
        trainer_cfg["mode_frame_threshold"] = args_cli.mode_frame_threshold
    trainer = trainer_cls(env=env, agents=agent, cfg=trainer_cfg)

    # checkpoint는 현재 task/intent_variant dimension과 맞아야 한다. 여기서 mismatch가
    # 나면 보통 다른 obs 또는 actor-input 규약으로 학습된 checkpoint라는 뜻이다.
    agent.load(checkpoint_path)
    if args_cli.intent_head_checkpoint:
        intent_path = retrieve_file_path(args_cli.intent_head_checkpoint)
        agent.load_intent_heads(intent_path)
        print(f"[INFO] Loaded offline latent-interaction-intent modules from: {intent_path}")
    trainer.eval()
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
