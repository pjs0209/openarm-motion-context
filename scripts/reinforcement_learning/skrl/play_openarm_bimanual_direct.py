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
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
_legacy_rl = Path(__file__).resolve().parents[3] / "compat" / "skrl_1_4_3"
if not (_legacy_rl / "skrl").is_dir():
    raise RuntimeError(
        "Pinned OpenArm RL runtime is missing; do not silently substitute skrl2 for old MAPPO checkpoints"
    )
sys.path.insert(0, str(_legacy_rl))

from isaaclab.app import add_launcher_args, launch_simulation

# 학습 script와 마찬가지로 simulator-heavy module import 전에 launcher argument를 parsing한다.
if "--headless" in sys.argv:
    sys.argv.remove("--headless")
    sys.argv.extend(["--viz", "none"])
sys.argv = [arg for arg in sys.argv if arg != "--enable_cameras"]

parser = argparse.ArgumentParser(description="Play OpenArm bimanual direct latent-interaction-intent MAPPO.")


def _str_to_bool(value):
    """선택형 boolean flag에서 자주 쓰는 문자열 표현을 bool로 바꾼다."""

    if isinstance(value, bool):
        return value
    normalized = str(value).lower()
    if normalized in ("1", "true", "yes", "on"):
        return True
    if normalized in ("0", "false", "no", "off"):
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got {value!r}.")


def _parse_cli_and_hydra_args(argument_parser, argv=None, launcher=None):
    """Keep Hydra overrides from becoming values of optional boolean flags."""

    cli_args = []
    overrides = []
    for token in sys.argv[1:] if argv is None else argv:
        is_override = token.lstrip("+~").startswith(("env.", "agent.", "hydra.")) and (
            "=" in token or token.startswith("~")
        )
        (overrides if is_override else cli_args).append(token)
    original_argv = sys.argv
    try:
        # AppLauncher also parses the current command line before adding its
        # own flags, so protect that preliminary parse as well.
        sys.argv = [original_argv[0]] + cli_args
        if launcher is not None:
            launcher.add_app_launcher_args(argument_parser)
        parsed, unknown = argument_parser.parse_known_args(cli_args)
    finally:
        sys.argv = original_argv
    return parsed, unknown + overrides


parser.add_argument("--mode", type=str, default=None, choices=("intent_share", "no_share"))
parser.add_argument(
    "--task",
    type=str,
    default="Isaac-Lift-OpenArm-Bimanual-Direct-PlusJoint-v0",
    help="Task name registered in gym.",
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments.")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint.")
parser.add_argument(
    "--use_current_config",
    action="store_true",
    help="Explicit migration/diagnostic replay using current task defaults instead of the saved training configs.",
)
parser.add_argument("--num_steps", type=int, default=1000, help="Number of evaluation steps.")
parser.add_argument(
    "--seed", type=int, default=None, help="Evaluation seed (defaults to the checkpoint task config seed)."
)
parser.add_argument(
    "--scenario_seed_start",
    type=int,
    default=None,
    help="Opt-in single-world evaluation: independently seed each Torch reset, preserving policy RNG.",
)
parser.add_argument("--debug_interval", type=int, default=25, help="Print debug stats every N steps.")
parser.add_argument(
    "--audit_checkpoints",
    type=Path,
    nargs="+",
    default=None,
    help="After wrist diagnostic evaluation, inspect these same-run checkpoints on the captured first-episode states.",
)
parser.add_argument(
    "--wrist_disturbance",
    type=Path,
    default=None,
    help="Opt-in JSON wrist pulse protocol; requires saved traces, one world and scenario reset seeds.",
)
parser.add_argument(
    "--intent_head_checkpoint", type=str, default=None, help="Path to latent-interaction-intent module weights."
)
parser.add_argument(
    "--diagnostic_mask",
    choices=("none", "context_zero", "motion_zero", "message_zero"),
    default="none",
    help="Inference-only masking of the partner message for Motion Context checkpoint diagnostics.",
)
parser.add_argument(
    "--save_motion_context_trace",
    nargs="?",
    const=True,
    default=False,
    type=_str_to_bool,
    help="Save motion-context communication traces during play/eval.",
)
parser.add_argument(
    "--motion_context_trace_dir",
    type=str,
    default=None,
    help="Override the trace directory. Defaults to logs/current/motion_context_traces/<task>_<mode>/<run number>.",
)
parser.add_argument(
    "--save_mode_trace",
    nargs="?",
    const=True,
    default=False,
    type=_str_to_bool,
    help="Deprecated alias for --save_motion_context_trace.",
)
parser.add_argument("--mode_trace_dir", type=str, default=None, help="Directory for motion-mode trace JSON files.")
parser.add_argument(
    "--mode_trace_env_index", type=int, default=None, help="Environment index to record in motion-mode traces."
)
parser.add_argument(
    "--mode_trace_format", type=str, default=None, choices=("json",), help="Trace format for motion-mode traces."
)
parser.add_argument(
    "--num_eval_episodes", type=int, default=None, help="Stop trace-saving eval after this many episodes."
)
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
parser.add_argument(
    "--mode_frame_threshold",
    type=float,
    default=None,
    help="Mode probability threshold for representative frame capture.",
)
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
add_launcher_args(parser)
args_cli, hydra_args = _parse_cli_and_hydra_args(parser, launcher=None)
sys.argv = [sys.argv[0]] + hydra_args


import gymnasium as gym

from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.io import dump_yaml

from isaaclab_rl.skrl import SkrlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.direct.openarm_motion_context.models.actor import (
    build_motion_context_mappo_agent as build_intent_mappo_agent,
)
from isaaclab_tasks.direct.openarm_motion_context.train.checkpoint_config import restore_checkpoint_configs
from isaaclab_tasks.direct.openarm_motion_context.train.trainer import MotionContextTrainer as IntentSequentialTrainer
from isaaclab_tasks.utils.hydra import resolve_task_config


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
    """checkpoint 실행에 필요한 agent builder와 trainer class를 결정한다."""

    runtime_cfg = agent_cfg.get("runtime", {}) if isinstance(agent_cfg, dict) else {}
    builder = build_intent_mappo_agent
    trainer_cls = IntentSequentialTrainer
    if runtime_cfg.get("agent_builder"):
        builder = _load_symbol(runtime_cfg["agent_builder"])
    if runtime_cfg.get("trainer"):
        trainer_cls = _load_symbol(runtime_cfg["trainer"])
    return builder, trainer_cls


def _evaluation_scenario_cfg(cli, save_trace):
    """Validate the optional reset protocol before constructing the trainer."""
    if cli.scenario_seed_start is None:
        return {}
    if not save_trace or cli.num_envs != 1 or not cli.num_eval_episodes or cli.num_eval_episodes < 1:
        raise ValueError("Scenario resets require one world, saved traces and a positive evaluation episode count")
    from isaaclab_tasks.direct.openarm_motion_context.eval.reset_scenarios import PROTOCOL

    return {"evaluation_reset_protocol": PROTOCOL, "evaluation_scenario_seed_start": cli.scenario_seed_start}


def _evaluate_with_reset_scenarios(trainer, task_env, trace_dir, seed_start, episodes):
    """Install after checkpoint loading; audit starts after the evaluator returns."""
    if seed_start is None:
        trainer.eval()
        return
    from isaaclab_tasks.direct.openarm_motion_context.eval.reset_scenarios import (
        finalize_reset_scenarios,
        install_reset_scenarios,
    )

    records = install_reset_scenarios(task_env, seed_start)
    eval_returned = False
    try:
        trainer.eval()
        eval_returned = True
    finally:
        finalize_reset_scenarios(trace_dir, records, seed_start, episodes, eval_returned)


def _canonical_intent_variant(variant: str | None) -> str | None:
    """최종 MotionMode variant 이름을 정규화한다."""

    if variant is None:
        return None
    return str(variant)


def _next_motion_context_trace_dir(task: str, communication_mode: str) -> str:
    """Reserve a new numbered trace directory for one task and communication mode."""

    task_lower = task.lower()
    if "lift" in task_lower:
        task_name = "lift"
    elif "peginhole" in task_lower:
        task_name = "peg_in_hole"
    else:
        task_name = re.sub(r"[^a-z0-9]+", "_", task_lower).strip("_")
    mode_name = re.sub(r"[^a-z0-9]+", "_", communication_mode.lower()).strip("_")
    base_dir = os.path.join("logs", "current", "motion_context_traces", f"{task_name}_{mode_name}")
    os.makedirs(base_dir, exist_ok=True)
    run_index = 1
    while True:
        trace_dir = os.path.join(base_dir, f"{run_index:03d}")
        try:
            os.mkdir(trace_dir)
            return trace_dir
        except FileExistsError:
            run_index += 1


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


def _validate_wrist_evaluation(cli):
    if cli.audit_checkpoints and cli.wrist_disturbance is None:
        raise ValueError("Checkpoint distribution audit requires wrist diagnostic observation capture")
    if cli.wrist_disturbance is None:
        return
    if cli.scenario_seed_start is None or cli.num_envs != 1 or not cli.num_eval_episodes:
        raise ValueError("Wrist diagnostics require one world, scenario_seed_start and num_eval_episodes")
    if not (cli.save_motion_context_trace or cli.save_mode_trace):
        raise ValueError("Wrist diagnostics require saved episode traces")


def _evaluate_with_wrist_disturbance(trainer, task_env, trace_dir, cli):
    disturbance = None
    if cli.wrist_disturbance is not None:
        import json

        from isaaclab_tasks.direct.openarm_motion_context.eval.wrist_disturbance import WristDisturbance

        disturbance = WristDisturbance(task_env, json.loads(cli.wrist_disturbance.read_text()), Path(trace_dir))
    completed = False
    try:
        _evaluate_with_reset_scenarios(trainer, task_env, trace_dir, cli.scenario_seed_start, cli.num_eval_episodes)
        completed = True
    finally:
        if disturbance is not None:
            disturbance.close(completed)
    if completed and cli.audit_checkpoints:
        from isaaclab_tasks.direct.openarm_motion_context.eval.checkpoint_audit import audit_checkpoint_distributions

        audit_checkpoint_distributions(
            trainer.agents,
            Path(trace_dir),
            cli.audit_checkpoints,
            Path(cli.checkpoint),
            float(task_env.cfg.gripper_switch_epsilon),
        )


def main(env_cfg, agent_cfg):
    """checkpoint evaluation을 위한 Hydra entry point."""

    _validate_wrist_evaluation(args_cli)
    checkpoint_path = retrieve_file_path(args_cli.checkpoint)
    checkpoint_agent_cfg_path = Path(checkpoint_path).parent.parent / "params" / "agent.yaml"
    training_seed = None
    if not args_cli.use_current_config:
        env_cfg, agent_cfg, training_seed = restore_checkpoint_configs(checkpoint_path, env_cfg, agent_cfg, hydra_args)
        print(f"[INFO] Restored checkpoint environment and agent configs from: {checkpoint_agent_cfg_path.parent}")
    elif checkpoint_agent_cfg_path.is_file():
        import yaml

        with checkpoint_agent_cfg_path.open("r", encoding="utf-8") as cfg_file:
            training_seed = (yaml.safe_load(cfg_file) or {}).get("seed")
    if args_cli.use_current_config:
        print("[WARN] Explicit current-config replay: this is not a matched-training-configuration evaluation.")
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    if args_cli.seed is not None:
        agent_cfg["seed"] = args_cli.seed
    env_cfg.seed = agent_cfg["seed"]
    if args_cli.mode is not None and hasattr(env_cfg, "sharing_mode"):
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

    print(f"[INFO] Using sharing mode for play: {env_cfg.sharing_mode}")
    if args_cli.intent_variant is not None:
        print(f"[INFO] Using intent_variant for play: {_canonical_intent_variant(args_cli.intent_variant)}")
    if args_cli.intent_arch is not None:
        print(f"[INFO] Using intent_arch for play: {args_cli.intent_arch}")

    save_trace = args_cli.save_motion_context_trace or args_cli.save_mode_trace
    trace_dir = args_cli.motion_context_trace_dir or args_cli.mode_trace_dir
    if save_trace and trace_dir is None:
        trace_dir = _next_motion_context_trace_dir(args_cli.task, str(env_cfg.communication_mode))
    if save_trace and any(Path(trace_dir).glob("motion_context_trace_episode_*.json")):
        raise FileExistsError(f"Evaluation traces already exist in {trace_dir}; choose a new directory.")

    # 선택적 trace frame 저장을 위해 play는 항상 rgb_array render mode로 env를 만든다.
    with launch_simulation(env_cfg, args_cli):
        env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array")
        task_env = _unwrap_task_env(env)
        if not hasattr(task_env, "possible_agents"):
            raise RuntimeError("Failed to unwrap multi-agent task env.")
        env = SkrlVecEnvWrapper(env, ml_framework="torch", wrapper="isaaclab-multi-agent")
        env = _attach_multi_agent_api(env, task_env)
        # Never write evaluation events/checkpoints into the restored training run.
        experiment_cfg = agent_cfg.setdefault("agent", {}).setdefault("experiment", {})
        experiment_cfg.update(write_interval=0, checkpoint_interval=0, wandb=False)
        agent_builder, trainer_cls = _runtime_components(agent_cfg)
        agent = agent_builder(env, agent_cfg)
        # 학습 시점 trainer config를 기반으로 play 전용 trace/eval 설정만 덧씌운다.
        trainer_cfg = dict(agent_cfg["trainer"])
        trainer_cfg["timesteps"] = args_cli.num_steps
        trainer_cfg["close_environment_at_exit"] = False
        trainer_cfg["debug_interval"] = args_cli.debug_interval
        trainer_cfg["mode_trace_task"] = args_cli.task
        trainer_cfg["mode_trace_checkpoint"] = checkpoint_path
        trainer_cfg["mode_trace_training_seed"] = training_seed
        trainer_cfg["diagnostic_mask"] = args_cli.diagnostic_mask
        if save_trace:
            trainer_cfg["save_mode_trace"] = True
        if trace_dir is not None:
            trainer_cfg["mode_trace_dir"] = trace_dir
            if save_trace:
                print(f"[INFO] Motion-context traces will be saved to: {os.path.abspath(trace_dir)}")
        if args_cli.mode_trace_env_index is not None:
            trainer_cfg["mode_trace_env_index"] = args_cli.mode_trace_env_index
        if args_cli.mode_trace_format is not None:
            trainer_cfg["mode_trace_format"] = args_cli.mode_trace_format
        if args_cli.num_eval_episodes is not None:
            trainer_cfg["num_trace_episodes"] = args_cli.num_eval_episodes
        trainer_cfg.update(_evaluation_scenario_cfg(args_cli, save_trace))
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
        if save_trace:
            # Record the actual evaluation world count, physics and trainer
            # overrides, not just a link to a mutable training configuration.
            dump_yaml(os.path.join(trace_dir, "params", "env.yaml"), env_cfg)
            dump_yaml(os.path.join(trace_dir, "params", "agent.yaml"), {**agent_cfg, "trainer": trainer_cfg})
        trainer = trainer_cls(env=env, agents=agent, cfg=trainer_cfg)

        # checkpoint는 현재 task/intent_variant dimension과 맞아야 한다. 여기서 mismatch가
        # 나면 보통 다른 obs 또는 actor-input 규약으로 학습된 checkpoint라는 뜻이다.
        agent.load(checkpoint_path)
        if args_cli.intent_head_checkpoint:
            intent_path = retrieve_file_path(args_cli.intent_head_checkpoint)
            agent.load_intent_heads(intent_path)
            print(f"[INFO] Loaded offline latent-interaction-intent modules from: {intent_path}")
        _evaluate_with_wrist_disturbance(trainer, task_env, trace_dir, args_cli)
        env.close()


if __name__ == "__main__":
    main(*resolve_task_config(args_cli.task, "skrl_mappo_cfg_entry_point"))
