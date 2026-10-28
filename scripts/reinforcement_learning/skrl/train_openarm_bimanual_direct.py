# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""skrl 기반 OpenArm 양팔 lift policy 학습 진입점.

이 script는 기존 Direct/interaction-intent task와 최종 MotionMode task가 함께
사용한다. skrl 학습을 시작하기 전에 세 가지 일을 한다.

1. 무거운 simulator module을 import하기 전에 Isaac Lab launcher argument를 parsing한다.
2. Hydra env/agent config를 읽고 compact CLI override를 적용한다.
3. YAML ``runtime`` section에 적힌 agent/trainer 조합을 동적으로 만든다.

최종 MotionMode task에서는 공식 CLI variant를 ``no_intent``와 ``share_intent``로
좁혀 둔다. mode hyperparameter는 여러 CLI flag로 흩뿌리지 않고
``motion_mode_agent_cfg.yaml``에서 읽는다.
"""

import argparse
import importlib
import json
import os
import random
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
_legacy_rl = Path(__file__).resolve().parents[3] / "compat" / "skrl_1_4_3"
if not (_legacy_rl / "skrl").is_dir():
    raise RuntimeError(
        "Pinned OpenArm RL runtime is missing; do not silently substitute skrl2 for old MAPPO checkpoints"
    )
sys.path.insert(0, str(_legacy_rl))

from isaaclab.app import add_launcher_args, launch_simulation


# Isaac Lab은 대부분의 Isaac/Omni import보다 AppLauncher argument parsing이 먼저
# 끝나야 한다. 그래서 parser는 파일 상단에 둔다.
if "--headless" in sys.argv:
    sys.argv.remove("--headless")
    sys.argv.extend(["--viz", "none"])
sys.argv = [arg for arg in sys.argv if arg != "--enable_cameras"]

parser = argparse.ArgumentParser(description="Train OpenArm bimanual direct latent-interaction-intent MAPPO.")
parser.add_argument("--mode", type=str, default="intent_share", choices=("intent_share", "no_share"))
parser.add_argument("--video", action="store_true", default=False, help="Record training videos.")
parser.add_argument("--video_length", type=int, default=200, help="Length of recorded training videos.")
parser.add_argument("--video_interval", type=int, default=2000, help="Training video recording interval.")
parser.add_argument("--debug_interval", type=int, default=100, help="Print debug stats every N steps.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments.")
parser.add_argument(
    "--task",
    type=str,
    default="Isaac-Lift-OpenArm-Bimanual-Direct-PlusJoint-v0",
    help="Task name registered in gym.",
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for training.")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint to resume from.")
parser.add_argument(
    "--resume_run_dir",
    type=str,
    default=None,
    help="Continue an interrupted run in its existing directory using --checkpoint agent_STEP.pt.",
)
parser.add_argument(
    "--policy_warmstart",
    type=str,
    default=None,
    help="Copy a Lift actor and actor scaler into WithBall; keep critic and optimizer fresh.",
)
parser.add_argument(
    "--intent_head_checkpoint", type=str, default=None, help="Path to latent-interaction-intent module weights."
)
parser.add_argument(
    "--freeze_intent_head", action="store_true", default=False, help="Freeze loaded latent-interaction-intent modules."
)
parser.add_argument(
    "--disable_online_intent_loss", action="store_true", default=False, help="Disable online timing-intent loss."
)
parser.add_argument("--max_iterations", type=int, default=None, help="Override max training iterations.")
parser.add_argument(
    "--mode_loss_warmup_timesteps", type=int, default=None, help="Override MotionModeIntent mode loss warmup steps."
)
parser.add_argument("--stable_hold_reward_scale", type=float, default=None)
parser.add_argument("--stable_hold_tilt_deg", type=float, default=None)
parser.add_argument("--stable_hold_height_margin", type=float, default=None, help=argparse.SUPPRESS)
parser.add_argument("--success_grasp_threshold", type=float, default=None)
parser.add_argument(
    "--intent_variant",
    type=str,
    default=None,
    choices=("no_intent", "share_intent"),
    help="Final MotionMode variant: no_intent baseline or share_intent proposed method.",
)
parser.add_argument(
    "--intent_arch",
    type=str,
    default=None,
    choices=("shared_intent_encoder",),
    help="share_intent architecture. Only shared_intent_encoder is supported.",
)
parser.add_argument("--experiment_tag", type=str, default=None, help="Optional experiment-name tag.")
add_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
if args_cli.video:
    args_cli.enable_cameras = True

sys.argv = [sys.argv[0]] + hydra_args


import gymnasium as gym

from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.io import dump_yaml
from isaaclab_rl.skrl import SkrlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.direct.openarm_motion_context.train.trainer import MotionContextTrainer as IntentSequentialTrainer
from isaaclab_tasks.direct.openarm_motion_context.models.actor import (
    build_motion_context_mappo_agent as build_intent_mappo_agent,
)
from isaaclab_tasks.utils.hydra import resolve_task_config


def _unwrap_task_env(env):
    """gym/skrl wrapper 아래에 있는 실제 task env를 찾는다."""

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
    """task metadata와 intent hook을 skrl wrapper에 노출한다.

    skrl 자체는 step 가능한 vector env만 필요하지만 custom agent는 task-specific
    dimension과 intent hook도 필요하다. wrapper에 복사해두면 agent builder가
    Isaac Lab wrapper 내부 구조에 덜 의존한다.
    """

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
    """agent YAML runtime block의 ``module:symbol`` 참조를 import한다."""

    module_name, symbol_name = path.split(":", 1)
    module = importlib.import_module(module_name)
    return getattr(module, symbol_name)


def _runtime_components(agent_cfg: dict):
    """YAML에서 agent builder와 trainer class를 가져오고, 없으면 legacy 기본값을 쓴다."""

    runtime_cfg = agent_cfg.get("runtime", {}) if isinstance(agent_cfg, dict) else {}
    builder = build_intent_mappo_agent
    trainer_cls = IntentSequentialTrainer
    if runtime_cfg.get("agent_builder"):
        builder = _load_symbol(runtime_cfg["agent_builder"])
    if runtime_cfg.get("trainer"):
        trainer_cls = _load_symbol(runtime_cfg["trainer"])
    return builder, trainer_cls


def _next_experiment_index(log_root_path: str) -> int:
    """skrl log directory에서 다음 001 형태의 run index를 계산한다."""

    if not os.path.isdir(log_root_path):
        return 1
    max_index = 0
    for entry in os.listdir(log_root_path):
        match = re.match(r"^(\d{3})_", entry)
        if match is not None:
            max_index = max(max_index, int(match.group(1)))
    return max_index + 1


def _canonical_intent_variant(variant: str | None) -> str | None:
    """최종 MotionMode intent variant 문자열을 정규화한다."""

    if variant is None:
        return None
    return str(variant)


def _set_env_attr_if_present(env_cfg, name: str, value) -> None:
    """env cfg가 해당 attribute를 가지고 있을 때만 값을 쓴다."""

    if value is not None and hasattr(env_cfg, name):
        setattr(env_cfg, name, value)


def _apply_compact_motion_mode_cfg(agent_cfg: dict, env_cfg=None) -> None:
    """압축된 ``mode_cfg`` YAML block을 skrl config로 펼친다.

    학습 명령어를 짧게 유지하면서도, 실제 최종 값은 dump되는 agent/env YAML에
    남겨 재현 가능하게 만든다.
    """

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


def main(env_cfg, agent_cfg):
    """env_cfg/agent_cfg가 로드된 뒤 Hydra가 호출하는 main entry point."""

    if args_cli.policy_warmstart and (args_cli.checkpoint or args_cli.resume_run_dir):
        raise ValueError("Policy warm-start cannot be combined with checkpoint resume")
    if args_cli.policy_warmstart and args_cli.task != "Isaac-OpenArm-LiftWithBall-v0":
        raise ValueError("Lift actor warm-start is currently restricted to WithBall")
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
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
    _set_env_attr_if_present(env_cfg, "stable_hold_reward_scale", args_cli.stable_hold_reward_scale)
    _set_env_attr_if_present(env_cfg, "stable_hold_tilt_deg", args_cli.stable_hold_tilt_deg)
    _set_env_attr_if_present(env_cfg, "stable_hold_height_margin", args_cli.stable_hold_height_margin)
    _set_env_attr_if_present(env_cfg, "success_grasp_threshold", args_cli.success_grasp_threshold)

    print(f"[INFO] Using sharing mode for train: {args_cli.mode}")
    if args_cli.intent_variant is not None:
        print(f"[INFO] Using intent_variant for train: {_canonical_intent_variant(args_cli.intent_variant)}")
    if args_cli.intent_arch is not None:
        print(f"[INFO] Using intent_arch for train: {args_cli.intent_arch}")

    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)
    agent_cfg["seed"] = args_cli.seed if args_cli.seed is not None else agent_cfg["seed"]
    env_cfg.seed = agent_cfg["seed"]

    if args_cli.max_iterations:
        agent_cfg["trainer"]["timesteps"] = args_cli.max_iterations * agent_cfg["agent"]["rollouts"]
    if args_cli.mode_loss_warmup_timesteps is not None:
        agent_cfg["agent"]["mode_loss_warmup_timesteps"] = args_cli.mode_loss_warmup_timesteps
    if hasattr(env_cfg, "__post_init__"):
        env_cfg.__post_init__()
    # Keep the archived agent configuration consistent with the actual runtime actor.
    # The actor reads this mode from task_env, while Hydra overrides only env_cfg.
    if hasattr(env_cfg, "communication_mode"):
        agent_cfg.setdefault("models", {}).setdefault("policy", {})["communication_mode"] = str(
            env_cfg.communication_mode
        )
    if getattr(env_cfg, "observe_control_state", False):
        policy_cfg = agent_cfg.setdefault("models", {}).setdefault("policy", {})
        policy_cfg["own_observation_dim"] = env_cfg.own_observation_dim
        policy_cfg["communication_feature_dim"] = env_cfg.communication_feature_dim
    if args_cli.disable_online_intent_loss or args_cli.freeze_intent_head:
        if "intent_loss_scale" in agent_cfg["agent"]:
            agent_cfg["agent"]["intent_loss_scale"] = 0.0
        if "lambda_temporal" in agent_cfg["agent"]:
            agent_cfg["agent"]["lambda_temporal"] = 0.0
    agent_cfg["trainer"]["close_environment_at_exit"] = False
    agent_cfg["trainer"]["debug_interval"] = args_cli.debug_interval

    # run directory는 증가 index를 사용한다. 기본적으로 train/play/analysis output이
    # 이전 실험을 덮어쓰지 않게 하기 위함이다.
    log_root_path = os.path.abspath(
        os.path.join("logs", "current", "skrl", agent_cfg["agent"]["experiment"]["directory"])
    )
    resume_step = None
    if args_cli.resume_run_dir:
        log_dir = os.path.realpath(args_cli.resume_run_dir)
        checkpoint = os.path.realpath(args_cli.checkpoint or "")
        match = re.fullmatch(r"agent_(\d+)\.pt", os.path.basename(checkpoint))
        if (
            os.path.dirname(log_dir) != os.path.realpath(log_root_path)
            or os.path.dirname(checkpoint) != os.path.join(log_dir, "checkpoints")
            or not os.path.isfile(checkpoint)
            or match is None
        ):
            raise ValueError("Resume requires an existing run and its agent_STEP.pt checkpoint")
        resume_step = int(match.group(1))
        if not 0 < resume_step < int(agent_cfg["trainer"]["timesteps"]):
            raise ValueError("Resume checkpoint must precede the planned final timestep")
        if os.path.exists(os.path.join(log_dir, "checkpoints", f"agent_{agent_cfg['trainer']['timesteps']}.pt")):
            raise ValueError("Run already has its final checkpoint")
        log_name = os.path.basename(log_dir)
    else:
        run_index = _next_experiment_index(log_root_path)
        log_suffix = args_cli.experiment_tag if args_cli.experiment_tag else args_cli.mode
        log_name = f"{run_index:03d}_{log_suffix}"
    agent_cfg["agent"]["experiment"]["directory"] = log_root_path
    agent_cfg["agent"]["experiment"]["experiment_name"] = log_name
    log_dir = os.path.join(log_root_path, log_name)

    if resume_step is None:
        dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
        dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    else:
        # Preserve the original configs and reject changed training settings.
        import yaml

        resume_params = os.path.join(log_dir, "params", f"resume_{resume_step}_{time.time_ns()}")
        dump_yaml(os.path.join(resume_params, "env.yaml"), env_cfg)
        dump_yaml(os.path.join(resume_params, "agent.yaml"), agent_cfg)
        for config_name in ("env.yaml", "agent.yaml"):
            with open(os.path.join(log_dir, "params", config_name)) as stream:
                original = yaml.load(stream, Loader=yaml.FullLoader)
            with open(os.path.join(resume_params, config_name)) as stream:
                resumed = yaml.load(stream, Loader=yaml.FullLoader)
            if original != resumed:
                raise ValueError(f"Resume configuration differs from saved {config_name}")
        with open(os.path.join(resume_params, "recovery.json"), "w") as stream:
            json.dump(
                {
                    "checkpoint": checkpoint,
                    "initial_timestep": resume_step,
                    "final_timestep": agent_cfg["trainer"]["timesteps"],
                    "physics_and_rng_restored": False,
                    "scheduler_state_restored": False,
                },
                stream,
                indent=2,
            )
        print(f"[INFO] Recovering interrupted run at {resume_step} steps; simulator/RNG and scheduler restart")
    env_cfg.log_dir = log_dir

    # Isaac Lab env를 만들고, 필요하면 video wrapper를 붙인 뒤 skrl multi-agent API로 감싼다.
    with launch_simulation(env_cfg, args_cli):
        env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
        if args_cli.video:
            env = gym.wrappers.RecordVideo(
                env,
                video_folder=os.path.join(log_dir, "videos", "train"),
                step_trigger=lambda step: step % args_cli.video_interval == 0,
                video_length=args_cli.video_length,
                disable_logger=True,
            )

        task_env = _unwrap_task_env(env)
        if not hasattr(task_env, "possible_agents"):
            raise RuntimeError("Failed to unwrap multi-agent task env.")
        env = SkrlVecEnvWrapper(env, ml_framework="torch", wrapper="isaaclab-multi-agent")
        env = _attach_multi_agent_api(env, task_env)
        # runtime block 덕분에 같은 script로 legacy intent component와 최종 MotionMode
        # model/trainer를 모두 구성할 수 있다.
        agent_builder, trainer_cls = _runtime_components(agent_cfg)
        agent = agent_builder(env, agent_cfg)
        trainer = trainer_cls(env=env, agents=agent, cfg=agent_cfg["trainer"])
        if resume_step is not None:
            trainer.initial_timestep = resume_step

        if args_cli.checkpoint:
            agent.load(retrieve_file_path(args_cli.checkpoint))
        if args_cli.policy_warmstart:
            from isaaclab_tasks.direct.openarm_motion_context.models.policy_transfer import initialize_from_lift

            manifest = initialize_from_lift(agent, retrieve_file_path(args_cli.policy_warmstart))
            trainer.policy_warmstart_manifest = manifest
            with open(os.path.join(log_dir, "params", "policy_warmstart.json"), "x") as stream:
                json.dump(manifest, stream, indent=2)
            print(f"[INFO] Actor warm-start from {manifest['source_checkpoint']}; critic and optimizer are fresh")
        if args_cli.intent_head_checkpoint:
            intent_path = retrieve_file_path(args_cli.intent_head_checkpoint)
            agent.load_intent_heads(intent_path)
            print(f"[INFO] Loaded offline latent-interaction-intent modules from: {intent_path}")
        if args_cli.freeze_intent_head:
            agent.freeze_intent_heads()
            print("[INFO] Frozen latent-interaction-intent module weights")

        start_time = time.time()
        trainer.train()
        print(f"Training time: {round(time.time() - start_time, 2)} seconds")
        env.close()


if __name__ == "__main__":
    main(*resolve_task_config(args_cli.task, "skrl_mappo_cfg_entry_point"))
