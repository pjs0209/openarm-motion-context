"""Prospective fixed-state/trend study; CPU-only commands and trace audits.

Lift is an independent confirmation of the previously selected absolute
representation. Camera Ball is a separate cold-training development pilot.
Each evaluation scenario starts in a fresh simulator process.
"""

import hashlib
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
NAME = "paper_motion_context_v2_20261019"
OUTPUT = ROOT / "logs/motion_context_analysis" / NAME
TRACES = ROOT / "logs/motion_context_traces" / NAME
CONDITIONS = ("absolute_fixed", "trend", "no_share")
TRAINING_SEEDS = {"lift": tuple(range(500, 505)), "ball": (500,)}
EVALUATION_SEEDS = {"lift": tuple(range(600, 630)), "ball": tuple(range(700, 710))}
TASKS = {"lift": "Isaac-OpenArm-Lift-v0", "ball": "Isaac-OpenArm-LiftWithBall-v0"}
BUILDERS = {
    "lift": "isaaclab_tasks.direct.openarm_motion_context.models.matched_communication:build_matched_communication_mappo_agent",
    "ball": "isaaclab_tasks.direct.openarm_motion_context.models.ball_center_goal_matched:build_center_goal_matched_mappo_agent",
}
TRAINER = "scripts.studies.paper_motion_context_v2_runtime:PaperMotionContextTrainer"
STEPS = 60000
SMOKE_STEPS = 64
SMOKE_SEED = 499
STATE_FIELDS = ("robot_root_state_w", "robot_joint_pos", "robot_joint_vel", "object_root_state_w")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def representation(condition):
    if condition not in CONDITIONS:
        raise ValueError("Unregistered communication condition")
    return "trend" if condition == "no_share" else condition


def tag(task, seed, condition, smoke=False, attempt=1):
    representation(condition)
    if task not in TASKS or type(seed) is not int or (seed != SMOKE_SEED and seed not in TRAINING_SEEDS[task]):
        raise ValueError("Unregistered task/training seed")
    if type(attempt) is not int or not 1 <= attempt <= 2:
        raise ValueError("At most one explicitly recorded cold retry is permitted")
    return f"{NAME}_{task}_seed{seed}_{condition}" + ("_smoke64" if smoke else "") + f"_attempt{attempt}"


def overrides(task, condition):
    sharing = "no_share" if condition == "no_share" else "intent_share"
    values = [
        "env.communication_mode=motion_context",
        f"env.sharing_mode={sharing}",
        f"env.motion_context_representation={representation(condition)}",
        "env.motion_context_trend_horizon=5",
        f"agent.runtime.agent_builder={BUILDERS[task]}",
        f"agent.runtime.trainer={TRAINER}",
        "+agent.models.policy.partner_action_max_ratio=0.5",
        "agent.models.policy.clip_actions=true",
    ]
    if task == "ball":
        values += [
            "env.ball_reward_profile=center_penalty",
            "env.ball_center_penalty_full_error=0.10",
            "env.camera_extra_observation_dim=4",
        ]
    return values


def train_command(task, seed, condition, *, smoke=False, attempt=1):
    run_tag = tag(task, seed, condition, smoke, attempt)
    if smoke != (seed == SMOKE_SEED):
        raise ValueError("Smoke and independent training seeds must be distinct")
    steps = SMOKE_STEPS if smoke else STEPS
    result = [
        "./isaaclab.sh",
        "-p",
        "scripts/reinforcement_learning/skrl/train_openarm_bimanual_direct.py",
        "--task",
        TASKS[task],
        "--num_envs",
        "4" if smoke else "128",
        "--seed",
        str(seed),
        "--mode",
        "no_share" if condition == "no_share" else "intent_share",
        "--experiment_tag",
        run_tag,
        "--headless",
    ]
    if task == "ball":
        result.append("--enable_cameras")
    return (
        result
        + overrides(task, condition)
        + [
            f"agent.trainer.timesteps={steps}",
            f"agent.agent.experiment.checkpoint_interval={steps if smoke else 5000}",
            "agent.agent.experiment.write_interval=32",
            "+agent.agent.learning_rate_scheduler_kwargs.max_lr=0.0003",
        ]
    )


def eval_command(task, seed, condition, checkpoint, directory, scenario_seed):
    tag(task, seed, condition)
    if scenario_seed not in EVALUATION_SEEDS[task] or Path(checkpoint).name != "agent_60000.pt":
        raise ValueError("Registered fresh-process final60k evaluation required")
    result = [
        "./isaaclab.sh",
        "-p",
        "scripts/reinforcement_learning/skrl/play_openarm_bimanual_direct.py",
        "--task",
        TASKS[task],
        "--num_envs",
        "1",
        "--num_steps",
        "12000",
        "--seed",
        str(scenario_seed),
        "--mode",
        "no_share" if condition == "no_share" else "intent_share",
        "--headless",
        "--num_eval_episodes",
        "1",
        "--deterministic_eval=true",
        "--save_motion_context_trace=true",
        "--motion_context_trace_dir",
        str(directory),
        "--checkpoint",
        str(checkpoint),
        "--debug_interval",
        "100000",
    ]
    if task == "ball":
        result.append("--enable_cameras")
    return result + overrides(task, condition)


def validate_physical_state(state, task):
    fields = set(STATE_FIELDS) | ({"ball_root_state_w"} if task == "ball" else set())
    if not isinstance(state, dict) or set(state) != fields:
        raise ValueError("Complete recorded robot/object physical state required")
    for key, values in state.items():
        if (
            not isinstance(values, list)
            or not values
            or any(type(x) not in (int, float) or not math.isfinite(x) for x in values)
        ):
            raise ValueError(f"Nonfinite/missing physical state: {key}")
        if key.endswith("root_state_w"):
            if len(values) != 13 or abs(sum(x * x for x in values[3:7]) - 1) > 0.002:
                raise ValueError("Rigid root needs position, unit quaternion and both velocities")
    if len(state["robot_joint_pos"]) != len(state["robot_joint_vel"]):
        raise ValueError("Joint position/velocity dimensions differ")


def compare_starts(first, second, task):
    validate_physical_state(first, task)
    validate_physical_state(second, task)
    differences = {}
    for key, left in first.items():
        right = second[key]
        if len(left) != len(right):
            raise ValueError("Recorded physical-state widths differ")
        errors = [abs(a - b) for a, b in zip(left, right)]
        if key.endswith("root_state_w"):
            quaternion_error = min(max(errors[3:7]), max(abs(a + b) for a, b in zip(left[3:7], right[3:7])))
            errors = errors[:3] + [quaternion_error] + errors[7:]
        differences[key] = max(errors)
    return {
        "recorded_start_equal_at_1e_minus_5": max(differences.values()) <= 1e-5,
        "max_absolute_differences": differences,
        "limits": "Recorded physical state only; hidden PhysX/contact/RNG state is not certified.",
    }


def audit_trace(trace, task, seed, condition, checkpoint, scenario_seed):
    own = 34 if task == "ball" else 30
    meta = trace["metadata"]
    expected = {
        "task": TASKS[task],
        "training_seed": seed,
        "evaluation_seed": scenario_seed,
        "sharing_mode": "no_share" if condition == "no_share" else "intent_share",
        "communication_mode": "motion_context",
        "motion_context_representation": representation(condition),
        "own_obs_dim": own,
        "actor_input_dim": own * 2,
        "communication_slot_dim": own,
        "communication_payload_dim": 6,
        "context_trend_horizon": 5,
        "deterministic_eval": True,
        "diagnostic_mask": "none",
        "actor_object_pose_source": "apriltag",
        "apriltag_measurement_source": "camera_transform",
        "context_calibration_enabled": False,
        "context_update_scale": False,
    }
    if any(meta.get(key) != value for key, value in expected.items()):
        raise ValueError("Evaluation identity/representation differs from protocol")
    if Path(meta["checkpoint"]).resolve() != Path(checkpoint).resolve():
        raise ValueError("Evaluation loaded another checkpoint")
    if trace.get("paper_protocol") != NAME or type(trace.get("success")) is not bool:
        raise ValueError("Missing versioned physical-state/terminal-success trace")
    validate_physical_state(trace["initial_physical_state"], task)
    n = trace["num_steps"]
    if type(n) is not int or n < 1:
        raise ValueError("Empty evaluation trajectory")
    nonzero = False
    delivery_error = 0.0
    for side, sender in (("left", "right"), ("right", "left")):
        rows = trace[f"{side}_policy_message"]
        contexts, motions = trace[f"{side}_motion_context"], trace[f"{side}_motion"]
        if len(rows) != n or len(contexts) != n or len(motions) != n:
            raise ValueError("Actor input and trace length differ")
        for context, motion in zip(contexts, motions):
            if len(context) != 3 or len(motion) != 3 or not all(math.isfinite(v) for v in context + motion):
                raise ValueError("Invalid context or motion trace")
            if representation(condition) == "trend":
                if any(abs(v) > 1.000001 for v in context):
                    raise ValueError("Trend outside its declared range")
            elif any(not 0 <= v <= 1.500001 for v in context[:2]) or not 0 <= context[2] <= 1.000001:
                raise ValueError("Absolute context outside its declared range")
        for index, row in enumerate(rows):
            if len(row) != own or not all(math.isfinite(v) for v in row) or any(abs(v) > 1e-7 for v in row[6:]):
                raise ValueError("Malformed/nonfinite/padded actor message")
            if condition == "no_share" and any(abs(v) > 1e-7 for v in row):
                raise ValueError("NoShare actor consumed a message")
            nonzero |= any(abs(v) > 1e-7 for v in row)
            if condition != "no_share":
                if index == 0 and any(abs(value) > 1e-7 for value in row):
                    raise ValueError("Initial actor observation must consume the reset-zero buffer")
                if index:
                    # Existing trace convention: motion is the previous
                    # prediction; context signals describe the current row's
                    # already-delivered buffer, sampled before env.step.
                    expected = trace[f"{sender}_motion"][index - 1] + trace[f"{sender}_motion_context"][index]
                    delivery_error = max(delivery_error, max(abs(a - b) for a, b in zip(row[:6], expected)))
    if (condition != "no_share" and not nonzero) or delivery_error > 1e-5:
        raise ValueError("Actor never consumed a message or sender delivery was misaligned")
    if task == "ball":
        if (
            meta.get("ball_observation_source") != "rendered_axial_depth_and_oracle_semantic_image_mask"
            or meta.get("ball_pose_ground_truth_in_actor") is not False
            or trace.get("paper_success_goal") != "centered_hold_v1"
        ):
            raise ValueError("Ball observation/strict success provenance mismatch")
        if len(trace["ball_camera_observation"]) != n:
            raise ValueError("Camera timeline length differs")
        for row in trace["ball_camera_observation"]:
            if len(row) != 4 or not all(math.isfinite(x) for x in row) or row[3] not in (0, 1):
                raise ValueError("Invalid camera observation")
            if row[3] == 0 and any(row[:3]):
                raise ValueError("Missing ball detection supplied nonzero coordinates")
        counts, centered = trace["success_hold_count"], trace["ball_center_ok"]
        if len(counts) != n or len(centered) != n or any(count > 0 and not ok for count, ok in zip(counts, centered)):
            raise ValueError("Strict ball hold counter contradicts centering")
        if trace["success"] and max(counts) < 59:
            raise ValueError("Ball success lacks recorded pre-terminal 60-step hold")
    return {
        "audit_passed": True,
        "success": trace["success"],
        "steps": n,
        "max_delivery_error": delivery_error,
        "initial_physical_state": trace["initial_physical_state"],
    }
