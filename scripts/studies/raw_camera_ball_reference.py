"""Unqueued raw-reference WithBall34 candidate; no performance result.

Reuse the original matched actor without later mean/noise/residual guards.
Keep rendered camera4, original Lift predicates AND central-ball hold60,
center-penalty reward and fresh featurewise camera counts. This builder does
not alter observations/physics, load a teacher, or allocate a training job.
Any future study must register its own source, own64 checks and matched runs.
"""

import argparse
import hashlib
import importlib
import json
from pathlib import Path

import torch

from scripts.studies import fresh_camera_scaler as camera
from scripts.studies import raw_reference_actor as reference

SOURCE_KIND = "common_raw_camera_ball_reference_v1"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare_actor_source(payload):
    """Clone a neutral NoShare-derived camera34 teacher, never critic/optimizer.

    Preserve all learned tensors and proprioception statistics. The source's
    new camera channels must already be neutral/fresh. Reset only the six
    newly activated peer counts, whose inherited NoShare history was zeros.
    This is source preparation, not a mutation of a running agent or artifact.
    """
    if set(payload) != {"left_arm", "right_arm"}:
        raise ValueError("Both matching camera34 actors required")
    result = {}
    for uid, checkpoint in payload.items():
        policy, scaler = checkpoint["policy"], checkpoint["state_preprocessor"]
        if any(not isinstance(v, torch.Tensor) or not torch.isfinite(v).all() for v in policy.values()) or any(
            any(marker in key for marker in reference.GUARDS) for key in policy
        ):
            raise ValueError("Finite uncapped original policy required")
        for key in ("own_backbone.0.weight", "partner_intent_encoder.0.weight"):
            value = policy[key]
            if value.ndim != 2 or value.shape[1] != 34 or value[:, 30:].count_nonzero():
                raise ValueError("Original camera34 encoders with zero new camera columns required")
        head = policy["arm_action_head.weight"]
        if (
            head.shape != (7, 160)
            or head[:, 128:].count_nonzero()
            or any(
                value.count_nonzero()
                for key, value in policy.items()
                if key.startswith("partner_intent_encoder.") and key.endswith("bias")
            )
        ):
            raise ValueError("Neutral NoShare-derived peer branch required")
        gripper = policy.get("gripper_action_head.0.weight")
        if (
            gripper is None
            or gripper.shape != (128, 167)
            or gripper[:, 128:160].count_nonzero()
            or policy.get("log_std_parameter", torch.empty(0)).shape != (8,)
        ):
            raise ValueError("Neutral original eight-action gripper peer branch required")
        if (
            set(scaler) != {"running_mean", "running_variance", "current_count"}
            or any(
                not isinstance(v, torch.Tensor)
                or v.shape != (68,)
                or not v.is_floating_point()
                or not torch.isfinite(v).all()
                for v in scaler.values()
            )
            or (scaler["running_variance"] < 0).any()
            or (scaler["current_count"] < 1).any()
        ):
            raise ValueError("Finite camera34 featurewise scaler required")
        for indexes in (slice(30, 34), slice(64, 68)):
            if (
                scaler["running_mean"][indexes].count_nonzero()
                or not (scaler["running_variance"][indexes] == 1).all()
                or not (scaler["current_count"][indexes] == 1).all()
            ):
                raise ValueError("New camera channels must be neutral and fresh")
        if scaler["running_mean"][34:40].count_nonzero() or not (scaler["running_variance"][34:40] == 1).all():
            raise ValueError("Neutral newly activated peer statistics required")
        cloned = {name: value.clone() for name, value in scaler.items()}
        cloned["current_count"][34:40] = 1
        result[uid] = {"policy": {name: value.clone() for name, value in policy.items()}, "state_preprocessor": cloned}
    return result


def validate_task(task, experiment_cfg):
    cfg = task.cfg
    expected = {
        "intent_task_label": "openarm_lift_with_ball",
        "own_observation_dim": 34,
        "actor_input_dim": 68,
        "camera_extra_observation_dim": 4,
        "actor_object_pose_source": "apriltag",
        "apriltag_measurement_source": "camera_transform",
        "ball_reward_profile": "center_penalty",
        "hold_required_steps": 60,
        "ball_center_success_threshold": 0.05,
        "communication_mode": "motion_context",
        "motion_context_representation": "trend",
        "motion_context_trend_horizon": 5,
    }
    if any(getattr(cfg, name, None) != value for name, value in expected.items()):
        raise ValueError("Exact rendered-camera34 TrendH5 centered-held60 Ball task required")
    if (
        experiment_cfg["models"]["policy"].get("clip_actions") is not False
        or experiment_cfg["agent"].get("clip_predicted_values") is not True
    ):
        raise ValueError("Explicit raw Gaussian actions and original project value clipping required")


def build_raw_camera_ball_reference_mappo_agent(env, experiment_cfg):
    # Lazy task import: a future runtime selects this only after AppLauncher.
    goal = importlib.import_module("isaaclab_tasks.direct.openarm_motion_context.envs.ball_center_goal")
    task = goal.unwrap_ball_task(env)
    validate_task(task, experiment_cfg)
    goal.install_centered_hold_goal(task)
    agent = reference.build_raw_reference_mappo_agent(env, experiment_cfg)
    camera.install_vector_counts(agent)
    return agent


def prepare_artifact(source, output):
    source, output = Path(source).resolve(), Path(output).resolve()
    sidecar = output.with_suffix(".json")
    if output.exists() or sidecar.exists():
        raise ValueError("Preserve existing source artifacts")
    parent_path = source.with_suffix(".json")
    parent = json.loads(parent_path.read_bytes())
    source_sha = digest(source)
    if (
        parent.get("kind") != "common_lift_actor_fresh_camera_counts_v1"
        or parent.get("derived_sha256") != source_sha
        or parent.get("source_provenance", {}).get("kind") != "common_noshare_lift_actor_initialization"
    ):
        raise ValueError("Verified NoShare-derived fresh-camera source required")
    original_actor = Path(parent["source"])
    original_checkpoint = Path(parent["source_provenance"]["source_checkpoint"])
    if (
        digest(original_actor) != parent["source_sha256"]
        or digest(original_checkpoint) != parent["source_provenance"]["source_sha256"]
    ):
        raise ValueError("Original teacher provenance changed")
    inputs = {str(path): digest(path) for path in (source, parent_path, original_actor, original_checkpoint)}
    sources = {str(Path(path).resolve()): digest(path) for path in (__file__, reference.__file__, camera.__file__)}
    derived = prepare_actor_source(torch.load(source, map_location="cpu", weights_only=True))
    if any(digest(path) != expected for path, expected in {**inputs, **sources}.items()):
        raise ValueError("Preparation source/input drift")
    output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive outputs: failures preserve partial evidence, never overwrite.
    with output.open("xb") as stream:
        torch.save(derived, stream)
    report = {
        "kind": SOURCE_KIND,
        "source": str(source),
        "source_sha256": source_sha,
        "source_provenance": parent,
        "derived": str(output),
        "derived_sha256": digest(output),
        "input_sha256": inputs,
        "preparation_source_sha256": sources,
        "torch_version": str(torch.__version__),
        "copied": ["policy", "state_preprocessor"],
        "actor_parameters_changed": False,
        "fresh_peer_feature_indices": list(range(34, 40)),
        "fresh_peer_count": 1,
        "limits": "CPU preparation only, not a queued runtime closure, simulator test or performance result. "
        "Same candidate source must be used by both methods. Original teacher was selected on exploratory Lift seed2, not held-out WithBall performance. "
        "Critic/optimizer are excluded. A future queue still needs a complete frozen runtime archive and own64 checks.",
    }
    with sidecar.open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare only the raw-camera Ball actor source; never launch training")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = prepare_artifact(args.source, args.output)
    print(
        json.dumps(
            {
                "kind": report["kind"],
                "derived": report["derived"],
                "derived_sha256": report["derived_sha256"],
                "queued_or_gpu_started": False,
            }
        )
    )
