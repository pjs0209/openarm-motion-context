# OpenArm Motion Context

This package contains the Isaac Lab implementation of Proprioceptive Motion
Context Sharing for OpenArm bimanual manipulation.

## Current main comparison

The Lift/Peg recovery study uses the common own-controller-state Actor/Critic
builder in `models/main_comparison.py`. Its four independently trained conditions
are NoShare, MotionOnly, Signed Twist and Full6D Trend. NoShare retains the partner
encoder and receives zeros from training start. Historical gate/decoder/control
variants elsewhere in this package are separate experiments.

The maintained configuration is
`scripts/studies/message_comparison_preflight.json` at the Isaac Lab checkout root;
the protocol and actual checks are described in `scripts/studies/RECOVERY_PROTOCOL.rst`.
The 512-step configuration checks integration and matched settings; it is not a
task-success result. Simulation observations with the optional own-control-state
inputs require a matching observation implementation before real deployment.

- `envs`: Isaac Lab task definitions and reward/state logic.
- `perception`: AprilTag transforms, multi-camera fusion, and grasp targets.
- `communication`: signed EE motion, motion context, and message construction.
- `models`: decentralized actors and centralized critic.
- `train`: MAPPO trainer, checkpoint state, and YAML configurations.
- `eval`: communication ablations, metrics, and trace analysis.
- `deploy`: ROS2 inference, calibration, and real-robot configuration.
- `common`: shared transform, observation, and action contracts.
- `scripts`: short command wrappers for training, evaluation, and deployment.

The Gym task IDs are:

- `Isaac-OpenArm-Lift-v0`
- `Isaac-OpenArm-LiftWithBall-v0`
- `Isaac-OpenArm-PegInHole-Circle-Grasp-v0` (contact and friction grasping)
- `Isaac-OpenArm-PegInHole-v0` (arrow compatibility alias)
- `Isaac-OpenArm-PegInHole-Arrow-v0`
- `Isaac-OpenArm-PegInHole-Circle-v0`
- `Isaac-OpenArm-PegInHole-Hexagon-v0`

WithBall has an optional Newton mesh/SDF ball collision profile. Add this Hydra
override to a new run:

```text
env.scene.environment.spawn.func=isaaclab_tasks.direct.openarm_motion_context.envs.ball_contact_mesh:spawn_openarm_mesh_ball_scene
```

It replaces only the invisible ball collider, retaining the rendered sphere,
camera observations, materials and analytic sphere mass properties. Its mesh
has a radial approximation budget of 50 micrometers. Original USD files and
the default analytic-sphere profile remain available. Keep the saved environment
configuration with each checkpoint; record the collision profile separately in
comparisons. Passive contact stability does not establish grasp or lift success.

Separate local-observation experiments use `Isaac-OpenArm-Lift-RotationObs-v0`
and `Isaac-OpenArm-LiftWithBall-RotationObs-v0`. They retain every original local
channel and append a three-component signed camera rotation error in robot-base
axes, normalized by pi. Lift has 33 own channels, a 66D actor input and a 79D
critic state; WithBall has 37 own channels, a 74D actor input and an 87D critic
state. The rendered ball measurement remains four channels. The partner payload
remains the same six proprioceptive motion/trend channels.

These task IDs use the Gaussian actor with the optional physical joint-target
guard and a trainer that records the added camera features against the actual
pre-action actor inputs. Saved checkpoints carry an observation-version marker.
Use matching task IDs and saved configurations for replay. Compare methods only
within the same local-observation, control and physics profile; these variants
do not establish a performance improvement or communication advantage.
The new local encoder is a simulation experiment; the ROS adapter needs the
matching signed-rotation channels before these checkpoints can be deployed.

An optional training reset curriculum is available through
`models.grasp_reset_curriculum:build_grasp_reset_curriculum_mappo_agent` and
`models.grasp_reset_curriculum:GraspResetCurriculumTrainer` as the saved runtime
builder and trainer. It uses the same signed camera-rotation plain Lift policy,
rewards and success conditions. Open-jaw canonical grasp starts are mixed with
the original randomized normal starts, without adding a curriculum signal to
the actor or partner payload. The original arm reference bounds remain centered
at the original default posture; the configured action range must include the
canonical warm poses.

The default schedule holds warm starts for 10,000 control steps and fades them
out early enough that all episodes from step 40,000 onward have original starts.
Evaluation always uses the original reset distribution and does not draw any
curriculum randomness. Checkpoints carry a separate reset-profile marker; replay
uses their saved runtime settings. This is an experimental training protocol:
warm-start successes are development results, and original-task performance
requires normal-start evaluation. It does not change the existing task defaults.

The separate builder
`models.grasp_reset_exploration:build_grasp_exploration_mappo_agent` uses the same
curriculum trainer and initializes the gripper Gaussian mean to the saved
`runtime.gripper_initial_mean` (default `-0.2`, toward closing). This is an
initialization experiment for exploring a slow gripper reference, rather than
an action override. Only the final gripper output layer is initialized; arm
outputs, exploration variance, message, controller and task conditions retain
their existing definitions. All weights remain trainable, and replay restores
the learned checkpoint weights. Apply the same initialization to every method
in a comparison. Command-space exploration measurements do not establish
physical grasp or learned-policy success.

Use `Isaac-OpenArm-PegInHole-Circle-Grasp-v0` for new Circle experiments.
Peg and hole are separate dynamic 0.2 kg bodies with gravity enabled. Reset
places them between the jaws once; subsequent movement comes from contact and
friction. Opening the grippers releases the parts. This task starts with a
pregrasp and does not include picking the parts up from a table.

Each arm has eight actions: seven joint increments and one gripper command.
The earlier Circle registrations retain their welded, seven-action behavior
for checkpoint replay. Their policies cannot be loaded into the grasp task.
CAD seating requires 18–21 mm insertion, a shaft radial envelope within
2.5 mm, axis alignment above 0.97, and both grasp points within 20 mm of their
commanded-closed grippers for 45 consecutive steps. A grasp-point error above
40 mm ends the episode as a drop. The grasp checks use poses and command state;
they do not measure contact forces.

From the repository root with `env_isaaclab` activated, visually check physical
holding followed by release:

```bash
uv run --no-project python scripts/environments/openarm_control_probe.py \
  --task Isaac-OpenArm-PegInHole-Circle-Grasp-v0 \
  --probe grasp_release --viz kit \
  --output /tmp/openarm_circle_grasp_release.json
```

Run a short Full 6D training pilot:

```bash
uv run --no-project python scripts/reinforcement_learning/skrl/train_openarm_bimanual_direct.py \
  --task Isaac-OpenArm-PegInHole-Circle-Grasp-v0 \
  --num_envs 8 --seed 42 --headless \
  --experiment_tag circle_contact_grasp_full6d \
  env.communication_mode=motion_context \
  agent.models.policy.initial_log_std=-1.0 \
  agent.agent.entropy_loss_scale=0.0 \
  agent.agent.experiment.checkpoint_interval=1024 \
  agent.trainer.timesteps=4096
```

Holding and transport probes validate the physical grasp. A complete insertion
rollout is a separate validation gate; passing the former does not establish
insertion success.

To view a trained contact-grasp policy in one environment, use its own saved
configuration and the grasp task ID:

```bash
uv run --no-project python scripts/reinforcement_learning/skrl/play_openarm_bimanual_direct.py \
  --task Isaac-OpenArm-PegInHole-Circle-Grasp-v0 \
  --num_envs 1 --viz kit --deterministic_eval \
  --checkpoint /path/to/contact-grasp/checkpoints/agent_60000.pt
```

Keep the run's `params/env.yaml` and `params/agent.yaml` alongside its
`checkpoints` directory. A legacy welded checkpoint has a different action
contract and requires its original task ID for replay.

An optional control ablation uses
`models.bounded_execution_mean:build_bounded_execution_mean_mappo_agent` as the
saved `runtime.agent_builder`. It bounds Gaussian arm means with softsign and
keeps the gripper distribution unchanged, including its close/open confidence.
Gaussian samples and their likelihoods remain unmodified; execution still
clips commands in the environment. This adds no trainable parameters and does
not change the 6D message, observations, rewards or exploration variance.
Apply the same builder to every method in a prospective comparison. Existing
configs keep the plain actor; this ablation is not evidence of improved task
performance until evaluated.

The separate control builder
`models.arm_reference_control:build_rate_limited_mappo_agent` keeps the plain
Gaussian policy and limits arm position-reference changes to each joint's
declared velocity limit times the control period. It supports Lift and
WithBall, updates references once per policy tick, and preserves the gripper
trajectory and task-owned reset buffers. It does not clamp actual velocities,
contact impulses or object states. Its saved runtime entry and checkpoint
markers prevent silent replay with the original controller. This controller
and the bounded-mean actor are separate experiments; neither is applied to the
frozen Peg comparison.

`models.arm_target_control:build_target_bounded_mappo_agent` is a separate Lift
and WithBall control experiment. It constrains arm references to the intersection
of the authored joint limits and the configured action range, with an optional
tracking-error limit (`runtime.arm_target_error_rad`, default 0.2 rad). It keeps
raw Gaussian actions, gripper trajectories, observations and rewards unchanged.
If physical joint state leaves the legal reference interval, the controller
commands recovery toward a legal target without rewriting that physical state.
Its checkpoint markers require matching playback control. Apply it consistently
to comparison methods; do not combine it with the reference-rate controller or
treat it as an improvement to the 6D communication representation. Deployment
support for these experimental controller markers has not been added.

For scenario comparisons, playback optionally accepts `--scenario_seed_start`
with one environment, `--num_eval_episodes` and saved traces. Each reset uses
an independent Torch seed while preserving the policy's random-generator state.
The evaluator checks recorded physical starts before adding scenario IDs and
writes `evaluation_reset_scenarios.json`; the final automatic reset is unscored.
Use identical scenario seeds across methods and keep these evaluations separate
from legacy rollout-dependent reset sequences. Equal recorded starts do not
establish determinism of Newton contact dynamics.
