OpenArm asymmetric recovery diagnostics
======================================

Scope
-----

Lift and contact-grasp circle Peg use one research question: whether compact
partner motion information adds value for recovery from asymmetric physical
forces. Lift includes approach, grasp, lift and hold. Peg starts with two free
parts placed between the fingers at reset; table pickup is outside its scope.
The current diagnostics do not establish communication superiority or hardware
success. Existing checkpoint physics and control settings are restored.

Single-policy evaluation
------------------------

Use the uv-managed training interpreter and the pinned skrl 1.4.3 playback
entry point required by these saved checkpoints. Example pulse JSON::

  {
    "onset_s": 0.5,
    "duration_s": 0.2,
    "force_n": 1.0,
    "events": [
      {"side": "left", "direction_w": [0, 1, 0]},
      {"side": "right", "direction_w": [0, -1, 0]}
    ]
  }

Directions are unit vectors in the simulation world, not the wrist frame.
The selected physical wrist link is ``openarm_left_link7`` or
``openarm_right_link7``. Force acts at its center of mass with zero applied
external torque. Events cycle by reset index without consuming policy RNG.
Onset and duration round upwards to whole physics steps; effective times are
saved. Zero N retains the same instrumentation and nominal event window.

Append these options to a normal saved-checkpoint playback command::

  --num_envs 1 --deterministic_eval --num_eval_episodes 2
  --scenario_seed_start 42100 --save_motion_context_trace
  --motion_context_trace_dir <new-directory>
  --wrist_disturbance <pulse.json>

The options must be supplied on one command line. This initial implementation
supports single-world Newton evaluation. It does not enable disturbances in
training or change default task observations, rewards, physics or actions.
Paired Torch reset seeds establish observed initial-state equality, not solver
determinism. Preserve independent training seeds as the statistical unit for
training reproducibility.

Recorded evidence
-----------------

``wrist_disturbance_substeps.jsonl`` records each physics-step input, both wrist
body names, world COM application points, requested forces and the actual Newton
``body_f`` input after asset submission. Wrench order is force XYZ then torque
XYZ. These inputs are not contact forces or net forces measured on hardware.
Zero input is explicitly submitted after the pulse. A mismatch raises an error.
The summary records delivered input impulse and partial pulses for each episode.
The final automatic reset is unscored.

``control_timing.jsonl`` records measured joints, accumulated arm targets,
gripper position commands and latches before and after action processing, plus
outgoing messages and context timing. The raw actor partner slot is checked
against the handed-off message. Trace reads are guarded against mutations of
motion-context tensor state.

A previously existing trace fallback advanced EMA/history before the first
message after reset. The collectors now report zero before any message has been
generated. Historical evaluations remain preserved; new diagnostics must identify
the corrected source version. This repair alone is not evidence that seed-specific
learning failures were caused by logging.

Campaigns and analysis
----------------------

``run_recovery_evaluation.py`` runs a declared manifest sequentially, archives
sources, records checkpoint hashes and exact commands, and stops if sources
change between cases. Each case declares its task, checkpoint, reset seeds,
evaluation seed, episode count and pulse. Use a new output directory::

  uv run python scripts/studies/run_recovery_evaluation.py \
    --manifest <pilot.json> --output <new-campaign-directory>

Analyze completed case directories with::

  uv run python scripts/studies/analyze_recovery_evaluation.py \
    <case-directory> [<case-directory> ...] --output <analysis-directory>

The analysis reports overall task completion, completion conditional on reaching
the scheduled onset, full-window exposure and early terminations. Early failures
remain in the overall denominator. Completion time after the pulse is separate
from geometric alignment recovery. The latter requires an aligned pre-event
state, observed alignment loss, and regained alignment held for 0.2 seconds.
Peg uses shaft clearance, axis alignment and grasp validity; Lift uses tilt and
grasp validity. No observed loss has no recovery-time value. Grasp validity here
uses the task's geometry/closed-command definition, not a contact-force sensor.

The 0/1 N, two-episode initial campaign validates implementation and diagnoses
failures. Its force, duration and seeds are declared before comparing methods.
It is not a sufficient performance study. Calibration of meaningful force ranges,
contact compliance checks, matched disturbance training and independent training
replications remain required. Normal-policy stress tests and policies trained
with disturbances must be reported separately.

Own controller-state profile
----------------------------

``env.observe_control_state=true`` adds ten local inputs after the original
30 channels: seven accumulated target-minus-measured joint errors in radians,
the mean rate-limited symmetric jaw reference in metres, the close latch, and
remaining reset command-suppression time in seconds. Camera rotation channels
remain last. No force, external-event flag, reward, success signal or partner
state is added. Default observations retain their previous dimensions.

Plain Lift and contact Peg then have own40/actor80. Signed-camera Lift has
own43/actor86/critic99; Peg has critic106. The fixed partner slot grows with the
own observation width, while MotionOnly/Twist/Trend payload definitions remain
unchanged and padded with zeros. Full-partner-observation explicitly includes
the sender's enlarged own observation. NoShare retains the same encoder width
with zero received messages. Compare communication methods with the same own
observation profile and history access.

Use ``models.control_state:build_control_state_mappo_agent`` for the plain
MAPPO profile, or ``models.control_state:build_control_state_grasp_exploration_agent``
for the signed-camera Lift reset/exploration profile (full module prefix:
``isaaclab_tasks.direct.openarm_motion_context``). The builders mark the new
layout in the checkpoint and reject incompatible weights. This opt-in profile
changes the information available to policies; it is a fresh-training experiment,
not a way to adapt legacy weights or evidence that local state improves success.

Checkpoint distribution audit
-----------------------------

Wrist diagnostic evaluation additionally captures the task's raw actor inputs in
``actor_observations.jsonl``. Appending ``--audit_checkpoints`` followed by
same-run checkpoint paths performs a post-evaluation distribution audit on the
first scored episode's captured states. Include the evaluated checkpoint in
that list. Each checkpoint's real policy and saved state standardizer are loaded;
no physics rollout is performed for the other checkpoints. The reference must
reproduce the observed mean actions within numerical tolerance.

``checkpoint_distributions.json`` reports gripper means, Gaussian standard
deviations, close-selection fractions and sampling probabilities. These describe
old policies on states visited by the reference policy; they do not measure the
old policies' task performance. The original evaluated checkpoint is restored
when the audit finishes. Raw inputs precede masking and normalization, so this
audit rejects diagnostic-mask runs.

Main four-condition comparison
------------------------------

``message_comparison_preflight.json`` declares one common configuration per
task. ``run_message_comparison.py`` expands it into independent fresh training
runs for NoShare, MotionOnly, Signed Twist and Full6D Trend. There are no
condition-specific controller, reward, curriculum, optimizer or model overrides.
The manifest's 512 steps are an integration check, not a learning budget for
performance claims. A longer study requires a new manifest and output directory.

================  ==================  =============  =====================
Condition         communication_mode  sharing_mode   Context representation
================  ==================  =============  =====================
NoShare           motion_context      no_share       trend (not transmitted)
MotionOnly        motion_only         intent_share   trend (not transmitted)
Signed Twist      motion_context      intent_share   signed_twist
Full6D Trend      motion_context      intent_share   trend
================  ==================  =============  =====================

All retain the same partner MLP and fixed input width. NoShare uses constant
raw zeros from the beginning of its own training; ``communication_mode=none``
would remove the encoder and is deliberately excluded. MotionOnly populates
the first three entries, Twist and Trend the first six, and remaining entries
are zero padding. The shared translation signal is the existing deterministic
base-frame signed EE motion expression, not a learned intent predictor. Twist
contains signed base-frame angular velocity divided by 3 rad/s and clipped to
[-1, 1]. Trend contains translation, rotation and action-stability summaries.
H=5 spans 5/60 seconds with the current 60 Hz controller. Trend's EMA also
retains information beyond that finite difference interval; H alone is not
an exact memory cutoff. No explicit own-observation history stack is currently
provided to any of these four conditions. A claim specific to temporal summary
efficiency needs a later comparison using matched raw histories.

Declare commands without launching, or add ``--execute`` for training::

  uv run python scripts/studies/run_message_comparison.py \
    --manifest scripts/studies/message_comparison_preflight.json \
    --output <new-campaign-directory>

``models.main_comparison:build_main_comparison_agent`` selects only the common
own-control-state builder for each task. Lift retains its common target-error
bound, signed camera observation, reset curriculum and gripper initialization.
Peg retains its pregrasp contact task and common task-specific geometry/control
logic. The preflight declares a 10 N gripper effort-cap sensitivity profile for
all four Peg conditions; this is not a calibrated hardware force specification.
It must not be pooled with old 333.33 N checkpoint results. Lift settings are
unchanged apart from the previously declared common own-control-state inputs.

Each run saves ``main_comparison_model.json`` with actual Actor/Critic classes,
state shapes, parameter counts and initial state hashes. After training the
runner compares saved ``env.yaml`` and ``agent.yaml`` across conditions for the
same task/seed. Only the three declared environment message fields, the policy's
communication mode and the run name may differ. Actual model shapes and initial
weights must match as well. Statistics learned from each run may differ; their
estimation rules remain common. Sources and YAML defaults are archived before
launch and may not change during a campaign. Preserve the archive, manifest,
saved configs, checkpoint hashes and evaluation seeds with reported results.

The learner is the custom ``MotionContextMAPPO`` over pinned skrl 1.4.3, not an
unmodified stock MAPPO implementation. Its explicit update computes per-arm
GAE with termination/truncation boundaries, normalized advantages, clipped PPO
surrogates, entropy and central-value losses, sums active per-arm losses for
backpropagation, and uses per-arm optimizers, gradient clipping and KL-adaptive
learning rates. Actor, shared-state and value running standardizers are separate.
Gaussian latent samples are scored for PPO while the environment limits executed
actions. The selected common builder adds no intent classifier, decoder cap,
learned communication gate, auxiliary loss or method-specific reward.

Physical contact audit
----------------------

Wrist diagnostic evaluation also writes ``physical_audit.jsonl`` and a summary.
These read Newton's MuJoCo Warp solved contacts at the final physics substep of
each control step, before terminal resets: signed contact distance, active solver
constraints and world contact forces. They are not averages over all substeps
or hardware force measurements. Negative solver distance is not an exact CAD
penetration measurement. For Circle Peg, the audit independently derives shaft
radius and tip/entrance/bottom geometry from the hashed USD meshes/transforms
and fresh native body poses. A reported task success that fails this independent
geometric check aborts the diagnostic. Part mass, free-body status and gravity
compensation are recorded separately from the task's grasp predicate.
