# Current Lift/Peg research status — 2026-10-28

## Main comparison

The intended comparison uses independently trained NoShare, MotionOnly, Signed
Twist and Full6D Trend policies. Within each task only the partner message varies.
All four keep the partner encoder, own controller-state observations, Actor/Critic,
optimizer, rewards and common task controller/curriculum. NoShare receives constant
zeros from the start of training. Historical gate/decoder variants are retained
as separate experiments and are not selected by this main configuration.

| Task | Own / partner slot | Actor parameters per arm | Critic parameters per arm |
| --- | --- | ---: | ---: |
| Signed-camera Lift | 43 / 43 | 63,888 | 91,649 |
| Contact-grasp Circle Peg | 40 / 40 | 63,408 | 93,441 |

All four message conditions retain the 160-wide arm-head input. The common
controller-state extension provides target errors, jaw reference, latch and reset
grace; it is not evidence that these inputs fix learning. No explicit own history
stack is used. Trend-specific claims require a later raw-history comparison.

## What has actually been checked

- Eight fresh integration runs: two tasks, four conditions, seed 42, eight worlds,
  512 control steps each. All completed and produced final checkpoints.
- Resolved environment/agent YAMLs match across conditions except the explicitly
  declared message fields and run identifier. Actual Actor/Critic classes, state
  shapes, parameter counts and initial weights also match for each task/seed.
- An independent receiving checkout at the pinned upstream commit passed the
  five relevant regression files: **25 passed**, 46 dependency deprecation
  warnings. This follow-up is not a rerun of the entire historical suite.
- Training sources and publication/installed sources were compared over 113
  files. Python executable ASTs match after excluding docstrings/positions; YAML
  defaults match byte-for-byte. Raw hashes and checkpoint hashes are recorded in
  [CURRENT_RESEARCH_PROVENANCE.json](CURRENT_RESEARCH_PROVENANCE.json).
- The preflight uses a common 10 N finger effort-cap sensitivity profile for Peg.
  This is a changed physics profile requiring fresh training, not a modification
  to legacy checkpoint evaluation or a hardware-calibrated setting.

Each of the eight checkpoints also completed 32 control steps on the independent
publication receiver. Actual policy messages retained their declared zero padding;
NoShare was zero throughout. Its saved training standardizers had zero partner
means, and raw zero still produced exact zero through the actual saved scaler.
All saved model/scaler tensors were finite. A Peg MotionOnly preflight dropped at
step 28 and reset; both resulting episodes were included in the 32-step input audit.
The initial audit incorrectly required 32 steps in the first episode; that harness
assumption was corrected without discarding the early failure.

The short runs validate construction, updates, saved settings and checkpoint
handling. They are not task-success rates, independent-seed reproducibility,
disturbance recovery evidence or hardware results.

## Diagnoses that remain relevant

The old Lift V44 seed-42 policy did not grasp normally in the diagnostic episode.
Actual training telemetry shows closing ratios falling while all resets were still
warm, before the curriculum faded. Saved checkpoint outputs on final-policy
states also drift toward opening. These fixed-input outputs are not evaluations
of old-policy task success and do not establish a single causal failure.

Old Peg seed-42 policies succeed in the small 0/1 N diagnostic; Full6D seed 43
fails. The independent USD geometric check agrees with a newly replayed seed-42
success, but solved-contact distances show up to about **6.22 mm overlap**.
Signed solver distance is not exact CAD penetration. Earlier hold/transport
checks at 10 N were feasible, while replaying the old high-cap policy at 10 N
failed. Contact realism and performance after fresh low-cap training remain open.
Do not use the old contact success alone to claim physically validated insertion.

## Remaining work

Diagnose normal grasp acquisition and closing behavior under the common Lift
profile; validate contact and insertion behavior under the declared Peg physics;
then run matched normal/disturbance evaluations and training with independent
seeds. Keep all failures in the overall denominator. Conditional recovery requires
observed loss and sustained recovery, rather than task completion alone.

The source overlay requires the separately licensed assets and matching simulator
environment described in [RESEARCH_SETUP.md](RESEARCH_SETUP.md). Source publication
and deployment support do not supply a certified successful hardware policy.
