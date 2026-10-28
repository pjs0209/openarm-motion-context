# OpenArm Isaac Lab 3 / Newton research source

This repository now includes the current Lift, LiftWithBall and Circle contact-grasp
tasks, communication and actor variants, training/replay/analysis entry points,
research studies, existing regressions, and the required local compatibility code.
It is a **research snapshot**, not a claim that every experiment succeeds or that
all historical results reproduce under these current defaults.

The ROS deployment instructions in the root README remain a separate Python 3.10
workflow. Simulation uses Python 3.12. Do not run a simulation installer inside the
ROS environment or replace the simulator's dependency pins with ROS dependencies.

## Install the source overlay

Use an independently installed Isaac Lab 3 checkout at the recorded upstream
commit. Follow its installation instructions for the Isaac Sim 6.2 / Newton
environment before installing this overlay. This command does not install system
packages, drivers, NVIDIA assets, or a Python environment:

```bash
git clone https://github.com/isaac-sim/IsaacLab.git IsaacLab-newton
git -C IsaacLab-newton checkout e3f7d9ff1f5d7f8280de3ece9000b7117f66bcf4
# From this openarm-motion-context repository, with env_isaaclab activated:
uv run --no-project python scripts/install_research_overlay.py --isaaclab /path/to/IsaacLab-newton
uv run --no-project python scripts/install_research_overlay.py --isaaclab /path/to/IsaacLab-newton --apply
```

The first command previews. The second copies only explicitly scoped OpenArm
source, study/deploy scripts, tests, changelogs and compatibility modules. It also
installs the fatal-CUDA shutdown correction in the upstream Kit launcher, rejecting
unrelated edits in that file. Existing different overlay files require the explicit
`--overwrite` option. Prefer a clean checkout; the installer does not remove files.
The root `third_party/IsaacLab` gitlink records the same upstream commit; initializing
the entire ROS submodule graph is not required for simulation.

Recorded workstation pins: Python 3.12, Isaac Sim 6.2.0.0, PyTorch 2.11.0,
Newton 1.6.1 and Warp 1.17.0. Use Isaac Lab's matching CUDA/renderer installation,
not a blind upgrade of an existing environment. The `compat/skrl_1_4_3` source and
its MIT license are included because these MAPPO/checkpoint contracts use skrl
1.4.3 rather than substituting skrl 2. Training/replay put that directory first
on the import path. The installed skrl 2 distribution may coexist for other tasks.

From the receiving Isaac Lab checkout, check the separate Newton site-package fix:

```bash
uv run --no-project python compat/apply_newton_161_sdf_barrier.py
# Only when the checker requests it:
uv run --no-project python compat/apply_newton_161_sdf_barrier.py --apply
```

See [NEWTON_SDF_BACKPORT.md](../compat/NEWTON_SDF_BACKPORT.md). The patcher checks
the exact source hash and version and refuses unrelated library changes. Do not
force this old patch onto a newer Newton version.

## Assets are separate, licensed inputs

USD/CAD/texture files are **not redistributed in this code snapshot**. The local
scenes combine custom modifications, NVIDIA asset references, camera CAD and
external layers without a complete public redistribution/license manifest. Do
not relabel those meshes or textures as this repository's BSD-3-Clause code.
The [asset inventory](LOCAL_ASSET_INVENTORY.tsv) records paths, sizes and hashes
without copying the assets.

Supply your existing licensed asset directories under the receiving checkout:

```text
source/isaaclab_assets/data/Robots/OpenArm/
source/isaaclab_assets/data/Environments/OpenArm/
source/isaaclab_assets/data/Objects/OpenArm/
source/isaaclab_assets/data/Sensors/OpenArm/
```

Alternatively set `OPENARM_ROBOT_ASSET_DIR`, `OPENARM_ENVIRONMENT_ASSET_DIR`,
`OPENARM_OBJECT_ASSET_DIR` and `OPENARM_SENSOR_ASSET_DIR` to those four category
directories. The old flat `OPENARM_ASSET_DIR` layout is not a substitute. Preserve
relative USD dependencies and access to the upstream NVIDIA materials/robot
layers. `OmniPBR.mdl`, remote AprilTag materials and some robot layers require the
appropriate USD/Isaac resolver and licensed asset installation; an ordinary USD
dependency walk does not certify all of them. The local legacy `openarm_robot.usd`
also contains an unresolved configuration-layer reference. This publication does
not repair assets or silently replace their collision geometry.

## Train and replay

Run these from the receiving Isaac Lab checkout after setup and asset preparation:

```bash
uv run --no-project python scripts/reinforcement_learning/skrl/train_openarm_bimanual_direct.py \
  --task Isaac-OpenArm-Lift-v0 --num_envs 64 --seed 42 --headless \
  --experiment_tag lift_full6d \
  env.communication_mode=motion_context \
  env.motion_context_representation=trend \
  env.motion_context_trend_horizon=5 \
  agent.trainer.timesteps=60000
```

Use `Isaac-OpenArm-LiftWithBall-v0` for the rendered-depth/segmentation ball
observation task; it needs an RTX-capable renderer. Use
`Isaac-OpenArm-PegInHole-Circle-Grasp-v0` for new contact-grasp studies. Old Circle
task IDs preserve welded seven-action behavior and are not interchangeable with
the eight-action contact-grasp policy. Rotation-observation and curriculum
experiments require their own recorded task and runtime settings; see the
[task package guide](../source/isaaclab_tasks/isaaclab_tasks/direct/openarm_motion_context/README.md).
These commands are baseline entry points, not reconstructions of every past run.

```bash
uv run --no-project python scripts/reinforcement_learning/skrl/play_openarm_bimanual_direct.py \
  --task Isaac-OpenArm-PegInHole-Circle-Grasp-v0 --num_envs 1 --viz kit \
  --deterministic_eval --checkpoint /path/to/run/checkpoints/agent_60000.pt
```

Keep `params/env.yaml`, `params/agent.yaml`, normalizers and any message sidecar
with the checkpoint. Replay restores saved contracts and rejects implicit backend
changes. Models under `artifacts/` are historical, not current Newton success
policies. Large models and original training/evaluation logs are not newly uploaded.

## Scope and known limitations

- Lift/WithBall normal-start learning is not certified successful by this source
  publication. WithBall parent-Lift flags are not centered-ball task success.
- Contact Peg begins from reset pregrasp, not table pickup. Pose/closed-command
  checks are not force-closure measurements; contact penetration realism remains
  a separate validation question.
- Current code includes disabled-by-default wrist-disturbance protocol code and
  its configuration test. This is not evidence of successful disturbance recovery.
- The ROS bridge remains limited to compatible Lift Trend checkpoints. Added
  signed-rotation observation profiles need a matching real observation encoder;
  WithBall and Peg require separate perception/grasp adapters.
- `scripts/studies` contains experimental and historical configurations, not
  endorsed final methods. NoShare, Absolute, Twist and Trend results must retain
  their own physics, reward, observation, checkpoint and seed protocols.
- No new learning run or long GPU validation is launched by installing this snapshot.

## Checks

Use the matching simulator environment for simulator-dependent tests. Put the
vendored skrl runtime first when invoking pytest directly:

```bash
PYTHONPATH="$PWD/compat/skrl_1_4_3:$PWD${PYTHONPATH:+:$PYTHONPATH}" \
  uv run --no-project python -m pytest source/isaaclab_tasks/test/test_openarm_policy_conventions.py -q
```

Run each other `test_openarm*.py` file in its own process using the same command.
Portable deployment tests in the standalone checkout do not establish GPU task
success. See the publication validation record for the actual checks run and
any pre-existing lint or environmental failures; do not infer a clean full-suite
result from the presence of tests. The [publication validation record](PUBLICATION_VALIDATION.md)
documents the actual checks and the remaining historical lint debt.
