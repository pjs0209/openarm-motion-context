# OpenArm Motion Context

OpenArm bimanual manipulation with decentralized MAPPO actors and a compact
partner motion message. The current Newton Lift deployment message is:

```text
z = [m_x, m_y, m_z, translation trend, rotation trend, stability trend]
```

The Lift actor uses its own 30D observation plus a 30-wide partner slot with
six populated values. The software interface, normalization and one-tick
communication delay match the Newton training workspace.

## Newton Lift deployment

See the maintained [deployment guide](source/isaaclab_tasks/isaaclab_tasks/direct/openarm_motion_context/deploy/README.md)
for ROS Humble setup, checkpoint export, per-camera AprilTag configuration,
OpenArm V1.0 bringup and dry-run validation.

This deployment update does not port the repository's historical simulation
overlay or certify a trained policy. Train in the validated Isaac Lab 3/Newton
workspace and copy a run checkpoint **together with `params/env.yaml` and
`params/agent.yaml`**. The older checkpoints already in `artifacts/` are retained
as historical artifacts and are not validated for the new deployment adapter.

Use a separate ROS Python environment. This workstation uses Python 3.10 for
ROS Humble and Python 3.12 for Isaac Lab training:

```bash
source /opt/ros/humble/setup.bash
uv venv --python /usr/bin/python3 --system-site-packages .venv-openarm-deploy
uv pip install --python .venv-openarm-deploy/bin/python \
  -r source/isaaclab_tasks/isaaclab_tasks/direct/openarm_motion_context/deploy/requirements.txt
```

Validate and export a checkpoint without ROS or Isaac Sim:

```bash
uv run --no-project --python .venv-openarm-deploy/bin/python python \
  scripts/deploy/openarm_lift_ros2.py \
  --config scripts/deploy/openarm_lift_deploy.yaml \
  --checkpoint /absolute/path/to/run/checkpoints/agent_60000.pt \
  --export-config /absolute/path/to/deployment/lift.yaml
```

The export validates both policies, restores their recorded settings and writes
camera-specific detector YAMLs. `--validate-checkpoint` only validates. Runtime
defaults to dry-run; `--publish` enables commands. Export does not recalibrate
physical camera mounts, prove grasp success, or change motor gains.

Portable software regressions:

```bash
uv pip install --python .venv-openarm-deploy/bin/python pytest
uv run --no-project --python .venv-openarm-deploy/bin/python python -m pytest \
  source/isaaclab_tasks/test/test_openarm_deployment_runtime.py -q
```

Reported standard 36h11 print sizes, including the one-cell white margin, are
50 mm for tag 0 and 20 mm for tags 1 and 2. The detector black-border edge sizes
are therefore **40, 16 and 16 mm**. This is based on the reported print layout;
check it against the actual print before metric pose calibration.

## Scope and evidence

- The ROS bridge and both policies initialized under ROS Humble/Python 3.10 in
  an isolated localhost domain, without sending motor commands.
- Current workspace regressions compare simulator and ROS 30D observations,
  checkpoint preprocessing, controller timing and backend decimation behavior.
- At zero and configured start joint poses, official V1 URDF and Newton USD
  hand-frame FK differ by about 1.414 mm and less than 1e-6 degrees. This is a
  model comparison, not a measured hardware calibration.
- Real motor dynamics differ from the simulated high PD gains and disabled
  robot gravity. Real tracking and gravity compensation still need validation.
- This adapter accepts Lift Trend policies. WithBall needs real ball perception;
  the welded Circle task needs a physical grasp adapter. They are not accepted
  as Lift policies.

The deployment implementation lives in
`source/isaaclab_tasks/isaaclab_tasks/direct/openarm_motion_context/deploy`.
The public `scripts/deploy` launchers remain supported. USD assets and new
training checkpoints are not part of this code update.

## License

See [LICENSE](LICENSE). Upstream OpenArm, Isaac Lab and ROS components retain
their own licenses.
