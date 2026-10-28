# Research-source publication validation — 2026-10-28

## Scope

This publishes the current source overlay, not a new algorithm experiment.
No training job, reward tuning, hardware command, upstream merge, or new success
rate evaluation was performed for publication. Existing source headers and the
vendored skrl MIT license were retained. The deployment/browser implementation
already on `main` was preserved, alongside the local gripper-rate/gravity contract
additions. The original working experiment directory was not reformatted.

The simulation overlay installs onto upstream Isaac Lab commit
`e3f7d9ff1f5d7f8280de3ece9000b7117f66bcf4`. The standalone publication branch is
based on `10fc0a4c77b7d1252854491237552976dd55d634`.

## Checks actually run

- Installed and verified 433 scoped source/config/test files into an independent,
  clean checkout of the pinned Isaac Lab commit. A second install changed zero
  files. Installation is a preview unless `--apply` is supplied; conflicting local
  overlay files require `--overwrite`. Unrelated launcher edits are rejected.
- Ran each `test_openarm*.py` and the Kit-launcher regression file in its own
  Python 3.12 process against that receiving checkout: **179 passed**, zero
  failures/errors, one ROS visualization module skipped because `rclpy` is not
  available in the simulator Python environment. The actual imported task path
  was checked to belong to the receiving checkout, not the original workspace.
- Reran the visualization and portable deployment files under the separate ROS
  Humble/Python 3.10 environment: **6 passed**. This includes three portable cases
  already covered above; do not add the counts as independent evidence. ROS tests
  used isolated localhost domains and synthetic messages, not motors/cameras.
- Applied Ruff 0.14.10 formatting to 71 publication-copy files; 155 scoped Python
  files subsequently passed the format check. A 176-file AST comparison, excluding
  docstring text and source positions, found no executable AST changes caused by
  this mechanical formatting. No import reordering or unused-code pruning was
  performed to disguise historical lint debt.
- The new overlay installer passed Ruff lint and format checks.
- The receiving checkout passed the Isaac Lab changelog gate after three legacy
  fragments were converted to the required bullet format in the publication copy.
- A scoped scan found no GitHub token, AWS access-key ID, or private-key marker
  in the candidate source. This is a pattern scan, not a complete secret audit.
- Reviewed staged paths and file sizes; no new checkpoints, logs, USD/CAD binaries,
  Python caches, generated plans, credentials, or local virtual environments were
  selected for publication. Historical checkpoints already on `main` were retained.

The simulator tests used the workstation's existing licensed assets through the
four `OPENARM_*_ASSET_DIR` overrides. This is **not** validation on an asset-free
fresh machine, a GPU rollout, a new 60k training run, or real manipulation.

## Checks that are not clean

The standard `isaaclab --format` wrapper could not install its missing pre-commit
dependency in the restricted environment. Direct, offline Ruff checks were run
instead using the pinned Isaac Lab configuration.

The existing research source has **119 lint findings** after formatting:

| Rule | Count |
| --- | ---: |
| I001 import ordering | 58 |
| F401 unused imports | 25 |
| E501 long lines | 17 |
| F841 unused variables | 5 |
| F811 repeated definitions/import names | 4 |
| C901 complexity | 3 |
| SIM105 suppressible exceptions | 3 |
| SIM910 redundant dictionary default | 2 |
| SIM114 equivalent branches | 1 |
| UP015 redundant open mode | 1 |

They predate this source publication; the initial unformatted scan had 266
findings. Keeping executable code unchanged takes precedence over unsafe cleanup
while publishing the experimental snapshot. The vendored upstream skrl tree is
not reformatted or graded as new project-authored code. This is not a clean-lint
release and must not be described as one. Any publication PR should remain a
draft until these limitations have been reviewed.

## Deliberate exclusions and limits

All reusable current task/study/training/replay/deployment code is included.
Large model/log archives, generated analysis results and figures, external USD
assets and retired Isaac Lab 2.3 workspaces are not mirrored. The asset inventory
records local filenames, sizes and SHA-256 values without redistributing mixed-
license data. See `RESEARCH_SETUP.md` for installation and outstanding dependency
requirements. Historical source snapshots embedded in old experiment logs are not
newly reconstructed or represented as current source.

The research results still require their original checkpoint/configuration/trace
provenance. Code publication does not establish Full 6D superiority, successful
WithBall centered holding, physically calibrated Peg contact, or sim-to-real success.


## Follow-up: common four-message protocol

The statements above describe the initial publication snapshot. The follow-up
adds the actual own-control-state profile, timing/trace repair, independent CAD
and solved-contact audit, checkpoint-distribution audit and common four-condition
runner. See [CURRENT_RESEARCH_STATUS.md](CURRENT_RESEARCH_STATUS.md) for the
separate eight-run GPU integration check and 25-test receiving-checkout result.
These short runs were performed explicitly for the follow-up; installation itself
does not launch them. New diagnostic modules and scripts pass scoped Ruff lint.
Historical lint debt and separate licensed asset requirements remain.
