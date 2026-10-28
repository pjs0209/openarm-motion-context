# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Preview or copy the OpenArm research overlay into a pinned Isaac Lab checkout.

This copies source only. It does not install packages, fetch assets, apply the
Newton site-package backport, or execute a policy or hardware command.
"""

import argparse
import hashlib
import shutil
import subprocess
from pathlib import Path

ISAACLAB_COMMIT = "e3f7d9ff1f5d7f8280de3ece9000b7117f66bcf4"
SOURCE_ROOT = Path(__file__).resolve().parents[1]
OVERLAY_TREES = (
    "source/isaaclab_tasks/isaaclab_tasks/direct",
    "source/isaaclab_tasks/changelog.d",
    "source/isaaclab_physx",
    "scripts/studies",
    "scripts/deploy",
    "compat",
)
OVERLAY_FILES = (
    "scripts/analyze_motion_context_trace.py",
    "scripts/environments/openarm_control_probe.py",
    "scripts/reinforcement_learning/skrl/train_openarm_bimanual_direct.py",
    "scripts/reinforcement_learning/skrl/play_openarm_bimanual_direct.py",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--isaaclab", type=Path, required=True, help="Existing Isaac Lab checkout to receive source")
    parser.add_argument("--apply", action="store_true", help="Copy files; the default only previews")
    parser.add_argument("--overwrite", action="store_true", help="Explicitly replace conflicting local overlay files")
    args = parser.parse_args()
    target = args.isaaclab.expanduser().resolve()
    if target == SOURCE_ROOT or not (target / "source/isaaclab/isaaclab").is_dir():
        parser.error("--isaaclab must be a separate, complete Isaac Lab source checkout")
    revision = subprocess.check_output(["git", "-C", str(target), "rev-parse", "HEAD"], text=True).strip()
    if revision != ISAACLAB_COMMIT:
        parser.error(f"Expected Isaac Lab {ISAACLAB_COMMIT}; found {revision}. Use a separate pinned checkout.")
    candidates = {SOURCE_ROOT / name for name in OVERLAY_FILES}
    for name in OVERLAY_TREES:
        candidates.update((SOURCE_ROOT / name).rglob("*"))
    candidates.update((SOURCE_ROOT / "source/isaaclab_tasks/test").glob("test_openarm*.py"))
    files = sorted(
        path
        for path in candidates
        if path.is_file() and not any(part in {"__pycache__", "node_modules", ".pytest_cache"} for part in path.parts)
    )
    conflicts = []
    changes = []
    for source in files:
        relative = source.relative_to(SOURCE_ROOT)
        destination = target / relative
        if not destination.resolve().is_relative_to(target):
            parser.error(f"Destination escapes the checkout through a symlink: {relative}")
        if destination.exists() and destination.read_bytes() == source.read_bytes():
            continue
        changes.append((source, destination))
        if destination.exists():
            # The tracked upstream launcher is deliberately patched. Reject
            # unrelated edits before replacing it, even with --overwrite.
            if str(relative) in {
                "source/isaaclab_physx/isaaclab_physx/app/kit_launcher.py",
                "source/isaaclab_physx/test/app/test_kit_launcher.py",
            }:
                baseline = subprocess.check_output(["git", "-C", str(target), "show", f"HEAD:{relative}"])
                if destination.read_bytes() != baseline:
                    parser.error(f"Unrelated changes in an upstream launcher file: {relative}")
            else:
                conflicts.append(str(relative))
    print(f"Pinned Isaac Lab: {revision}")
    print(f"Overlay files: {len(files)}; changed destinations: {len(changes)}")
    for relative in conflicts:
        print(f"Existing different file: {relative}")
    if not args.apply:
        print("Preview only; no files written. Add --apply to copy source.")
        return
    if conflicts and not args.overwrite:
        parser.error("Conflicting overlay files exist; preserve them or explicitly add --overwrite")
    for source, destination in changes:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        if hashlib.sha256(source.read_bytes()).digest() != hashlib.sha256(destination.read_bytes()).digest():
            raise RuntimeError(f"Copy verification failed: {destination.relative_to(target)}")
    print(f"Copied and verified {len(changes)} files. Assets and Python dependencies remain separate inputs.")


if __name__ == "__main__":
    main()
