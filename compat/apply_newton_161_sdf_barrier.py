# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Apply the two SDF loop barriers from Newton PR4859 to the pinned 1.6.1 runtime."""

import argparse
import hashlib
import importlib.metadata
import importlib.util
import os
import stat
from pathlib import Path

ORIGINAL_SHA256 = "3ae93d6fca86e5d6abf6e209e1fd6a88b8bba9cd2235980f7f6edd0c8888e5a7"
UPSTREAM_COMMIT = "28d4b4f89be91dd79182c595b24c811ae8ab4cb0"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="Write the checked backport; otherwise only report status")
    args = parser.parse_args()
    if importlib.metadata.version("newton") != "1.6.1":
        raise RuntimeError("This backport is only for Newton 1.6.1; inspect upstream support for another version")
    spec = importlib.util.find_spec("newton")
    if spec is None or spec.origin is None:
        raise RuntimeError("Newton installation not found")
    path = Path(spec.origin).parent / "_src/geometry/sdf_contact.py"
    source = path.read_text()
    replacements = []
    for start in ("0", "edge_start"):
        original = f"                wp.tile_scatter_masked(progress, 0, {start}, t == 0)\n"
        fixed = (
            f"                # OpenArm backport of Newton PR4859 ({UPSTREAM_COMMIT[:7]}).\n"
            "                wp.tile_stack_clear(edge_stack)\n" + original
        )
        replacements.append((original, fixed))
    restored = source
    for original, fixed in replacements:
        restored = restored.replace(fixed, original)
    if hashlib.sha256(restored.encode()).hexdigest() != ORIGINAL_SHA256:
        raise RuntimeError(f"Unrecognized Newton SDF source; refusing to overwrite unrelated changes: {path}")
    patched = restored
    for original, fixed in replacements:
        if patched.count(original) != 1:
            raise RuntimeError("Expected exactly one reset for each SDF directional loop")
        patched = patched.replace(original, fixed)
    if source == patched:
        print(f"Newton 1.6.1 SDF barrier backport already applied: {path}")
        return
    if not args.apply:
        raise SystemExit("Newton 1.6.1 needs PR4859 barriers; run this script with --apply before training")
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(patched)
        temporary.chmod(stat.S_IMODE(path.stat().st_mode))
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"Applied Newton PR4859 two-barrier backport: {path}")
    print(f"Patched SHA256: {hashlib.sha256(patched.encode()).hexdigest()}")


if __name__ == "__main__":
    main()
