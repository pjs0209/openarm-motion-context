# Newton 1.6.1 SDF loop synchronization

The installed Newton 1.6.1 mesh-SDF kernels reset a shared edge-progress
counter for each shape-pair pass without first synchronizing the previous
pass's readers. A warp can read the new counter during its previous pass's
exit test, leaving the block's cooperative barriers out of step. The upstream
fix adds `wp.tile_stack_clear(edge_stack)` immediately before each pass reset.
It does not change collision geometry, material, contact sampling or rewards.

Source: [Newton PR4859](https://github.com/newton-physics/newton/pull/4859),
merged commit `28d4b4f89be91dd79182c595b24c811ae8ab4cb0` on 2026-10-27.

For the current pinned environment:

```bash
conda activate env_isaaclab
uv run --no-project --python "$CONDA_PREFIX/bin/python" python \
  compat/apply_newton_161_sdf_barrier.py --apply
```

The installer checks the version and original source SHA256, refuses unrelated
library modifications, and is idempotent. Run without `--apply` to check status.
Restart simulator processes after applying: already imported modules keep their
old kernel definitions. Reapply after reinstalling Newton 1.6.1. For a newer
version, inspect its upstream fix instead of forcing this backport.

This is separate from the OpenArm mesh-overlap BVH workaround: that kernel was
localized by synchronous CUDA checking; the OpenArm configuration selects SAH
while preserving the same CAD/SDF collision surfaces. Do not replace bores or
box interiors with convex hulls to avoid this crash.
