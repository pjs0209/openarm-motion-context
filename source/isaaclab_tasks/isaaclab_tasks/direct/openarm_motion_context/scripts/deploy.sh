#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../../.." && pwd)"
CONFIG="${1:-${REPO_ROOT}/source/isaaclab_tasks/isaaclab_tasks/direct/openarm_motion_context/deploy/configs/lift_real.yaml}"

exec "${REPO_ROOT}/isaaclab.sh" -p -m \
  isaaclab_tasks.direct.openarm_motion_context.deploy.deploy_node \
  --config "${CONFIG}"
