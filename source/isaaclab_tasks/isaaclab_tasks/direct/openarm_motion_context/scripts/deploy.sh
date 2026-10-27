#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../../.." && pwd)"
CONFIG="${1:-${REPO_ROOT}/source/isaaclab_tasks/isaaclab_tasks/direct/openarm_motion_context/deploy/configs/lift_real.yaml}"
if (( $# > 0 )); then
  shift
fi

exec uv run --no-project --python "${OPENARM_DEPLOY_PYTHON:-${REPO_ROOT}/.venv-openarm-deploy/bin/python}" python \
  "${REPO_ROOT}/source/isaaclab_tasks/isaaclab_tasks/direct/openarm_motion_context/deploy/deploy_node.py" \
  --config "${CONFIG}" "$@"
