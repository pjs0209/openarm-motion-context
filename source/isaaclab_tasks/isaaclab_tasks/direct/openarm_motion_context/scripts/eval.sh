#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: $0 TASK CHECKPOINT [COMMUNICATION_MODE] [TRACE_DIR]" >&2
  exit 2
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../../.." && pwd)"
TASK="$1"
CHECKPOINT="$2"
MODE="${3:-motion_context}"
TRACE_DIR="${4:-logs/motion_context_traces/${MODE}}"

exec "${REPO_ROOT}/isaaclab.sh" -p \
  "${REPO_ROOT}/scripts/reinforcement_learning/skrl/play_openarm_bimanual_direct.py" \
  --task "${TASK}" \
  --num_envs 1 \
  --num_steps 9000 \
  --num_eval_episodes 10 \
  --deterministic_eval \
  --save_motion_context_trace \
  --motion_context_trace_dir "${TRACE_DIR}" \
  --checkpoint "${CHECKPOINT}" \
  "env.communication_mode=${MODE}"
