#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../../.." && pwd)"
TASK="${1:-Isaac-OpenArm-Re-Lift-Paper-v0}"
MODE="${2:-motion_context}"

exec "${REPO_ROOT}/isaaclab.sh" -p \
  "${REPO_ROOT}/scripts/reinforcement_learning/skrl/train_openarm_bimanual_direct.py" \
  --task "${TASK}" \
  --intent_variant share_intent \
  --experiment_tag "${MODE}" \
  --num_envs 128 \
  --headless \
  "env.communication_mode=${MODE}"
