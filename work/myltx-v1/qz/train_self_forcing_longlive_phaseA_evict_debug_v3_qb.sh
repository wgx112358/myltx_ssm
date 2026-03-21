#!/usr/bin/env bash
set -euo pipefail

REPO="/inspire/hdd/project/agileapplication/zhangkaipeng-24043/wgx/myltx-v1"
BASE="/inspire/hdd/project/agileapplication/zhangkaipeng-24043/wgx/myltx"
CONFIG="configs/self_forcing_longlive_phaseA_evict_debug_v3_qb.yaml"

cd "$REPO"
source "$BASE/.venv/bin/activate"
export PYTHONPATH="packages/ltx-core/src:packages/ltx-trainer/src:packages/ltx-pipelines/src:scripts"

EXTRA_ARGS=()
if [[ -n "${LR:-}" ]]; then
  EXTRA_ARGS+=(--lr "$LR")
fi
if [[ -n "${STEPS:-}" ]]; then
  EXTRA_ARGS+=(--steps "$STEPS")
fi

python scripts/train_self_forcing.py "$CONFIG" "${EXTRA_ARGS[@]}"
