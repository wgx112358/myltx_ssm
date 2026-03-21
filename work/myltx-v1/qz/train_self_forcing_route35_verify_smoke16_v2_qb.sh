#!/usr/bin/env bash
set -euo pipefail

REPO="/inspire/hdd/project/agileapplication/zhangkaipeng-24043/wgx/myltx-v1"
BASE="/inspire/hdd/project/agileapplication/zhangkaipeng-24043/wgx/myltx"
CONFIG="configs/self_forcing_route35_verify_smoke16_v2_qb.yaml"

cd "$REPO"
source "$BASE/.venv/bin/activate"
export PYTHONPATH="packages/ltx-core/src:packages/ltx-trainer/src:packages/ltx-pipelines/src:scripts"

EXTRA_ARGS=()
if [[ -n "${STEPS:-}" ]]; then
  EXTRA_ARGS+=(--steps "$STEPS")
fi
if [[ -n "${LR:-}" ]]; then
  EXTRA_ARGS+=(--lr "$LR")
fi

python scripts/train_self_forcing.py "$CONFIG" "${EXTRA_ARGS[@]}"
