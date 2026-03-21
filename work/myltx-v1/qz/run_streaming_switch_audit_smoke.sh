#!/usr/bin/env bash
set -euo pipefail

REPO="/inspire/hdd/project/agileapplication/zhangkaipeng-24043/wgx/myltx-v1"
BASE="/inspire/hdd/project/agileapplication/zhangkaipeng-24043/wgx/myltx"

STREAMING_DIR="${STREAMING_DIR:-$REPO/outputs/streaming_switch_ssm_smoke_v1}"
SUMMARY_JSON="${SUMMARY_JSON:-$STREAMING_DIR/streaming_score_summary.json}"

cd "$REPO"
source "$BASE/.venv/bin/activate"
export PYTHONPATH="packages/ltx-core/src:packages/ltx-trainer/src:packages/ltx-pipelines/src:scripts"

nice -n "${NICE_LEVEL:-10}" python scripts/streaming_switch_audit.py \
  --streaming-output-dir "$STREAMING_DIR" \
  --output-json "$SUMMARY_JSON" \
  --clip-model /inspire/qb-ilm/project/agileapplication/zhangkaipeng-24043/wgx/hf-cache/openai--clip-vit-base-patch32 \
  --frames-per-clip "${FRAMES_PER_CLIP:-4}" \
  --max-episodes "${MAX_EPISODES:-0}" \
  --metric-device "${METRIC_DEVICE:-cpu}"
