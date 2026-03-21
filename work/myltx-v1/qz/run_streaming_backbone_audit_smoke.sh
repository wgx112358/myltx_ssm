#!/usr/bin/env bash
set -euo pipefail

REPO="/inspire/hdd/project/agileapplication/zhangkaipeng-24043/wgx/myltx-v1"
BASE="/inspire/hdd/project/agileapplication/zhangkaipeng-24043/wgx/myltx"
QB_ROOT="/inspire/qb-ilm/project/agileapplication/zhangkaipeng-24043/wgx/myltx-v1"

STREAMING_DIR="${STREAMING_DIR:-$QB_ROOT/outputs/official_2stage_persistent_ssm_backbone_smoke_preview}"
SUMMARY_JSON="${SUMMARY_JSON:-$STREAMING_DIR/streaming_backbone_summary.json}"

cd "$REPO"
source "$BASE/.venv/bin/activate"
export PYTHONPATH="packages/ltx-core/src:packages/ltx-trainer/src:packages/ltx-pipelines/src:scripts"

nice -n "${NICE_LEVEL:-10}" python scripts/streaming_backbone_audit.py \
  --streaming-output-dir "$STREAMING_DIR" \
  --output-json "$SUMMARY_JSON" \
  --clip-model /inspire/qb-ilm/project/agileapplication/zhangkaipeng-24043/wgx/hf-cache/openai--clip-vit-base-patch32 \
  --frames-per-clip "${FRAMES_PER_CLIP:-4}" \
  --max-episodes "${MAX_EPISODES:-0}" \
  --metric-device "${METRIC_DEVICE:-cpu}" \
  --audio-metric-device "${AUDIO_METRIC_DEVICE:-cpu}"
