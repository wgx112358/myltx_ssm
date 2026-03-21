#!/usr/bin/env bash
set -euo pipefail

REPO="/inspire/hdd/project/agileapplication/zhangkaipeng-24043/wgx/myltx-v1"
BASE="/inspire/hdd/project/agileapplication/zhangkaipeng-24043/wgx/myltx"
EPISODE_ID="${EPISODE_ID:-episode_0000}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/streaming_switch_smoke_v1}"
OUTPUT_PATH="${OUTPUT_DIR}/${EPISODE_ID}.mp4"

cd "$REPO"
source "$BASE/.venv/bin/activate"
export PYTHONPATH="packages/ltx-core/src:packages/ltx-trainer/src:packages/ltx-pipelines/src:scripts"
read -r DEFAULT_HEIGHT DEFAULT_WIDTH <<< "$(python scripts/official_generation_defaults.py --preset small --format values)"
HEIGHT="${HEIGHT:-$DEFAULT_HEIGHT}"
WIDTH="${WIDTH:-$DEFAULT_WIDTH}"

python scripts/streaming_inference.py \
  --mode switch \
  --manifest ode/switch_episodes_smoke.jsonl \
  --episode-id "$EPISODE_ID" \
  --output "$OUTPUT_PATH" \
  --model-checkpoint /inspire/hdd/project/agileapplication/zhangkaipeng-24043/wgx/myltx/model/ltx-2.3-22b-distilled.safetensors \
  --text-encoder-path /inspire/hdd/project/agileapplication/zhangkaipeng-24043/wgx/myltx/model/gemma \
  --height "$HEIGHT" \
  --width "$WIDTH" \
  --chunk-num-frames "${CHUNK_NUM_FRAMES:-17}" \
  --frame-rate 8 \
  --num-inference-steps "${INFER_STEPS:-12}" \
  --guidance-scale 1.0 \
  --chunks-per-segment "${CHUNKS_PER_SEGMENT:-1}" \
  --reference-window-chunks "${REFERENCE_WINDOW_CHUNKS:-1}" \
  --reference-max-frames "${REFERENCE_MAX_FRAMES:-17}" \
  --prompt-cache-device cuda \
  --prompt-cache-load-in-8bit \
  --device cuda \
  --overwrite \
  "$@"
