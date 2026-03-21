#!/usr/bin/env bash
set -euo pipefail

REPO="/inspire/hdd/project/agileapplication/zhangkaipeng-24043/wgx/myltx-v1"
BASE="/inspire/hdd/project/agileapplication/zhangkaipeng-24043/wgx/myltx"

cd "$REPO"
source "$BASE/.venv/bin/activate"
export PYTHONPATH="packages/ltx-core/src:packages/ltx-trainer/src:packages/ltx-pipelines/src:scripts"
read -r DEFAULT_HEIGHT DEFAULT_WIDTH <<< "$(python scripts/official_generation_defaults.py --preset small --format values)"
HEIGHT="${HEIGHT:-$DEFAULT_HEIGHT}"
WIDTH="${WIDTH:-$DEFAULT_WIDTH}"

python scripts/baseline_audit.py \
  --manifest ode/switch_episodes_smoke.jsonl \
  --checkpoint /inspire/hdd/project/agileapplication/zhangkaipeng-24043/wgx/myltx/model/ltx-2.3-22b-distilled.safetensors \
  --text-encoder-path /inspire/hdd/project/agileapplication/zhangkaipeng-24043/wgx/myltx/model/gemma \
  --output-dir outputs/baseline_audit_manifest16_v1 \
  --max-episodes 16 \
  --height "$HEIGHT" \
  --width "$WIDTH" \
  --num-frames 17 \
  --frame-rate 8 \
  --num-inference-steps 12 \
  --guidance-scale 1.0 \
  --stg-scale 0.0 \
  --seed 42 \
  --frames-per-clip 4 \
  --clip-model /inspire/qb-ilm/project/agileapplication/zhangkaipeng-24043/wgx/hf-cache/openai--clip-vit-base-patch32 \
  --device cuda \
  --prompt-cache-device cuda \
  --prompt-cache-load-in-8bit \
  --metric-device cpu \
  --skip-audio \
  --overwrite \
  "$@"
