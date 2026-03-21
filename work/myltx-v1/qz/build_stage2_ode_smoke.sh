#!/usr/bin/env bash
set -euo pipefail

REPO="/inspire/hdd/project/agileapplication/zhangkaipeng-24043/wgx/myltx-v1"
BASE="/inspire/hdd/project/agileapplication/zhangkaipeng-24043/wgx/myltx"

cd "$REPO"
source "$BASE/.venv/bin/activate"
export PYTHONPATH="packages/ltx-core/src:packages/ltx-trainer/src:packages/ltx-pipelines/src:scripts"

python scripts/build_switch_manifest.py \
  --input-csv "$BASE/datagen/ltx_prompts_100_categorized.csv" \
  --output "$REPO/ode/switch_episodes_smoke.jsonl" \
  --max-episodes "${MAX_EPISODES:-16}" \
  --segments-per-episode 3 \
  --seed 42

python ode/convert_ode_pt_to_precomputed.py \
  --input-dir "$BASE/ode/data_distilled" \
  --output-dir "$REPO/ode/data_distilled_stage2_ode_smoke" \
  --stage stage2 \
  --trajectory-step all_non_last \
  --export-mode ode_regression \
  --limit "${LIMIT:-8}" \
  --overwrite \
  --model-path "$BASE/model/ltx-2.3-22b-distilled.safetensors" \
  --text-encoder-path "$BASE/model/gemma" \
  --device cuda
