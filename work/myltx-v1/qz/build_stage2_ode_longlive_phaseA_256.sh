#!/usr/bin/env bash
set -euo pipefail

REPO="/inspire/hdd/project/agileapplication/zhangkaipeng-24043/wgx/myltx-v1"
BASE="/inspire/hdd/project/agileapplication/zhangkaipeng-24043/wgx/myltx"

OUT_DIR="/inspire/hdd/project/agileapplication/zhangkaipeng-24043/wgx/myltx-v1/ode/data_distilled_stage2_ode_phaseA_256"
MANIFEST_OUT="/inspire/hdd/project/agileapplication/zhangkaipeng-24043/wgx/myltx-v1/ode/switch_episodes_longlive_phaseA_256.jsonl"

cd "$REPO"
source "$BASE/.venv/bin/activate"
export PYTHONPATH="packages/ltx-core/src:packages/ltx-trainer/src:packages/ltx-pipelines/src:scripts"

# Contract: exactly 256 episodes in sorted sample order (00000..00255).
python scripts/build_switch_manifest.py \
  --mode longlive_two_segment \
  --prompt-source "$BASE/datagen/ltx_prompts_12000.csv" \
  --prompt-column text_prompt \
  --output "$MANIFEST_OUT" \
  --max-episodes 256 \
  --episode-duration-seconds 15.0 \
  --switch-choices-seconds "5.0,10.0" \
  --seed 42

# Contract: exactly 256 sorted raw samples exported to precomputed format.
python ode/convert_ode_pt_to_precomputed.py \
  --input-dir "/inspire/hdd/project/agileapplication/zhangkaipeng-24043/wgx/myltx/ode/data_distilled" \
  --output-dir "$OUT_DIR" \
  --stage stage2 \
  --trajectory-step all_non_last \
  --export-mode ode_regression \
  --limit 256 \
  --overwrite \
  --model-path "$BASE/model/ltx-2.3-22b-distilled.safetensors" \
  --text-encoder-path "$BASE/model/gemma" \
  --device cuda
