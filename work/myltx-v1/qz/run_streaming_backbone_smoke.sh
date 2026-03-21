#!/usr/bin/env bash
set -euo pipefail

REPO="/inspire/hdd/project/agileapplication/zhangkaipeng-24043/wgx/myltx-v1"
BASE="/inspire/hdd/project/agileapplication/zhangkaipeng-24043/wgx/myltx"
QB_ROOT="/inspire/qb-ilm/project/agileapplication/zhangkaipeng-24043/wgx/myltx-v1"
DISTILLED_CHECKPOINT="/inspire/hdd/project/agileapplication/zhangkaipeng-24043/wgx/myltx/model/ltx-2.3-22b-distilled.safetensors"
GEMMA_ROOT="/inspire/hdd/project/agileapplication/zhangkaipeng-24043/wgx/myltx/model/gemma"
SPATIAL_UPSAMPLER="/inspire/hdd/project/agileapplication/zhangkaipeng-24043/wgx/myltx/model/ltx-2.3-spatial-upscaler-x2-1.0.safetensors"

MANIFEST_PATH="${MANIFEST_PATH:-$REPO/ode/streaming_backbone_smoke.jsonl}"
EPISODE_ID="${EPISODE_ID:-episode_0000}"
USE_SSM_STREAMING="${USE_SSM_STREAMING:-1}"
SSM_CHECKPOINT="${SSM_CHECKPOINT:-/inspire/qb-ilm/project/agileapplication/zhangkaipeng-24043/wgx/myltx-v1/outputs/self_forcing_longlive_phaseA_decay025_v3b_qb_shard16_cap48/ssm_weights_step_00200.pt}"
CHUNK_NUM_FRAMES="${CHUNK_NUM_FRAMES:-121}"
FRAME_RATE="${FRAME_RATE:-24}"
CHUNKS_PER_SEGMENT="${CHUNKS_PER_SEGMENT:-6}"
WINDOW_BLOCKS="${WINDOW_BLOCKS:-2}"
OFFICIAL_PRESET="${OFFICIAL_PRESET:-small}"

if [[ -z "${REFERENCE_WINDOW_CHUNKS:-}" ]]; then
  if [[ "$USE_SSM_STREAMING" == "1" ]]; then
    REFERENCE_WINDOW_CHUNKS=0
  else
    REFERENCE_WINDOW_CHUNKS=1
  fi
fi
REFERENCE_MAX_FRAMES="${REFERENCE_MAX_FRAMES:-$CHUNK_NUM_FRAMES}"

if [[ -z "${OUTPUT_DIR:-}" ]]; then
  if [[ "$USE_SSM_STREAMING" == "1" ]]; then
    OUTPUT_DIR="$QB_ROOT/outputs/official_2stage_persistent_ssm_backbone_smoke_preview"
  elif [[ "$REFERENCE_WINDOW_CHUNKS" == "0" ]]; then
    OUTPUT_DIR="$QB_ROOT/outputs/streaming_backbone_no_memory_smoke_v2"
  else
    OUTPUT_DIR="$QB_ROOT/outputs/streaming_backbone_short_context_smoke_v2"
  fi
fi
OUTPUT_PATH="${OUTPUT_DIR}/${EPISODE_ID}.mp4"

cd "$REPO"
source "$BASE/.venv/bin/activate"
export PYTHONPATH="packages/ltx-core/src:packages/ltx-trainer/src:packages/ltx-pipelines/src:scripts"
read -r DEFAULT_HEIGHT DEFAULT_WIDTH <<< "$(python scripts/official_generation_defaults.py --preset "$OFFICIAL_PRESET" --format values)"
HEIGHT="${HEIGHT:-$DEFAULT_HEIGHT}"
WIDTH="${WIDTH:-$DEFAULT_WIDTH}"

echo "[streaming_backbone_smoke] manifest=$MANIFEST_PATH episode=$EPISODE_ID use_ssm=$USE_SSM_STREAMING height=$HEIGHT width=$WIDTH chunk_num_frames=$CHUNK_NUM_FRAMES frame_rate=$FRAME_RATE chunks_per_segment=$CHUNKS_PER_SEGMENT window_blocks=$WINDOW_BLOCKS reference_window_chunks=$REFERENCE_WINDOW_CHUNKS reference_max_frames=$REFERENCE_MAX_FRAMES output_dir=$OUTPUT_DIR"

if [[ "$USE_SSM_STREAMING" == "1" ]]; then
  python scripts/official_2stage_backbone_orchestrator.py \
    --manifest "$MANIFEST_PATH" \
    --episode-id "$EPISODE_ID" \
    --output "$OUTPUT_PATH" \
    --distilled-checkpoint-path "$DISTILLED_CHECKPOINT" \
    --gemma-root "$GEMMA_ROOT" \
    --spatial-upsampler-path "$SPATIAL_UPSAMPLER" \
    --ssm-checkpoint "$SSM_CHECKPOINT" \
    --preset "$OFFICIAL_PRESET" \
    --chunk-num-frames "$CHUNK_NUM_FRAMES" \
    --frame-rate "$FRAME_RATE" \
    --chunks-per-segment "$CHUNKS_PER_SEGMENT" \
    --window-blocks "$WINDOW_BLOCKS" \
    --seed "${SEED:-42}" \
    --overwrite \
    "$@"
else
  python scripts/streaming_inference.py \
    --mode switch \
    --manifest "$MANIFEST_PATH" \
    --episode-id "$EPISODE_ID" \
    --output "$OUTPUT_PATH" \
    --model-checkpoint "$DISTILLED_CHECKPOINT" \
    --text-encoder-path "$GEMMA_ROOT" \
    --height "$HEIGHT" \
    --width "$WIDTH" \
    --chunk-num-frames "$CHUNK_NUM_FRAMES" \
    --frame-rate "$FRAME_RATE" \
    --num-inference-steps "${INFER_STEPS:-12}" \
    --guidance-scale 1.0 \
    --window-blocks "$WINDOW_BLOCKS" \
    --chunks-per-segment "$CHUNKS_PER_SEGMENT" \
    --reference-window-chunks "$REFERENCE_WINDOW_CHUNKS" \
    --reference-max-frames "$REFERENCE_MAX_FRAMES" \
    --prompt-cache-device cuda \
    --prompt-cache-load-in-8bit \
    --device cuda \
    --overwrite \
    "$@"
fi
