#!/usr/bin/env bash
set -euo pipefail

REPO="/inspire/hdd/project/agileapplication/zhangkaipeng-24043/wgx/myltx-v1"
BASE="/inspire/hdd/project/agileapplication/zhangkaipeng-24043/wgx/myltx"
QB_ROOT="/inspire/qb-ilm/project/agileapplication/zhangkaipeng-24043/wgx/myltx-v1"

MANIFEST="${MANIFEST:-$REPO/ode/switch_episodes_longlive_phaseA_256.jsonl}"
EPISODE_ID="${EPISODE_ID:-episode_0000}"

MODEL_CHECKPOINT="/inspire/hdd/project/agileapplication/zhangkaipeng-24043/wgx/myltx/model/ltx-2.3-22b-distilled.safetensors"
TEXT_ENCODER_PATH="/inspire/hdd/project/agileapplication/zhangkaipeng-24043/wgx/myltx/model/gemma"
CLIP_MODEL="/inspire/qb-ilm/project/agileapplication/zhangkaipeng-24043/wgx/hf-cache/openai--clip-vit-base-patch32"

NUM_FRAMES=17
FRAME_RATE=8
INFER_STEPS=12
read -r DEFAULT_HEIGHT DEFAULT_WIDTH <<< "$(python "$REPO/scripts/official_generation_defaults.py" --preset small --format values)"
HEIGHT="${HEIGHT:-$DEFAULT_HEIGHT}"
WIDTH="${WIDTH:-$DEFAULT_WIDTH}"

DEFAULT_SSM_STEP="00300"
SSM_CKPT_ROOT="/inspire/hdd/project/agileapplication/zhangkaipeng-24043/wgx/myltx-v1/outputs/self_forcing_longlive_phaseA_decay025_v1"
SSM_STEP_RAW="$DEFAULT_SSM_STEP"
SSM_CHECKPOINT=""

GEN_ROOT="${GEN_ROOT:-$QB_ROOT/outputs/route35_s20_streaming_min_v20260320}"
EVAL_ROOT="${EVAL_ROOT:-$QB_ROOT/outputs/route35_s20_matched_eval_min_v20260320}"

NON_SSM_RECACHE_DIR=""
NON_SSM_NO_RECACHE_DIR=""
PURE_SSM_DIR=""

SKIP_GENERATE=0
FORCE_GENERATE=0

usage() {
  cat <<USAGE
Usage: $0 [options]

Options:
  --ssm-step <00300|300|step_00300>
  --ssm-checkpoint <path>
  --gen-root <path>
  --eval-root <path>
  --non-ssm-recache-dir <path>
  --non-ssm-no-recache-dir <path>
  --pure-ssm-dir <path>
  --skip-generate
  --force-generate
  -h, --help
USAGE
}

normalize_step() {
  local raw="$1"
  raw="${raw#step_}"
  if ! [[ "$raw" =~ ^[0-9]+$ ]]; then
    return 1
  fi
  printf "%05d" "$((10#$raw))"
}

has_required_chunks() {
  local dir="$1"
  [[ -f "$dir/$EPISODE_ID/chunk_000.mp4" && -f "$dir/$EPISODE_ID/chunk_001.mp4" ]]
}

run_non_ssm_recache() {
  local out_dir="$1"
  EPISODE_ID="$EPISODE_ID" OUTPUT_DIR="$out_dir" \
    bash "$REPO/qz/run_streaming_switch_smoke.sh" \
    --manifest "$MANIFEST" \
    --chunk-num-frames "$NUM_FRAMES" \
    --frame-rate "$FRAME_RATE" \
    --num-inference-steps "$INFER_STEPS"
}

run_non_ssm_no_recache() {
  local out_dir="$1"
  EPISODE_ID="$EPISODE_ID" OUTPUT_DIR="$out_dir" \
    bash "$REPO/qz/run_streaming_switch_smoke.sh" \
    --manifest "$MANIFEST" \
    --chunk-num-frames "$NUM_FRAMES" \
    --frame-rate "$FRAME_RATE" \
    --num-inference-steps "$INFER_STEPS" \
    --disable-switch-recache
}

run_pure_ssm_ckpt() {
  local out_dir="$1"
  EPISODE_ID="$EPISODE_ID" MANIFEST_PATH="$MANIFEST" OUTPUT_DIR="$out_dir" SSM_CHECKPOINT="$SSM_CHECKPOINT" \
    bash "$REPO/qz/run_streaming_switch_ssm_phasea_ckpt_smoke.sh" \
    --chunk-num-frames "$NUM_FRAMES" \
    --frame-rate "$FRAME_RATE" \
    --num-inference-steps "$INFER_STEPS"
}

stage_chunks_for_baseline() {
  local system_name="$1"
  local source_dir="$2"
  local score_dir="$EVAL_ROOT/$system_name"
  local staged_episode_dir="$score_dir/$EPISODE_ID"

  mkdir -p "$staged_episode_dir"
  rm -f "$staged_episode_dir"/segment_*.mp4

  local found=0
  local chunk_path
  shopt -s nullglob
  for chunk_path in "$source_dir/$EPISODE_ID"/chunk_*.mp4; do
    local filename chunk_idx seg_idx seg_name
    filename="$(basename "$chunk_path")"
    chunk_idx="${filename#chunk_}"
    chunk_idx="${chunk_idx%.mp4}"
    if ! [[ "$chunk_idx" =~ ^[0-9]+$ ]]; then
      continue
    fi
    seg_idx="$((10#$chunk_idx))"
    printf -v seg_name "segment_%02d.mp4" "$seg_idx"
    cp -f "$chunk_path" "$staged_episode_dir/$seg_name"
    found=1
  done
  shopt -u nullglob

  if [[ "$found" -ne 1 ]]; then
    echo "No chunk_*.mp4 found for $system_name under: $source_dir/$EPISODE_ID" >&2
    exit 1
  fi

  if [[ ! -f "$staged_episode_dir/segment_00.mp4" || ! -f "$staged_episode_dir/segment_01.mp4" ]]; then
    echo "Staging failed for $system_name: segment_00.mp4 and segment_01.mp4 are required" >&2
    exit 1
  fi
}

run_baseline_score_only() {
  local system_name="$1"
  local score_dir="$EVAL_ROOT/$system_name"

  python scripts/baseline_audit.py \
    --manifest "$MANIFEST" \
    --checkpoint "$MODEL_CHECKPOINT" \
    --text-encoder-path "$TEXT_ENCODER_PATH" \
    --output-dir "$score_dir" \
    --max-episodes 1 \
    --height "$HEIGHT" \
    --width "$WIDTH" \
    --num-frames "$NUM_FRAMES" \
    --frame-rate "$FRAME_RATE" \
    --num-inference-steps "$INFER_STEPS" \
    --guidance-scale 1.0 \
    --stg-scale 0.0 \
    --seed 42 \
    --frames-per-clip 4 \
    --clip-model "$CLIP_MODEL" \
    --device cuda \
    --prompt-cache-device cuda \
    --prompt-cache-load-in-8bit \
    --metric-device cpu \
    --score-only
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --ssm-step)
      SSM_STEP_RAW="${2:-}"
      shift 2
      ;;
    --ssm-checkpoint)
      SSM_CHECKPOINT="${2:-}"
      shift 2
      ;;
    --gen-root)
      GEN_ROOT="${2:-}"
      shift 2
      ;;
    --eval-root)
      EVAL_ROOT="${2:-}"
      shift 2
      ;;
    --non-ssm-recache-dir)
      NON_SSM_RECACHE_DIR="${2:-}"
      shift 2
      ;;
    --non-ssm-no-recache-dir)
      NON_SSM_NO_RECACHE_DIR="${2:-}"
      shift 2
      ;;
    --pure-ssm-dir)
      PURE_SSM_DIR="${2:-}"
      shift 2
      ;;
    --skip-generate)
      SKIP_GENERATE=1
      shift
      ;;
    --force-generate)
      FORCE_GENERATE=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown arg: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ ! -f "$MANIFEST" ]]; then
  echo "Manifest not found: $MANIFEST" >&2
  exit 1
fi

if [[ -z "$SSM_CHECKPOINT" ]]; then
  if ! STEP_PADDED="$(normalize_step "$SSM_STEP_RAW")"; then
    echo "Invalid --ssm-step: $SSM_STEP_RAW" >&2
    exit 2
  fi
  SSM_CHECKPOINT="$SSM_CKPT_ROOT/ssm_weights_step_${STEP_PADDED}.pt"
else
  STEP_PADDED="custom"
fi

if [[ ! -f "$SSM_CHECKPOINT" ]]; then
  echo "SSM checkpoint not found: $SSM_CHECKPOINT" >&2
  exit 1
fi

if [[ -z "$NON_SSM_RECACHE_DIR" ]]; then
  NON_SSM_RECACHE_DIR="$GEN_ROOT/non_ssm_recache"
fi
if [[ -z "$NON_SSM_NO_RECACHE_DIR" ]]; then
  NON_SSM_NO_RECACHE_DIR="$GEN_ROOT/non_ssm_no_recache"
fi
if [[ -z "$PURE_SSM_DIR" ]]; then
  PURE_SSM_DIR="$GEN_ROOT/pure_ssm_checkpoint_step_${STEP_PADDED}"
fi

mkdir -p "$GEN_ROOT" "$EVAL_ROOT"

cd "$REPO"
source "$BASE/.venv/bin/activate"
export PYTHONPATH="packages/ltx-core/src:packages/ltx-trainer/src:packages/ltx-pipelines/src:scripts"

if has_required_chunks "$NON_SSM_RECACHE_DIR" && [[ "$FORCE_GENERATE" -ne 1 ]]; then
  echo "[reuse] non_ssm_recache: $NON_SSM_RECACHE_DIR"
else
  if [[ "$SKIP_GENERATE" -eq 1 ]]; then
    echo "Missing non_ssm_recache chunks and --skip-generate is set: $NON_SSM_RECACHE_DIR" >&2
    exit 1
  fi
  mkdir -p "$NON_SSM_RECACHE_DIR"
  echo "[generate] non_ssm_recache -> $NON_SSM_RECACHE_DIR"
  run_non_ssm_recache "$NON_SSM_RECACHE_DIR"
fi

if has_required_chunks "$NON_SSM_NO_RECACHE_DIR" && [[ "$FORCE_GENERATE" -ne 1 ]]; then
  echo "[reuse] non_ssm_no_recache: $NON_SSM_NO_RECACHE_DIR"
else
  if [[ "$SKIP_GENERATE" -eq 1 ]]; then
    echo "Missing non_ssm_no_recache chunks and --skip-generate is set: $NON_SSM_NO_RECACHE_DIR" >&2
    exit 1
  fi
  mkdir -p "$NON_SSM_NO_RECACHE_DIR"
  echo "[generate] non_ssm_no_recache -> $NON_SSM_NO_RECACHE_DIR"
  run_non_ssm_no_recache "$NON_SSM_NO_RECACHE_DIR"
fi

if has_required_chunks "$PURE_SSM_DIR" && [[ "$FORCE_GENERATE" -ne 1 ]]; then
  echo "[reuse] pure_ssm_checkpoint: $PURE_SSM_DIR"
else
  if [[ "$SKIP_GENERATE" -eq 1 ]]; then
    echo "Missing pure_ssm_checkpoint chunks and --skip-generate is set: $PURE_SSM_DIR" >&2
    exit 1
  fi
  mkdir -p "$PURE_SSM_DIR"
  echo "[generate] pure_ssm_checkpoint -> $PURE_SSM_DIR"
  run_pure_ssm_ckpt "$PURE_SSM_DIR"
fi

for system_name in non_ssm_recache non_ssm_no_recache pure_ssm_checkpoint; do
  case "$system_name" in
    non_ssm_recache)
      source_dir="$NON_SSM_RECACHE_DIR"
      ;;
    non_ssm_no_recache)
      source_dir="$NON_SSM_NO_RECACHE_DIR"
      ;;
    pure_ssm_checkpoint)
      source_dir="$PURE_SSM_DIR"
      ;;
    *)
      echo "Unknown system: $system_name" >&2
      exit 2
      ;;
  esac

  echo "[stage] $system_name <- $source_dir"
  stage_chunks_for_baseline "$system_name" "$source_dir"

  echo "[score] $system_name"
  run_baseline_score_only "$system_name"
done

SUMMARY_JSON="$EVAL_ROOT/route35_s20_matched_eval_min_summary.json"
python - "$SUMMARY_JSON" "$EVAL_ROOT" "$MANIFEST" "$EPISODE_ID" "$NUM_FRAMES" "$FRAME_RATE" "$INFER_STEPS" "$SSM_CHECKPOINT" "$STEP_PADDED" <<PY
import json
import sys
from pathlib import Path

summary_path = Path(sys.argv[1])
eval_root = Path(sys.argv[2])
manifest = sys.argv[3]
episode_id = sys.argv[4]
num_frames = int(sys.argv[5])
frame_rate = float(sys.argv[6])
infer_steps = int(sys.argv[7])
ssm_checkpoint = sys.argv[8]
ssm_step = sys.argv[9]

systems = {
    "non_ssm_recache": eval_root / "non_ssm_recache" / "baseline_summary.json",
    "non_ssm_no_recache": eval_root / "non_ssm_no_recache" / "baseline_summary.json",
    "pure_ssm_checkpoint": eval_root / "pure_ssm_checkpoint" / "baseline_summary.json",
}

payload = {
    "manifest": manifest,
    "episode_id": episode_id,
    "decode_config": {
        "num_frames": num_frames,
        "frame_rate": frame_rate,
        "num_inference_steps": infer_steps,
    },
    "ssm": {
        "checkpoint": ssm_checkpoint,
        "step": ssm_step,
    },
    "systems": {},
}

for name, path in systems.items():
    item = {
        "baseline_summary_json": str(path),
        "exists": path.exists(),
    }
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
        item["aggregate"] = data.get("aggregate", {})
    payload["systems"][name] = item

summary_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
PY

echo "[done] route35_s20 matched eval summary: $SUMMARY_JSON"
echo "[done] per-system baseline summary roots: $EVAL_ROOT/{non_ssm_recache,non_ssm_no_recache,pure_ssm_checkpoint}/baseline_summary.json"
