#!/usr/bin/env bash
set -euo pipefail

REPO="/inspire/hdd/project/agileapplication/zhangkaipeng-24043/wgx/myltx-v1"
DEFAULT_CKPT_ROOT="/inspire/hdd/project/agileapplication/zhangkaipeng-24043/wgx/myltx-v1/outputs/self_forcing_longlive_phaseA_decay025_v1"
DEFAULT_OUT_ROOT="/inspire/qb-ilm/project/agileapplication/zhangkaipeng-24043/wgx/myltx-v1/outputs/streaming_switch_ssm_phasea_ckpt_progression"

usage() {
  echo "Usage: $0 --step <00050|50|step_00050> [--output-dir <dir>]"
}

STEP_RAW=""
OUTPUT_DIR=""
SSM_CHECKPOINT_ARG=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --step)
      STEP_RAW="${2:-}"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="${2:-}"
      shift 2
      ;;
    --ssm-checkpoint)
      SSM_CHECKPOINT_ARG="${2:-}"
      shift 2
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

if [[ -z "$STEP_RAW" ]]; then
  echo "--step is required" >&2
  usage >&2
  exit 2
fi

STEP_NUM="${STEP_RAW#step_}"
if ! [[ "$STEP_NUM" =~ ^[0-9]+$ ]]; then
  echo "Invalid step: $STEP_RAW" >&2
  exit 2
fi
STEP_PADDED="$(printf "%05d" "$((10#$STEP_NUM))")"

if [[ -n "$SSM_CHECKPOINT_ARG" ]]; then
  SSM_CHECKPOINT="$SSM_CHECKPOINT_ARG"
else
  SSM_CHECKPOINT="${CKPT_ROOT:-$DEFAULT_CKPT_ROOT}/ssm_weights_step_${STEP_PADDED}.pt"
fi

if [[ -z "$OUTPUT_DIR" ]]; then
  OUTPUT_DIR="${OUT_ROOT:-$DEFAULT_OUT_ROOT}/step_${STEP_PADDED}"
fi

if [[ ! -f "$SSM_CHECKPOINT" ]]; then
  echo "Checkpoint not found: $SSM_CHECKPOINT" >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR"

SSM_CHECKPOINT="$SSM_CHECKPOINT" OUTPUT_DIR="$OUTPUT_DIR" \
  bash "$REPO/qz/run_streaming_switch_ssm_phasea_ckpt_smoke.sh"
