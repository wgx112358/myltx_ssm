#!/usr/bin/env bash
set -euo pipefail

REPO="/inspire/hdd/project/agileapplication/zhangkaipeng-24043/wgx/myltx-v1"
RUN_ONE_SCRIPT="$REPO/qz/run_streaming_switch_ssm_phasea_ckpt_s19_scriptargs_v20260320.sh"
DEFAULT_CKPT_ROOT="/inspire/qb-ilm/project/agileapplication/zhangkaipeng-24043/wgx/myltx-v1/outputs/self_forcing_longlive_phaseA_decay025_v3b_qb_shard16_cap48"
DEFAULT_OUT_ROOT="/inspire/qb-ilm/project/agileapplication/zhangkaipeng-24043/wgx/myltx-v1/outputs/streaming_switch_ssm_phasea_ckpt_progression_s19_v3b_cap48"
DEFAULT_STEPS="00050,00100,00200,00300"

usage() {
  cat <<USAGE
Usage: $0 [--steps <00050,00100,00200,00300>] [--ckpt-root <dir>] [--out-root <dir>]
USAGE
}

STEPS_CSV="$DEFAULT_STEPS"
CKPT_ROOT="$DEFAULT_CKPT_ROOT"
OUT_ROOT="$DEFAULT_OUT_ROOT"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --steps)
      STEPS_CSV="${2:-}"
      shift 2
      ;;
    --ckpt-root)
      CKPT_ROOT="${2:-}"
      shift 2
      ;;
    --out-root)
      OUT_ROOT="${2:-}"
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

if [[ -z "$STEPS_CSV" ]]; then
  echo "--steps must not be empty" >&2
  exit 2
fi

IFS=, read -r -a STEPS <<< "$STEPS_CSV"
mkdir -p "$OUT_ROOT"

for step_raw in "${STEPS[@]}"; do
  step_trimmed="${step_raw// /}"
  step_num="${step_trimmed#step_}"
  if ! [[ "$step_num" =~ ^[0-9]+$ ]]; then
    echo "Invalid step: $step_trimmed" >&2
    exit 2
  fi
  step_padded="$(printf "%05d" "$((10#$step_num))")"
  ssm_checkpoint="$CKPT_ROOT/ssm_weights_step_${step_padded}.pt"
  output_dir="$OUT_ROOT/step_${step_padded}"

  if [[ ! -f "$ssm_checkpoint" ]]; then
    echo "Checkpoint not found for step ${step_padded}: $ssm_checkpoint" >&2
    exit 1
  fi

  echo "[run] step=${step_padded} ckpt=${ssm_checkpoint} out=${output_dir}"
  bash "$RUN_ONE_SCRIPT" --ssm-checkpoint "$ssm_checkpoint" --output-dir "$output_dir"
done

echo "[done] progression outputs under: $OUT_ROOT"
