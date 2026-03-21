#!/usr/bin/env bash
set -euo pipefail

REPO="/inspire/hdd/project/agileapplication/zhangkaipeng-24043/wgx/myltx-v1"
RUN_SCRIPT="$REPO/qz/run_route35_s20_matched_eval_min_v20260320.sh"

DEFAULT_SSM_CKPT_ROOT="/inspire/qb-ilm/project/agileapplication/zhangkaipeng-24043/wgx/myltx-v1/outputs/self_forcing_longlive_phaseA_decay025_v3b_qb_shard16_cap48"
DEFAULT_SSM_STEP="00300"
DEFAULT_GEN_ROOT="/inspire/qb-ilm/project/agileapplication/zhangkaipeng-24043/wgx/myltx-v1/outputs/route35_s20_streaming_min_v3b_cap48"
DEFAULT_EVAL_ROOT="/inspire/qb-ilm/project/agileapplication/zhangkaipeng-24043/wgx/myltx-v1/outputs/route35_s20_matched_eval_min_v3b_cap48"

usage() {
  cat <<USAGE
Usage: $0 [--ssm-step <00300|300|step_00300>] [--ssm-checkpoint <path>] [--gen-root <dir>] [--eval-root <dir>] [extra args passed through]
USAGE
}

SSM_STEP_RAW="$DEFAULT_SSM_STEP"
SSM_CHECKPOINT=""
GEN_ROOT="$DEFAULT_GEN_ROOT"
EVAL_ROOT="$DEFAULT_EVAL_ROOT"
PASS_ARGS=()

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
    -h|--help)
      usage
      exit 0
      ;;
    *)
      PASS_ARGS+=("$1")
      shift
      ;;
  esac
done

if [[ -z "$SSM_CHECKPOINT" ]]; then
  step_num="${SSM_STEP_RAW#step_}"
  if ! [[ "$step_num" =~ ^[0-9]+$ ]]; then
    echo "Invalid --ssm-step: $SSM_STEP_RAW" >&2
    exit 2
  fi
  step_padded="$(printf "%05d" "$((10#$step_num))")"
  SSM_CHECKPOINT="$DEFAULT_SSM_CKPT_ROOT/ssm_weights_step_${step_padded}.pt"
fi

if [[ ! -f "$SSM_CHECKPOINT" ]]; then
  echo "SSM checkpoint not found: $SSM_CHECKPOINT" >&2
  exit 1
fi

echo "[run] s20 matched eval with checkpoint: $SSM_CHECKPOINT"
mkdir -p "$GEN_ROOT" "$EVAL_ROOT"

bash "$RUN_SCRIPT" \
  --ssm-checkpoint "$SSM_CHECKPOINT" \
  --gen-root "$GEN_ROOT" \
  --eval-root "$EVAL_ROOT" \
  "${PASS_ARGS[@]}"
