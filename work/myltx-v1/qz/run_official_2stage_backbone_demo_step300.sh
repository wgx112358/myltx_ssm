#!/usr/bin/env bash
set -euo pipefail

REPO="/inspire/hdd/project/agileapplication/zhangkaipeng-24043/wgx/myltx-v1"
QB_ROOT="/inspire/qb-ilm/project/agileapplication/zhangkaipeng-24043/wgx/myltx-v1"

export USE_SSM_STREAMING=1
export EPISODE_ID="${EPISODE_ID:-episode_0000}"
export SSM_CHECKPOINT="${SSM_CHECKPOINT:-$QB_ROOT/outputs/self_forcing_longlive_phaseA_decay025_v3b_qb_shard16_cap48/ssm_weights_step_00300.pt}"
export OUTPUT_DIR="${OUTPUT_DIR:-$QB_ROOT/outputs/official_2stage_persistent_ssm_backbone_demo_step300}"
export CHUNK_NUM_FRAMES="${CHUNK_NUM_FRAMES:-121}"
export FRAME_RATE="${FRAME_RATE:-24}"
export CHUNKS_PER_SEGMENT="${CHUNKS_PER_SEGMENT:-6}"
export WINDOW_BLOCKS="${WINDOW_BLOCKS:-2}"
export OFFICIAL_PRESET="${OFFICIAL_PRESET:-small}"
export SEED="${SEED:-42}"

cd "$REPO"
bash qz/run_streaming_backbone_smoke.sh "$@"
