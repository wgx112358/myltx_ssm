#!/usr/bin/env bash
set -euo pipefail

ROOT="/inspire/hdd/project/agileapplication/zhangkaipeng-24043/wgx"
python3 "$ROOT/.claude/skills/qz/scripts/qz_cli.py" submit \
  --script "qz/run_streaming_switch_ssm_phasea_ckpt_progression_s19_v3b_cap48.sh" \
  --version "v3b_cap48" \
  --experiment "streaming-switch-ssm-phasea-ckpt-progression-s19-v3b-cap48" \
  --gpus 1 \
  --machine "h200-2" \
  --priority 3 \
  --conda-env "" \
  --workdir "$ROOT/myltx-v1"
