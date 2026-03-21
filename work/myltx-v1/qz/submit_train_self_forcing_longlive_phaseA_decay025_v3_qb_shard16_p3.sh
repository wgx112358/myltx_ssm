#!/usr/bin/env bash
set -euo pipefail

ROOT="/inspire/hdd/project/agileapplication/zhangkaipeng-24043/wgx"
python3 "$ROOT/.claude/skills/qz/scripts/qz_cli.py" submit \
  --script "qz/train_self_forcing_longlive_phaseA_decay025_v3_qb_shard16.sh" \
  --version "decay025_v3_qb_shard16" \
  --experiment "self-forcing-longlive-phaseA-decay025-v3-qb-shard16" \
  --gpus 1 \
  --machine "h200-2" \
  --priority 3 \
  --conda-env "" \
  --workdir "$ROOT/myltx-v1"
