#!/usr/bin/env bash
set -euo pipefail

ROOT="/inspire/hdd/project/agileapplication/zhangkaipeng-24043/wgx"
python3 "$ROOT/.claude/skills/qz/scripts/qz_cli.py" submit \
  --script "qz/train_self_forcing_route35_verify_smoke16_v2_qb.sh" \
  --version "route35-verify-smoke16-v2-qb" \
  --experiment "self-forcing-route35-verify-smoke16-v2-qb" \
  --gpus 1 \
  --machine "h200-2" \
  --priority 3 \
  --conda-env "" \
  --workdir "$ROOT/myltx-v1"
