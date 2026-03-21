#!/usr/bin/env bash
set -euo pipefail

ROOT="/inspire/hdd/project/agileapplication/zhangkaipeng-24043/wgx"
python3 "$ROOT/.claude/skills/qz/scripts/qz_cli.py" submit \
  --script "qz/train_self_forcing_longlive_smoke.sh" \
  --version "v6" \
  --experiment "self-forcing-longlive-smoke" \
  --gpus 1 \
  --machine "h200-2" \
  --priority 3 \
  --conda-env "" \
  --workdir "$ROOT/myltx-v1"
