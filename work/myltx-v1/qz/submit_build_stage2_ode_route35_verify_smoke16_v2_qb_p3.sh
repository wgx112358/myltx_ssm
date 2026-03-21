#!/usr/bin/env bash
set -euo pipefail

ROOT="/inspire/hdd/project/agileapplication/zhangkaipeng-24043/wgx"
python3 "$ROOT/.claude/skills/qz/scripts/qz_cli.py" submit \
  --script "qz/build_stage2_ode_route35_verify_smoke16_v2_qb.sh" \
  --version "v2_qb" \
  --experiment "build-stage2-ode-route35-verify-smoke16-v2-qb" \
  --gpus 1 \
  --machine "h200-2" \
  --priority 3 \
  --conda-env "" \
  --workdir "$ROOT/myltx-v1"
