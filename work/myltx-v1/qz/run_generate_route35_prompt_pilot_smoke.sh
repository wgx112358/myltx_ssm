#!/usr/bin/env bash
set -euo pipefail

ROOT="/inspire/hdd/project/agileapplication/zhangkaipeng-24043/wgx"
OUT_ROOT="/inspire/qb-ilm/project/agileapplication/zhangkaipeng-24043/wgx"

python3 "$ROOT/myltx-v1/scripts/generate_route3_prompt_pilot.py" \
  --output-root "$OUT_ROOT" \
  --run-name "route35_prompt_pilot_smoke_v1" \
  --target-v2-groups 12 \
  --split-counts 8,2,2 \
  --seed 20260320
