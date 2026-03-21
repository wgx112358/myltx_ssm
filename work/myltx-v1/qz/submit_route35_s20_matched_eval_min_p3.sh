#!/usr/bin/env bash
set -euo pipefail

ROOT="/inspire/hdd/project/agileapplication/zhangkaipeng-24043/wgx"
SCRIPT_ARGS="${*:-}"

CMD=(
  python3 "$ROOT/.claude/skills/qz/scripts/qz_cli.py" submit
  --script "qz/run_route35_s20_matched_eval_min_v20260320.sh"
  --version "v20260320"
  --experiment "route35-s20-matched-eval-min"
  --gpus 1
  --machine "h200-2"
  --priority 3
  --conda-env ""
  --workdir "$ROOT/myltx-v1"
)

if [[ -n "$SCRIPT_ARGS" ]]; then
  CMD+=(--script-args "$SCRIPT_ARGS")
fi

"${CMD[@]}"
