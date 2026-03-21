#!/usr/bin/env bash
set -euo pipefail

REPO="/inspire/hdd/project/agileapplication/zhangkaipeng-24043/wgx/myltx-v1"
BASE="/inspire/hdd/project/agileapplication/zhangkaipeng-24043/wgx/myltx"
QB_ROOT="/inspire/qb-ilm/project/agileapplication/zhangkaipeng-24043/wgx/myltx-v1"

SRC_V1_MANIFEST="/inspire/qb-ilm/project/agileapplication/zhangkaipeng-24043/wgx/route35_prompt_pilot_verify_gen_v1/route3_v1_switch_episodes.jsonl"
MAX_EPISODES="${MAX_EPISODES:-16}"
LIMIT="${LIMIT:-16}"
SKIP_EXPORT="${SKIP_EXPORT:-0}"

OUT_DIR="$QB_ROOT/ode/data_distilled_stage2_ode_route35_verify_smoke16_v2_qb"
MANIFEST_OUT="$QB_ROOT/ode/switch_episodes_route35_verify_smoke16_v2_qb.jsonl"

mkdir -p "$OUT_DIR" "$(dirname "$MANIFEST_OUT")"

cd "$REPO"
source "$BASE/.venv/bin/activate"
export PYTHONPATH="packages/ltx-core/src:packages/ltx-trainer/src:packages/ltx-pipelines/src:scripts"
export SRC_V1_MANIFEST MANIFEST_OUT MAX_EPISODES

# Build a smoke V1 manifest from the verified prompt-pilot artifact.
# To avoid duplicate-key ambiguity in sample_id matching, rewrite sample_id as
# deterministic row index and preserve original sample_id in source_sample_id.
python - <<'PY'
from __future__ import annotations

import json
import os
from pathlib import Path

src = Path(os.environ["SRC_V1_MANIFEST"])
out = Path(os.environ["MANIFEST_OUT"])
max_episodes = int(os.environ.get("MAX_EPISODES", "16"))

if max_episodes <= 0:
    raise ValueError("MAX_EPISODES must be > 0")
if not src.exists():
    raise FileNotFoundError(f"source manifest not found: {src}")

rows = []
with src.open("r", encoding="utf-8") as handle:
    for line in handle:
        line = line.strip()
        if not line:
            continue
        payload = json.loads(line)
        if isinstance(payload, dict):
            rows.append(payload)
        if len(rows) >= max_episodes:
            break

if not rows:
    raise ValueError(f"no rows loaded from {src}")

with out.open("w", encoding="utf-8") as handle:
    for idx, row in enumerate(rows):
        row = dict(row)
        old_sample_id = str(row.get("sample_id", "")).strip()
        row["source_sample_id"] = old_sample_id if old_sample_id else None
        row["sample_id"] = f"{idx:05d}"

        # Keep schedule contract if explicit top-level fields are absent.
        segments = row.get("segments")
        if isinstance(segments, list) and len(segments) >= 2:
            first = segments[0]
            second = segments[1]
            if row.get("prefix_prompt") in (None, ""):
                row["prefix_prompt"] = first.get("prompt", "")
            if row.get("switch_prompt") in (None, ""):
                row["switch_prompt"] = second.get("prompt", "")
            if row.get("switch_time_seconds") in (None, ""):
                try:
                    row["switch_time_seconds"] = float(second.get("start_seconds", 0.0))
                except (TypeError, ValueError):
                    row["switch_time_seconds"] = 0.0

        handle.write(json.dumps(row, ensure_ascii=False) + "\n")

print(f"wrote {len(rows)} smoke V1 episodes to {out}")
PY

# Export smoke precomputed samples to qb-ilm using existing v2_qb conversion chain.
if [ "$SKIP_EXPORT" = "1" ]; then
  echo "SKIP_EXPORT=1 set, skip convert_ode_pt_to_precomputed"
  exit 0
fi

python ode/convert_ode_pt_to_precomputed.py \
  --input-dir "/inspire/hdd/project/agileapplication/zhangkaipeng-24043/wgx/myltx/ode/data_distilled" \
  --output-dir "$OUT_DIR" \
  --stage stage2 \
  --trajectory-step all_non_last \
  --export-mode ode_regression \
  --limit "$LIMIT" \
  --overwrite \
  --model-path "$BASE/model/ltx-2.3-22b-distilled.safetensors" \
  --text-encoder-path "$BASE/model/gemma" \
  --device cuda
