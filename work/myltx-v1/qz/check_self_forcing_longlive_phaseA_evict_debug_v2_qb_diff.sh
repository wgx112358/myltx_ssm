#!/usr/bin/env bash
set -euo pipefail

OUTPUT_DIR="${OUTPUT_DIR:-/inspire/qb-ilm/project/agileapplication/zhangkaipeng-24043/wgx/myltx-v1/outputs/self_forcing_longlive_phaseA_evict_debug_v2_qb}"
CKPT_A="${CKPT_A:-$OUTPUT_DIR/ssm_weights_step_00001.pt}"
CKPT_B="${CKPT_B:-$OUTPUT_DIR/ssm_weights_step_00002.pt}"

python3 - "$CKPT_A" "$CKPT_B" <<PY
import sys
from pathlib import Path

import torch


def unwrap(payload):
    if isinstance(payload, dict):
        for key in ("state_dict", "model_state_dict", "model"):
            nested = payload.get(key)
            if isinstance(nested, dict):
                return nested
    return payload

ckpt_a = Path(sys.argv[1])
ckpt_b = Path(sys.argv[2])
if not ckpt_a.exists():
    raise SystemExit(f"Missing checkpoint: {ckpt_a}")
if not ckpt_b.exists():
    raise SystemExit(f"Missing checkpoint: {ckpt_b}")

state_a = unwrap(torch.load(ckpt_a, map_location="cpu"))
state_b = unwrap(torch.load(ckpt_b, map_location="cpu"))

common_keys = sorted(set(state_a) & set(state_b))
diff_keys = 0
max_abs_diff = 0.0

for key in common_keys:
    tensor_a = state_a[key]
    tensor_b = state_b[key]
    if not (torch.is_tensor(tensor_a) and torch.is_tensor(tensor_b)):
        continue
    if tensor_a.shape != tensor_b.shape:
        diff_keys += 1
        max_abs_diff = float("inf")
        continue
    abs_diff = float((tensor_a.float() - tensor_b.float()).abs().max().item())
    if abs_diff > 0.0:
        diff_keys += 1
        if abs_diff > max_abs_diff:
            max_abs_diff = abs_diff

print(f"checkpoint_a={ckpt_a}")
print(f"checkpoint_b={ckpt_b}")
print(f"common_tensor_keys={len(common_keys)}")
print(f"diff_keys={diff_keys}")
print(f"max_abs_diff={max_abs_diff}")
PY
