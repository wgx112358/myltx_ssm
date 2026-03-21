# Route 3 Audit: OmniForcing + LingBot-World Transferability (2026-03-19)

## 1) What was found

### OmniForcing (`/inspire/hdd/project/agileapplication/zhangkaipeng-24043/wgx/omniforcing_v1`)
- The public repo currently contains project-page assets and README only; no released training/inference code yet.
- Mechanism-level claims in README that matter for Route 3:
  - causal streaming conversion from bidirectional teacher;
  - joint self-forcing for long-horizon error correction;
  - modality-independent rolling memory/cache behavior;
  - explicit handling of modality asymmetry.

### LingBot-World (`/inspire/hdd/project/agileapplication/zhangkaipeng-24043/wgx/lingbot_world_v1`)
- Code is available, but main public path is long-horizon single-pass generation (large `frame_num`) with control signals (camera/action), not Route-3-style online prompt-switch streaming over chunks.
- Useful transferable pattern at design level: explicit runtime control knobs for conditioning behavior (clear external controls), rather than hidden implicit behavior.

## 2) Mapping to current Route 3 pure SSM streaming

- Current weakness in Route 3: prompt-switch chunks carry full prior SSM memory unchanged, which risks semantic inertia after switches.
- This directly matches the missing "switch-aware state control" already identified in Route 3 planning.
- A bounded high-leverage optimization is to attenuate SSM carry state only on prompt-switch boundaries (soft memory reset), not a full architecture rewrite.

## 3) Changes implemented now (bounded)

### Switch-aware SSM state decay control
- Added `SSMState.scale_(factor)` helper in:
  - `packages/ltx-core/src/ltx_core/model/transformer/ssm_memory.py`

### Inference integration (Route 3 streaming)
- Added configurable switch-time memory attenuation in:
  - `scripts/streaming_inference.py`
- New config/CLI control:
  - `ssm_switch_state_decay` / `--ssm-switch-state-decay` (default `1.0`, no-op)
- Behavior:
  - if `--ssm-streaming` and chunk is a prompt-switch chunk, the carried `stream_state.ssm_state` is decayed by factor before generating that chunk;
  - metadata now records decay config and per-chunk application flag (`ssm_switch_state_decay_applied`).

### Training integration (switch-aware objective path)
- Added matching switch-time decay in training chunk loop:
  - `scripts/train_self_forcing.py`
- New config field in `SelfForcingConfig`:
  - `ssm_switch_state_decay` (default `1.0`, no-op)
- Metrics now include:
  - `switch_state_decay_chunks`.

### Regression tests added/updated
- `tests/test_streaming_inference.py`
  - new unit test for decay helper;
  - new end-to-end mocked streaming test to verify state passed into post-switch chunk is decayed and metadata flag is set.
- `tests/test_train_self_forcing_schedule.py`
  - updated metrics expectation for new field;
  - added test verifying training loop applies switch-state decay before switched chunk forward.

## 4) What remains speculative / not implemented

- OmniForcing-specific mechanisms like asymmetric block-causal masking and audio sink token cannot be audited at code level from the current public repo (code not released there yet).
- LingBot-World long-memory internals are not directly a chunked prompt-switch control implementation, so no direct code transplant was justified.
- No large architectural rewrite (KV cache redesign, external memory bank, RL control) was attempted by design.

