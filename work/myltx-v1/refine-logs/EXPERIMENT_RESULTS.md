# Initial Experiment Results

**Date**: 2026-03-20  
**Plan**: `EXPERIMENT_PLAN.md`

## M0: Stage A Backbone Scaffold

- Implemented a dedicated no-switch audit at `scripts/streaming_backbone_audit.py`.
- Added a Stage A no-switch smoke manifest at `ode/streaming_backbone_smoke.jsonl`.
- Added no-switch smoke wrappers:
  - `qz/train_self_forcing_longlive_backbone_smoke.sh`
  - `qz/run_streaming_backbone_smoke.sh`
  - `qz/run_streaming_backbone_audit_smoke.sh`
- Added submit wrappers:
  - `qz/submit_train_self_forcing_longlive_backbone_smoke_p3.sh`
  - `qz/submit_streaming_backbone_smoke_p3.sh`
- Remote verification passed:
  - `PYTHONDONTWRITEBYTECODE=1 python -B -m pytest -q --cache-clear tests/test_streaming_backbone_audit.py`
  - Result: `2 passed`

## Active Run

- `job-1e218dd2-8c24-463f-b327-374b3e5066c7`
  - Name: `wgx-train-h200-1g-p3-v2-streaming-backbone-persistent-ssm-smoke`
  - Status at 2026-03-20 19:44 (Asia/Shanghai): `job_queuing`
  - Purpose: Corrected Stage A no-switch persistent-SSM smoke generation with 5-second chunks

## Invalidated Run

- `streaming_backbone_smoke_v1` is not a valid Stage A backbone result.
- Original job: `job-c7f3f2da-b19a-4a5e-a1ab-36a67eae2023`
- Root cause:
  - it ran with `ssm_streaming_enabled=false`
  - it used `reference_video` instead of persistent state carry
  - it used `17` frames at `8` fps per chunk, so each chunk was only about `2.1s`
- Observed failure mode:
  - same-prompt near-repaint across 6 chunks rather than true continuation

## Corrective Action

- `qz/run_streaming_backbone_smoke.sh` V2 now defaults to:
  - `USE_SSM_STREAMING=1`
  - `CHUNK_NUM_FRAMES=41`
  - `FRAME_RATE=8`
  - output root `streaming_backbone_persistent_ssm_smoke_v2`
- The newly submitted V2 smoke should be treated as the first valid Stage A backbone run.

## Summary

- Stage A bridge code is implemented and remotely validated.
- The first smoke artifact exposed a real configuration bug rather than a method result.
- The corrected persistent-SSM V2 smoke is the next meaningful execution target.
- Prompt-switch evaluation remains intentionally deferred to Stage B.

## Next Step

- Once `job-1e218dd2-8c24-463f-b327-374b3e5066c7` finishes, run `qz/run_streaming_backbone_audit_smoke.sh` on the V2 output and inspect whether the chunks now exhibit actual temporal continuation.
