# Execution Mode Log

## Phase Roadmap

- Phase A: Complete AV baseline on untouched base model. Status: done.
- Phase B: Build real runtime prompt-switching inference over chunked AV generation. Status: done.
- Phase C: Run low-priority smoke job for runtime switching inference. Status: done.
- Phase D: Compare LongLive-style switch recache against the current reference-video baseline. Status: done.
- Phase E: Upgrade from chunked runtime switching prototype to stronger memory-backed streaming inference. Status: done.

## Current Phase

### Phase E: Native SSM-Backed Streaming Inference Preparation
- Objective: move from the verified decoded-frame fallback into the repo-native persistent-state path without pretending that KV / cross-attention cache recache already exists.
- Implementation status:
  - the original `reference_video` baseline smoke already succeeded as `job-54c2d7f0-d7af-45be-92fb-c5806ab3e58d`.
  - the v2 feasibility audit is complete: the current inference path exposes no direct KV / cross-attention cache API for LongLive-style recache.
  - repo-native persistent state does exist via the SSM path (`SSMState`, `SSMAugmentedLTXModel`), but current inference plumbing does not yet thread state through `ValidationSampler` / `X0Model`.
  - the strongest non-invasive fallback is now implemented: prompt switches recache bounded recent history under the new prompt and feed that clip back through `reference_video`.
  - local verification is green: `7` targeted tests passed in `tests/test_streaming_inference.py`, and `14` broader regression tests passed across the streaming + self-forcing schedule suite.
  - native SSM streaming wiring now lands in `X0Model`, `ValidationSampler`, and `scripts/streaming_inference.py`; remote regression is green: `pytest tests/test_streaming_inference.py tests/test_train_self_forcing_data.py tests/test_train_self_forcing_schedule.py -q` -> `15 passed in 48.78s`.
  - low-priority v2 smoke job `job-0fda36de-9fcc-44b9-91b4-5dd9a31a3ccd` succeeded and wrote refreshed artifacts to `outputs/streaming_switch_smoke_v1/`.
  - `outputs/streaming_switch_smoke_v1/episode_0000.json` records `reference_source=switch_recache` with `switch_recache_frames=17` on chunks `1` and `2`; `outputs/streaming_switch_smoke_v1/verification_summary_v2.json` captures the metadata-level verification summary for that run.
- Current low-priority job:
  - none active at this checkpoint.
- Latest completed SSM run:
  - initial smoke `job-28e1b799-fc4c-4751-a33f-28317de6a453` failed at the first chunk on a real dtype mismatch between bf16 chunk latents and fp32 SSM compression weights.
  - fix applied: `ValidationSampler` now moves the transformer wrapper to `dtype=torch.bfloat16` before denoising, matching the inference path used by the base model.
  - retry job `job-8a7ee4e2-7ed0-4a5c-a38a-21d311a585f0` succeeded and wrote `outputs/streaming_switch_ssm_smoke_v1/episode_0000.mp4`, `outputs/streaming_switch_ssm_smoke_v1/episode_0000.json`, and `outputs/streaming_switch_ssm_smoke_v1/verification_summary.json`.
  - result: all three chunks record `reference_source=none`, `stream_state_enabled=true`, and `stream_state_returned=true`, so this run is a pure SSM-memory smoke rather than decoded-frame carryover.
- Latest completed control run:
  - id: `job-aede8bad-af03-4191-9b07-a96b2a3afa42`
  - name: `wgx-train-h200-1g-p3-v2norecache-streaming-switch-ablation`
  - gpu_count: `1`
  - priority: `3`
  - status: `job_succeeded`
  - log: `/inspire/hdd/project/agileapplication/zhangkaipeng-24043/wgx/qz/logs/wgx-train-h200-1g-p3-v2norecache-streaming-switch-ablation.log`
  - result: `outputs/streaming_switch_smoke_no_recache_v1/episode_0000.json` records `reference_source=history` on prompt-switch chunks, providing the direct control against the v2 `switch_recache` fallback.
- Next code surface:
  - `packages/ltx-core/src/ltx_core/model/transformer/model.py`
  - `packages/ltx-trainer/src/ltx_trainer/validation_sampler.py`
  - `scripts/streaming_inference.py`
  - `scripts/train_self_forcing.py`
- Exit criteria for the current phase:
  - keep `X0Model` backward-compatible while adding optional stream-state return support
  - thread SSM state through validation sampling and chunk-level streaming inference
  - prepare the next low-priority smoke submission against existing SSM checkpoint artifacts

- Phase E completion evidence:
  - `pytest tests/test_streaming_inference.py tests/test_train_self_forcing_data.py tests/test_train_self_forcing_schedule.py -q` -> `15 passed in 48.78s`
  - post-fix focused rerun `pytest tests/test_streaming_inference.py tests/test_train_self_forcing_schedule.py -q` -> `10 passed in 44.97s`
  - pure SSM smoke `job-8a7ee4e2-7ed0-4a5c-a38a-21d311a585f0` succeeded on low-priority 1-GPU remote execution
