# Online Prompt Switching Inference Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a real chunked inference path that updates prompts during generation time, carries forward generated visual context across chunks, and emits stitched AV outputs plus chunk metadata.

**Architecture:** Extend `scripts/streaming_inference.py` with a `switch` mode that consumes a prompt-switch episode, precomputes prompt embeddings once, then generates chunk-by-chunk with runtime prompt updates. Reuse `ValidationSampler` reference-video conditioning to propagate recent generated context into the next chunk, save every chunk AV clip, and stitch a final AV artifact plus execution metadata.

**Tech Stack:** Python, PyTorch, `ltx_trainer.validation_sampler`, existing switch-manifest utilities, pytest, QZ low-priority submission wrappers.

---

## Status Snapshot

- `2026-03-19 06:16 +08:00`: `scripts/streaming_inference.py --mode switch` implemented with per-chunk prompt updates, `reference_video` carryover, plan-only metadata output, per-chunk mp4 paths, and stitched-final mp4 output.
- Local verification:
  - `python -m pytest -q tests/test_streaming_inference.py` -> `5 passed in 16.07s`
  - `python -m pytest -q tests/test_streaming_inference.py tests/test_train_self_forcing_data.py tests/test_train_self_forcing_schedule.py` -> `12 passed in 30.29s`
  - `bash qz/run_streaming_switch_smoke.sh --plan-only` -> wrote `outputs/streaming_switch_smoke_v1/episode_0000.json`
- Remote smoke submission:
  - Job: `job-54c2d7f0-d7af-45be-92fb-c5806ab3e58d`
  - Name: `wgx-train-h200-1g-p3-v1-streaming-switch-smoke`
  - GPU count: `1`
  - Priority: `3`
  - Current status: `job_succeeded`

### Task 1: Plan and schedule helpers

**Files:**
- Modify: `scripts/self_forcing_data.py`
- Create: `tests/test_streaming_inference.py`

- [x] **Step 1: Write the failing tests**
- [x] **Step 2: Run test to verify it fails**
- [x] **Step 3: Write minimal implementation**
- [x] **Step 4: Run test to verify it passes**

### Task 2: Streaming switch runtime helpers

**Files:**
- Modify: `scripts/streaming_inference.py`
- Test: `tests/test_streaming_inference.py`

- [x] **Step 1: Write the failing tests**
- [x] **Step 2: Run test to verify it fails**
- [x] **Step 3: Write minimal implementation**
- [x] **Step 4: Run test to verify it passes**

### Task 3: Online switch generation mode

**Files:**
- Modify: `scripts/streaming_inference.py`
- Test: `tests/test_streaming_inference.py`

- [x] **Step 1: Write the failing tests**
- [x] **Step 2: Run test to verify it fails**
- [x] **Step 3: Write minimal implementation**
- [x] **Step 4: Run targeted verification**

### Task 4: Remote smoke verification and experiment logging

**Files:**
- Modify: `refine-logs/EXPERIMENT_TRACKER.md`
- Modify: `AUTO_REVIEW.md`
- Modify: `refine-logs/EXECUTION_MODE.md`
- Create: `qz/run_streaming_switch_smoke.sh`
- Create: `qz/submit_streaming_switch_smoke_p3.sh`

- [x] **Step 1: Add remote smoke wrapper**
- [x] **Step 2: Run local plan-only verification**
- [x] **Step 3: Submit low-priority smoke job**
- [x] **Step 4: Verify artifact and update logs after job completion**

## Remaining Exit Condition

- Smoke verification complete: `job-54c2d7f0-d7af-45be-92fb-c5806ab3e58d` succeeded, wrote 3 chunk mp4s plus the stitched final mp4, and verification is recorded in `outputs/streaming_switch_smoke_v1/verification_summary.json`.


## Addendum: v2 Switch-Recache Follow-Up

- `2026-03-19 07:31 +08:00`: Completed the feasibility audit for true LongLive-style cache recache on the current LTX inference stack. The active inference path exposes no direct KV / cross-attention cache interface; the only native persistent-state mechanism already present in-repo is the SSM path used by `SSMAugmentedLTXModel`.
- `2026-03-19 07:31 +08:00`: Implemented the strongest non-invasive fallback in `scripts/streaming_inference.py`: on prompt switches, regenerate a bounded recent-history clip under the new prompt and feed that recached clip back through `reference_video` for the next chunk.
- Runtime metadata now records `reference_source`, `switch_recache_source_frames`, and `switch_recache_frames` for every chunk.
- Local verification:
  - `python -m pytest -q tests/test_streaming_inference.py` -> `7 passed in 17.26s`
  - `python -m pytest -q tests/test_streaming_inference.py tests/test_train_self_forcing_data.py tests/test_train_self_forcing_schedule.py` -> `14 passed in 45.16s`
- Remote smoke submission:
  - Job: `job-0fda36de-9fcc-44b9-91b4-5dd9a31a3ccd`
  - Name: `wgx-train-h200-1g-p3-v2-streaming-switch-smoke`
  - GPU count: `1`
  - Priority: `3`
  - Current status: `job_succeeded`
- Remote smoke evidence:
  - the job log records `reference=none frames=0` for chunk `0`, then `Recaching switch history` followed by `reference=switch_recache frames=17` for chunks `1` and `2`
  - chunkwise metadata is written to `outputs/streaming_switch_smoke_v1/episode_0000.json`
  - metadata-level verification summary is written to `outputs/streaming_switch_smoke_v1/verification_summary_v2.json`

### Task 5: Switch-recache fallback completion

**Files:**
- Modify: `scripts/streaming_inference.py`
- Modify: `tests/test_streaming_inference.py`
- Modify: `refine-logs/EXPERIMENT_TRACKER.md`
- Modify: `refine-logs/EXECUTION_MODE.md`
- Modify: `AUTO_REVIEW.md`

- [x] **Step 1: Audit true cache-recache feasibility in the current inference stack**
- [x] **Step 2: Implement bounded switch-recache fallback and per-chunk metadata**
- [x] **Step 3: Run targeted and broader local regression**
- [x] **Step 4: Submit low-priority v2 smoke job and inspect the remote outputs**

### Task 6: Native SSM-Backed Streaming Inference (Next)

**Files:**
- Modify: `packages/ltx-core/src/ltx_core/model/transformer/model.py`
- Modify: `packages/ltx-trainer/src/ltx_trainer/validation_sampler.py`
- Modify: `scripts/streaming_inference.py`
- Modify: `scripts/train_self_forcing.py`

- [ ] **Step 1: Keep `X0Model` backward-compatible while adding optional `ssm_state` threading**
- [ ] **Step 2: Extend `ValidationSampler` to carry stream state and compress evicted tokens**
- [ ] **Step 3: Add an SSM memory mode to `scripts/streaming_inference.py` with explicit metadata**
- [x] **Step 4: Submit the first low-priority SSM smoke with `outputs/self_forcing_phase1_curve/ssm_weights_step_00050.pt` into `outputs/streaming_switch_ssm_smoke_v1/` (job `job-28e1b799-fc4c-4751-a33f-28317de6a453`)**
- [x] **Parallel control: low-priority `--disable-switch-recache` ablation completed as `job-aede8bad-af03-4191-9b07-a96b2a3afa42`, writing `outputs/streaming_switch_smoke_no_recache_v1/episode_0000.json` with `reference_source=history` on switch chunks**

## Updated Exit Condition

- Fallback switch-recache smoke is complete: `job-0fda36de-9fcc-44b9-91b4-5dd9a31a3ccd` succeeded and `outputs/streaming_switch_smoke_v1/episode_0000.json` records `reference_source=switch_recache` on chunks `1` and `2`.
- The next upgrade path is native SSM-state streaming inference, not a claim of already having LongLive-equivalent internal cache recache.

- Native SSM smoke completion: initial job `job-28e1b799-fc4c-4751-a33f-28317de6a453` revealed a dtype mismatch during SSM compression; after forcing the wrapped transformer to bf16 in validation sampling, retry job `job-8a7ee4e2-7ed0-4a5c-a38a-21d311a585f0` succeeded and wrote `outputs/streaming_switch_ssm_smoke_v1/episode_0000.json` plus `outputs/streaming_switch_ssm_smoke_v1/verification_summary.json`.
