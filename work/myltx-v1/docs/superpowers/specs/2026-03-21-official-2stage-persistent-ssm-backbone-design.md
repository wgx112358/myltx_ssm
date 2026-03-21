# Official Two-Stage Persistent-SSM Backbone Design

## Goal

Replace the current Stage A `persistent_ssm backbone smoke` runtime with a new streaming backbone that reuses the official LTX distilled two-stage generation logic for each chunk, while moving chunk memory and streaming lifecycle out of `ValidationSampler` and into a dedicated orchestrator.

This is a Stage A backbone-only migration. It is not a full replacement of the existing streaming runtime, and it must not be presented as complete official streaming alignment.

## Problem

The current Stage A backbone smoke path uses [`ValidationSampler`](/Users/nick/ai_research_codex_v1/work/myltx-v1/packages/ltx-trainer/src/ltx_trainer/validation_sampler.py), which was originally designed as a training validation sampler, not as the canonical runtime generation path. It diverges from the official distilled pipeline in several important ways:

- It samples full-resolution latents directly instead of using the official two-stage distilled flow.
- It does not load or use the spatial upsampler.
- It allows non-official geometry and smoke-specific defaults.
- It mixes generation, streaming state management, delayed compression, and runtime orchestration in one place.

We now have strong evidence that delayed-compression persistent SSM is the correct Stage A memory schedule, but we do not yet have an implementation that combines that schedule with the official two-stage distilled generation core.

## Scope

### In Scope

- Replace only the `persistent_ssm backbone smoke` path.
- Reuse official two-stage distilled chunk generation for Stage A backbone smoke.
- Keep persistent SSM semantics, including delayed compression, outside the chunk runner.
- Restrict the first migrated path to the local `small` alias only, where `small` means the official distilled standard output geometry `1024x1536`.
- Keep output layout comparable with existing Stage A smoke artifacts.

### Out of Scope

- Prompt-switch runtime migration.
- `switch_recache` and prompt-switch decay behavior.
- `baseline_audit` migration.
- `reference_video` migration.
- Renaming or deleting `ValidationSampler` in this phase.
- Full official runtime parity claims.

## Design Principles

1. Official two-stage logic stays intact at the chunk level.
2. Streaming lifecycle lives outside the chunk runner.
3. The Stage A delayed-compression schedule must stay behaviorally aligned with the current validated implementation.
4. Phase 1 must not introduce a new memory-topology experiment.
5. The first migration must minimize the patch surface and isolate regression risk.
6. Names must reflect scope and avoid overclaiming.

## Proposed Architecture

### 1. Official Two-Stage Chunk Runner

Add a new chunk-oriented runner object that wraps the official distilled two-stage logic rather than reusing `ValidationSampler`.

Suggested location:

- `scripts/official_2stage_chunk_runner.py`

Public boundary:

- `run_chunk(...)`

Internal helpers may split stage 1 and stage 2, but the orchestrator must call a single public chunk-runner entrypoint so queue ownership and memory contracts stay outside the runner.

Responsibilities:

- Validate that geometry matches the phase-1 official distilled standard output `1024x1536`.
- Validate that `num_frames = 8*K + 1`.
- Load the distilled checkpoint, Gemma, and spatial upsampler using the official pipeline path.
- Construct these heavy components once per runner/orchestrator lifetime and reuse them across all chunk calls in the same run.
- Run the official two-stage distilled flow for exactly one chunk:
  - stage 1 at half resolution
  - `upsample_video(...)`
  - stage 2 refinement at target resolution
- Reuse the official distilled invariants:
  - stage-1 `DISTILLED_SIGMA_VALUES`
  - stage-2 `STAGE_2_DISTILLED_SIGMA_VALUES`
  - stage-2 handoff `initial_video_latent=upscaled_video_latent`
  - stage-2 handoff `initial_audio_latent=stage1_audio_latent`
  - stage-2 `noise_scale=stage_2_sigmas[0]`
- In phase 1, attach persistent SSM only to the stage-2 chunk denoising path.
- Accept a single persistent `ssm_state` as input and return updated `ssm_state`.
- Return compressor-compatible evictable stage-2 chunk tokens after chunk completion.

Non-responsibilities:

- No manifest handling.
- No chunk plan generation.
- No switch logic.
- No final stitch logic.
- No delayed-compression queue ownership.
- No new memory topology design.

### 2. Stateful Denoiser Wrapper

Do not rewrite the generic sampler utilities. Add a thin stateful denoiser wrapper around the official distilled denoising step so the official chunk runner can call the existing Euler denoising loop while carrying `ssm_state`.

Minimal requirement:

- A state-aware version of the distilled denoising closure that:
  - accepts current `ssm_state`
  - calls `X0Model(..., ssm_state=..., return_ssm_state=True)`
  - threads the returned state into the next diffusion step
  - returns final `next_ssm_state` after stage completion

State contract for phase 1:

- During the Euler loop, `ssm_state` is query-threaded step to step.
- Long-term persistent memory is written only by chunk-end delayed compression.
- The implementation must not add per-step compression or other new memory writes inside the diffusion loop.

This avoids changing `utils/samplers.py` or the generic `DenoisingFunc` protocol in phase 1.

### 3. Persistent Memory State

Phase 1 must not introduce a mandatory new stage-local memory topology.

Instead, it keeps one persistent memory carrier owned by the orchestrator:

- `persistent_ssm_state`
- `pending_video_chunks`
- `pending_audio_chunks`

In phase 1, that memory carrier is defined over the stage-2 chunk token representation only. Stage 1 stays an official chunk-local generation step and does not introduce its own cross-chunk memory path.

Reason:

- The current validated Stage A result uses one persistent memory carrier.
- Forcing a stage1/stage2 split in the first migration would turn runtime replacement into a new memory-design experiment.
- Compressing final stage-2 chunk tokens is closer to the current full-resolution persistent memory semantics than adding a second low-resolution memory path.

This is the one deliberate approximation introduced in phase 1. It must be treated as a parity hypothesis to verify against the current validated delayed-compression backbone path, not as a pre-validated semantic equivalence.

Future stage-local memory splits may be explored later as a separate ablation, but they are not part of phase 1.

### 4. Streaming Chunk Orchestrator

Add a new orchestrator dedicated to Stage A backbone smoke.

Suggested location:

- `scripts/official_2stage_backbone_orchestrator.py`

Responsibilities:

- Read the Stage A backbone manifest.
- Build the no-switch chunk plan.
- Resolve official preset geometry.
- Maintain chunk seeds and chunk-local metadata.
- Own the single persistent SSM state and delayed-compression queue.
- Call the official chunk runner once per chunk.
- Save per-chunk MP4s.
- Stitch the final episode output.
- Emit metadata and observability logs.

This layer is the only place that should know about:

- `window_blocks`
- pending queue length
- delayed compression timing
- artifact naming
- Stage A smoke output directories
- detached queue snapshots

### 5. Legacy Boundary for ValidationSampler

`ValidationSampler` remains in place for:

- training validation
- standalone sanity checks
- existing trainer-side validation utilities

Phase 1 policy:

- No new Stage A runtime features should be added to `ValidationSampler`.
- New official two-stage streaming work must not depend on it.
- A later cleanup phase may rename it to reduce runtime confusion, but not in this migration.

## Data Flow

For each chunk:

1. The orchestrator selects the official preset and chunk timing.
2. The orchestrator calls `run_chunk(...)` on the chunk runner.
3. Inside `run_chunk(...)`, the runner:
   - executes official stage 1 generation without a cross-chunk memory carrier
   - upsamples the stage-1 video latent
   - reuses the stage-1 audio latent for stage-2 initialization
   - executes official stage 2 refinement with the persistent `ssm_state`
4. The orchestrator receives from the single `run_chunk(...)` call:
   - final chunk audio/video
   - `next_ssm_state`
   - evictable stage-2 chunk tokens
5. The orchestrator updates the delayed-compression queue and calls `compress_evicted_tokens(...)` only when a chunk leaves the local window.
6. The orchestrator saves the chunk artifact and appends final-stitch tensors.
7. After the last chunk, the orchestrator stitches the episode output and writes metadata.

## Minimal Public Interfaces

### Chunk Runner Config

The first phase runner should expose only the minimal config needed for Stage A:

- distilled checkpoint path
- Gemma path
- spatial upsampler path
- local preset name: `small` only in phase 1
- chunk frame count
- frame rate
- chunk prompt
- chunk seed
- `ssm_streaming_enabled`
- `ssm_d_state`
- `ssm_gate_bias`
- optional `ssm_checkpoint_path`

Not included in phase 1:

- arbitrary height/width override
- HQ mode
- prompt-switch controls
- recache controls
- reference video controls

### Orchestrator State

Suggested state bundle:

- `persistent_ssm_state`
- `pending_video_chunks`
- `pending_audio_chunks`

Queue contract:

- queue items are detached snapshots, never borrowed live tensors
- `window_blocks` counts queued whole chunks, matching the currently validated delayed-compression behavior
- `evictable_video_tokens` and `evictable_audio_tokens` are patchified stage-2 chunk tokens in the compressor-compatible token space expected by `compress_evicted_tokens(...)`
- `stage1_video_latent_for_stage2` is an unpatchified stage-1 VAE latent
- `final_chunk_video` and `final_chunk_audio` are decoded media outputs

## Naming

Phase 1 names must stay explicit and narrow. Preferred naming should include:

- `official_2stage`
- `persistent_ssm`
- `backbone`
- `preview` or `trial`

Examples:

- `official_2stage_persistent_ssm_backbone_smoke_preview`
- `official_2stage_backbone_persistent_ssm_trial`

Avoid in phase 1:

- `official_streaming`
- `official_backbone`
- `two_stage_streaming_runtime`

Those names overclaim scope.

## Migration Plan

### Phase 1

- Build the new official two-stage chunk runner.
- Build the Stage A-only orchestrator.
- Switch only the `persistent_ssm backbone smoke` wrapper to the new orchestrator.
- Keep artifact directory structure compatible with current Stage A smoke review.
- Keep phase 1 on local preset `small` only.

### Phase 2

- Migrate `no_memory` and `short_context` Stage A baselines to the same orchestrator.
- Keep the same geometry, chunking, and artifact schema across all three Stage A variants.

### Phase 3

- Migrate prompt-switch runtime and related audit paths only after Stage A parity is stable.

## Testing Strategy

### Unit Tests

- chunk runner config validation accepts only phase-1 `small`
- stage-2 token outputs match the compressor contract
- the runner keeps one public `run_chunk(...)` boundary
- delayed-compression queue eviction happens at the expected window boundary
- wrapper defaults point to the new official-two-stage backbone preview path

### Integration Tests

- one-episode no-switch smoke plan produces chunk outputs and final stitched output
- metadata contains the local `small` alias, resolved official geometry, and persistent-memory mode
- persistent queue stats are logged

### Regression Checks

- compare new preview output against current validated delayed-compression behavior for:
  - successful completion
  - no black final video
  - stable chunk count
  - sane state magnitude trajectory

## Risks

### Risk 1: Delayed Compression Semantics Drift

If the new orchestrator changes eviction timing or token shape, the current validated `persistent_ssm` behavior may regress even if the official two-stage chunk runner is correct.

Mitigation:

- keep delayed-compression ownership entirely in the orchestrator
- preserve the current window semantics
- explicitly test queue lengths and eviction boundaries

### Risk 2: Token Shape Mismatch for Compression

`compress_evicted_tokens(...)` expects tokens in a compatible latent shape. If the official chunk runner emits a different representation, compression will silently become incorrect.

Mitigation:

- add a narrow adapter only if needed
- do not push adaptation logic back into `ValidationSampler`

### Risk 3: Overclaiming Official Alignment

This migration only covers Stage A persistent-SSM backbone smoke first.

Mitigation:

- enforce narrow naming
- document remaining legacy paths
- avoid calling the result a full official streaming runtime

## Rollback

If the new official two-stage preview path regresses black-screen stability or breaks delayed compression semantics:

- keep the current validated delayed-compression `ValidationSampler` runtime as the fallback Stage A backbone path
- keep the new chunk runner isolated behind its own wrapper
- debug token compatibility and state handoff before attempting broader migration

## Success Criteria

Phase 1 is successful when all of the following are true:

- the `persistent_ssm backbone smoke` path no longer depends on `ValidationSampler`
- chunk generation reuses official distilled two-stage logic
- delayed-compression persistent SSM still completes successfully on Stage A smoke
- the migrated path preserves the current Stage A backbone success bar: successful completion, non-black final output, and stable queue/SSM observability
- the implementation is clearly named as a backbone-only preview/trial path
- existing Stage A audit comparability is preserved
