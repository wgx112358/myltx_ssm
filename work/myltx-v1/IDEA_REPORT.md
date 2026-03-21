# Idea Discovery Report

**Direction**: Audio-video joint generation, streaming long video, interactive prompt switching, memory mechanism on top of LTX-2.3-distilled.
**Date**: 2026-03-19
**Pipeline**: idea-discovery -> implementation -> auto-review-loop

## Executive Summary

The recommended idea is an orthogonal memory-control streaming extension for LTX-2.3-distilled: keep long-horizon audiovisual state in per-layer SSM memory, while treating prompt switches as control changes that only recache short-range conditioning/KV state. This direction is feasible because the repo already has SSM integration and large distilled AV ODE trajectories; the main missing pieces were real-data wiring, prompt-switch episodes, and low-priority smoke execution.

## Literature Landscape

- Recent long-video work is converging on hybrid local-cache plus compressed global memory rather than pure full-attention continuation.
- State-space memory is becoming the most natural candidate for long-horizon video because it preserves a cheap persistent state while leaving short-term detail to a local window.
- Interactive prompt-switch editing is still much less standardized than long-context continuation; most open pipelines handle single-prompt clips or offline editing, not streaming control changes over minute-scale AV generation.
- LTX-2 already gives a strong synchronized AV base model and distilled fast inference path, which makes it a pragmatic substrate for a streaming-memory paper instead of training a new foundation model from scratch.

## Ranked Ideas

### 1. Orthogonal AV Memory-Control Decomposition — RECOMMENDED

**Core thesis**
Persist long-term AV dynamics in SSM memory and handle prompt switches by refreshing only the short-range control path.

**Why this is the best fit here**
- The repo already contains `SSMAugmentedLTXModel` and a block-causal streaming scaffold.
- Distilled ODE latent trajectories already exist at scale, so we can bootstrap training without collecting new AV data.
- The main novelty is not “use memory” alone, but the explicit separation between memory persistence and control recaching for interactive AV generation.

**Current empirical signal**
- Positive infrastructure signal: SSM modules exist; raw distilled ODE data exists; prompt-switch manifest generation now exists; low-priority Qizhi smoke submission exists.
- Main remaining risk: current raw data is single-prompt, so true switch supervision still needs dataset-side schedule support or an augmented condition format.

### 2. Latent Replay Streaming Baseline — BACKUP

Replay real Stage-2 distilled latents block by block, compress evicted blocks into SSM, and use this as a non-generative streaming smoke/eval path.

**Why backup only**
- Good for debugging memory mechanics and decode/replay.
- Weak paper story by itself because it does not solve prompt-conditioned generation.

### 3. Full-clip Distilled Continuation + Chunk Stitching — BACKUP

Generate a full clip with the existing distilled pipeline and then split/reuse it as a pseudo-streaming baseline.

**Why backup only**
- Easy baseline.
- Limited novelty and weak interaction story.

## Eliminated Ideas

- End-to-end refactor of `ltx_pipelines` to natively accept `SSMAugmentedLTXModel`: too much engineering for the current iteration.
- Hidden-state-space replay at transformer hidden dimension as the primary path: mismatched with existing latent-space ODE data and too fragile for a first smoke.
- Full prompt-switch supervision without condition-format or sample-writer changes: currently unsupported by the dataset.

## Refined Proposal

- Proposal: `refine-logs/FINAL_PROPOSAL.md`
- Experiment plan: `refine-logs/EXPERIMENT_PLAN.md`
- Tracker: `refine-logs/EXPERIMENT_TRACKER.md`

## Next Steps

- [x] Add prompt-switch episode builder.
- [x] Add precomputed-triplet discovery/loading utilities.
- [x] Rewire `train_self_forcing.py` to prefer real `.precomputed` ODE data and keep synthetic fallback only as smoke backup.
- [x] Add low-priority Qizhi smoke wrappers.
- [ ] Finish exporting a tiny `.precomputed` smoke set.
- [ ] Launch self-forcing smoke once the export job completes.
- [ ] Run auto-review-loop and iterate on the cheapest blocking weaknesses.
