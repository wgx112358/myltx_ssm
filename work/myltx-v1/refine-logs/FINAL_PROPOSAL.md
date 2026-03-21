# Final Proposal

## Problem Anchor

Open LTX-2.3-distilled can generate synchronized audio-video clips, but it does not yet provide a clean mechanism for minute-scale streaming generation with interactive prompt switches while preserving long-horizon scene/audio memory.

## Method Thesis

Use an orthogonal memory-control design:
- Memory path: per-layer SSM states compress evicted AV tokens and persist across the stream.
- Control path: prompt switches update only the short-range conditioning/KV path.
- Training path: start from distilled ODE-regression triplets, then extend conditions to schedule-aware prompt-switch supervision.

## Minimal Method

1. Export small `.precomputed` ODE-regression triplets from existing distilled trajectories.
2. Train the SSM-augmented transformer on chunked real latents instead of synthetic placeholders.
3. Generate prompt-switch episode manifests from categorized prompts.
4. Extend condition payloads later with switch schedules so each chunk can select the correct prompt embedding.
5. Evaluate whether SSM persistence helps continuity while prompt recache improves editability.

## Dominant Contribution

The contribution is not just “memory for long video”; it is the explicit decomposition between persistent memory and recacheable control for synchronized audio-video streaming on top of a strong open distilled model.

## Main Risks

- Existing distilled data is single-prompt, so schedule-aware supervision is not yet present.
- `streaming_inference.py` is still a scaffold and needs either replay mode or true denoising integration.
- Cheap smoke results may validate plumbing before they validate the full interaction hypothesis.
