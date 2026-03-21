# Route 3: Pure SSM Streaming Research Plan

## Scope Freeze

- Primary route: pure SSM streaming only.
- De-prioritized routes: `history` and `switch-recache` remain as existing baselines only.
- Goal: make LTX 2.3-distilled support online chunked continuation and prompt switching without `reference_video` and without recache.

## Current Status

- Pure SSM streaming smoke has already succeeded on Qizhi low-priority job `job-8a7ee4e2-7ed0-4a5c-a38a-21d311a585f0`.
- Artifact: `outputs/streaming_switch_ssm_smoke_v1/episode_0000.mp4`.
- Metadata confirms `reference_source=none`, `stream_state_enabled=true`, and `stream_state_returned=true` across chunks.
- Local regression already passed for the streaming and self-forcing tests.
- This means the route-3 inference path exists and is validated at smoke level; the next bottleneck is switch-aware training data and switch-aware training objectives.

## Problem Definition

- Bottom-line problem: enable long-horizon, chunked, interactive generation with prompt updates during generation time.
- Must-solve bottleneck: preserve temporal continuity across chunk boundaries while responding quickly to prompt switches.
- Non-goals:
  - no KV-cache / cross-attention cache redesign in the current phase
  - no external frame recache as the main method
  - no full-model large-scale retraining before the memory path is stable
- Success condition: pure SSM streaming should beat the no-switch-training control and become competitive with or better than `switch-recache` on the continuity-versus-switch-response tradeoff.

## Method Thesis

- Thesis: a lightweight internal memory path based on threaded SSM state is sufficient to support online prompt switching in a chunked distilled video generator, if memory carryover and switch response are trained explicitly.
- Smallest adequate intervention:
  - keep the current SSM threading path
  - add switch-aware state control
  - add switch-aware training curriculum and losses
- Explicitly excluded for now:
  - external memory banks
  - RL control
  - large auxiliary planners

## Planned Method Components

### A. State Threading Backbone

- Keep the current chunk-to-chunk SSM state threading as the only persistent memory mechanism.
- Maintain backward-compatible inference and training entry points.

### B. Switch-Aware State Control

- Add a lightweight control mechanism at prompt-switch boundaries.
- Preferred direction: soft reset / gated interpolation over hard reset.
- Purpose: reduce semantic inertia after a prompt switch without destroying useful continuity.

### C. Switch Curriculum

- Start with sparse boundary-aligned switches.
- Increase switch density and randomize switch positions after the base memory path is stable.
- Mix non-switch and switch episodes to avoid degrading short-horizon quality.

### D. Boundary and Response Objectives

- Boundary consistency objective: discourage visible chunk-boundary discontinuities.
- Switch response objective: emphasize alignment to the new prompt in a short post-switch window.
- Audio objectives are deferred until the video-only route is stable.

## Training Plan

### Phase A: Video-only, frozen backbone

- Train only SSM-related parameters and switch-control parameters.
- Use switch episodes with moderate horizon and prompt changes.
- Primary goal: stable long-horizon continuation plus measurable switch response gains.

### Phase B: Video-only, partial unfreeze

- Unfreeze a small set of upper temporal blocks if Phase A saturates.
- Run the main ablations and produce the first paper-grade comparison against the existing baselines.

### Phase C: Audio-video joint training

- Reuse the same episode structure with `audio_latents` preserved.
- Introduce AV consistency and synchronization objectives only after the video route is stable.

## Data Plan

- Existing training path already supports `.precomputed` triplets plus a `switch_episode_manifest`.
- LongLive-style training data does not require extra labeled switch videos; the minimal pattern is `(prompt_a, prompt_b, switch_time)`.
- For route 3, we can keep the current `.precomputed` triplets and only replace the manifest generation path with two-segment single-switch episodes.
- The current loader can stay unchanged as long as the generated manifest order is aligned with the prompt/sample order used by the exported data.
- Immediate missing piece: generate LongLive-style switch manifests that are more faithful than the current random category-stitching smoke manifest.

## LongLive Audit

- Official repo cloned at `/inspire/hdd/project/agileapplication/zhangkaipeng-24043/wgx/longlive_v1` (HEAD `2462895`).
- LongLive long-stage training uses paired prompts plus a sampled single switch time, not a separately annotated long-video switch dataset.
- The minimal transferable data pattern for route 3 is `(prompt_a, prompt_b, switch_time)` with switch timing chosen on chunk boundaries.
- This means route 3 can keep the current `.precomputed` triplets and only replace the manifest generation path.
- New smoke artifact: `ode/switch_episodes_longlive_smoke.jsonl`.

## Evaluation Gates

- Gate 1: pure SSM runs stably for more chunks without `reference_video`.
- Gate 2: switch response improves over the no-switch-training control.
- Gate 3: route 3 is competitive with `switch-recache` while keeping the cleaner inference path.
- Gate 4: audio-video joint demo is stable and reproducible.

## Immediate Next Actions

- Use the new LongLive-style manifest path for route-3 switch-aware training.
- Add route-3-specific switch-control and loss terms only if the pure manifest change is insufficient.
- Launch the next low-priority training round for Phase A.
- Expand evaluation to continuity, switch latency, and later AV synchronization.

## Phase A Pilot Update (2026-03-19)

- Phase A export+manifest contract path is fixed to:
  - raw input: `/inspire/hdd/project/agileapplication/zhangkaipeng-24043/wgx/myltx/ode/data_distilled`
  - precomputed output: `/inspire/hdd/project/agileapplication/zhangkaipeng-24043/wgx/myltx-v1/ode/data_distilled_stage2_ode_phaseA_256`
  - manifest: `ode/switch_episodes_longlive_phaseA_256.jsonl` (256 episodes)
- Ordering caveat (important): current loader maps switch episodes by sample index modulo episode count and does not explicitly key by manifest `sample_id`; therefore manifest row order must stay aligned with exported sorted sample order (`00000..00255`).
- New Phase A train config and scripts:
  - config: `configs/self_forcing_longlive_phaseA_decay025_v1.yaml`
  - train shell: `qz/train_self_forcing_longlive_phaseA_decay025_v1.sh`
  - submit shell: `qz/submit_train_self_forcing_longlive_phaseA_decay025_v1_p3.sh`
- Selected switch-state control:
  - `ssm_switch_state_decay: 0.25`
  - `total_steps: 300` (>=200), `checkpoint_interval: 50`
  - output dir: `outputs/self_forcing_longlive_phaseA_decay025_v1`
  - expected first checkpoint artifact: `outputs/self_forcing_longlive_phaseA_decay025_v1/ssm_weights_step_00050.pt`
- Critical-path submission state:
  - export submit script: `qz/submit_build_stage2_ode_longlive_phaseA_256_p3.sh`
  - submitted low-priority job: `job-5d2bf5ba-d83f-47ec-a89c-ca5cc6c0abf0`
  - log: `/inspire/hdd/project/agileapplication/zhangkaipeng-24043/wgx/qz/logs/wgx-train-h200-1g-p3-v2-build-stage2-ode-longlive-phasea-256.log`
  - training submission should wait until the phaseA_256 precomputed export is complete.
