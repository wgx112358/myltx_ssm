# Experiment Plan

**Topic**: Edit-Scoped Persistent Memory for streaming long-horizon audio-video generation on LTX-2.3-distilled  
**Date**: 2026-03-20

## Problem Anchor

Open LTX-2.3-distilled can generate strong short synchronized audio-video clips, but it still lacks a principled mechanism for:

- streaming extension to long horizons,
- interactive prompt switching mid-generation,
- preserving persistent world state,
- while selectively applying only the requested changes.

## Thesis

Separate generation into:

- **persistent world state**,
- **switchable prompt-conditioned control**,
- **edit scope** that determines what may change.

The mechanism should preserve:

- character identity,
- voice timbre / speaking style,
- scene and background layout,
- overall style and atmosphere,

while still changing:

- only the attributes named by the prompt switch.

## Claims

### Claim 1

Selective persistent memory beats both:

- recache-only switching,
- and global-memory-decay switching,

on long-horizon prompt-switch consistency.

### Claim 2

Factoring memory into entity / scene / audio banks improves selective editability over a monolithic memory state.

### Claim 3

Joint AV persistent state improves voice and identity stability without sacrificing post-switch prompt adherence.

## What Already Exists In Code

- SSM memory wrapper over the LTX transformer.
- Streaming switch runtime scaffold.
- Prompt-switch manifest and chunk schedule builder.
- Switch-aware chunk losses.
- Real AV distilled ODE-regression data plumbing.

## Minimal Implementation Delta

### Phase A: Scope-aware data and metadata

Add `edit_scope` to switch manifests and schedules.

Suggested scope vocabulary:

- `entity`
- `scene`
- `style`
- `audio`
- `entity+style`
- `scene+style`
- `entity+audio`

### Phase B: Selective memory banks

Replace one monolithic switch-decay rule with three banks:

- `entity_memory`
- `scene_memory`
- `audio_memory`

Each chunk should carry:

- `prompt_switch_flag`
- `edit_scope`
- per-bank update / preserve mask

### Phase C: Losses

Keep current chunk losses, but add structured objectives:

- `preserve_loss`: unchanged attributes should stay stable,
- `edit_success_loss`: targeted attributes should change,
- `binding_loss`: voice identity should stay aligned with the intended speaker or entity.

### Phase D: Evaluation harness

Add a switch benchmark with 3 edit families:

- identity-preserving local edits,
- style / atmosphere edits,
- speech or voice-style edits.

## Baselines

### Runtime baselines

1. Chunked LTX-2.3-distilled with no memory.
2. Chunked generation with recache only.
3. Chunked generation with current scalar `ssm_switch_state_decay`.
4. ESPM with selective bank updates.

### Ablations

1. No edit scope, update all banks.
2. No persistent memory, recache only.
3. No audio bank.
4. Monolithic memory instead of banked memory.
5. Always freeze memory after switch.

## Metrics

### Core selective-edit metrics

- **Edit Success**: did the requested attribute change?
- **Non-Target Preservation**: did unrelated attributes remain stable?
- **Switch Recovery Latency**: how many chunks until the new prompt takes effect?

### Video stability metrics

- face or entity embedding consistency,
- scene/background similarity,
- temporal stability around switch boundaries.

### Audio metrics

- speaker embedding consistency,
- speaking-style consistency,
- sync quality proxy for AV alignment.

### Joint AV metrics

- voice-to-entity binding consistency,
- post-switch AV sync retention.

### Standard external evaluation

- VBench,
- VBench-2.0 where feasible,
- EvalCrafter where feasible,
- targeted human study for "changed what should change, preserved what should stay."

## Datasets / Episode Construction

### Smoke benchmark

Use existing switch manifests and distilled ODE data to create a tiny scope-labeled benchmark.

### Main benchmark

Construct multi-turn switch episodes with explicit edit types:

1. `entity-stable scene-stable action-change`
2. `entity-stable scene-stable speaking-style-change`
3. `entity-stable scene-change`
4. `entity-change with scene preserved`
5. `voice-style-change with identity preserved`

## Run Order

1. Add `edit_scope` to manifest format and chunk schedule metadata.
2. Add selective per-bank update logic to current SSM path.
3. Run smoke training on a tiny scope-labeled set.
4. Run switch-mode inference with plan dumps and metadata inspection.
5. Evaluate recache-only vs scalar-decay vs ESPM.
6. Add audio-specific stability and binding metrics.
7. Launch the first larger ablation on real data.

## Success Criteria For The First Round

- training runs end-to-end on real data,
- chunk metadata contains correct scope labels,
- selective bank updates change runtime behavior,
- ESPM beats recache-only and scalar-decay baselines on at least one selective-edit benchmark,
- qualitative demos clearly show:
  - identity preserved,
  - voice preserved,
  - scene preserved when appropriate,
  - prompt edits applied quickly when requested.

## Non-Goals For The First Round

- full minute-scale SOTA quality claims,
- full pipeline rewrite inside `ltx_pipelines`,
- broad world-modeling claims,
- 3D scene reconstruction claims.
