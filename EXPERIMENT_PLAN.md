# Experiment Plan

**Topic**: Counterfactual edit isolation with cross-modal typed persistent memory for streaming long-horizon audio-video generation on LTX-2.3-distilled  
**Date**: 2026-03-20

## Problem Anchor

Open LTX-2.3-distilled can generate strong short synchronized audio-video clips, but it still lacks a principled mechanism for:

- streaming extension to long horizons,
- interactive prompt switching mid-generation,
- preserving typed persistent world state,
- while selectively applying only the requested changes.

The immediate engineering blocker is simpler than the final paper claim:

- before studying prompt-switch isolation,
- we need a stable long-horizon streaming joint AV backbone,
- with persistent visual state, auditory state, and cross-modal binding that do not collapse over time.

## Thesis

Separate generation into:

- **typed persistent world state**,
- **switchable prompt-conditioned control**,
- **edit scope** that determines what may change.

The mechanism should preserve:

- character identity,
- voice timbre / speaking style,
- scene and background layout,
- ambient acoustic scene and persistent sound sources,
- overall style and atmosphere,

while still changing:

- only the attributes named by the prompt switch.

## Claims

### Claim 1

A stable streaming joint AV backbone requires explicit persistent state for:

- visual world state,
- auditory world state,
- and cross-modal binding.

### Claim 2

Once the long-horizon backbone is stable, counterfactual edit isolation improves:

- edit success,
- non-target preservation,
- and post-switch AV consistency

over recache-only and global scalar decay baselines.

## What Already Exists In Code

- SSM memory wrapper over the LTX transformer.
- Streaming switch runtime scaffold.
- Prompt-switch manifest and chunk schedule builder.
- Switch-aware chunk losses.
- Real AV distilled ODE-regression data plumbing.

## Experimental Strategy

### Stage A: Long-Horizon Joint AV Backbone First

Before prompt-switch experiments, build and validate a no-switch streaming backbone.

**Goal**

Show that the system can continue generation for long horizons while preserving:

- visual identity and scene continuity,
- auditory continuity,
- cross-modal role / speaker binding,
- AV sync after many chunks.

**Why this stage comes first**

- It removes the confound of switch data design.
- It tells us whether persistent auditory state is even being carried.
- It is the cheapest path to a usable training and evaluation scaffold.

**Important constraint**

This stage is a bridge milestone, not the final paper novelty. By itself, "longer streaming AV generation" is too crowded.

### Stage B: Prompt-Switch Isolation on Top of the Backbone

Once Stage A is stable, add intervention-specific machinery:

- typed `edit_scope`,
- counterfactual continuation,
- selective typed state mutation,
- preservation-edit frontier evaluation.

### Minimal Implementation Delta for Stage A

1. Keep the current streaming stack and real ODE-regression data path.
2. Train and evaluate long-horizon continuation without prompt switches first.
3. Make persistent state explicit for:
   - visual continuity,
   - auditory continuity,
   - cross-modal binding.
4. Add parseable metrics and logging for drift over horizon.

### Minimal Implementation Delta for Stage B

1. Add `edit_scope` to switch manifests and schedules.
2. Replace scalar switch decay with typed mutation policies.
3. Add counterfactual continuation targets.
4. Measure the preservation-edit frontier directly.

## Baselines

### Runtime baselines

1. Chunked LTX-2.3-distilled with no memory.
2. Chunked generation with recache only.
3. Chunked generation with current scalar `ssm_switch_state_decay`.
4. Typed persistent memory without prompt switches.
5. Counterfactual edit isolation with typed state mutation.

### Ablations

1. No edit scope, update all banks.
2. No persistent memory, recache only.
3. No audio bank.
4. Monolithic memory instead of banked memory.
5. Always freeze memory after switch.

### Stage A-specific ablations

1. No auditory persistent state.
2. Shared AV memory vs typed memory.
3. No explicit binding state.
4. Horizon length sweep.

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
- ambient acoustic continuity,
- persistent sound-source continuity,
- sync quality proxy for AV alignment.

### Joint AV metrics

- voice-to-entity binding consistency,
- turn-taking / role persistence,
- post-switch AV sync retention.

### Stage A backbone metrics

- long-horizon visual drift,
- long-horizon audio drift,
- long-horizon binding drift,
- horizon-to-failure under no-switch continuation.

### Standard external evaluation

- VBench,
- VBench-2.0 where feasible,
- EvalCrafter where feasible,
- targeted human study for "changed what should change, preserved what should stay."

## Datasets / Episode Construction

### Stage A smoke benchmark

Use existing real distilled ODE data to create a no-switch long-horizon continuation benchmark first.

### Stage B switch benchmark

Use existing switch manifests and then extend them with typed scope labels.

### Main benchmark

Construct multi-turn switch episodes with explicit edit types:

1. `entity-stable scene-stable action-change`
2. `entity-stable scene-stable speaking-style-change`
3. `entity-stable scene-change`
4. `entity-change with scene preserved`
5. `voice-style-change with identity preserved`

## Run Order

### Stage A: Backbone

1. Run no-switch long-horizon sanity training on real data.
2. Add explicit metrics for visual, auditory, and binding drift over horizon.
3. Compare:
   - no memory
   - scalar decay
   - typed persistent state
4. Validate that audio persistence is real, not incidental.

### Stage B: Prompt-switch isolation

5. Add `edit_scope` to manifest format and chunk schedule metadata.
6. Add typed state mutation logic and counterfactual continuation targets.
7. Run switch-mode inference with plan dumps and metadata inspection.
8. Evaluate recache-only vs scalar-decay vs typed mutation.
9. Measure the preservation-edit frontier.
10. Launch the first larger ablation on real data.

## Success Criteria For The First Round

### Stage A success

- training runs end-to-end on real data,
- long-horizon continuation works without prompt switches,
- audio drift is measurable and improved by explicit auditory state,
- cross-modal binding does not collapse over horizon,
- results are saved in parseable form.

### Stage B success

- chunk metadata contains correct scope labels,
- typed state mutation changes runtime behavior,
- the method beats recache-only and scalar-decay baselines on at least one selective-edit benchmark,
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

## Frozen Execution Decision

The first implementation milestone will **not** start with prompt switches.

It will start with:

- a no-switch long-horizon streaming joint AV backbone,
- explicit auditory persistence checks,
- explicit cross-modal binding checks.

Prompt-switch experiments remain the second milestone, after the backbone is stable.
