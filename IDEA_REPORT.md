# Idea Discovery Report

**Direction**: Selective persistent memory for streaming long-horizon audio-video generation with interactive prompt switching on top of LTX-2.3-distilled.  
**Date**: 2026-03-20  
**Pipeline**: research-lit -> idea-creator -> novelty-check -> research-review -> implementation handoff

## Executive Summary

The strongest direction is no longer "add memory to long video generation." The sharper and more defensible thesis is:

> Build an **Edit-Scoped Persistent Memory** mechanism that separates **persistent world state** from **switchable prompt control**, so a streaming audio-video generator can preserve character identity, voice, scene style, and background atmosphere across long horizons while still applying user prompt changes selectively.

This wedge is promising because the closest lines of work are still fragmented as of **March 20, 2026**:
- streaming long-form generation papers mostly focus on continuation and cache stability,
- memory papers mostly focus on coherence and retrieval,
- editing papers focus on instruction following and preservation,
- joint audio-video papers focus on sync and quality,
- but the combination of **streaming + selective mid-stream editing + persistent AV world state** remains under-served.

The current `myltx-v1` codebase already contains the right substrate:
- per-layer SSM memory integration,
- streaming switch planning and recache logic,
- switch-aware chunk schedules,
- switch-weighted training losses,
- real AV distilled ODE data plumbing.

The missing leap is that the current switch logic is still essentially a **global decay / global recache** mechanism, not a true selective persistence mechanism.

## Literature Landscape

### 1. Streaming and long-horizon generation

- **StreamingT2V** (arXiv March 2024, CVPR June 2025): chunkwise long video generation with appearance-preservation memory.  
  Source: <https://arxiv.org/abs/2403.14773>
- **Self Forcing** (arXiv June 2025): addresses autoregressive exposure bias for video diffusion, strong streaming relevance but not selective prompt editing.  
  Source: <https://self-forcing.github.io/>
- **Rolling Forcing** (arXiv September 2025, ICLR 2026): real-time multi-minute autoregressive diffusion with attention-sink style anchoring.  
  Source: <https://arxiv.org/abs/2509.25161>
- **Rolling Sink** (arXiv February 2026): explicit long-horizon extrapolation beyond training horizon.  
  Source: <https://arxiv.org/abs/2602.07775>

### 2. Persistent memory and state mechanisms

- **MemFlow** (arXiv December 2025): adaptive memory for long-context consistency in video generation.  
  Source: <https://arxiv.org/abs/2512.14699>
- **MemRoPE** (arXiv March 2026): training-free evolving memory tokens for effectively infinite generation.  
  Source: <https://arxiv.org/abs/2603.12513>
- **StoryMem** (arXiv December 2025): explicit shot-level memory for long-form storytelling and ST-Bench.  
  Source: <https://arxiv.org/abs/2512.19539>
- **Long-Context State-Space Video World Models** (arXiv May 2025, ICCV 2025): state-space sequence modeling for long-context video retention and retrieval.  
  Source: <https://arxiv.org/abs/2505.20171>

### 3. Interactive prompt-based editing

- **InteractiveVideo** (arXiv February 2024): interactive control during generation rather than one-shot prompting.  
  Source: <https://arxiv.org/abs/2402.03040>
- **VIVA** (arXiv December 2025): VLM-guided instruction video editing.  
  Source: <https://arxiv.org/abs/2512.16906>
- **O-DisCo-Edit** (arXiv September 2025): realistic video editing with explicit distortion control.  
  Source: <https://arxiv.org/abs/2509.01596>
- **Memory-V2V** (arXiv January 2026): memory-augmented video-to-video editing.  
  Source: <https://arxiv.org/abs/2601.16296>

### 4. Identity consistency and character persistence

- **StoryDiffusion** (NeurIPS 2024): long-range identity consistency via consistent self-attention.  
  Source: <https://proceedings.neurips.cc/paper_files/paper/2024/file/c7138635035501eb71b0adf6ddc319d6-Paper-Conference.pdf>
- **ContextAnyone** (arXiv December 2025): character-consistent text-to-video from text plus a reference image.  
  Source: <https://arxiv.org/abs/2512.07328>
- **AnyCrowd** (arXiv March 2026): multi-character identity-pose binding.  
  Source: <https://arxiv.org/abs/2603.15415>

### 5. Joint audio-video generation and voice preservation

- **LTX-2** (arXiv January 2026): open-weight DiT-based audio-video foundation model.  
  Source: <https://arxiv.org/abs/2601.03233>
- **Movie Gen** (arXiv October 2024): synchronized text-to-audio-video generation.  
  Source: <https://arxiv.org/abs/2410.13720>
- **SyncFlow** (arXiv December 2024): temporally aligned joint audio-video generation.  
  Source: <https://arxiv.org/abs/2412.15220>
- **JavisDiT** (arXiv March 2025) and **JavisDiT++** (February 2026): joint AV DiT with explicit sync priors.  
  Source: <https://arxiv.org/abs/2503.23377>
- **ALIVE** (arXiv February 2026): adapts a pretrained T2V backbone to synchronized AV generation.  
  Source: <https://arxiv.org/abs/2602.08682>
- **EmoDubber** (arXiv December 2024, CVPR June 2025): strong voice identity preservation reference via dubbing and voice cloning.  
  Source: <https://arxiv.org/abs/2412.08988>

## Ranked Ideas

### 1. Edit-Scoped Persistent Memory for Streaming AV Generation — RECOMMENDED

**Core thesis**

Persist world state across the stream, but make prompt-switch updates selective instead of global.

**World state to persist**

- character identity,
- voice timbre and speaking style,
- scene/background layout,
- style and atmosphere.

**What changes on a prompt switch**

- only the state dimensions implied by the prompt delta,
- while unrelated state remains locked or strongly regularized.

**Why this is novel**

- It is not just "memory for long video."
- It is not just "prompt recache for interaction."
- It is not just "identity preservation."
- It is the explicit decomposition of:
  - **persistent state**,
  - **switchable control**,
  - **edit scope**,
  on top of an open joint AV foundation model.

**Why this fits the codebase**

- `packages/ltx-core/src/ltx_core/model/transformer/ssm_integration.py` already injects per-layer SSM memory.
- `packages/ltx-core/src/ltx_core/model/transformer/ssm_memory.py` already treats SSM state as orthogonal to local cache.
- `scripts/streaming_inference.py` already supports switch plans, history references, recache windows, and SSM streaming mode.
- `scripts/train_self_forcing.py` already supports chunkwise switch schedules and switch-aware losses.

**Main gap**

Current code only provides **global switch decay** via `ssm_switch_state_decay`, not selective state retention by state type.

### 2. Cross-Modal Identity Binding Memory — BACKUP

Keep voice identity and visual identity explicitly bound across long-horizon generation and prompt switches.

**Why backup only**

- Stronger as a sub-contribution or evaluation axis than as the main paper.
- Reviewer risk: can be read as a narrower talking-head or personalization add-on.

### 3. Persistent World-State Tokens Across Prompt Recache — BACKUP

Separate recacheable prompt-conditioned control from persistent world-state tokens that survive every switch.

**Why backup only**

- Good clean mechanism.
- Slightly less concrete than the edit-scoped framing.
- Easier for reviewers to reduce to "just another memory token trick."

## Closest Competitors We Must Beat

| Cluster | Representative papers | What they cover | What remains open for us |
|---|---|---|---|
| Streaming long-form generation | StreamingT2V, Self Forcing, Rolling Forcing, Rolling Sink | continuation, drift reduction, long rollout | selective prompt switching with preserved AV world state |
| Memory for long-form generation | MemFlow, MemRoPE, StoryMem | coherence, compression, retrieval | explicit edit-scoped persistence under interaction |
| Interactive editing | InteractiveVideo, VIVA, O-DisCo-Edit, Memory-V2V | instruction following, editable video | streaming long-horizon joint AV state preservation |
| Character consistency | StoryDiffusion, ContextAnyone, AnyCrowd | visual identity consistency | voice + scene + style + interaction all together |
| Joint AV generation | LTX-2, Movie Gen, SyncFlow, JavisDiT, ALIVE | synchronized audio-video quality | long streaming persistence plus selective mid-stream edits |

## Codebase Fit

### What already exists

- `scripts/streaming_inference.py`
  - switch-mode planning,
  - history reference video selection,
  - switch recache path,
  - SSM streaming toggle,
  - `ssm_switch_state_decay`.
- `scripts/train_self_forcing.py`
  - chunked SSM training loop,
  - prompt-switch schedule,
  - keep/edit/switch loss accounting,
  - prompt-cache reuse.
- `scripts/self_forcing_data.py`
  - switch episode manifests,
  - chunk-level prompt schedule construction.
- `packages/ltx-core/src/ltx_core/model/transformer/ssm_memory.py`
  - additive persistent SSM memory per layer and modality.

### What is still missing

- no representation of **state type** inside the memory path,
- no representation of **edit scope** inside switch manifests,
- no per-scope update / decay / freeze policy,
- no metric pipeline yet for:
  - target-change success,
  - non-target preservation,
  - voice identity persistence,
  - voice-face binding consistency.

## Recommended Mechanism

### Name

**Edit-Scoped Persistent Memory (ESPM)**

### Minimal mechanism

1. Factor persistent memory into three banks:
   - `entity_memory`,
   - `scene_memory`,
   - `audio_memory`.
2. At each prompt switch, compute a lightweight `edit_scope`:
   - `entity`,
   - `scene`,
   - `style`,
   - `audio`,
   - combinations of the above.
3. Use `edit_scope` to decide:
   - which memory banks can update,
   - which banks are decayed,
   - which banks are preserved,
   - which banks contribute retrieval at the next chunk.
4. Keep short-range prompt control and recache logic separate from persistent memory.
5. Train with counterfactual multi-turn edits so the model learns:
   - change the targeted attributes,
   - preserve the untargeted attributes.

### Why this is better than the current scalar decay

The current `ssm_switch_state_decay` answers only one question:
"How much should all memory fade when the prompt changes?"

ESPM answers the real research question:
"Which parts of memory should change, which parts should stay, and how do we prove that distinction helps?"

## Reviewer Risks

- "This is just cache engineering with a new name."
- "Edit scope is heuristic and may not generalize."
- "Identity preservation can conflict with editability."
- "Audio identity claims are weak without strong voice metrics."
- "Long-horizon claims need stronger evaluation than short-clip quality benchmarks."

## How To Defend The Novelty

- Make the decomposition explicit:
  - prompt control is recacheable,
  - world state is persistent,
  - edit scope decides which persistent state is mutable.
- Show that memory alone is insufficient:
  - compare against always-preserve and always-update baselines.
- Show that recache alone is insufficient:
  - compare against recache-only switch handling.
- Make audio-video part real:
  - include voice identity and cross-modal binding metrics, not only video demos.

## Auto-Selected Idea

With `AUTO_PROCEED=true`, the selected idea is:

**Idea 1: Edit-Scoped Persistent Memory for Streaming AV Generation**

## Immediate Next Step

Do not rewrite the whole streaming stack. The highest-leverage next implementation delta is:

1. extend switch manifests with `edit_scope`,
2. propagate chunk-level scope metadata through the current schedule builder,
3. replace scalar switch decay with per-bank selective update rules,
4. add preservation-vs-edit metrics and a first switch benchmark.
