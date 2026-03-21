# Idea Discovery Report

**Direction**: Counterfactual edit isolation with cross-modal typed persistent memory for streaming long-horizon audio-video generation with interactive prompt switching on top of LTX-2.3-distilled.  
**Date**: 2026-03-20  
**Pipeline**: research-lit -> idea-creator -> novelty-check -> research-review -> implementation handoff

## Executive Summary

The strongest direction is no longer "add memory to long video generation." The sharper and more defensible thesis is:

> Build a **Counterfactual Edit Isolation** mechanism on top of **Cross-Modal Typed Persistent Memory**, so a streaming audio-video generator changes only the intended visual and auditory factors under prompt switches while preserving non-target state.

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

The missing leap is that the current switch logic is still essentially a **global decay / global recache** mechanism, not a typed state-mutation mechanism. In particular, `audio` is still treated too much like a side-effect of video continuity, rather than a first-class persistent state.

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

### 1. Counterfactual Edit Isolation with Cross-Modal Typed Persistent Memory — RECOMMENDED

**Core thesis**

At every prompt switch, compare the edited continuation against a counterfactual "would-have-continued" branch, and only commit changes to the typed persistent states that are meant to change.

**World state to persist**

- `visual world state`
  - character appearance
  - scene/background layout
  - style and atmosphere
- `auditory world state`
  - speaker timbre
  - speaking style / prosody
  - ambient acoustic scene
  - persistent sound sources
- `cross-modal binding state`
  - who is speaking
  - who that voice belongs to
  - turn-taking / interruption / response roles

**What changes on a prompt switch**

- only the state dimensions implied by the prompt delta,
- while unrelated state remains locked or strongly regularized.

**Why this is novel**

- It is not just "memory for long video."
- It is not just "prompt recache for interaction."
- It is not just "identity preservation."
- It is an explicit claim about:
  - **typed state mutability under intervention**
  - **auditory state as a first-class persistent variable**
  - **counterfactual edit isolation**
  on top of an open joint AV foundation model.

**Why this fits the codebase**

- `packages/ltx-core/src/ltx_core/model/transformer/ssm_integration.py` already injects per-layer SSM memory.
- `packages/ltx-core/src/ltx_core/model/transformer/ssm_memory.py` already treats SSM state as orthogonal to local cache.
- `scripts/streaming_inference.py` already supports switch plans, history references, recache windows, and SSM streaming mode.
- `scripts/train_self_forcing.py` already supports chunkwise switch schedules and switch-aware losses.

**Main gap**

Current code only provides **global switch decay** via `ssm_switch_state_decay`, not selective state retention by type or counterfactual edit isolation.

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

**Counterfactual Edit Isolation with Cross-Modal Typed Persistent Memory**

### Minimal mechanism

1. Factor persistent memory into three typed state classes:
   - `visual world state`
   - `auditory world state`
   - `cross-modal binding state`
2. At each prompt switch, compute a lightweight `edit_scope` over these state classes.
3. Keep a counterfactual continuation target that estimates what the stream would have done without the edit.
4. Commit updates only to the typed states the edit is allowed to change.
5. Preserve non-target states by explicitly comparing the edited rollout against the counterfactual branch.

### Why this is better than the current scalar decay

The current `ssm_switch_state_decay` answers only one question:
"How much should all memory fade when the prompt changes?"

The frozen idea answers the real research question:
"Which state types should change, which should stay, and how do we prove that auditory and binding state are preserved unless the edit truly targets them?"

## Reviewer Risks

- "This is just cache engineering with a new name."
- "Edit scope is heuristic and may not generalize."
- "Identity preservation can conflict with editability."
- "Audio memory is just a side branch of visual memory."
- "Long-horizon claims need stronger evaluation than short-clip quality benchmarks."

## How To Defend The Novelty

- Make the decomposition explicit:
  - prompt control is transient,
  - persistent state is typed,
  - auditory state is first-class,
  - edit scope decides which typed state is mutable.
- Show that memory alone is insufficient:
  - compare against always-preserve and always-update baselines.
- Show that recache alone is insufficient:
  - compare against recache-only switch handling.
- Make audio-video part real:
  - include voice identity, ambient persistence, and cross-modal binding metrics, not only video demos.

## Auto-Selected Idea

With `AUTO_PROCEED=true`, the selected idea is:

**Idea 1: Counterfactual Edit Isolation with Cross-Modal Typed Persistent Memory**

## Immediate Next Step

Do not rewrite the whole streaming stack. The highest-leverage next implementation delta is:

1. extend switch manifests with typed `edit_scope`,
2. propagate chunk-level scope metadata through the current schedule builder,
3. replace scalar switch decay with typed mutation policies,
4. make `audio` and `binding` explicit persistent state targets,
5. measure the preservation-edit frontier directly.

## Refinement After Novelty Check

The broad ESPM framing is too wide. The idea should be narrowed from:

- "better persistent memory for streaming AV generation"

to:

- "selective state mutability under prompt switches in streaming joint AV generation"

### What to drop

- Do not sell `persistent world state vs prompt control` as the main novelty.
- Do not sell generic memory banks as the main novelty.
- Do not sell identity consistency or AV sync as the main contribution.

Those are now crowded by recent work such as interactive streaming memory, memory retrieval, and personalized joint AV generation.

### What to keep

- Prompt switches are the core intervention.
- The important question is not how much history to keep, but which state types should change and which should not.
- Audio must be first-class, not an add-on.
- Evaluation should center on `Edit Success vs Non-Target Preservation`.

## Refined Idea Variants

### 1. Anchor-Conditioned Typed Memory Writes — RECOMMENDED

**Thesis**

Use prompt-switch anchors and typed edit scope as explicit write permissions for persistent memory, so different AV state types update selectively rather than uniformly.

**Why this is stronger**

- Directly addresses the closest streaming baseline family.
- Moves the contribution from "more memory" to "controlled memory mutation."
- Reuses the current stack: switch manifests, recache path, SSM state, and prompt-switch metadata.

**Main risk**

- If gains mostly look like better continuity, reviewers may collapse it into anchor/re-cache work.

### 2. Counterfactual Edit Memory

**Thesis**

Keep a frozen "would-have-continued" branch and only commit memory updates that improve the target edit while preserving non-target content against that counterfactual.

**Why this is stronger**

- Makes preservation a causal comparison, not only a heuristic.
- Gives a cleaner story for non-target preservation.

**Main risk**

- More expensive to implement and evaluate.
- Can look like a regularization trick unless the counterfactual branch changes outcomes clearly.

### 3. Audio-Video Scope-Coupled Memory

**Thesis**

Represent prompt switches as typed AV edit scopes and update persistent memory only when visual and audio evidence agree on the intended change.

**Why this is stronger**

- Pushes novelty into cross-modal edit binding, where close video-memory baselines are weaker.
- Keeps audio first-class.

**Main risk**

- Requires stronger scope labels or proxies.
- If the AV coupling signal is weak, it will read as a narrow extension.

### 4. Reversible Memory Commits

**Thesis**

Treat post-switch memory writes as provisional and roll them back if later evidence shows the change was transient or leaked outside scope.

**Why this is stronger**

- Sharpens the contribution from retention to memory governance.

**Main risk**

- May be over-engineered if rollback events are rare or hard to define.

### 5. Edit-Locality Benchmark + Objective

**Thesis**

Formalize long-horizon prompt-switch generation around locality: target change, collateral drift, cross-modal leakage, and recovery latency.

**Why this is stronger**

- Survives even if broad method novelty is partly collapsed.
- Gives a clean reviewer-facing target that current papers under-measure.

**Main risk**

- Benchmark-centric papers are often judged as less ambitious unless the protocol is clearly indispensable.

## Recommended Narrowing

If the goal is the best novelty-to-effort ratio, the idea should be narrowed to:

### Typed Memory Mutation Under Prompt Switches

**Paper-level claim**

Streaming joint AV generation fails not because it cannot retain history, but because it lacks explicit control over which visual, auditory, and binding states may mutate under prompt switches.

**Mechanism**

- typed edit scope,
- counterfactual continuation target,
- typed mutation of persistent state,
- explicit AV preservation-edit metrics.

**Evaluation**

- compare against recache-only,
- compare against scalar global decay,
- compare against anchor-guided streaming baselines,
- report the preservation-edit frontier instead of isolated consistency numbers,
- include non-audio edits that should preserve audio state,
- include audio edits that should preserve visual state.

## Frozen Version

### Problem Anchor

Interactive streaming joint AV generation fails because current systems do not model `auditory state` and `cross-modal binding` as persistent, selectively mutable state variables.

### Method Thesis

Use counterfactual edit isolation with cross-modal typed persistent memory so prompt switches update only the intended visual, auditory, and binding states.

### Dominant Contribution

Not better generic memory retention, but **typed cross-modal state mutation under intervention**, with auditory memory as a first-class component.
4. add preservation-vs-edit metrics and a first switch benchmark.
