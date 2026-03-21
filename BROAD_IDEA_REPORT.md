# Broad Idea Report

**Direction**: Streaming, interactive, joint audio-video generation  
**Date**: 2026-03-20  
**Scope**: Alternative research ideas beyond generic persistent memory

## Landscape Summary

As of 2026-03-20, the field is already crowded in four nearby lanes:

- **Streaming / causalization / real-time systems**:
  [LongLive](https://arxiv.org/abs/2509.22622),
  [Anchor Forcing](https://arxiv.org/abs/2603.13405),
  [OmniForcing](https://arxiv.org/abs/2603.11647)
- **Memory retrieval / long-context consistency**:
  [MemFlow](https://arxiv.org/abs/2512.14699),
  [Context as Memory](https://arxiv.org/abs/2506.03141),
  [MemRoPE](https://arxiv.org/abs/2603.12513)
- **Multi-turn editing with memory**:
  [Memory-V2V](https://arxiv.org/abs/2601.16296)
- **Joint AV identity / personalization**:
  [Identity as Presence](https://arxiv.org/abs/2603.17889)

This means broad ideas like:

- "better streaming AV generation,"
- "better memory for long video,"
- "better interactive prompt switching,"
- "better AV identity consistency"

are weak by default unless narrowed aggressively.

## Recommended Ideas

### 1. Memory Governance for Interactive AV

**Thesis**

The central problem is not retaining more history, but governing the memory lifecycle: what to write, what to preserve, what to forget, and what evidence each memory write came from.

**Why it matters**

Recent work mostly optimizes retention and reuse. Memory hygiene, selective forgetting, and provenance remain under-served.

**Closest pressure**

- Anchor Forcing
- MemFlow
- MemRoPE

**Contribution type**

Method + evaluation

**Main risk**

Can drift into systems engineering unless governance decisions are made explicit and measurable.

### 2. Counterfactual Edit Memory

**Thesis**

Maintain a frozen "would-have-continued" branch and only accept edited memory writes if they improve the target edit without increasing off-target drift.

**Why it matters**

This turns non-target preservation into a causal comparison instead of a weak consistency metric.

**Closest pressure**

- Anchor Forcing
- MemFlow
- Memory-V2V

**Contribution type**

Method

**Main risk**

If the counterfactual branch behaves like a regularizer only, the paper will look incremental.

### 3. Branch-Consistent Interactive Generation

**Thesis**

Treat interactive generation as a branching history problem with fork / undo / redo, not a single linear prompt stream.

**Why it matters**

Current interactive methods are mostly linear. Creative workflows are not.

**Closest pressure**

- Memory-V2V

**Contribution type**

Problem setting + method

**Main risk**

Requires a convincing protocol and benchmark, otherwise it can look like product framing.

### 4. Persistent Social / Character State in Joint AV

**Thesis**

Track not just who appears and sounds how, but long-horizon turn-taking, relationships, role assignments, and conversational continuity across scene changes.

**Why it matters**

OmniForcing and Identity-as-Presence open the joint-AV lane, but persistent social state is still weakly handled.

**Closest pressure**

- OmniForcing
- Identity as Presence

**Contribution type**

Method + benchmark

**Main risk**

Will collapse into personalization unless the “social state” component is explicit.

### 5. Deferred Commitments / Story-State Memory

**Thesis**

Explicitly track unresolved commitments, goals, and pending events, then measure whether later generation pays them off.

**Why it matters**

Most current memory methods retrieve past cues; few model future obligations.

**Closest pressure**

- RELIC
- Context as Memory

**Contribution type**

Problem formulation + method

**Main risk**

Needs a clean operational definition of commitment and payoff.

### 6. Edit-Locality Frontier for Streaming Joint AV

**Thesis**

Define the task around the frontier between `Edit Success` and `Non-Target Preservation`, plus cross-modal leakage and recovery latency.

**Why it matters**

Current papers still under-measure collateral drift under interaction, especially in joint AV.

**Closest pressure**

- Anchor Forcing
- LongLive
- Memory-V2V

**Contribution type**

Benchmark + objective

**Main risk**

Benchmark-heavy contributions need strong protocols to feel indispensable.

### 7. Event-Centric Audio Control in Streaming AV

**Thesis**

Model audio not only as sync and voice identity, but as a stream of persistent and editable event commitments: ambience, sound sources, causal sound effects, and foreground/background salience.

**Why it matters**

Most AV work still privileges video semantics. Audio is usually sync, voice, or realism, not editable event structure.

**Closest pressure**

- OmniForcing
- Identity as Presence

**Contribution type**

Method

**Main risk**

Needs strong audio-side evaluation or it will read as niche.

## Selected Direction To Carry Forward

After narrowing, the recommended direction to carry forward is:

### Counterfactual Edit Isolation with Cross-Modal Typed Persistent Memory

**Why this version**

- It preserves the strongest part of the original memory idea.
- It avoids competing on generic long-memory or generic streaming quality.
- It gives `audio` a real research role instead of treating it as sync or identity residue.

**Frozen problem statement**

Streaming joint AV generation still lacks a mechanism that treats `auditory state` and `cross-modal binding` as persistent state variables that can be selectively mutated under prompt switches.

**Frozen mechanism sketch**

- maintain typed persistent state:
  - visual world state
  - auditory world state
  - cross-modal binding state
- compare edited continuation against a counterfactual no-edit continuation
- commit changes only to the state classes the edit is meant to touch

**Why this is better than broad ESPM**

- stronger novelty axis: `edit isolation` instead of generic memory
- stronger AV story: `auditory state` is first-class
- stronger evaluation target: preservation-edit frontier

## Best Ideas By Goal

### Best novelty wedge

1. Memory Governance for Interactive AV
2. Branch-Consistent Interactive Generation
3. Deferred Commitments / Story-State Memory

### Best novelty-to-feasibility ratio

1. Counterfactual Edit Memory
2. Edit-Locality Frontier for Streaming Joint AV
3. Memory Governance for Interactive AV

### Best fit to current `myltx-v1` stack

1. Edit-Locality Frontier for Streaming Joint AV
2. Counterfactual Edit Memory
3. Memory Governance via typed write / decay / skip-write policies

## Directions To Avoid

- Generic "better long memory for AV generation"
- Generic "better streaming joint AV generation"
- Generic "better identity consistency in AV generation"
- Generic "banked persistent memory" without a sharper task definition

## Recommended Execution Order

If the goal is a publishable idea with the current stack, start with:

1. **Counterfactual Edit Isolation + typed memory mutation**
2. **Edit-Locality Frontier for Streaming Joint AV**
3. **Memory Governance / selective forgetting**

If the goal is a more ambitious longer-horizon bet, explore:

1. **Branch-consistent interaction**
2. **Persistent social state**
3. **Deferred commitments / story-state memory**
