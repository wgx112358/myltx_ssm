# LongLive Switch Manifest Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the minimum LongLive-style switch-data path needed for route-3 pure SSM training in the current LTX repo.

**Architecture:** Keep the current `.precomputed triplet + switch_episode_manifest` training interface unchanged. Extend the existing manifest builder so it can generate LongLive-style two-segment single-switch episodes while preserving the current category-based smoke-manifest mode.

**Tech Stack:** Python, JSONL, pytest, existing `scripts/build_switch_manifest.py` utilities.

---

### Task 1: Extend the existing manifest builder with a LongLive-style mode

**Files:**
- Modify: `scripts/build_switch_manifest.py`
- Test: `tests/test_build_switch_manifest.py`

- [x] **Step 1: Write failing tests for a two-segment single-switch episode builder**
- [x] **Step 2: Run the targeted tests to verify they fail**
- [x] **Step 3: Add a LongLive-style builder that emits one episode per prompt with a sampled switch time**
- [x] **Step 4: Keep the current category-based smoke builder compatible**
- [x] **Step 5: Add or extend the CLI so both modes can be selected explicitly**
- [x] **Step 6: Re-run the targeted tests to verify both modes pass**

### Task 2: Regression and artifact generation

**Files:**
- Modify: `refine-logs/ROUTE3_PURE_SSM_STREAMING_PLAN.md`
- Modify: `refine-logs/EXPERIMENT_TRACKER.md`

- [x] **Step 1: Run focused regression for `tests/test_build_switch_manifest.py`**
- [x] **Step 2: Generate one LongLive-style smoke manifest artifact from the current prompt source**
- [x] **Step 3: Record the new data path in the route-3 plan and tracker**

## Exit Condition

- The existing builder still supports the current category-based smoke-manifest workflow.
- A LongLive-style two-segment single-switch manifest mode exists.
- Focused tests for both builder modes pass.
- One smoke LongLive-style manifest artifact is generated and recorded.
