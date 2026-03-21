# Experiment Plan

## Claims

### Claim 1
Real `.precomputed` ODE-regression data is sufficient to replace the current synthetic self-forcing smoke path.

### Claim 2
A persistent SSM memory state is a plausible long-horizon memory carrier for streaming AV generation on top of block-causal local attention.

### Claim 3
Prompt-switch manifests and schedule-aware conditions can be layered on top of the existing data path without refactoring the entire LTX pipeline stack.

## Run Order

1. Build a tiny Stage-2 ODE-regression smoke dataset (`limit=8`) plus prompt-switch manifest.
2. Run 1-GPU self-forcing smoke for 2 steps and verify:
   - model load
   - `.precomputed` discovery
   - chunking/positions
   - optimizer step/checkpoint path
3. Add replay-mode streaming smoke or equivalent non-fake evaluation stub.
4. Add schedule-aware condition keys for switch supervision.
5. Launch the first real switch-training ablation.

## Required Artifacts

- `ode/data_distilled_stage2_ode_smoke/.precomputed/*`
- `ode/switch_episodes_smoke.jsonl`
- `outputs/self_forcing_phase1/*`
- `AUTO_REVIEW.md`

## Metrics / Checks

- Smoke success: job completes, sample loads, loss is finite, checkpoint saves.
- Data integrity: no dropped audio remainder during chunking.
- Switch data sanity: each episode has category changes across segments.
- Review readiness: remaining blockers are written down explicitly after each round.

## Non-goals For This Iteration

- Full minute-scale generation quality claims.
- Full streaming decode stack rewrite.
- Paid/high-priority compute.
