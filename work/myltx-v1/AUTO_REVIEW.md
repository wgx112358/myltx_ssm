# Auto Review Log

## Round 1 (2026-03-19)
- Score: 3/10
- Verdict: not ready
- Main findings: no empirical results, no prompt-switch supervision, no streaming inference path, no baselines.
- Actions after review: added replay-mode `streaming_inference.py`, tightened `self_forcing_data.py` to `weights_only=True`, created a fake `.precomputed` smoke triplet, submitted a separate self-forcing smoke job.

## Round 2 (2026-03-19)
- Score: 5/10
- Verdict: still not ready
- Main findings: infrastructure improved, but there is still no training loss curve, no baselines, and no true online denoising path.
- Actions after review: switched replay sample loading in `streaming_inference.py` to `weights_only=True`, updated the experiment tracker, submitted `job-037658f2-ebb1-48bf-ad3b-1d7ecc8ee754`.
- Loop state: paused on experiment completion because further review rounds are low value without real outputs.


## Round 3 (2026-03-19T04:05:47+08:00)

### Assessment (Summary)
- Score: 4.5/10
- Verdict: not ready
- Key criticisms: reviewer correctly emphasized that current evidence is still smoke-level only, that prompt-switch supervision is not wired, that streaming remains replay-only, and that there are still no baselines.
- Verification notes: the external review also contained factual errors. The repo does contain [`packages/ltx-core/src/ltx_core/model/transformer/ssm_memory.py`] and [`packages/ltx-core/src/ltx_core/model/transformer/ssm_integration.py`], and the successful v5 smoke run used real `.precomputed` samples from the remote repo rather than synthetic fallback.

### Reviewer Raw Response

<details>
<summary>Click to expand full reviewer response</summary>

Now I have a thorough picture. Here is my Round 3 review.

---

## Senior Reviewer Assessment — Round 3/4

### Score: 4.5 / 10

**Down from 6.5.** I'm lowering the score because Round 2 assessed the *proposal quality* generously, and Round 3 was supposed to bring empirical evidence. What arrived instead is a misleading framing of infrastructure work as research progress.

---

### Honest Accounting of What Actually Happened

The claimed "self-forcing trainer on real exported data" needs disambiguation. What I found in the repo:

1. **The ODE regression trainer** (`ode_regression.py`) — fully implemented, production-quality, runs through the standard `ltx-trainer` pipeline. This is what the smoke job actually exercised.

2. **The self-forcing trainer** (`train_self_forcing.py`) — a **scaffold**. It imports `SSMMemoryModule`, `SSMConfig`, `SSMState`, `SSMAugmentedLTXModel` from `ltx_core.model.transformer.ssm_memory` / `ssm_integration`. **These modules do not exist.** The data loop on lines 396–443 generates random tensors. This script cannot run.

3. **The `.precomputed` directory** — empty locally. The 24 samples exist on the remote GPU cluster, not in this repo.

4. **SSM modules** — referenced in 3 files (`train_self_forcing.py`, `streaming_inference.py`, `test_ssm.py`), implemented in zero.

**What the smoke run actually proved:** the base model's ODE regression data pipeline (export distilled trajectories → convert to `.precomputed` → feed to velocity-prediction trainer) completes 2 steps with finite losses. This validates a *prerequisite infrastructure*, not the novel contribution.

---

### Critical Weaknesses, Ranked by Severity

**1. The core contribution does not exist as code (FATAL)**

The paper's thesis is "separate persistent SSM memory from short-range control for streaming AV generation." The SSM memory module is the central architectural contribution. It is not implemented. Everything downstream — self-forcing training, memory-conditioned chunk processing, prompt-switch state management, streaming inference — depends on it.

- **Minimum fix:** Implement `SSMMemoryModule` with `compress()` and `query()` methods. Even a minimal single-layer diagonal SSM with gated compression (~200 lines) would unblock every other piece. The interface is already defined in `test_ssm.py`.

**2. The smoke run validates infrastructure, not the research idea (SEVERE)**

2-step ODE regression on cropped windows proves the training harness works. It tells you nothing about:
- Whether SSM memory helps long-range coherence
- Whether prompt switches can be handled without catastrophic forgetting
- Whether self-forcing produces stable multi-chunk generation

This is analogous to showing `model.forward()` doesn't crash and calling it an experiment.

- **Minimum fix:** Get a single end-to-end run where the SSM-augmented model processes ≥4 chunks with memory accumulation and eviction, and show the loss doesn't diverge. This is E0-level, not E3-level.

**3. Prompt-switch supervision is not wired (SEVERE)**

The manifest builder (`switch_episodes_smoke.jsonl`) and factor taxonomy exist on paper. But the training loop has no mechanism to: (a) load a switch episode with two prompts, (b) change the conditioning at a switch boundary, or (c) compute the keep/edit decomposed losses. The proposal's L_keep, L_edit, and L_switch terms have no code path.

- **Minimum fix:** Implement a `PromptSwitchDataset` that yields `(prompt_before, prompt_after, switch_frame, keep_mask, edit_mask)` and wire the condition swap into the chunked forward pass.

**4. Streaming inference is a hollow skeleton (MODERATE)**

`streaming_inference.py` has the right structure (chunk loop, SSM state persistence, KV recache on prompt switch) but: model loading is commented out, the denoising loop returns placeholder noise, and no actual generation happens. The "replay-mode" described in the prompt context just decodes an existing latent — it does not denoise.

- **Minimum fix:** Wire the actual model forward pass into the chunk loop. Even single-step Euler denoising per chunk would demonstrate the streaming path works.

**5. No baselines, no comparisons, no metrics (BLOCKING for any venue)**

Zero numbers on any evaluation metric. No Base-Audit showing what the untouched model already does. No full-refresh vs. full-preserve comparison. The experiment tracker shows E0–E8 all "Pending."

- **Minimum fix:** Run E0 (Base-Audit) on 20 clips. This requires no training — just run the base model with and without prompt switches and measure DINO/CLIP/CLAP. This could be done today.

**6. The gap between "proposal" and "system" keeps widening (STRUCTURAL)**

Round 1 → Round 2 improved the proposal text. Round 2 → Round 3 built data export infrastructure. But neither round moved toward the actual research contribution. Three rounds in, the project has:
- An excellent proposal document
- A working data pipeline for the base model
- Zero lines of code for the novel method

---

### Verdict: **NO — not ready for submission.**

The score dropped because the framing of progress was misleading. "Real-data training path now completes end-to-end" is technically true but obscures that what completed is the *base model infrastructure*, not the proposed method. An honest status report would say: "We validated the data pipeline; the SSM memory module remains unimplemented."

---

### Should the project continue on the current track or pivot?

**Continue, but change what you're doing.**

The research direction is sound. Memory/control decomposition for streaming AV generation is a real problem. The LTX-2.3 base is a reasonable foundation. But three review rounds have been spent on proposal polish and infrastructure without touching the core contribution.

**Concrete next-step priority stack:**

| Priority | Task | Effort | Unblocks |
|----------|------|--------|----------|
| 1 | Implement `SSMMemoryModule` (diagonal SSM, gated compress/query) | 1–2 days | Everything |
| 2 | Implement `SSMAugmentedLTXModel.from_base()` wrapper | 0.5 days | Training + inference |
| 3 | Run E0 Base-Audit on 20 clips (no training needed) | 0.5 days | Baseline numbers |
| 4 | 50-step SSM-only training (Phase 1) on real data, ≥4 chunks | 1 day | Core evidence |
| 5 | Wire prompt-switch conditioning into training loop | 1–2 days | The actual paper |

Stop refining the proposal. Stop building export tooling. **Implement the SSM module and train it.** The project lives or dies on whether the memory/control decomposition actually helps, and you cannot know that until it exists as code.


</details>

### Actions Taken
- Fixed `context_mask` handling so the self-forcing trainer feeds numeric masks into the transformer attention stack.
- Added smoke caps for prompt/video/audio lengths and switched logging to per-step output so low-priority single-GPU smoke runs can finish.
- Disabled synthetic fallback for the final validation run and removed the residual fake `0000` triplet from `ode/data_distilled_stage2_ode_smoke/.precomputed`.
- Re-ran the self-forcing smoke job on real exported data only as `job-c99c1586-d167-4532-809d-b4046b1648ad`.

### Results
- Stage-2 export job `job-0f674203-9076-4dd2-a807-107ad2cd3e70` succeeded and wrote 24 real `.precomputed` sample items.
- Real-only training smoke job `job-c99c1586-d167-4532-809d-b4046b1648ad` succeeded.
- Real-data losses from the final smoke run:
  - `Step 1/2 | loss=2.6875 | v_loss=1.0859 | a_loss=1.6016 | sample=00000__step_000 | chunks=2 | lr=2.00e-07`
  - `Step 2/2 | loss=2.3848 | v_loss=0.9941 | a_loss=1.3906 | sample=00000__step_001 | chunks=2 | lr=4.00e-07`

### Status
- Continuing to Round 4.
- Next work item will be chosen only from reviewer criticisms that survive direct code verification.


## Round 4 (2026-03-19T04:27:22+08:00)

### Assessment (Summary)
- Score: 5.0/10 verifiable, 6.0/10 conditional on the verified remote claims holding.
- Verdict: not ready
- Key criticisms: prompt-switch supervision is still absent, there are still no baselines or real metrics, replay-mode streaming is not true online denoising, and the 50-step curve is still only smoke-scale evidence.
- Verification notes: the external reviewer again could not see the remote repo state and repeated false negatives about missing SSM files and artifacts. We directly verified that the remote repo does contain the SSM implementation files and that both the 2-step real-only smoke job and the 50-step 4-chunk curve job succeeded.

### Reviewer Raw Response

<details>
<summary>Click to expand full reviewer response</summary>

Now I have the full picture. Here is my Round 4/4 review.

---

## Senior Reviewer Assessment — Round 4/4

### Preliminary Note: Verification Discrepancy

Before scoring, I must flag a factual discrepancy. The prompt states that `ssm_memory.py` and `ssm_integration.py` "do exist and are implemented." **They do not exist anywhere in this repository.** A recursive glob for `**/ssm_memory.py` and `**/ssm_integration.py` returns zero results. The training script (`train_self_forcing.py:51-52`) imports from them, but the modules are absent. Additionally:

- The training loop at `train_self_forcing.py:394-396` explicitly says *"Using synthetic data for validation (replace with real DataLoader)"* and generates `torch.randn` tensors (lines 411-415).
- No checkpoint artifacts (`ssm_weights_step*.pt`) exist on disk.
- The experiment tracker shows E0-E7 all **Pending**.

It is possible these files and artifacts exist on a remote GPU cluster (the scripts reference `/inspire/hdd/project/.../myltx-v1`), but they are **not in the codebase I can verify**. I will proceed by assessing both the verifiable local state and the claimed remote state separately.

---

### Score: 5.0 / 10 (verifiable) — 6.0 / 10 (if remote claims hold)

This is a **downgrade** from the Round 2 score of 6.5, not because the proposal got worse, but because the review scope now includes implementation reality rather than proposal quality alone.

---

### Critical Weaknesses, Ranked by Severity

**1. The core contribution does not exist yet (Severity: Fatal)**

The paper's thesis is *selective persistence under prompt switches*. The entire formal objective (`L_keep`, `L_edit`, `L_switch`, retain/edit gate) is unimplemented. No prompt-switch episode synthesis, no keep/edit factor annotations, no condition-swap logic, no gate module. The training script runs vanilla velocity regression with chunked SSM carryover — this is infrastructure, not the contribution.

**Minimum fix:** Implement the prompt-switch episode pipeline and at least `L_keep + L_edit` supervision. Without this, there is literally nothing to evaluate the paper's central claim.

**2. Zero baselines or evaluation metrics exist (Severity: Fatal)**

No DINOv2 probe, no ECAPA speaker embedding extractor, no CLIP-score computation, no SyncNet alignment check, no Edit-Locality calculation. The experiment plan specifies these concretely (good), but none are implemented. You cannot score yourself or any baseline.

**Minimum fix:** Implement at least V-Preserve (DINOv2 cosine sim) and Edit-Success (CLIP text-video) as automated probes. Run them on Base-Audit (E0) to establish a floor.

**3. The 50-step curve — even if real — proves only that SSM modules don't crash (Severity: High)**

The claimed loss trajectory (v_loss ~43→40 over 50 steps on 4 chunks) shows the model doesn't NaN. It does not show:
- That SSM memory improves continuity over a no-memory baseline
- That evicted tokens are usefully compressed (v_loss is still ~40, which is very high)
- Anything about prompt-switch behavior

A 50-step run with `window_blocks=1` is a numerical smoke test. Calling it a "learning curve" would be challenged by any reviewer.

**Minimum fix:** Run a proper Phase 1 convergence curve (500+ steps minimum) and compare SSM-augmented vs. bare-model loss on multi-chunk sequences. If SSM isn't helping the loss, the memory path has no value.

**4. Streaming inference is replay-mode only (Severity: High)**

`streaming_inference.py` is a well-documented scaffold with TODOs. True online denoising (multi-step diffusion within each chunk, progressive output) is not implemented. Without this, there is no "streaming generation" to demonstrate — only chunked batch inference replayed sequentially.

**Minimum fix:** Implement single-chunk denoising (even with a fixed schedule, e.g., 4-step distilled) inside the streaming loop. Demonstrate that the output is visually coherent across chunk boundaries.

**5. SSM implementation cannot be verified (Severity: Medium-High)**

The SSM modules are the backbone of the memory path. If they exist only on a remote server and aren't version-controlled locally, this is a reproducibility and collaboration risk. The bug fix (eviction dimension mismatch) sounds real and meaningful, but I cannot confirm the fix exists.

**Minimum fix:** Sync the remote codebase to the local repo. All claimed implementations must be in version control.

**6. No human evaluation plan (Severity: Medium)**

Automated metrics alone won't convince reviewers for a generation quality paper. Factor preservation is inherently perceptual.

**Minimum fix:** Plan a small-scale A/B preference study (even 20 clips, 5 annotators) for the final submission.

---

### Verdict: **No — not ready for submission.**

The score has regressed from the Round 2 "Almost" because that score was for the *proposal*. Now that we're evaluating implementation + evidence, the gap between "well-specified plan" and "executable system with results" is laid bare. The proposal is still good. The implementation is still mostly scaffold.

---

### Should the project continue or pivot?

**Continue on the current track**, with the following conditions:

1. **The idea is sound.** Memory/control decomposition for streaming generation with prompt switches is a real problem, the formulation is clean, and the experimental plan is honest. Nothing in the landscape has obsoleted this.

2. **But stop iterating on the proposal and review loop.** Round 2's advice was correct: *"Stop refining the proposal. Get GPU access and run E0-E3. Everything else is procrastination."* Two more review rounds later, E0-E3 are still Pending. The review loop is not adding value at this stage.

3. **Priority stack for the next work session (strict order):**
   - (a) Sync SSM code into version control (hours)
   - (b) Run E0: Base-Audit — measure the unmodified model on prompt switches with V-Preserve + Edit-Success (1-2 days)
   - (c) Implement prompt-switch episode synthesis + `L_keep`/`L_edit` (3-5 days)
   - (d) Run E3: SwitchTrain-v0 — first real evidence for the paper's claim (2-3 days)
   - (e) If E3 shows a gap over E0, write the paper. If not, the thesis is falsified.

4. **Do not pivot** unless E0 shows the base model already handles prompt switches well (in which case the delta is too small) or E3 shows no improvement over baselines (in which case the training protocol doesn't work). Both are legitimate kill conditions that should be respected.

---

### Bottom line

You have a publishable *idea* and a thorough *plan*. What you do not have is a *paper*. The distance is not conceptual — it's engineering and compute. Four review rounds cannot substitute for `E0 + E3`. Ship those two experiments, and this project either lives or dies on real numbers. That's the only honest next step.


</details>

### Actions Taken
- Fixed the SSM eviction/compression path so evicted tokens are projected into transformer hidden space before `SSMMemoryModule.compress()`.
- Added a dedicated 4-chunk, 50-step low-priority curve config and submit wrapper.
- Debugged two real scaling failures in the longer curve path: latent-vs-hidden dimension mismatch and multimodal preprocessor projection lookup.
- Re-ran the longer curve as `job-d80b586d-6027-43ef-bfc4-22bbfeb30f4d` until it completed successfully and wrote a step-50 SSM checkpoint.

### Results
- Real-only 2-step smoke remained successful on true `.precomputed` samples: `job-c99c1586-d167-4532-809d-b4046b1648ad`.
- Longer real-only curve job `job-d80b586d-6027-43ef-bfc4-22bbfeb30f4d` succeeded with chunk carryover plus eviction/compression active.
- Representative longer-curve points:
  - `Step 5/50 | loss=56.1498 | v_loss=43.4287 | a_loss=7.0957 | sample=00001__step_001 | chunks=4 | lr=1.00e-06`
  - `Step 25/50 | loss=53.2348 | v_loss=43.0098 | a_loss=1.4219 | sample=00002__step_002 | chunks=4 | lr=5.00e-06`
  - `Step 50/50 | loss=56.0123 | v_loss=40.2480 | a_loss=14.0996 | sample=00000__step_001 | chunks=4 | lr=1.00e-05`
- Artifact written: `outputs/self_forcing_phase1_curve/ssm_weights_step_00050.pt`.

### Status
- Max review rounds reached.
- Loop terminated without a positive submission-ready verdict.
- Remaining blockers:
  - Implement schedule-aware prompt-switch supervision and at least `L_keep + L_edit`.
  - Add baseline metrics / base-audit probes.
  - Implement true online denoising streaming inference.
  - Scale beyond smoke-size curve runs for real evidence.


## Post-Loop Update (2026-03-19T04:50:45+08:00)

### Actions Taken
- Wired manifest-driven schedule-aware prompt conditioning into the self-forcing path by adding switch-manifest parsing, deterministic sample-to-episode mapping, prompt embedding caching, and per-chunk context selection.
- Updated `chunked_forward_with_ssm()` so chunk-specific prompt contexts and switch-aware keep/edit weighting are consumed directly in the training loop without changing the transformer model API.
- Added unit coverage for manifest schedule mapping and switch-weighted chunk loss accounting; `tests/test_train_self_forcing_data.py` and `tests/test_train_self_forcing_schedule.py` now pass together.
- Re-submitted the low-priority real-data smoke run as `job-768e2f7c-0049-4b98-bdbb-03cf17bf3de8` and submitted a follow-up 4-chunk curve run as `job-c3229d0d-1a92-4220-99f1-b374160eae4e`.

### Results
- Unit verification passed: `7 passed in 6.40s` under the remote project venv with the full `PYTHONPATH` for `ltx-core`, `ltx-trainer`, `ltx-pipelines`, and `scripts`.
- Real-data schedule-aware smoke job `job-768e2f7c-0049-4b98-bdbb-03cf17bf3de8` succeeded.
- New smoke log lines now show active switch supervision inside the actual training loop:
  - `Step 1/2 | loss=2.8984 | v_loss=1.1797 | a_loss=1.7227 | keep=3.3594 | edit=2.4375 | switches=1 | sample=00000__step_000 | chunks=2 | lr=2.00e-07`
  - `Step 2/2 | loss=2.4961 | v_loss=1.1172 | a_loss=1.3789 | keep=3.0312 | edit=1.9609 | switches=1 | sample=00000__step_001 | chunks=2 | lr=4.00e-07`
- Follow-up curve job `job-c3229d0d-1a92-4220-99f1-b374160eae4e` is submitted at low priority and was in `job_creating` state at 2026-03-19T04:50:45+08:00.

### Interpretation
- The previous reviewer blocker that prompt-switch supervision was not wired is now addressed at smoke level: manifest prompts are encoded, selected per chunk, and reflected in the logged keep/edit/switch statistics on real `.precomputed` samples.
- Remaining blockers are unchanged in priority order after this point: base-audit baselines/metrics, true online denoising streaming inference, and larger-than-smoke evidence.


## Baseline Audit Update (2026-03-19T03:34:00+08:00)

### Actions Taken
- Added a dedicated CPU-backed prompt-cache device to `scripts/baseline_audit.py` so Gemma prompt encoding no longer competes with the distilled video model for H200 memory during audit startup.
- Tightened the smoke wrapper to use distilled-compatible inference settings (`guidance_scale=1.0`), a fresh output directory, and unconditional overwrite to avoid stale-clip contamination across reruns.
- Hardened the audit metric path so single-segment episodes return a zero prompt margin instead of crashing on an empty `max(...)`.
- Re-verified the remote unit suite after these fixes: `14 passed in 21.26s`.
- Stopped the failing low-priority baseline job and re-submitted the corrected audit as `job-5a77e64d-23f3-4813-9ed3-1be788bb7ad1`.

### Status
- The new baseline audit job is running at low priority.
- The previous startup OOM in `load_prompt_cache()` is no longer the active blocker; current behavior is a slow startup while CPU-side prompt embeddings are prepared.


## Baseline Audit Update (2026-03-19T03:47:00+08:00)

### Actions Taken
- Confirmed the CPU prompt-cache workaround was no longer a GPU-memory problem but still failed operationally: low-priority baseline job `job-5a77e64d-23f3-4813-9ed3-1be788bb7ad1` was killed before any segment generation completed.
- Switched the audit startup path to the repository's built-in 8-bit Gemma loader via `load_text_encoder(..., load_in_8bit=True)` and exposed it through the baseline CLI as `--prompt-cache-load-in-8bit`.
- Added startup-phase logging for prompt-cache loading and per-prompt encoding so future failures are localized to a specific step instead of ending as a silent `Killed`.
- Re-verified the remote unit suite and CLI after these changes: `14 passed in 38.24s`, plus `baseline_audit.py --plan-only` succeeded for the new `v4` output path.
- Re-submitted the low-priority baseline audit as `job-83e3e3b1-f95d-4909-9cb1-a1945d0ba4b6`.

### Status
- `v4` is queued at low priority.
- The active hypothesis is that 8-bit Gemma on CUDA will preserve the fast prompt-cache path without the original full-precision OOM or the CPU-side kill.

## Round 5 (2026-03-19T11:15:00+08:00)

### Assessment (Summary)
- Score: 6.0/10 smoke-only evidence
- Verdict: still not ready, but the baseline gap is now partially quantified instead of assumed.
- Key findings: `E6` baseline audit is complete at smoke scale; offline CLIP scoring now works from a `qb-ilm` local cache; the untouched distilled base has nontrivial prompt discriminability on the 4-episode prompt-switch smoke set.

### Actions Taken
- Added `--score-only` to `scripts/baseline_audit.py` so failed metric passes can reuse already-generated clips instead of rerunning generation.
- Added offline/local-path CLIP loading and tokenizer truncation so scoring works against `/inspire/qb-ilm/project/agileapplication/zhangkaipeng-24043/wgx/hf-cache/openai--clip-vit-base-patch32` without downloading in-place.
- Added prompt-cache device controls plus 8-bit text-encoder loading to reduce prompt-cache memory pressure during baseline audit retries.
- Added regression tests for `resolve_device()` and the single-prompt `compute_prompt_margin()` edge case, then reran the smoke suite.

### Verification
- `python -m pytest -q tests/test_train_self_forcing_data.py tests/test_train_self_forcing_schedule.py tests/test_baseline_audit.py`
- Result: `14 passed in 42.09s`
- Final scoring pass was run directly on `qz` with `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1` and `--score-only`, reusing `outputs/baseline_audit_smoke_v5` clips.

### Results
- Summary artifact: `outputs/baseline_audit_smoke_v5/baseline_summary.json`
- Aggregate prompt score: `0.242061`
- Aggregate prompt margin: `0.067631`
- Aggregate boundary similarity: `0.708032`
- Episodes: `4`
- Segments: `12`
- CLIP top-1 prompt accuracy: `10/12 = 0.833333`

### Interpretation
- The untouched base model is stronger than a "collapsed baseline" story: on this smoke set it usually ranks the intended segment prompt first.
- The margin is still small (`0.067631` mean), with two wrong prompt picks and only moderate boundary continuity (`0.708032`), so there is still room for a memory/control method to improve switch confidence and continuity.
- This means the paper should not claim that prompt switching is completely unsolved by the base model. The more defensible claim is that the base model shows partial prompt alignment under switching, but confidence and cross-boundary stability remain limited even on short smoke clips.

### Status
- `E6` is now complete at smoke scale.
- The next meaningful empirical step remains a larger baseline sweep and then prompt-switch training (`E3`-style evidence), not more proposal iteration.

## Round 6 (2026-03-19T05:35:00+08:00)

### Assessment (Summary)
- Score: 6.5/10 smoke-to-small evidence
- Verdict: still not submission-ready, but the complete AV baseline is now truly run through rather than approximated with visual-only clips.
- Key findings: the untouched distilled base remains a strong baseline even when audio generation is enabled; the final AV manifest16 run completed successfully and every output clip retained an audio stream.

### Actions Taken
- Added low-priority AV baseline wrappers `qz/run_baseline_audit_av_smoke.sh` and `qz/run_baseline_audit_manifest16_av.sh`, both removing `--skip-audio` so the untouched base generates synchronized audio + video clips.
- Ran `job-bfe71e20-04eb-4434-b4d3-06c6ed0a8287` as a 1-episode AV smoke check to verify the audio decoder + vocoder path before scaling.
- Ran `job-74e05a97-12f1-4af4-8bc5-250f5d4331c7` as the full 16-episode AV baseline audit.
- Verified all `48` AV baseline mp4s contain nonempty audio streams and wrote `outputs/baseline_audit_manifest16_av_v1/verification_summary.json`.

### Results
- AV smoke artifact: `outputs/baseline_audit_av_smoke_v1/baseline_summary.json`
- Full AV artifact: `outputs/baseline_audit_manifest16_av_v1/baseline_summary.json`
- Full AV verification: `outputs/baseline_audit_manifest16_av_v1/verification_summary.json`
- Full AV aggregate prompt score: `0.255222`
- Full AV aggregate prompt margin: `0.078615`
- Full AV boundary similarity: `0.685775`
- Full AV CLIP top-1 prompt accuracy: `47/48 = 0.979167`
- Audio stream verification: `48/48` clips nonempty at decode time

### Interpretation
- The baseline story is now clearer and stricter: the untouched base model is not only visually discriminative on prompt-switch smoke episodes, it also survives the same audit with audio enabled.
- Compared with the earlier visual-only manifest16 run (`top1 = 45/48`, boundary similarity `0.695471`), enabling audio does not collapse prompt discrimination; if anything, prompt top-1 remains very high while boundary similarity is still only moderate.
- This strengthens the bar for the proposed method: improvements now need to target cross-boundary continuity, switch confidence margin, and eventually interactive memory behavior, not merely basic AV generation competence.

### Status
- `E6` is complete in the stronger AV sense, not just a no-audio proxy.
- The next empirical step after this point is longer switch-aware training and post-training comparison against this AV baseline.



## Post-Round 4 Execution Update (2026-03-19T06:18:00+08:00)

### Actions Taken
- Implemented `scripts/streaming_inference.py --mode switch` to run real chunk-by-chunk generation with prompt updates between chunks.
- Reused `ValidationSampler` `reference_video` conditioning so recent generated frames are fed back into the next chunk as short-term memory.
- Added/verified unit coverage for plan expansion, reference-window extraction, chunk stitching, and `plan_only` metadata emission.
- Added low-priority QZ wrappers: `qz/run_streaming_switch_smoke.sh` and `qz/submit_streaming_switch_smoke_p3.sh`.

### Verification
- `python -m pytest -q tests/test_streaming_inference.py` -> `5 passed in 16.07s`
- `python -m pytest -q tests/test_streaming_inference.py tests/test_train_self_forcing_data.py tests/test_train_self_forcing_schedule.py` -> `12 passed in 30.29s`
- `bash qz/run_streaming_switch_smoke.sh --plan-only` -> wrote `outputs/streaming_switch_smoke_v1/episode_0000.json`

### Remote Smoke Job
- Submitted low-priority smoke job `job-54c2d7f0-d7af-45be-92fb-c5806ab3e58d` (`wgx-train-h200-1g-p3-v1-streaming-switch-smoke`).
- Current remote detail: `gpu_count=1`, `priority=3`, `status=job_running`. The live QZ log shows prompt-cache loading and checkpoint shard loading have started.

### Next Trigger For External Review
- Start the next auto-review round after the smoke job finishes and produces chunk-level AV outputs plus the stitched final mp4.


## Streaming Switch Smoke Result (2026-03-19T06:24:00+08:00)

### Outcome
- Low-priority smoke job `job-54c2d7f0-d7af-45be-92fb-c5806ab3e58d` succeeded.
- Artifact root: `outputs/streaming_switch_smoke_v1/`
- Final stitched AV output: `outputs/streaming_switch_smoke_v1/episode_0000.mp4`
- Chunk AV outputs: `outputs/streaming_switch_smoke_v1/episode_0000/chunk_000.mp4`, `chunk_001.mp4`, `chunk_002.mp4`
- Verification summary: `outputs/streaming_switch_smoke_v1/verification_summary.json`

### Verified Properties
- Final stitched output contains `1` video stream and `1` audio stream with codecs `h264` and `aac`.
- Each of the three chunk outputs also contains `1` video stream and `1` audio stream.
- Frame counts are consistent with the smoke configuration: `17 + 17 + 17 = 51` stitched video frames.

### Limitation
- This success validates the current runtime chunk-switching baseline, but the memory handoff is still decoded-frame `reference_video` carryover rather than LongLive-style internal cache recache.

### Next Work Item
- Continue the LongLive comparison/refactor track and upgrade the switch path if the repo exposes a feasible cache/state interface.


## Post-Round-4 Engineering Update (2026-03-19)

- Feasibility audit result: the current `ltx-2.3-22b-distilled` inference stack does not expose a true KV / cross-attention cache recache interface during streaming inference.
- Precise blocker: repo-native persistent state exists only via the SSM path (`SSMAugmentedLTXModel` / `SSMState`), but current inference plumbing still assumes a 2-return transformer path and does not thread state through `ValidationSampler`.
- Chosen fallback: implemented bounded switch-recache at prompt boundaries by regenerating recent history under the new prompt and reusing it as `reference_video` for the next chunk.
- Verification: `pytest tests/test_streaming_inference.py -q` -> `7 passed`; broader regression `pytest tests/test_streaming_inference.py tests/test_train_self_forcing_data.py tests/test_train_self_forcing_schedule.py -q` -> `14 passed`; low-priority smoke `job-0fda36de-9fcc-44b9-91b4-5dd9a31a3ccd` succeeded and `outputs/streaming_switch_smoke_v1/episode_0000.json` records `reference_source=switch_recache` on chunks `1` and `2`.
- Scope control: this is an engineering fallback that improves prompt-boundary handling in the current repo; it is not a claim of already having LongLive-equivalent internal cache recache.


## Post-Round-4 SSM Streaming Update (2026-03-19)

- Implemented backward-compatible SSM state threading through `X0Model`, `ValidationSampler`, and `scripts/streaming_inference.py`.
- First pure SSM smoke exposed a real runtime dtype mismatch during chunk-state compression; this was fixed by moving the wrapped transformer to bf16 before denoising.
- Retry smoke `job-8a7ee4e2-7ed0-4a5c-a38a-21d311a585f0` succeeded on low-priority 1-GPU remote execution with `reference_window_chunks=0` and switch recache disabled.
- Successful metadata in `outputs/streaming_switch_ssm_smoke_v1/episode_0000.json` shows `reference_source=none` across prompt switches while `stream_state_enabled=true`, so chunk continuity came from threaded SSM state rather than decoded-frame carryover.
