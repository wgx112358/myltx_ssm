# Official Two-Stage Persistent-SSM Backbone Preview Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Phase-1 Stage A backbone preview that replaces the current `persistent_ssm backbone smoke` runtime with an official distilled two-stage chunk runner plus a dedicated streaming orchestrator, while preserving the validated delayed-compression memory schedule.

**Architecture:** Keep the official distilled two-stage generation semantics inside a reusable chunk runner module and move chunk planning, delayed-compression queue ownership, artifact saving, and stitching into a standalone Stage A orchestrator. Route only the `USE_SSM_STREAMING=1` backbone smoke wrapper to the new preview path; keep `ValidationSampler`, `streaming_inference.py`, `no_memory`, `short_context`, `switch`, and `baseline_audit` out of scope for this phase.

**Tech Stack:** Python, PyTorch, `ltx_pipelines` distilled utilities, `ltx_core` SSM integration, existing Stage A manifest/wrapper scripts, pytest, QZ submission wrappers.

---

## Implementation Rules

- Follow `@superpowers:test-driven-development` for every behavior change.
- Follow `@superpowers:verification-before-completion` before claiming parity or success.
- If the preview path diverges from the current delayed-compression behavior, use `@superpowers:systematic-debugging` before changing the design.

## File Structure

- Create: `packages/ltx-pipelines/src/ltx_pipelines/distilled_streaming.py`
  Purpose: official distilled chunk-runner module with a single public `run_chunk(...)` boundary, stateful stage-2 denoiser threading, and compressor-compatible token outputs.
- Create: `scripts/official_2stage_backbone_orchestrator.py`
  Purpose: Stage A no-switch orchestrator that owns chunk planning, delayed-compression queue, chunk saving, final stitching, metadata, and observability.
- Create: `tests/test_distilled_streaming.py`
  Purpose: unit tests for the chunk-runner contract, official-small validation, stage-2 handoff, state threading, and compressor token shape.
- Create: `tests/test_official_2stage_backbone_orchestrator.py`
  Purpose: orchestration tests for delayed-compression queue behavior, detached snapshots, metadata layout, and chunk runner interaction.
- Modify: `qz/run_streaming_backbone_smoke.sh`
  Purpose: route only the `persistent_ssm` Stage A smoke branch to the new orchestrator preview path while leaving legacy `no_memory` and `short_context` branches unchanged.
- Modify: `qz/run_streaming_backbone_audit_smoke.sh`
  Purpose: point the default audit target to the preview output directory so Stage A smoke verification reads the right artifact.
- Modify: `qz/submit_streaming_backbone_smoke_p3.sh`
  Purpose: update experiment/version naming so the remote run is visibly a backbone preview, not a generic legacy smoke.
- Modify: `tests/test_qz_wrappers.py`
  Purpose: lock wrapper routing, output directory defaults, and audit defaults.
- Modify: `../../RESEARCH_LOG.md`
  Purpose: record implementation completion, smoke evidence, and rollback point after remote verification.

## Task 1: Lock the Official Chunk-Runner Contract

**Files:**
- Create: `tests/test_distilled_streaming.py`
- Create: `packages/ltx-pipelines/src/ltx_pipelines/distilled_streaming.py`

- [ ] **Step 1: Write the failing tests for the public runner contract**

```python
def test_chunk_config_accepts_only_phase1_small_geometry() -> None:
    config = OfficialDistilledChunkConfig(
        preset="small",
        distilled_checkpoint_path="/tmp/ltx.safetensors",
        gemma_root="/tmp/gemma",
        spatial_upsampler_path="/tmp/upsampler.safetensors",
        num_frames=41,
        frame_rate=8.0,
        prompt="test",
        seed=42,
    )
    assert config.preset == "small"


def test_chunk_config_rejects_non_8k_plus_1_frames() -> None:
    with pytest.raises(ValueError):
        OfficialDistilledChunkConfig(
            preset="small",
            distilled_checkpoint_path="/tmp/ltx.safetensors",
            gemma_root="/tmp/gemma",
            spatial_upsampler_path="/tmp/upsampler.safetensors",
            num_frames=40,
            frame_rate=8.0,
            prompt="test",
            seed=42,
        )


def test_runner_exposes_single_run_chunk_boundary() -> None:
    runner = OfficialDistilledChunkRunner(...)
    assert callable(runner.run_chunk)
```

- [ ] **Step 2: Run the targeted tests to verify they fail**

Run:

```bash
source .venv/bin/activate && python -m pytest tests/test_distilled_streaming.py -q
```

Expected:

- Fails with `ModuleNotFoundError`, `ImportError`, or missing symbol errors for `OfficialDistilledChunkConfig` / `OfficialDistilledChunkRunner`.

- [ ] **Step 3: Add the minimal module, dataclasses, and validation helpers**

```python
@dataclass
class OfficialDistilledChunkConfig:
    preset: str
    distilled_checkpoint_path: str
    gemma_root: str
    spatial_upsampler_path: str
    num_frames: int
    frame_rate: float
    prompt: str
    seed: int
    ssm_streaming_enabled: bool = False
    ssm_d_state: int = 64
    ssm_gate_bias: float = -2.0
    ssm_checkpoint_path: str = ""
```

Implementation notes:

- Keep the public boundary as `OfficialDistilledChunkRunner.run_chunk(...)`.
- Validate only the phase-1 local alias `small`, and resolve `1024x1536` internally from the preset.
- Reject any `num_frames` that do not satisfy `8*K+1`.
- Do not implement full generation yet; only define the contract and validation surface needed by the tests.

- [ ] **Step 4: Run the targeted tests to verify the contract passes**

Run:

```bash
source .venv/bin/activate && python -m pytest tests/test_distilled_streaming.py -q
```

Expected:

- PASS for the new contract/validation tests.

- [ ] **Step 5: Commit the contract-only checkpoint**

```bash
git add packages/ltx-pipelines/src/ltx_pipelines/distilled_streaming.py tests/test_distilled_streaming.py
git commit -m "feat: add official two-stage chunk runner contract"
```

## Task 2: Implement the Official Distilled Chunk Runner

**Files:**
- Modify: `packages/ltx-pipelines/src/ltx_pipelines/distilled_streaming.py`
- Modify: `tests/test_distilled_streaming.py`

- [ ] **Step 1: Write failing tests for distilled stage handoff and stage-2 SSM threading**

```python
def test_run_chunk_uses_exact_distilled_stage2_handoff(monkeypatch) -> None:
    result = runner.run_chunk(config, ssm_state=None)
    assert result.evictable_video_tokens.shape[-1] == 128
    assert result.final_chunk_video is not None


def test_run_chunk_threads_ssm_state_only_through_stage2(monkeypatch) -> None:
    result = runner.run_chunk(config, ssm_state="state0")
    assert result.next_ssm_state == "state1"
```

Test design requirements:

- Monkeypatch `ModelLedger`, `encode_prompts`, `denoise_audio_video`, and `upsample_video` so the test observes exact call arguments.
- Assert these distilled invariants explicitly:
  - stage 1 uses `DISTILLED_SIGMA_VALUES`
  - stage 2 uses `STAGE_2_DISTILLED_SIGMA_VALUES`
  - stage 2 receives `initial_video_latent=upscaled_video_latent`
  - stage 2 receives `initial_audio_latent=stage1_audio_latent`
  - stage 2 receives `noise_scale=stage_2_sigmas[0]`
- Assert the runner constructs heavy ledger/components once and reuses them across multiple `run_chunk(...)` calls.
- Assert `evictable_video_tokens` / `evictable_audio_tokens` are patchified stage-2 chunk tokens, not decoded media and not unpatchified VAE tensors.
- Assert the runner exposes `compress_evicted_tokens(...)` so the orchestrator can own delayed-compression queue eviction without reaching into private transformer fields.

- [ ] **Step 2: Run the targeted tests to verify they fail**

Run:

```bash
source .venv/bin/activate && python -m pytest tests/test_distilled_streaming.py -k "handoff or ssm or reuse" -q
```

Expected:

- FAIL because `run_chunk(...)` does not yet execute the official two-stage flow or return the required token outputs.

- [ ] **Step 3: Implement the runner internals with no generic sampler rewrites**

```python
class OfficialDistilledChunkRunner:
    def __init__(...):
        self.model_ledger = ModelLedger(...)
        self.pipeline_components = PipelineComponents(...)

    def run_chunk(self, config: OfficialDistilledChunkConfig, ssm_state=None) -> OfficialDistilledChunkResult:
        stage1_video_state, stage1_audio_state = self._run_stage1_chunk(config)
        return self._run_stage2_chunk(config, stage1_video_state, stage1_audio_state, ssm_state)

    def compress_evicted_tokens(self, ssm_state, evicted_video, evicted_audio):
        ...
```

Implementation requirements:

- Keep one public `run_chunk(...)` boundary.
- Do not modify `utils/samplers.py`.
- Add only a thin state-aware stage-2 denoiser wrapper that:
  - calls `X0Model(..., ssm_state=..., return_ssm_state=True)`
  - threads `next_ssm_state` across diffusion steps
  - does not compress/write long-term memory inside the Euler loop
- Keep stage 1 chunk-local with no cross-chunk memory path in phase 1.
- Return:
  - `final_chunk_video`
  - `final_chunk_audio`
  - `next_ssm_state`
  - `evictable_video_tokens`
  - `evictable_audio_tokens`
- Expose a runner-owned `compress_evicted_tokens(...)` method that delegates to the wrapped SSM transformer.

- [ ] **Step 4: Re-run the targeted runner tests**

Run:

```bash
source .venv/bin/activate && python -m pytest tests/test_distilled_streaming.py -q
```

Expected:

- PASS for contract, distilled handoff, reuse, and state-threading tests.

- [ ] **Step 5: Commit the runner implementation**

```bash
git add packages/ltx-pipelines/src/ltx_pipelines/distilled_streaming.py tests/test_distilled_streaming.py
git commit -m "feat: add official distilled chunk runner"
```

## Task 3: Build the Stage A Backbone Orchestrator

**Files:**
- Create: `scripts/official_2stage_backbone_orchestrator.py`
- Create: `tests/test_official_2stage_backbone_orchestrator.py`

- [ ] **Step 1: Write failing tests for orchestration and delayed-compression ownership**

```python
def test_orchestrator_updates_single_pending_queue_by_chunk_window(monkeypatch, tmp_path) -> None:
    result = run_backbone_preview(...)
    assert result["window_blocks"] == 2


def test_orchestrator_queues_detached_chunk_snapshots(monkeypatch, tmp_path) -> None:
    metadata = json.loads(output_json.read_text())
    assert metadata["memory_mode"] == "persistent_ssm"
```

Test design requirements:

- Stub the chunk runner so each chunk returns deterministic tensors and a mock `next_ssm_state`.
- Assert the orchestrator:
  - owns one `persistent_ssm_state`
  - owns one delayed-compression queue
  - treats `window_blocks` as queued whole chunks
  - stores detached snapshots in the pending queue
  - calls `compress_evicted_tokens(...)` only after queue overflow
  - preserves the existing `episode_0000/chunk_000.mp4` + `episode_0000.mp4` layout
- Assert metadata contains:
  - local preset alias `small`
  - `memory_mode`
  - resolved geometry
  - `chunk_num_frames`
  - `frame_rate`
  - per-chunk output records

- [ ] **Step 2: Run the targeted orchestrator tests to verify they fail**

Run:

```bash
source .venv/bin/activate && python -m pytest tests/test_official_2stage_backbone_orchestrator.py -q
```

Expected:

- FAIL because the new orchestrator entrypoint does not exist yet.

- [ ] **Step 3: Implement the orchestrator CLI and runtime loop**

```python
def run_backbone_preview(config: BackbonePreviewConfig) -> dict[str, Any]:
    runner = OfficialDistilledChunkRunner(...)
    for chunk in chunk_plan:
        result = runner.run_chunk(chunk_config, ssm_state=state.ssm_state)
        state = update_delayed_queue(
            state,
            result.evictable_video_tokens,
            result.evictable_audio_tokens,
            compress_fn=runner.compress_evicted_tokens,
        )
        save_chunk(...)
    stitch_episode(...)
```

Implementation requirements:

- Reuse the existing Stage A manifest and no-switch chunk-plan conventions.
- Keep queue ownership in the orchestrator, not in the runner.
- Thread the required official assets through the orchestrator CLI/config:
  - `--distilled-checkpoint-path`
  - `--gemma-root`
  - `--spatial-upsampler-path`
  - optional `--ssm-checkpoint`
- Keep output layout compatible with `scripts/streaming_backbone_audit.py`.
- Preserve observability:
  - queue length
  - state presence
  - output paths
- Do not reuse `ValidationSampler` or `scripts/streaming_inference.py`.

- [ ] **Step 4: Run the targeted orchestrator tests to verify they pass**

Run:

```bash
source .venv/bin/activate && python -m pytest tests/test_official_2stage_backbone_orchestrator.py -q
```

Expected:

- PASS for queue ownership, detached snapshots, metadata, and chunk layout tests.

- [ ] **Step 5: Commit the orchestrator**

```bash
git add scripts/official_2stage_backbone_orchestrator.py tests/test_official_2stage_backbone_orchestrator.py
git commit -m "feat: add official two-stage backbone orchestrator"
```

## Task 4: Cut Over the Persistent-SSM Smoke Wrapper

**Files:**
- Modify: `qz/run_streaming_backbone_smoke.sh`
- Modify: `qz/run_streaming_backbone_audit_smoke.sh`
- Modify: `qz/submit_streaming_backbone_smoke_p3.sh`
- Modify: `tests/test_qz_wrappers.py`

- [ ] **Step 1: Write failing wrapper tests for preview-path routing**

```python
def test_streaming_backbone_smoke_routes_persistent_ssm_to_official_preview() -> None:
    script = script_text()
    assert "official_2stage_backbone_orchestrator.py" in script


def test_streaming_backbone_audit_defaults_to_preview_output_dir() -> None:
    script = audit_script_text()
    assert "official_2stage" in script
```

Test requirements:

- Assert `USE_SSM_STREAMING=1` now routes to the new orchestrator preview path.
- Assert legacy `REFERENCE_WINDOW_CHUNKS==0` and short-context branches still point to the old runtime in phase 1.
- Assert the default persistent output directory and audit directory are preview-specific.
- Assert the submit wrapper experiment/version naming reflects preview scope.
- Assert the wrapper passes the required official asset paths and the optional SSM checkpoint path into the new orchestrator CLI.

- [ ] **Step 2: Run the wrapper tests to verify they fail**

Run:

```bash
source .venv/bin/activate && python -m pytest tests/test_qz_wrappers.py -q
```

Expected:

- FAIL because the wrapper scripts still point at the legacy runtime and legacy output bucket.

- [ ] **Step 3: Update the wrappers with the narrow cutover only**

```bash
if [[ "$USE_SSM_STREAMING" == "1" ]]; then
  python scripts/official_2stage_backbone_orchestrator.py \
    --distilled-checkpoint-path /inspire/hdd/project/agileapplication/zhangkaipeng-24043/wgx/myltx/model/ltx-2.3-22b-distilled.safetensors \
    --gemma-root /inspire/hdd/project/agileapplication/zhangkaipeng-24043/wgx/myltx/model/gemma \
    --spatial-upsampler-path /inspire/hdd/project/agileapplication/zhangkaipeng-24043/wgx/myltx/model/ltx-2.3-spatial-upscaler-x2-1.0.safetensors \
    --ssm-checkpoint "$SSM_CHECKPOINT" \
    ...
else
  python scripts/streaming_inference.py ...
fi
```

Implementation requirements:

- Do not migrate `no_memory` or `short_context` in this phase.
- Use preview naming in output paths and submit metadata.
- Keep audit output schema unchanged; only change the default directory.

- [ ] **Step 4: Re-run the wrapper tests**

Run:

```bash
source .venv/bin/activate && python -m pytest tests/test_qz_wrappers.py -q
```

Expected:

- PASS for routing, output-directory, and preview-naming assertions.

- [ ] **Step 5: Commit the wrapper cutover**

```bash
git add qz/run_streaming_backbone_smoke.sh qz/run_streaming_backbone_audit_smoke.sh qz/submit_streaming_backbone_smoke_p3.sh tests/test_qz_wrappers.py
git commit -m "feat: route persistent backbone smoke to official preview"
```

## Task 5: Full Verification, Remote Smoke, and Research Log Update

**Files:**
- Modify: `../../RESEARCH_LOG.md`

- [ ] **Step 1: Run the full local regression pack**

Run:

```bash
source .venv/bin/activate && python -m pytest \
  tests/test_distilled_streaming.py \
  tests/test_official_2stage_backbone_orchestrator.py \
  tests/test_qz_wrappers.py \
  tests/test_official_generation_defaults.py \
  tests/test_streaming_inference.py \
  -q
```

Expected:

- PASS with `0 failed`.

- [ ] **Step 2: Sync the changed files to `qz`**

Run:

```bash
scp packages/ltx-pipelines/src/ltx_pipelines/distilled_streaming.py qz:/inspire/hdd/project/agileapplication/zhangkaipeng-24043/wgx/myltx-v1/packages/ltx-pipelines/src/ltx_pipelines/distilled_streaming.py
scp scripts/official_2stage_backbone_orchestrator.py qz:/inspire/hdd/project/agileapplication/zhangkaipeng-24043/wgx/myltx-v1/scripts/official_2stage_backbone_orchestrator.py
scp qz/run_streaming_backbone_smoke.sh qz:/inspire/hdd/project/agileapplication/zhangkaipeng-24043/wgx/myltx-v1/qz/run_streaming_backbone_smoke.sh
scp qz/run_streaming_backbone_audit_smoke.sh qz:/inspire/hdd/project/agileapplication/zhangkaipeng-24043/wgx/myltx-v1/qz/run_streaming_backbone_audit_smoke.sh
scp qz/submit_streaming_backbone_smoke_p3.sh qz:/inspire/hdd/project/agileapplication/zhangkaipeng-24043/wgx/myltx-v1/qz/submit_streaming_backbone_smoke_p3.sh
scp tests/test_distilled_streaming.py qz:/inspire/hdd/project/agileapplication/zhangkaipeng-24043/wgx/myltx-v1/tests/test_distilled_streaming.py
scp tests/test_official_2stage_backbone_orchestrator.py qz:/inspire/hdd/project/agileapplication/zhangkaipeng-24043/wgx/myltx-v1/tests/test_official_2stage_backbone_orchestrator.py
scp tests/test_qz_wrappers.py qz:/inspire/hdd/project/agileapplication/zhangkaipeng-24043/wgx/myltx-v1/tests/test_qz_wrappers.py
```

Expected:

- All files copy cleanly into the remote repo under `/inspire/hdd/project/agileapplication/zhangkaipeng-24043/wgx/myltx-v1/`.

- [ ] **Step 3: Run the remote regression pack**

Run:

```bash
ssh qz 'cd /inspire/hdd/project/agileapplication/zhangkaipeng-24043/wgx/myltx-v1 && source /inspire/hdd/project/agileapplication/zhangkaipeng-24043/wgx/myltx/.venv/bin/activate && python -m pytest tests/test_distilled_streaming.py tests/test_official_2stage_backbone_orchestrator.py tests/test_qz_wrappers.py tests/test_official_generation_defaults.py -q'
```

Expected:

- PASS with `0 failed`.

- [ ] **Step 4: Submit and inspect the preview smoke job**

Run:

```bash
SUBMIT_OUTPUT=$(ssh qz 'cd /inspire/hdd/project/agileapplication/zhangkaipeng-24043/wgx/myltx-v1 && bash qz/submit_streaming_backbone_smoke_p3.sh')
JOB_ID=$(printf "%s\n" "$SUBMIT_OUTPUT" | rg -o 'job-[a-f0-9-]+' -m 1)
PREVIEW_DIR=/inspire/qb-ilm/project/agileapplication/zhangkaipeng-24043/wgx/myltx-v1/outputs/official_2stage_persistent_ssm_backbone_smoke_preview
ssh qz "cd /inspire/hdd/project/agileapplication/zhangkaipeng-24043/wgx && python3 .claude/skills/qz/scripts/qz_cli.py detail $JOB_ID"
JOB_NAME=$(ssh qz "cd /inspire/hdd/project/agileapplication/zhangkaipeng-24043/wgx && python3 .claude/skills/qz/scripts/qz_cli.py detail $JOB_ID" | python3 -c 'import json,sys; print(json.load(sys.stdin)[\"name\"])')
ssh qz "tail -n 120 /inspire/hdd/project/agileapplication/zhangkaipeng-24043/wgx/qz/logs/${JOB_NAME}.log"
ssh qz "ls -lah $PREVIEW_DIR"
ssh qz 'cd /inspire/hdd/project/agileapplication/zhangkaipeng-24043/wgx/myltx-v1 && bash qz/run_streaming_backbone_audit_smoke.sh'
```

Expected:

- Job reaches `job_succeeded`.
- Output directory contains chunk mp4s plus the stitched final mp4 and metadata.
- No black-final-video signature.
- Queue/state observability is present in logs or metadata.
- Audit summary is written under the preview output directory.

- [ ] **Step 5: Record the result and rollback point in the research log**

Update `../../RESEARCH_LOG.md` with:

- exact preview job id and name
- output directory
- evidence for success or failure
- whether stage-2-only persistent SSM matched the current delayed-compression baseline closely enough
- explicit rollback point back to the validated legacy Stage A path if parity fails

Parity check requirements for this step:

- Compare against the validated official-small delayed-compression baseline:
  - `/inspire/qb-ilm/project/agileapplication/zhangkaipeng-24043/wgx/myltx-v1/outputs/streaming_backbone_persistent_ssm_smoke_v5_official_smallres/episode_0000.json`
  - `/inspire/hdd/project/agileapplication/zhangkaipeng-24043/wgx/qz/logs/wgx-train-h200-1g-p3-v5-streaming-backbone-persistent-ssm-delayed-window2-official-smallres.log`
- Require exact parity for:
  - resolved geometry `1024x1536`
  - `chunk_num_frames=41`
  - `frame_rate=8`
  - `window_blocks=2`
  - chunk count `6`
- Require parity checks for the migration hypothesis:
  - final video is not black
  - all chunk files and the stitched output exist
  - memory becomes active by the same queue-overflow phase, i.e. no later than chunk `2`
  - logged state magnitudes remain finite and stay within the same order of magnitude as the baseline chunk-5 values (`mean_abs≈1.87`, `max_abs≈104.5`)
- If any exact-parity item fails, record the preview as a migration regression and use the legacy `v5_official_smallres` artifact as the rollback point.

- [ ] **Step 6: Commit the verified preview path**

```bash
git add packages/ltx-pipelines/src/ltx_pipelines/distilled_streaming.py \
  scripts/official_2stage_backbone_orchestrator.py \
  qz/run_streaming_backbone_smoke.sh \
  qz/run_streaming_backbone_audit_smoke.sh \
  qz/submit_streaming_backbone_smoke_p3.sh \
  tests/test_distilled_streaming.py \
  tests/test_official_2stage_backbone_orchestrator.py \
  tests/test_qz_wrappers.py \
  ../../RESEARCH_LOG.md
git commit -m "feat: add official two-stage persistent backbone preview"
```

## Exit Condition

- `persistent_ssm backbone smoke` no longer routes through `ValidationSampler`.
- The new path uses the official distilled two-stage chunk semantics at `1024x1536`.
- Only the `persistent_ssm` Stage A branch is migrated; `no_memory`, `short_context`, `switch`, and `baseline_audit` remain untouched.
- Local and remote regression suites pass.
- A preview smoke job succeeds on `qz`, writes comparable Stage A artifacts, and is recorded in `../../RESEARCH_LOG.md`.
