# Route35 / Route3 Rebuild Next Actions (2026-03-20)

## Key conclusion

The v3 eviction-debug result (`diff_keys=216`, `max_abs_diff=6.103515625e-05`) establishes that the training path can update `ssm_layers` once eviction is actually active.

The previous Route3 PhaseA path failed for two separate reasons:

1. `window_blocks=4` was eviction-invalid for the available phaseA exports.
2. Full-manifest prompt caching (`510` prompts) caused startup OOM.

## 1) Make PhaseA training eviction-valid and reproducible

### Eviction-valid rule

For the current phaseA exports, a sample has `16` latent frames. With the current chunking rule:

- chunk 0 = first frame
- each later chunk = `block_size=6` frames
- therefore `16` frames produce `4` video chunks total

Eviction only affects training if there is at least one later chunk after compression. That means:

- required condition: `num_chunks >= window_blocks + 2`
- with `16`-frame exports and `4` chunks total, PhaseA training must use `window_blocks <= 2`

So the current `window_blocks=4` PhaseA config can never train on evicted memory, even without smoke caps.

### Reproducible path

Use a two-stage path:

1. **Gate run**: always run the existing v3 single-episode eviction-debug first.
   - single-episode manifest
   - `max_data_samples=1`
   - smoke caps small enough to fit memory, but large enough to keep `3` chunks
   - verify checkpoint diff before any longer rebuild

2. **Rebuild run**: run a small longlive shard, not the full `256`-episode manifest, until prompt-cache handling is improved.
   - keep real HDD phaseA data root
   - use a small manifest shard so startup prompt caching stays bounded
   - keep `window_blocks=2` (or `1`) so the `16`-frame data remains eviction-valid
   - write outputs to `qb-ilm`

Without code changes, manifest sharding is the practical way to keep prompt-cache memory bounded.

## 2) Minimal config fields that need to change

From the old PhaseA v1 path, the minimum required field changes are:

- `window_blocks`: change from `4` to `2` for real `16`-frame phaseA rebuilds
- `switch_episode_manifest`: stop using the full `256`-episode manifest for rebuild smoke / pilot; use a small shard manifest
- `output_dir`: point rebuild outputs to `qb-ilm`
- `checkpoint_interval`: keep explicit checkpoint saves aligned to S19 (`50`, `100`, `200`, `300` for the real rebuild; `1` for gate runs)
- `max_data_samples`: use `1` for gate, then increase for rebuild once the shard is stable

For the v3 gate config specifically, the minimal stable smoke caps are:

- `smoke_max_prompt_tokens: 64`
- `smoke_max_video_frames: 8`
- `smoke_max_video_height: 12`
- `smoke_max_video_width: 16`
- `smoke_max_audio_time_steps: 24`

Why these still trigger eviction:

- `8` frames with `block_size=6` produce `3` chunks: `1 + 6 + 1`
- with `window_blocks=1`, chunk 2 causes compression of chunk 1, and chunk 3 is the first chunk that can consume evicted memory

## 3) Minimal S19 / S20 rerun chain

### Step A: run the gate first

Use the existing v3 gate and confirm checkpoint divergence:

```bash
bash qz/submit_train_self_forcing_longlive_phaseA_evict_debug_v3_qb_p3.sh
OUTPUT_DIR=/inspire/qb-ilm/project/agileapplication/zhangkaipeng-24043/wgx/myltx-v1/outputs/self_forcing_longlive_phaseA_evict_debug_v3_qb \
  bash qz/check_self_forcing_longlive_phaseA_evict_debug_v2_qb_diff.sh
```

Success condition: `diff_keys > 0`.

### Step B: rebuild a real PhaseA pilot on a small manifest shard

Next config to add should be a real-data rebuild pilot with:

- data root = `/inspire/hdd/project/agileapplication/zhangkaipeng-24043/wgx/myltx-v1/ode/data_distilled_stage2_ode_phaseA_256`
- manifest = a small longlive shard, not the full 256-episode manifest
- `window_blocks=2`
- no video smoke-frame cap if memory allows; otherwise only lower caps that still preserve `num_chunks >= 4`
- `total_steps=300`
- `checkpoint_interval=50`
- output to `qb-ilm`

This is the smallest rebuild that can produce S19 checkpoints with actual eviction-active training.

### Step C: rerun S19 progression on the rebuilt checkpoints

Reuse the existing S19 progression wrapper, but point it at the new rebuild checkpoints:

```bash
bash qz/run_streaming_switch_ssm_phasea_ckpt_s19_scriptargs_v20260320.sh --step 00050 --ssm-checkpoint <new_step_00050.pt> --output-dir <new_s19_out>/step_00050
bash qz/run_streaming_switch_ssm_phasea_ckpt_s19_scriptargs_v20260320.sh --step 00100 --ssm-checkpoint <new_step_00100.pt> --output-dir <new_s19_out>/step_00100
bash qz/run_streaming_switch_ssm_phasea_ckpt_s19_scriptargs_v20260320.sh --step 00200 --ssm-checkpoint <new_step_00200.pt> --output-dir <new_s19_out>/step_00200
bash qz/run_streaming_switch_ssm_phasea_ckpt_s19_scriptargs_v20260320.sh --step 00300 --ssm-checkpoint <new_step_00300.pt> --output-dir <new_s19_out>/step_00300
```

Then score those outputs with `scripts/streaming_switch_audit.py` as already memoized.

### Step D: rerun S20 matched eval on the rebuilt latest checkpoint

Reuse the existing matched wrapper, but pass the rebuilt step-300 checkpoint explicitly:

```bash
bash qz/run_route35_s20_matched_eval_min_v20260320.sh --ssm-checkpoint <new_step_00300.pt>
```

That keeps the non-SSM recache / non-SSM no-recache baselines unchanged, while swapping only the pure-SSM system to the rebuilt checkpoint.

## Recommended immediate sequence

1. Treat v3 as the standing eviction-valid gate.
2. Add one small longlive-shard rebuild config with `window_blocks=2` and `checkpoint_interval=50`.
3. Rebuild to `300` steps on that shard.
4. Run S19 on `50/100/200/300`.
5. Run S20 on rebuilt `step_00300`.

This is the minimum chain that converts the v3 debugging result into a Route3 / Route35 rebuild with quantitatively comparable S19/S20 artifacts.
