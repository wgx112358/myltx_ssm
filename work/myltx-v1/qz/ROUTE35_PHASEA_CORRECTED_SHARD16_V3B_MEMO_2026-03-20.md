# Route35 corrected PhaseA shard16 v3b memo

This v3b patch is a minimal safety fix for the shard16 corrected PhaseA pilot.

- Manifest remains `/inspire/hdd/project/agileapplication/zhangkaipeng-24043/wgx/myltx-v1/ode/switch_episodes_longlive_phaseA_256_shard16_decay025_v3_qb.jsonl`.
- The manifest sample_id prefix is `00000` through `00015`.
- Precomputed samples are sorted so the first 48 entries are exactly `00000__step_000 .. 00015__step_002`.
- `max_data_samples: 48` therefore restricts the loader to the shard16-aligned sample-step prefix and prevents silent fallback/modulo remapping into later episodes.

Nothing in loader code is changed. This is only a config-level cap plus renamed train/submit wrappers and a new output directory.
