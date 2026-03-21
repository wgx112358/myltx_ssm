# Route35 PhaseA v3b cap48 Memo (2026-03-20)

This variant (`decay025_v3b_qb_shard16_cap48`) keeps the v3 shard16 route unchanged except an explicit cap:
- `max_data_samples=48`

Why this removes sample-manifest mismatch:
- The shard16 slice is `00000..00015` (16 sample ids).
- Each sample contributes 3 temporal steps (`step_000/001/002`).
- Effective available training units are therefore `16 x 3 = 48`.
- Capping to 48 prevents the loader from requesting beyond the shard16/manifest-supported set.
- `allow_synthetic_fallback=false` keeps the run strict: no synthetic backfill that could hide indexing mismatch.

Result: data loader demand is aligned to the manifest-backed shard cardinality by construction.
