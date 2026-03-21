# Route35 SSM Evict Debug Realdata Memo

This branch uses the only verified existing phaseA/longlive exports in the repo:

- data root: `/inspire/hdd/project/agileapplication/zhangkaipeng-24043/wgx/myltx-v1/ode/data_distilled_stage2_ode_phaseA_256`
- manifest: `/inspire/hdd/project/agileapplication/zhangkaipeng-24043/wgx/myltx-v1/ode/switch_episodes_longlive_phaseA_256.jsonl`

It does not depend on the missing qb-ilm `*_v2_qb` data paths.
Training reads from HDD, but writes debug checkpoints to:

- output dir: `/inspire/qb-ilm/project/agileapplication/zhangkaipeng-24043/wgx/myltx-v1/outputs/self_forcing_longlive_phaseA_evict_debug_realdata_v1_qb`

Purpose: minimal eviction-trigger debug run to test whether `ssm_weights_step_00001.pt` and `ssm_weights_step_00002.pt` start diverging once real phaseA data is used and eviction is guaranteed.
