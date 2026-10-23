# MISSING (exact names, needed columns, substitutes, follow-up cost)

| Missing | Needed for | Needed columns | Substitute shipped | Follow-up cost (estimate) |
|---|---|---|---|---|
| SpatialVLA RGB observation frames for P1-D (`rec[].img` not stored; only `img_hash`) | V4 start/middle/end frames | per-step RGB (task, seed, config, arm, t) | TCP paths, success, image hashes | Replay of stored float64 actions from `reset(seed)` in SimplerEnv with rendering; deterministic state restore was verified (restore_ok 100 %). ≈ 80 steps × ≤ 24 selected episodes, renderer needs a GPU/Vulkan device: ≈ 0.5 GPU-h. Not run (no new GPU work in this package). |
| Exact model input tensors (4 history frames × front/front-left/front-right after preprocessing) | V1 "exact observation" | resized image tensors | current-time jpg of CAM_F0/L0/R0 (raw, unmodified) | CPU re-run of `autovla/dataset_utils` preprocessing; ≈ minutes. History frames exist on disk only if fetched; not checked frame by frame. |
| Camera projection of trajectories | V1 overlay | verified ego→camera chain (lidar2ego + sensor2lidar + intrinsics + distortion) | side-by-side frame + BEV | CPU only; needs a verified transform test against a known landmark; ≈ 0.5 day. |
| Learned extrapolator time series | V2 baseline curve | per-pose predictions of `_extrap_models` | endpoint values only | CPU re-evaluation of stored models; ≈ 1 h. |
| Day-vehicle component IDs for the dev mechanism set | V2 `component_id` | component per unit | empty column with reason | CPU graph over (day, vehicle); not defined in the original analyses. |
| Experiment 34 actual TCP | V4 legacy | TCP per step | none (P1-D used instead) | would need re-running exp 34 with TCP logging (GPU). |
| Intervals for the time-resolved curves | V2 | – | point estimates + n per horizon | needs a validated pointwise/simultaneous method; not attempted. |
