# NUMERICAL_CHECKS (minimum checks of this package; no repetition of completed statistical audits)

| Check | Result | File |
|---|---|---|
| V1 paired IDs: NAT_REF and NAT_NSRC exist for every B2 unit (n_old = 1) | 2,160 / 2,160 | tables/V1_checks.json |
| V1 emitted prefix identical (tokens before the anchor, prefix sha, poses before the anchor) | 2,160 / 2,160 for each | tables/V1_checks.json |
| V1 FDE5 recomputed from poses and the P1-C GT trajectory vs stored `fde5` | max abs diff 6.9e-6 m (stored values rounded to 5 decimals) | tables/V1_checks.json |
| V1 unit difference vs stored C_NAT in unit_contrasts.parquet (sign flipped) | max abs diff 0 | tables/V1_checks.json |
| V1 F cohort | 711 units / 100 scenes / 18 drives; 0 invalid; 0 without CAM_F0; 711/711 CAM_F0 files in P1-C inference keep lists | tables/V1_checks.json |
| V1 candidate manifest ↔ selected cases | every selected unit flagged `selected_unit` in CASE_CANDIDATES.csv | CASE_CANDIDATES.csv |
| V2 S0 rows of review_followup5 B_NUMERIC equal the P1-A raw rows | 1,453 / 1,453 units (U and OC poses) | tables/V2_checks.json |
| V2 unit-level D_model(k = 9) and D_b(k = 9) vs review_followup3 P2 p2_units.parquet | max abs diff ≤ 3.6e-15 m (all five baselines) | tables/V2_checks.json |
| V2 scene-weighted endpoint E_b vs P2 MOTION_BASELINE_RESULTS est_scene (A−, A+) | max abs diff 8.9e-16 m | tables/V2_endpoint_checks.csv |
| V2 endpoint E_b for S0/SE/SF vs review_followup5 B_NUMERIC_RESULTS | max abs diff 8.9e-16 m | tables/V2_endpoint_checks.csv |
| V2 D_model(j) vs P1-A deviation_curves.csv (OC) | equal at j ≤ 1–3; up to 0.062 m (A+, j = 9) later. **Cause:** deviation_curves measures OC against the stored B = 1 reference; this package measures OC against the U row of the same B = 10 batch (the pairing used by P2/B_NUMERIC). U differs from the reference in 160 / 1,453 units (batch numerics; max 3.29 m). Same n_scenes. No claim affected (the P2/B_NUMERIC endpoint numbers are the ones reproduced exactly). | tables/V2_endpoint_checks.csv |
| V3 B2 scene → stratum aggregation vs MATCHED_PREFIX_RESULTS theta (C_GT, C_NAT, INT, C_PROV; F and P) | max abs diff 8e-17 | tables/V3_reproduction_check.csv |
| V3 interval display follows cvpr_next FINAL_INTERVAL_POLICY / PAPER_READY_NUMBERS | E RV, E4 C_NAT, F INT withheld (point only); E4 C_src, E DM: no interval | EFFECTS_FOR_PLOTTING.csv (`ci_drawn`) |
| V4 paired TCP area mean per config (R−N, C−F) vs P1-D paired_effects.csv | max abs diff 5e-6 cm·step (stored rounding) | tables/V4_checks.csv |
| V4 paired success difference vs paired_effects.csv | max abs diff 3e-6 | tables/V4_checks.csv |
| V4 success rates per config/arm/task vs condition_summary.csv | max abs diff 3.3e-5 (stored 4-decimal rounding) | tables/V4_checks.csv |
| V4 restore | 3,840 / 3,840 branch episodes restore_ok; 960 natural roots have no restore field | SPATIAL_EPISODES.csv |
| V4 physical states | coke 120 states; move_near 10 states for 120 units | SPATIAL_EPISODES.csv |
| Coordinates / units | AutoVLA: ego frame at observation, x forward / y left, m, 0.5 s per pose (p2_kinematics.DT); SpatialVLA TCP: sapien world frame, m, quaternion w,x,y,z | – |

Not verifiable here (UNVERIFIABLE): that the shipped jpg bytes equal the decoded tensors fed to the model (preprocessing not stored);
the camera projection chain (not attempted).
No new error was found in existing results.
