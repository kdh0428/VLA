# ANALYSIS_PLAN — CVPR visual evidence (post-hoc; written 2026-10-18 for the request dated 2026-10-19)

Status: **fixed before any V1 case image was chosen and before any V2/V4 curve was computed.**
Everything below is a **post-hoc descriptive re-analysis of stored outputs**. It is not a new experiment, not a new
independent replication and not preregistered. No GPU, no model inference, no training. Inputs are read only.

Inspection done before this plan was written (disclosed): file listings, record schemas, row counts, the B2 cell
fields of one unit (`b5ee0c4a00765073|alt0`), the existing aggregate tables (FINAL_PAPER_NUMBERS, PAPER_READY_NUMBERS,
FINAL_INTERVAL_POLICY, R8R9 condition_summary, P1-D condition_summary) and a check that the 100 B2-F scenes have a
CAM_F0 frame on disk (100/100). No per-scene FDE ranking and no figure was produced before this file.

## V1 — driving cases (detail in CASE_SELECTION_RULE.md)
- Cohort: B2 primary cohort (P1-C intervention sample ∩ E4), **F stratum**, primary anchor n_old = 1 (a = t*+3):
  100 scenes / 711 units / 18 drives. Source `review_followup4/B/B2/p1c_main/cells.jsonl` +
  `p1c_analysis/unit_contrasts.parquet`.
- Contrast: **NAT_REF (treatment, latest context position = GT) vs NAT_NSRC (reference, latest context position =
  frozen Normal-source token)**; both behind the model's own natural older position and the same emitted prefix.
  This is the manuscript C_NAT with the opposite sign (treatment − reference = −C_NAT). Availability decided the
  choice (all cells, poses, tokens and frames exist); the repeated-RN fallback is not needed.
- Plot: open-loop 5 s plan poses (10 poses, 0.5 s apart, ego frame at the observation time, x forward / y left, m),
  GT 5 s trajectory, shared emitted prefix (poses 0..a−1), suffix after the anchor. Edited context tokens are drawn as
  separate markers (their single-segment motion), never as part of the emitted trajectory.
- Camera: the stored front camera frame (CAM_F0) at the observation token; CAM_L0/CAM_R0 of the same frame as context.
  No projection into the image (calibration exists, but the projection chain is not verified in this work);
  side-by-side only. No lanes, objects or future frames are added.

## V2 — time-resolved response (G-family, P1-A)
- Units: P1-A family G (GT-free perturbation value and reference; position t_p = t* is GT-informed), strata A− (52
  scenes / 364 units / 16 logs; failure-selected), A+ (156 / 1,089 / 26) and the pooled selected mechanism set
  (208 scenes). Source `review_followup/P1A_gt_free_reference/run_20261008/raw/records.jsonl` (setting S0 = original
  run). Settings S0/SE/SF from `review_followup5/B_NUMERIC/run/records.jsonl` with the same code where U and OC rows exist.
- Pair: P = OC row (output and context perturbed at t_p), U = U row (unperturbed; equals the reference).
  D_model(h) = ||P(h) − U(h)||, D_b(h) = ||P_b(h) − U_b(h)||, E_b(h) = D_model(h) − D_b(h).
  Baselines with full h: Geo (privileged: reference future composed after the perturbed token), CM (body-frame constant
  motion), CV0, CTRV-K3, CTRA-2 (exact functions of `scripts/review_followup3/p2_kinematics.py`). Learned extrapolator:
  only endpoint values are stored → endpoint only (no CPU re-evaluation in this package).
- Time axis: pose index k = 0..9, horizon_s = 0.5·(k+1) (codebook segment = 0.5 s, `p2_kinematics.DT`).
  step_after_anchor j = k − t_p. Primary curves by j (anchors differ by unit, so the sample shrinks with j);
  a common-sample version uses units with t_p ≤ 4 and j = 0..5. Missing j are not filled or interpolated.
- Aggregation: unit → scene mean → equal scene weight within the named stratum (the dev mechanism set has no
  sampling weights). Point estimates and n_units / n_scenes / n_drives per j only; **no intervals**.
- GT error ||P−GT||, ||U−GT|| reported in a separate table/panel; response size and accuracy are different metrics.
- Endpoint check: k = 9 must reproduce `review_followup3/P2/MOTION_BASELINE_RESULTS.csv` (family G, metric end,
  est_scene, source primary) to 1e-9, and S0/SE/SF must reproduce `review_followup5/B_NUMERIC/B_NUMERIC_RESULTS.csv`
  where that file has the same quantity. Differences are reported, not hidden.
- No 3 m severe threshold on short horizons.

## V2-c — correction schedule
- Diagram: repeated Recent-GT (sliding: at step k only context position k−1 ← GT) vs R9 window w (positions
  t*+1..t*+w hold GT for every later step; retained, never released). Source definitions:
  `scripts/temporal_feedback_window.py`, `scripts/action_history_causal.py`.
- Window values: R9 raw recovery `condition_summary.csv` (A−, win1..win8, normal, gt_history; same 365 units / 52 scenes
  in every row). Plotted as discrete points; win w ≥ 8 − t* equals Full GT-history (exposure truncated by t*).
  No line connecting windows is presented as an efficiency curve; no post-release persistence is inferred.

## V3 — effects for plotting
- Re-visualisation of existing aggregates only (no new inference): dev A− RN/RV, P1-C F RN/RV, P1-C E RN/RV,
  B2 F and E4 C_GT / C_NAT / INT / C_PROV(=C_src). Intervals follow `cvpr_next_validation_20261019/
  FINAL_INTERVAL_POLICY.csv` and PAPER_READY_NUMBERS: WITHHOLD → no interval drawn; APPROXIMATE → drawn and labelled.
- New descriptive unit-level tables: B2 scene-level C_NAT/C_GT (scene means) and drive summaries that keep the scene
  weights inside a drive (drive points are heterogeneity displays, not independent samples).

## V4 — SpatialVLA (P1-D)
- Source `review_followup/P1D_dose_control/run_20261008/rollouts/main/episodes.jsonl` (4,800 episodes).
  Arms N/F/C/R per protocol table (§2.2); configs r1e1/r2e1/r4e1/r1e4; FD/CD/null/gen kept in the episode table only.
- Fields: generated (ensemble predictions `preds`, chunk tokens `exec`), executed action (`action`, float64),
  actual TCP pose (`pose`: position m + quaternion, sapien order w,x,y,z), gripper, success.
- Paired TCP translation deviation per (task, seed, config): R−N and C−F, t = 8..79, cm·step; checked against
  the stored P1-D outputs where the same quantity exists.
- Success rates from all eligible episodes (restore_ok), overall and per task; move_near physical states counted
  by `init.state_hash` — never treated as 120 independent states.
- No RGB frames were stored → TCP/success only; no re-render (would need the simulator).

## Changes after this plan
Recorded in START_HERE.md (section "Changes after the plan").
