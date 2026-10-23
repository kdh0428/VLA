# FIGURE_CAPTIONS (English drafts; all figures are post-hoc descriptive re-analyses of stored outputs)

Every quantitative figure has PDF + SVG (editable text) + 300 dpi PNG in `figures/`. Plot inputs are the CSV/Parquet files named below.

## V1_case_<group>_<rank> (cases/<group>_<rank>/)
**Caption.** *Open-loop plans from one fixed observation (AutoVLA, B2 failure stratum F, anchor a = t*+3).* Left: the stored
front-camera frame at the observation time (unmodified). Middle: bird's-eye view in the ego frame at observation (forward up,
metres, 0.5 s between poses). Grey: emitted prefix shared by both arms (identical tokens). After the anchor, blue (NAT_REF) is
generated with the GT token in the latest context position and orange (NAT_NSRC) with the frozen Normal-source token; the older
context position is the model's own token in both. Hollow diamonds show the edited latest-context value (not an executed action).
Dotted: GT 5 s trajectory. Right: forward position versus plan time. These are planned trajectories, not closed-loop driving. The
case was chosen by a deterministic post-hoc rule (closest to the group median of the scene-mean FDE5 difference; groups by the sign
of the scene-mean severe-event difference) and is illustrative, not representative.
**Supports.** What "latest-context value changes the generated suffix behind an identical emitted prefix" looks like.
**Limits.** 87/100 F scenes are in the improved group, 12 unchanged-severe, 1 worsened (the worsened group has a single scene, so
its case is that scene by construction). Unchanged-severe means the severe judgment did not change, not identical trajectories.
Suggested main-text set: improved_1, unchanged_severe_1, worsened_1. Inputs: `cases/*/trajectories.csv`, `case.json`, `case_arrays.npz`.

## V2_response_vs_baselines
**Caption.** *Response to one GT-free perturbed token over the plan horizon (P1-A family G, setting S0).* Top: AutoVLA response
D_model = ||P − U||. Middle: response of motion baselines to the same two inputs (constant motion, coordinate accumulation
[privileged: composes the reference future], CTRA-2). Bottom: E_b = D_model − D_b. Points are unit → scene mean → equal scene weight.
Solid: all observable units (the sample shrinks at later times because the perturbation position t_p varies; n = scenes);
dashed: common sample (t_p ≤ 4, first 3 s). Left: A− failure-selected stratum; right: the full selected mechanism set (not a natural
population). No intervals.
**Supports.** In A−, the model response exceeds the constant-motion and accumulation responses and the gap grows with horizon; in the
pooled set the excess over constant motion is small (≈ 0.08 m at 2.5 s in the common sample).
**Limits.** The perturbation value and reference are GT-free; the position t_p = t* and the A−/A+ strata are GT-defined. The last
points of the solid curves rest on few scenes (A−: 17 at 4.5 s) and change composition. Response size is not accuracy (see next figure).
Inputs: `TIME_SERIES_SUMMARY.csv`, `AUTOVLA_TIME_SERIES.parquet`.

## V2_gt_error_separate
**Caption.** Distance to GT of the perturbed (OC) and unperturbed (U) plans over time, same units and aggregation. Shown separately
because response size and accuracy change are different metrics. No severe-event threshold is applied at short horizons.

## V2c_correction_schedules
**Caption.** *Which context positions hold GT while each step is generated (illustration t* = 1).* Recent-GT repeatedly replaces only
the most recent context position (earlier positions return to the model's own tokens); the R9 window w = 3 keeps GT at positions
t*+1..t*+3 for every later step (retained, never released); Full GT-history replaces all post-t* positions. Executed actions are
always the model's own tokens.
**Limits.** Schematic; no data.

## V2c_window_points
**Caption.** *Severe-event rate by retained window width (R9 raw recovery, same units in every row).* Discrete points; numbers mark the
units for which window w already equals Full GT-history (w ≥ 8 − t*), so exposure differs across units. Retained windows are not
GT release, and these values do not measure persistence after correction ends or task success. No interval drawn
(log-cluster bootstrap not validated). A−: Normal 45.1 %, w = 3 7.6 %, Full GT 3.4 % (scene means).

## V3_effects_composite
**Caption.** *Different contrasts and estimands (severe events, pp).* Left: existing aggregate effects for repeated RN/RV (dev
failure-selected set, P1-C failure stratum F, P1-C weighted eligible frame E) and matched-prefix C_GT, C_NAT, INT and C_src
(B2 F and weighted E4). Bars: approximate cluster-robust t 95 % intervals where the latest interval policy allows; "interval withheld"
rows (E RV, E4 C_NAT, F INT) show the point estimate only. Right: B2 C_NAT per scene (dots, size ∝ E4 scene weight) and per drive
(ticks; scene weights kept inside a drive) with the stratum estimate (black). F and E/E4 are different estimands; E and E4 have
different eligibility. Drive points share days/vehicles and are not independent; with 6 day–vehicle components the smallest attainable
p is 0.03125 (Holm 0.125). C_src changes value and source together.
Inputs: `EFFECTS_FOR_PLOTTING.csv`, `tables/V3_B2_scene_effects.csv`, `tables/V3_B2_drive_summary.csv`.

## V4_spatialvla_tcp_success
**Caption.** *SpatialVLA P1-D: trajectory change and success are different outcomes.* Left: mean actual TCP translation difference
between paired episodes from the same initial physical state (R−N, C−F; r4e1 solid, r1e1 dashed; injections at t = 8, 12, 16, 20).
Middle: TCP deviation area (t = 8–79) for r4e1 grouped by the paired success change. Right: r4e1 success of N/F/C/R over all eligible
episodes (120 per task per arm; move_near uses 10 physical states).
**Limits.** Success precision is about ±10 pp for the two-task average only (not per task). Physical states are shared across
instructions/seeds in move_near. Inputs: `tables/V4_paired_units.csv`, `tables/V4_success_rates.csv`.

## V4_spatialvla_cases_tcp
**Caption.** Actual TCP x/y paths (simulator world frame) of paired episodes for r4e1, chosen by CASE_SELECTION_RULE.md §B (success
worse / same / better, closest to the group median area). No RGB frames were stored. Inputs: `SPATIAL_TIME_SERIES.parquet`,
`tables/V4_selected_cases.csv`.
