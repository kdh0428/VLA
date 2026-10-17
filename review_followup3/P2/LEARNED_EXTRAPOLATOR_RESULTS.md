# LEARNED_EXTRAPOLATOR_RESULTS (P2-B)

**Status: PROVISIONAL.** These are AutoVLA numbers from the P1-A rollouts, the C1 input path. They stay provisional until the AutoVLA P1 no-op/cache parity gate is judged (`outputs/review_followup3/P1/NOOP_CACHE_AUDIT.*`; present when this report was generated). Section 6.2 and Figure 1 of the manuscript are held until the gate passes. Tags: driving domain, AutoVLA, open-loop 5 s plan, POST-HOC analysis of existing rollouts.

This is a small learned **motion** extrapolator: a non-autoregressive MLP that predicts the whole suffix from the prefix. It is **not** a non-AR version of the VLA.

**Protocol:** `LEARNED_EXTRAPOLATOR_PROTOCOL.md`, locked at **2026-10-14T14:44:28Z**.
- The lock preceded training-data extraction, training and every P2-A new-baseline result.
- The protocol was not amended.
- One implementation fix was made before any sample existed: OpenScene `frame_idx` restarts in every scene, so plan frames are ordered by timestamp within a log and adjacency is checked with Δt only. A first run that used frame_idx adjacency produced 0 samples.

## Data and drive separation (`extrap_drive_separation.json`)
- **Source:** OpenScene v1.1 trainval metadata, i.e. the AutoVLA checkpoint training split. It was streamed (7.05 GB over the network, never written to disk). Only per-frame ego fields were kept, 56 MB on disk.
- **Drives:**
  - 171 trainval drives;
  - 0 excluded for overlap with test-split, dev, P1-C or P5-pool drives (there is none);
  - **145 training drives** and **26 validation drives**.
- **Drive overlap with reference sets:** {'dev': 0, 'p1c': 0, 'p5': 0, 'p1a_eval': 0, 'test': 0}; train–validation drive overlap: 0.
- **Reference set sizes (drives):** {'dev_discovery': 20, 'p1c_eval_or_pilot': 26, 'p5_candidates': 46, 'p1a_eval_units': 19, 'openscene_test_all': 46}.
- **Descriptive day overlap, training vs P1-A evaluation drives:** ['2021.10.06']. Vehicle overlap: 10 vehicles. Drive-disjointness is the requirement; this overlap is reported, not required to be zero.
- **Plan frames:** 352722, of which 296080 training and 56642 validation. Each frame gives 9 rows, j = 0..8. Dropped for Δt gaps: 3233.
- **GT→token quantisation residual:** median 0.023 m, p99 0.194 m.
- The evaluation script asserts that no P1-A evaluation drive or log appears in the samples.

## Unit tests (`extrap_tests_output.txt`)

```
[PASS] X1 prefix_features invariant to poses after j (no future leakage): max err 0.000e+00 (tol 0e+00)
[PASS] X2 SE(2) to_frame/from_frame round trip: max err 4.263e-14 (tol 1e-09)
[PASS] X3 targets/assemble round trip on training plans: max err 2.383e-06 (tol 1e-05)
[PASS] X4 GT->token port vs stored AutoVLA GT tokens (208 scenes): share mismatching tokens: max err 0.000e+00 (tol 0e+00)
[PASS] X5 tokenizer running pose == decoder rollout: max err 9.232e-06 (tol 1e-04)
[PASS] X6 ego speed feature == |velocity| from the scene JSON: max err 3.261e-09 (tol 1e-06)

ALL PASS
```

## Hyperparameter selection (validation drives only, primary input, seed 0)

|   H |     lr |   final_val_mean_pos_err_m |   final_train_loss | selected   |
|----:|-------:|---------------------------:|-------------------:|:-----------|
| 512 | 0.001  |                   0.72409  |          0.031107  | True       |
| 256 | 0.001  |                   0.734379 |          0.0314992 | False      |
| 512 | 0.0003 |                   0.749104 |          0.032208  | False      |
| 256 | 0.0003 |                   0.765846 |          0.0328106 | False      |

## Validation-drive accuracy (descriptive; mean position error over suffix poses, m)

These are token-domain GT plans; 5,000 validation plans per j.

|   j |    CM |   CTRA-2 |   CTRV-K3 |   CV0 |   prefix_ego_s0 |   prefix_ego_s1 |   prefix_ego_s2 |   prefix_s0 |   prefix_s1 |   prefix_s2 |
|----:|------:|---------:|----------:|------:|----------------:|----------------:|----------------:|------------:|------------:|------------:|
|   0 | 1.561 |    1.561 |     1.561 | 1.66  |           0.976 |           0.973 |           0.975 |       1.573 |       1.572 |       1.574 |
|   1 | 1.336 |    1.041 |     1.445 | 1.388 |           0.82  |           0.817 |           0.819 |       0.872 |       0.872 |       0.877 |
|   2 | 1.094 |    0.871 |     1.328 | 1.134 |           0.663 |           0.661 |           0.662 |       0.681 |       0.681 |       0.682 |
|   3 | 0.866 |    0.675 |     1.081 | 0.897 |           0.512 |           0.507 |           0.511 |       0.522 |       0.523 |       0.523 |
|   4 | 0.658 |    0.505 |     0.85  | 0.69  |           0.384 |           0.38  |           0.383 |       0.387 |       0.389 |       0.389 |
|   5 | 0.474 |    0.359 |     0.639 | 0.5   |           0.275 |           0.273 |           0.275 |       0.275 |       0.277 |       0.275 |
|   6 | 0.324 |    0.243 |     0.457 | 0.346 |           0.192 |           0.191 |           0.191 |       0.188 |       0.189 |       0.189 |
|   7 | 0.204 |    0.156 |     0.305 | 0.219 |           0.128 |           0.131 |           0.13  |       0.126 |       0.125 |       0.127 |
|   8 | 0.119 |    0.099 |     0.185 | 0.127 |           0.09  |           0.09  |           0.087 |       0.085 |       0.085 |       0.084 |

## Coverage of the evaluated prefixes vs the training prefix distribution (`extrap_coverage.csv`)

- Distance: nearest-neighbour distance in standardised primary-feature space.
- Out of range: the perturbed or unperturbed prefix is beyond the validation q99 for the same j.
- **Degenerate reference at j ≤ 1.** Prefixes are decoded tokens, so at j = 0 (and mostly at j = 1) validation prefixes have exact token matches in the training sample. The validation q99 is then 0 (j = 0) or small. 'Out of range' there means the prefix (token sequence) does not occur in the 100k-row training reference sample.
- The ratio columns are NaN where the validation q99 is 0. This is the locked rule, applied as written.

| family   | group   |   n_units |   nn_pert_q50 |   nn_pert_q90 |   nn_pert_max |   ratio_pert_to_val_q99_q50 |   ratio_pert_to_val_q99_q90 |   share_pert_out_of_range |   share_unpert_out_of_range |   share_unit_out_of_range |   n_units_in_range |   n_drives_in_range |
|:---------|:--------|----------:|--------------:|--------------:|--------------:|----------------------------:|----------------------------:|--------------------------:|----------------------------:|--------------------------:|-------------------:|--------------------:|
| G        | A-      |       364 |         0.07  |         0.927 |         4.484 |                     nan     |                     nan     |                     0.082 |                       0.019 |                     0.088 |                332 |                  13 |
| G        | A+      |      1089 |         0.055 |         0.823 |         6.825 |                     nan     |                     nan     |                     0.078 |                       0.006 |                     0.079 |               1003 |                  19 |
| G        | all     |      1453 |         0.062 |         0.839 |         6.825 |                     nan     |                     nan     |                     0.079 |                       0.009 |                     0.081 |               1335 |                  19 |
| E        | A-      |       313 |         0.155 |         1.644 |         5.467 |                     nan     |                     nan     |                     0.23  |                       0.016 |                     0.23  |                241 |                  13 |
| E        | A+      |       887 |         0.09  |         1.279 |        11.644 |                     nan     |                     nan     |                     0.156 |                       0.007 |                     0.157 |                748 |                  19 |
| E        | all     |      1200 |         0.103 |         1.366 |        11.644 |                     nan     |                     nan     |                     0.175 |                       0.009 |                     0.176 |                989 |                  19 |
| GF1      | A-      |       374 |         0.129 |         0.595 |        34.096 |                       0.259 |                       1.199 |                     0.12  |                       0     |                     0.12  |                329 |                  13 |
| GF1      | A+      |      1096 |         0.112 |         0.475 |         2.223 |                       0.226 |                       0.957 |                     0.096 |                       0.005 |                     0.098 |                989 |                  19 |
| GF1      | all     |      1470 |         0.115 |         0.505 |        34.096 |                       0.232 |                       1.019 |                     0.102 |                       0.004 |                     0.103 |               1318 |                  19 |
| GF3      | A-      |       362 |         0.514 |         1.403 |         4.159 |                       0.297 |                       0.811 |                     0.044 |                       0     |                     0.044 |                346 |                  13 |
| GF3      | A+      |      1068 |         0.56  |         1.374 |         3.685 |                       0.324 |                       0.794 |                     0.044 |                       0.011 |                     0.05  |               1015 |                  19 |
| GF3      | all     |      1430 |         0.55  |         1.39  |         4.159 |                       0.318 |                       0.803 |                     0.044 |                       0.008 |                     0.048 |               1361 |                  19 |

Per-j reference distribution:

|   j |   n_train_ref |   n_val |   val_nn_q50 |   val_nn_q90 |   val_nn_q99 |   n_eval_units |
|----:|--------------:|--------:|-------------:|-------------:|-------------:|---------------:|
|   0 |        100000 |   20000 |        0     |        0     |        0     |            871 |
|   1 |        100000 |   20000 |        0     |        0     |        0.496 |           2610 |
|   2 |        100000 |   20000 |        0     |        0.362 |        1.051 |            348 |
|   3 |        100000 |   20000 |        0.098 |        0.686 |        1.73  |           1530 |
|   4 |        100000 |   20000 |        0.229 |        1.036 |        2.403 |             46 |
|   5 |        100000 |   20000 |        0.391 |        1.43  |        3.378 |             95 |
|   6 |        100000 |   20000 |        0.585 |        1.956 |        4.223 |              0 |
|   7 |        100000 |   20000 |        0.839 |        2.567 |        5.695 |             53 |
|   8 |        100000 |   20000 |        1.159 |        3.519 |        7.924 |              0 |

## Primary: family G, M1 endpoint displacement norm, Δ_model − Δ_E (m)

| stratum | extrapolator | analysis | Δ_model − Δ_E [95% CI] | sign-flip p | Holm p (m=2) | drives | seed est. min / max (SD) | judgment |
|---|---|---|---|---|---|---|---|---|
| A- | E:prefix | sensitivity: out-of-range units excluded | +1.91 [+0.97, +2.84] | 0.0073 | NA | 13 | — | — |
| A- | E:prefix_ego | sensitivity: out-of-range units excluded | +2.18 [+1.34, +3.08] | 2.4e-04 | NA | 13 | — | — |
| A- | E:prefix | primary (all units, seed-mean Delta_E) | +1.97 [+1.06, +2.87] | 0.0044 | 0.0044 | 13 | +1.95 / +1.99 (0.021) | model > extrapolator |
| A- | E:prefix_ego | primary (all units, seed-mean Delta_E) | +2.27 [+1.50, +3.10] | 2.4e-04 | 4.9e-04 | 13 | +2.26 / +2.27 (0.004) | model > extrapolator |
| A+ | E:prefix | sensitivity: out-of-range units excluded | -1.18 [-1.54, -0.90] | 5.3e-05 | NA | 19 | — | — |
| A+ | E:prefix_ego | sensitivity: out-of-range units excluded | -0.67 [-0.98, -0.43] | 5.1e-04 | NA | 19 | — | — |
| A+ | E:prefix | primary (all units, seed-mean Delta_E) | -1.11 [-1.44, -0.80] | 1.4e-04 | 2.9e-04 | 19 | -1.12 / -1.09 (0.014) | model < extrapolator |
| A+ | E:prefix_ego | primary (all units, seed-mean Delta_E) | -0.56 [-0.86, -0.29] | 0.0029 | 0.0029 | 19 | -0.58 / -0.54 (0.022) | model < extrapolator |
| all | E:prefix | sensitivity: out-of-range units excluded | -0.42 [-0.84, -0.11] | 0.019 | NA | 19 | — | — |
| all | E:prefix_ego | sensitivity: out-of-range units excluded | +0.04 [-0.37, +0.33] | 0.85 | NA | 19 | — | — |
| all | E:prefix | primary (all units, seed-mean Delta_E) | -0.34 [-0.77, -0.01] | 0.074 | 0.15 | 19 | -0.35 / -0.33 (0.010) | not detected (equivalence not tested) |
| all | E:prefix_ego | primary (all units, seed-mean Delta_E) | +0.15 [-0.28, +0.45] | 0.47 | 0.47 | 19 | +0.13 / +0.16 (0.016) | not detected (equivalence not tested) |

Per-seed rows (seed variance, family G, M1):

| stratum | extrapolator | seed | Δ_model − Δ_E [95% CI] |
|---|---|---|---|
| A- | E:prefix | 0 | +1.96 [+1.08, +2.85] |
| A- | E:prefix | 1 | +1.95 [+1.05, +2.84] |
| A- | E:prefix | 2 | +1.99 [+1.10, +2.88] |
| A- | E:prefix_ego | 0 | +2.26 [+1.49, +3.11] |
| A- | E:prefix_ego | 1 | +2.26 [+1.51, +3.10] |
| A- | E:prefix_ego | 2 | +2.27 [+1.48, +3.14] |
| A+ | E:prefix | 0 | -1.09 [-1.41, -0.79] |
| A+ | E:prefix | 1 | -1.11 [-1.43, -0.81] |
| A+ | E:prefix | 2 | -1.12 [-1.45, -0.82] |
| A+ | E:prefix_ego | 0 | -0.57 [-0.86, -0.29] |
| A+ | E:prefix_ego | 1 | -0.54 [-0.85, -0.27] |
| A+ | E:prefix_ego | 2 | -0.58 [-0.87, -0.32] |
| all | E:prefix | 0 | -0.33 [-0.75, -0.01] |
| all | E:prefix | 1 | -0.35 [-0.78, -0.02] |
| all | E:prefix | 2 | -0.34 [-0.78, -0.01] |
| all | E:prefix_ego | 0 | +0.14 [-0.26, +0.45] |
| all | E:prefix_ego | 1 | +0.16 [-0.26, +0.47] |
| all | E:prefix_ego | 2 | +0.13 [-0.29, +0.44] |

Levels Δ_E (seed mean; family G, M1):

| stratum | extrapolator | Δ_E [95% CI] |
|---|---|---|
| A- | E:prefix | +2.94 [+2.49, +3.30] |
| A- | E:prefix_ego | +2.64 [+2.05, +2.97] |
| A+ | E:prefix | +3.35 [+2.96, +3.66] |
| A+ | E:prefix_ego | +2.80 [+2.49, +3.02] |
| all | E:prefix | +3.24 [+2.90, +3.53] |
| all | E:prefix_ego | +2.76 [+2.44, +2.97] |

## Secondary metrics and families (primary analysis, seed mean)

| family | metric | stratum | E:prefix | E:prefix_ego |
|---|---|---|---|---|
| G | sfx | A- | +0.37 [+0.24, +0.48] * | +0.36 [+0.23, +0.48] * |
| G | sfx | A+ | -0.00 [-0.03, +0.03] | -0.01 [-0.03, +0.02] |
| G | sfx | all | +0.09 [+0.04, +0.12] * | +0.09 [+0.04, +0.12] * |
| G | dref | A- | +3.63 [+2.68, +4.46] * | +3.93 [+2.96, +4.84] * |
| G | dref | A+ | +0.45 [+0.06, +0.75] | +0.94 [+0.42, +1.34] * |
| G | dref | all | +1.24 [+0.67, +1.63] * | +1.68 [+1.02, +2.14] * |
| G | dgt | A- | -2.83 [-3.83, -1.72] * | -2.51 [-3.48, -1.36] * |
| G | dgt | A+ | -0.76 [-1.06, -0.41] * | -0.28 [-0.64, +0.12] |
| G | dgt | all | -1.28 [-1.65, -0.82] * | -0.84 [-1.19, -0.39] * |
| G | area | A- | +3.08 [+1.46, +4.77] * | +3.39 [+2.10, +5.09] * |
| G | area | A+ | -1.74 [-2.33, -1.27] * | -1.04 [-1.61, -0.57] * |
| G | area | all | -0.54 [-1.32, +0.05] | +0.07 [-0.64, +0.63] |
| E | end | A- | +1.08 [-0.06, +2.11] | +0.96 [-0.29, +2.06] |
| E | end | A+ | -1.60 [-1.97, -1.23] * | -1.40 [-1.97, -0.95] * |
| E | end | all | -0.93 [-1.45, -0.51] * | -0.81 [-1.52, -0.26] * |
| E | sfx | A- | +0.34 [+0.19, +0.46] * | +0.27 [+0.10, +0.41] * |
| E | sfx | A+ | -0.06 [-0.09, -0.02] * | -0.08 [-0.14, -0.04] * |
| E | sfx | all | +0.04 [-0.02, +0.08] | +0.01 [-0.07, +0.06] |
| GF1 | end | A- | +0.74 [+0.05, +1.99] | +1.48 [+0.65, +2.73] * |
| GF1 | end | A+ | -1.67 [-2.09, -1.28] * | -0.56 [-0.87, -0.23] * |
| GF1 | end | all | -1.07 [-1.47, -0.69] * | -0.05 [-0.38, +0.31] |
| GF1 | sfx | A- | +0.19 [+0.11, +0.32] * | +0.24 [+0.16, +0.37] * |
| GF1 | sfx | A+ | -0.06 [-0.09, -0.02] * | +0.00 [-0.03, +0.04] |
| GF1 | sfx | all | +0.01 [-0.03, +0.04] | +0.06 [+0.03, +0.10] * |
| GF3 | end | A- | +1.08 [+0.49, +2.12] * | +1.43 [+0.83, +2.41] * |
| GF3 | end | A+ | -0.67 [-0.86, -0.46] * | -0.16 [-0.36, +0.04] |
| GF3 | end | all | -0.23 [-0.43, -0.03] | +0.24 [+0.02, +0.45] |
| GF3 | sfx | A- | +0.25 [+0.16, +0.41] * | +0.28 [+0.19, +0.44] * |
| GF3 | sfx | A+ | +0.01 [-0.02, +0.04] | +0.04 [+0.01, +0.08] |
| GF3 | sfx | all | +0.07 [+0.04, +0.10] * | +0.10 [+0.07, +0.14] * |
(* = Holm m=2 p < 0.05 and CI excludes 0)

## Combined interpretation (protocol section 8; family G, M1)

| stratum   | exceeds_CM   | exceeds_CV0   | exceeds_CTRV-K3   | exceeds_CTRA-2   | exceeds_E_prefix   | sens_sign_kept_E_prefix   | exceeds_E_prefix_ego   | sens_sign_kept_E_prefix_ego   | beyond_motion_extrapolation   | comparators_not_exceeded                    |
|:----------|:-------------|:--------------|:------------------|:-----------------|:-------------------|:--------------------------|:-----------------------|:------------------------------|:------------------------------|:--------------------------------------------|
| A-        | True         | True          | True              | True             | True               | True                      | True                   | True                          | True                          |                                             |
| A+        | False        | True          | False             | False            | False              | True                      | False                  | True                          | False                         | CM, CTRV-K3, CTRA-2, E_prefix, E_prefix_ego |
| all       | False        | True          | True              | False            | False              | True                      | False                  | True                          | False                         | CM, CTRA-2, E_prefix, E_prefix_ego          |

- **A-:** the model's endpoint response exceeds all four kinematic controls and both learned extrapolators, and the out-of-range-excluded sensitivity keeps the sign. 'Beyond motion extrapolation' may be stated for this stratum only, as an open-loop AutoVLA driving result. It is PROVISIONAL until the gate.
- **A+:** not distinguished from motion extrapolation. The model does not exceed: CM, CTRV-K3, CTRA-2, E_prefix, E_prefix_ego. The §6.2 statement must be narrowed for this stratum.
- **all:** not distinguished from motion extrapolation. The model does not exceed: CM, CTRA-2, E_prefix, E_prefix_ego. The §6.2 statement must be narrowed for this stratum.
