# MOTION_BASELINE_RESULTS (P2-A)

**Status: PROVISIONAL.** These are AutoVLA numbers from the P1-A rollouts, the C1 input path. They stay provisional until the AutoVLA P1 no-op/cache parity gate is judged (`outputs/review_followup3/P1/NOOP_CACHE_AUDIT.*`; present when this report was generated). Section 6.2 and Figure 1 of the manuscript are held until the gate passes. Tags: driving domain, AutoVLA, open-loop 5 s plan, POST-HOC analysis of existing rollouts.

**Quantity.** Δ_model − Δ_b, where Δ = φ(perturbed) − φ(unperturbed) is computed per method.
- Units are paired; the horizon is 10 poses.
- Scenes are equally weighted.
- Inference uses drive clusters: bootstrap with B = 10,000 and an exact drive sign-flip test.
- Holm m = 4 over the kinematic baselines within each (family, stratum, metric).
- Geo is privileged: it uses the model's unperturbed future r[t+1:]. It is reported outside the Holm family.
- A positive value means the model moves more than the baseline. 'Not detected' is not equivalence.
- Source: `MOTION_BASELINE_RESULTS.csv`. Figure: `fig_p2_motion_baselines.pdf/.png`.

## Primary: family G, M1 endpoint displacement norm (m)

| stratum | baseline | Δ_model − Δ_b [95% CI] | sign-flip p | Holm p (m=4) | drives | judgment |
|---|---|---|---|---|---|---|
| A- | CM | +2.35 [+1.27, +3.19] | 7.3e-04 | 0.0015 | 13 | model > baseline |
| A- | CV0 | +4.02 [+2.94, +4.83] | 2.4e-04 | 9.8e-04 | 13 | model > baseline |
| A- | CTRV-K3 | +3.02 [+1.93, +3.87] | 2.4e-04 | 9.8e-04 | 13 | model > baseline |
| A- | CTRA-2 | +1.43 [+0.44, +2.33] | 0.023 | 0.023 | 13 | model > baseline |
| A- | Geo | +4.37 [+3.33, +5.20] | 2.4e-04 | NA | 13 | CI excludes 0 (unadjusted; not in Holm family) |
| A+ | CM | -0.52 [-0.82, -0.27] | 0.0019 | 0.0039 | 19 | model < baseline |
| A+ | CV0 | +1.35 [+0.98, +1.63] | 1.5e-05 | 6.1e-05 | 19 | model > baseline |
| A+ | CTRV-K3 | +0.17 [-0.14, +0.45] | 0.31 | 0.31 | 19 | not detected (equivalence not tested) |
| A+ | CTRA-2 | -1.56 [-1.86, -1.27] | 3.1e-05 | 9.2e-05 | 19 | model < baseline |
| A+ | Geo | +1.64 [+1.28, +1.93] | 3.8e-06 | NA | 19 | CI excludes 0 (unadjusted; not in Holm family) |
| all | CM | +0.20 [-0.33, +0.56] | 0.51 | 0.51 | 19 | not detected (equivalence not tested) |
| all | CV0 | +2.01 [+1.46, +2.37] | 7.6e-06 | 3.1e-05 | 19 | model > baseline |
| all | CTRV-K3 | +0.88 [+0.37, +1.25] | 0.0025 | 0.005 | 19 | model > baseline |
| all | CTRA-2 | -0.81 [-1.24, -0.51] | 1.9e-04 | 5.7e-04 | 19 | model < baseline |
| all | Geo | +2.32 [+1.78, +2.68] | 3.8e-06 | NA | 19 | CI excludes 0 (unadjusted; not in Holm family) |

Levels Δ_method (M1, family G; descriptive):

| stratum | method | Δ [95% CI] |
|---|---|---|
| A- | model | +4.91 [+3.84, +5.72] |
| A- | CM | +2.56 [+2.34, +2.80] |
| A- | CV0 | +0.88 [+0.85, +0.91] |
| A- | CTRV-K3 | +1.89 [+1.72, +2.14] |
| A- | CTRA-2 | +3.47 [+2.89, +3.88] |
| A- | Geo | +0.53 [+0.43, +0.61] |
| A+ | model | +2.24 [+1.85, +2.54] |
| A+ | CM | +2.76 [+2.61, +2.89] |
| A+ | CV0 | +0.89 [+0.86, +0.92] |
| A+ | CTRV-K3 | +2.07 [+1.87, +2.23] |
| A+ | CTRA-2 | +3.80 [+3.45, +4.09] |
| A+ | Geo | +0.60 [+0.54, +0.64] |
| all | model | +2.91 [+2.34, +3.28] |
| all | CM | +2.71 [+2.59, +2.81] |
| all | CV0 | +0.89 [+0.87, +0.91] |
| all | CTRV-K3 | +2.03 [+1.86, +2.14] |
| all | CTRA-2 | +3.72 [+3.40, +3.95] |
| all | Geo | +0.58 [+0.53, +0.62] |

## Secondary metrics, family G (kept separate; they never override M1)

### M2 suffix local-motion change (m/0.5 s)

| stratum | CM | CV0 | CTRV-K3 | CTRA-2 | Geo |
|---|---|---|---|---|---|
| A- | +0.48 [+0.34, +0.59] * | +0.50 [+0.36, +0.61] * | +0.52 [+0.38, +0.63] * | +0.35 [+0.20, +0.48] * | +0.58 [+0.44, +0.70] |
| A+ | +0.12 [+0.09, +0.16] * | +0.15 [+0.11, +0.18] * | +0.16 [+0.13, +0.20] * | -0.02 [-0.06, +0.01] | +0.23 [+0.19, +0.26] |
| all | +0.21 [+0.15, +0.25] * | +0.24 [+0.17, +0.28] * | +0.25 [+0.19, +0.29] * | +0.07 [+0.02, +0.11] * | +0.32 [+0.26, +0.36] |

### M3 signed error change vs reference r (m)

| stratum | CM | CV0 | CTRV-K3 | CTRA-2 | Geo |
|---|---|---|---|---|---|
| A- | +4.02 [+2.96, +4.80] * | +4.69 [+3.67, +5.47] * | +4.36 [+3.34, +5.11] * | +3.62 [+2.51, +4.36] * | +4.28 [+3.25, +5.08] |
| A+ | +1.21 [+0.80, +1.53] * | +2.06 [+1.68, +2.35] * | +1.49 [+1.13, +1.78] * | +0.64 [+0.17, +0.99] * | +1.54 [+1.19, +1.80] |
| all | +1.91 [+1.28, +2.34] * | +2.71 [+2.16, +3.09] * | +2.21 [+1.60, +2.63] * | +1.38 [+0.71, +1.82] * | +2.22 [+1.70, +2.59] |

### M4 signed GT error change (m)

| stratum | CM | CV0 | CTRV-K3 | CTRA-2 | Geo |
|---|---|---|---|---|---|
| A- | -2.57 [-3.68, -1.53] * | -2.12 [-3.11, -1.05] * | -2.35 [-3.44, -1.25] * | -2.84 [-3.95, -1.71] * | -2.07 [-3.08, -0.99] |
| A+ | -0.13 [-0.48, +0.26] | +0.70 [+0.41, +0.99] * | +0.14 [-0.20, +0.50] | -0.62 [-0.97, -0.22] | +0.61 [+0.33, +0.89] |
| all | -0.74 [-1.14, -0.27] * | -0.00 [-0.34, +0.39] | -0.49 [-0.89, -0.01] | -1.18 [-1.60, -0.69] * | -0.06 [-0.39, +0.34] |

### M5 path area (m·s)

| stratum | CM | CV0 | CTRV-K3 | CTRA-2 | Geo |
|---|---|---|---|---|---|
| A- | +3.74 [+1.85, +5.16] * | +6.53 [+4.56, +8.00] * | +4.87 [+2.83, +6.27] * | +2.33 [+0.70, +3.88] * | +7.38 [+5.51, +8.86] |
| A+ | -0.74 [-1.29, -0.26] * | +2.39 [+1.70, +2.95] * | +0.41 [-0.18, +0.92] | -2.37 [-2.91, -1.91] * | +3.11 [+2.40, +3.67] |
| all | +0.38 [-0.53, +1.02] | +3.43 [+2.45, +4.07] * | +1.52 [+0.59, +2.20] * | -1.19 [-1.90, -0.68] * | +4.18 [+3.20, +4.84] |

(* = Holm m=4 p < 0.05 and CI excludes 0)

## Other perturbation families, M1 (secondary)

Family E is GT-anchored. Families GF1 and GF3 are fixed-position and GT-free.

| family | stratum | CM | CV0 | CTRV-K3 | CTRA-2 | Geo |
|---|---|---|---|---|---|---|
| E | A- | +1.89 [+0.28, +3.25] | +4.77 [+3.66, +5.74] * | +3.04 [+1.48, +4.42] * | -0.12 [-1.77, +1.18] | +5.63 [+4.37, +6.77] |
| E | A+ | -0.94 [-1.27, -0.68] * | +1.30 [+0.88, +1.60] * | -0.00 [-0.25, +0.20] | -2.64 [-3.23, -2.08] * | +1.79 [+1.33, +2.13] |
| E | all | -0.24 [-0.87, +0.26] | +2.17 [+1.51, +2.58] * | +0.76 [+0.21, +1.20] * | -2.01 [-2.83, -1.36] * | +2.75 [+2.04, +3.22] |
| GF1 | A- | +1.52 [+0.48, +2.60] * | +3.51 [+2.84, +4.44] * | +2.69 [+1.88, +3.61] * | -0.13 [-1.80, +1.62] | +3.78 [+3.07, +4.72] |
| GF1 | A+ | -0.62 [-1.01, -0.32] * | +1.31 [+0.96, +1.58] * | +0.53 [+0.16, +0.82] * | -2.48 [-2.90, -2.01] * | +1.60 [+1.27, +1.88] |
| GF1 | all | -0.09 [-0.54, +0.26] | +1.86 [+1.47, +2.14] * | +1.07 [+0.65, +1.38] * | -1.89 [-2.47, -1.30] * | +2.15 [+1.77, +2.44] |
| GF3 | A- | +1.38 [+0.73, +2.36] * | +2.48 [+1.75, +3.37] * | +2.31 [+1.64, +3.23] * | +0.40 [-0.26, +1.65] | +2.70 [+2.04, +3.62] |
| GF3 | A+ | -0.41 [-0.68, -0.19] * | +0.90 [+0.65, +1.14] * | +0.63 [+0.37, +0.86] * | -1.45 [-1.74, -1.12] * | +1.07 [+0.83, +1.30] |
| GF3 | all | +0.04 [-0.28, +0.30] | +1.30 [+1.00, +1.51] * | +1.05 [+0.73, +1.29] * | -0.99 [-1.29, -0.61] * | +1.48 [+1.19, +1.69] |

## Decision (locked rule 2.4, kinematic part only; family G, M1)

| stratum | exceeds CM | exceeds CV0 | exceeds CTRV-K3 | exceeds CTRA-2 |
|---|---|---|---|---|
| A- | True | True | True | True |
| A+ | False | True | False | False |
| all | False | True | True | False |

The combined verdict, which adds the learned extrapolator, is in `LEARNED_EXTRAPOLATOR_RESULTS.md`.
