# STATISTICAL_CHECK_PLAN — weighted E / E4 and INT interval check (CVPR next validation, 2026-10-19)

Fixed and hashed (`STATISTICAL_CHECK_PLAN_LOCK.json`) **before** any result of this check. Post-hoc: every contrast below
has been analysed before. CPU only. Code: `scripts/cvpr_next_validation_20261019/sim_weighted.py`.

## 1. Estimands

- **Finite-population (design-based) E / E4 effect:** scene-weighted mean, over all eligible scenes of the frame, of the
  scene-mean paired difference (treatment − reference). F scenes are a census (π = 1); S scenes are sampled per log
  (segment) without replacement, m per log, weight N_S,log / m (Hajek ratio Σ w d / Σ w).
- **Superpopulation (new drives) effect:** the same ratio for drives drawn from the population of drives that the 24
  observed drives represent (operationalised by resampling the observed drives with replacement).
- F-only variants (census within drive) are run under the superpopulation design for comparison.

## 2. Pseudo-population (assumption, not data)

Outcomes of non-sampled S scenes are unknown. For every log the S part of the frame (N_S,log eligible scenes) is filled by
cycling that log's sampled S scenes (each pseudo-scene carries the observed unit-level differences of its source scene).
F scenes are the observed census. This hypothetical population has the real frame sizes, real per-log sampling fractions,
real unit counts per scene and real weights; its S outcome heterogeneity is limited to the observed sampled scenes.

## 3. Effect scenarios via within-unit label swaps

Swapping the two arms of a unit flips the sign of its difference (for INT: swapping the GT and NAT older-prefix labels of
all four cells flips the sign of the unit's difference-of-differences, keeping the joint four-cell structure). With swap
probability 1 − q per unit (keep with probability q):
- q = 0.5 → true effect 0 (null); q = 0.75 → half the observed-scale effect, same sign; q = 0.25 → half, opposite sign;
  q = 1 → observed scale. True θ(q) = (2q − 1) · θ_pop, θ_pop = pseudo-population mean of the observed scene means
  (reported; no clipping occurs in this construction).
- Two coupling assumptions: **U** = independent swaps per unit; **S** = one swap per scene (all units of a scene together;
  keeps within-scene correlation).

## 4. Designs

- **D3 (finite population, fixed drives):** per replicate, S scenes re-sampled per log (m without replacement), F census,
  label swaps.
- **D2 (superpopulation):** per replicate, 24 (or the cohort's G) drives re-sampled with replacement (each copy with its own
  independent within-drive sampling and swaps), then D3 inside each drive.

## 5. Contrasts (all scene-weighted, drive clusters)

| id | estimand | contrast | weights |
|---|---|---|---|
| E_RN | P1-C E (2,191 frame; 353 sampled) | recent_gt − normal (H7), amplification | P1-C design (m = 4 per log) |
| E_RV | P1-C E | reverse@emb − gt_history (H8) | P1-C design |
| E4_CNAT | B2 E4 (1,775 frame; 312 sampled) | NAT_NSRC − NAT_REF, severe | B2 E4 design (m per log from the cohort file) |
| E4_CGT | B2 E4 | GT_NSRC − GT_REF | same |
| E4_INT | B2 E4 | (GT_NSRC − GT_REF) − (NAT_NSRC − NAT_REF), unit level | same |
| F_RN | P1-C F (105 / 18 drives) | as E_RN | weight 1 (D2 only) |
| F_CNAT | B2 F (100 / 18 drives) | as E4_CNAT | weight 1 (D2 only) |
| F_INT | B2 F | as E4_INT | weight 1 (D2 only) |

## 6. Methods (only the two already reported; no method shopping)

1. Cluster-robust t (ratio linearisation, G − 1 df) — the interval used so far for F rows.
2. Drive percentile bootstrap (B = 999) — the interval reported originally.

## 7. Replicates and performance

Cluster-t: R = 5,000 per scenario; bootstrap: R = 1,000 per scenario (cost). Report coverage, bias (mean estimate − θ),
mean width, P(zero width), failure rate (non-finite interval), MC SE and Wilson 95% MC interval. Seed 20261019.

## 8. Decision rule (fixed now)

An interval is called **adequate in the checked scenarios** only if, for both designs D3 / D2 (D2 only for F) and both
couplings, every scenario's Wilson MC upper bound is ≥ 0.94 (nominal 95%). Otherwise **APPROXIMATE_ONLY** if the minimum
coverage is ≥ 0.90, else **WITHHOLD**. No method, scenario or parameter is changed after results are seen; changes go to
`AMENDMENTS.md` with what was seen.

## 9. Not in scope

New drives' scene-structure distribution beyond the observed 24 drives; finite-population variation of F (census);
other weightings; CI recovery for day–vehicle or component clustering.
