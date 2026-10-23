# ESTIMAND_AND_METHOD_AUDIT — AutoVLA effects, weights and intervals (CVPR next validation, 2026-10-19)

CPU audit of code and stored outputs; no estimate changed. Inputs and hashes: `INPUT_HASHES.csv`.

## 1. What each result estimates

| result | data | estimand as computed | what a drive-cluster CI targets |
|---|---|---|---|
| dev A− RN −39.8 pp / RV +28.6 pp (exp 7 / R8) | the 52 A− scenes of the selected dev mechanism set (all A− of the 2,747 dev scenes, π = 1), 13 drives | descriptive effect on the observed selected set (scene-weighted mean of scene-mean differences) | a superpopulation of drives like these 13, **conditional on the A− selection rule**; the 13 drives are not a probability sample |
| P1-C F RN −34.2 pp / RV +26.8 pp | census of the 105 F scenes of the P1-C frame (π = 1), 18 drives | finite-population F effect: no scene-sampling variance, only outcome stochasticity | superpopulation of drives (the 24 P1-C drives were chosen as new evaluation logs, not randomly sampled) |
| P1-C E RN −4.1 pp / RV +3.6 pp | 2,191-scene eligible frame E; F census (105) + S sampled per log (m = 4) → 353 scenes, 24 drives | finite-population E effect, Hajek ratio Σ w d / Σ w | finite population **and** superpopulation readings differ (see §4) |
| B2 F C_GT / C_NAT / INT / C_PROV | 100 F scenes (census of E4 F), 711 unit-anchors, 18 drives | finite-population F effect within the B2 E4 domain | superpopulation of drives |
| B2 E4 | 1,775-scene domain (t* ≤ 4); F census 100 + S sampled per log (m_S_dom) → 312 scenes, 24 drives | finite-population E4 effect, Hajek with weights recomputed for the domain | as for E |
| GT-free endpoint excess (B_NUMERIC) | 208 selected dev scenes (A− 52, A+ 156), family G | descriptive effect on the selected set per stratum | superpopulation of drives, conditional on selection |

## 2. How the code builds E / E4

- **Stage-1 selection** (`P1C_new_log_replication/run_20261008/selection/shard_*.json`): per log, `denominators` holds
  `eligible`, `F`, `S_eligible`, `S_sampled`; each `selected` scene carries `stratum`, `pi`, `weight` = N_S,seg / m_seg,
  `N_S_seg`, `m_seg` (segment = log, m = 4), and the sampling order key (salted sha256). F scenes have π = 1, weight 1.
- **Analysis** (`scripts/review_followup/p1c/analyze_p1c.py`): `scene_table` pairs the two conditions per unit, averages
  to the scene (`d` = scene mean, `w` = scene weight); `estimand_table` keeps F, S or all (P = E); `cluster_arrays`
  (scene weighting) forms drive totals N_g = Σ w_s d_s and D_g = Σ w_s; θ = Σ N / Σ D (Hajek); cluster-t uses ratio
  linearisation with G − 1 df; the bootstrap resamples drives (F and S of a drive together).
- **B2 E4** (`review_followup4/B/B2/p1c_cohort_t4.json`, `scripts/review_followup4/b/b2_analyze.py`): per scene
  `stratum`, `weight` = N_S_dom / m_S_dom (recomputed for the t* ≤ 4 domain), `pi`, `log`, `drive`, `day_vehicle`;
  `est()` calls the same `analyze_p1c.estimate` with drive (or day–vehicle) clusters.
- **INT** (`b2_analyze.int_tab`): the unit-level contrasts INT_a = GT_NSRC − GT_REF and INT_b = NAT_NSRC − NAT_REF are
  joined **by unit_id**, INT_u = INT_a − INT_b is formed per unit (all four cells of the same unit), then averaged to the
  scene and aggregated to drives. The joint four-cell structure is therefore kept; INT is not computed as a difference of
  two separately estimated contrasts.

## 3. p values vs intervals: different assumptions

- The δ = 0 sign-flip p (exact enumeration of 2^G drive sign patterns) is valid under the null that drive totals are
  independent and sign-symmetric. It is not a randomization test (treatment was not randomly assigned to drives) and it
  says nothing about the size of the effect.
- An effect-size interval needs a model of the sampling distribution of θ̂ under non-null effects (cluster-t: approximate
  normality of the linearised ratio with G − 1 df; bootstrap: drives as an i.i.d. sample). These assumptions are stronger
  and can fail with few, unequal, heterogeneous clusters and design weights. Withholding or qualifying an interval does
  **not** certify the p value in general; the p value rests on its own (symmetry / independence) assumptions.

## 4. Scope of the simulations

- review_followup5 and the author-query check (2026-10-19) **fixed the observed scenes** and generated within-scene
  outcomes; coverage was judged against the replicate's conditional true value. That speaks to the finite-population /
  observed-scene estimand, with between-drive variation entering only through an assumed random drive shift.
- This round adds a **superpopulation design** (D2: drives re-sampled with replacement, each with its own S re-sampling and
  outcomes) and a **finite-population design** (D3: S re-sampled within the real frame of each log), both with the real
  weights (`STATISTICAL_CHECK_PLAN.md`). D2 is still bounded by the 24 observed drives (their scene structures and
  effect heterogeneity); it is not a guarantee for arbitrary new drives.
