# CI_VALIDITY_AUDIT — degenerate sign-flip-inversion CIs for paired success rates (review_followup4 A1)

Date: 2026-10-17. Status: **POST-HOC CORRECTION / SENSITIVITY.** CPU only (`nice -n 19`), no model run, no GPU. Nothing in review_followup2/3, numbers.json or the paper package was modified. The original analysis stays on record and is shown next to the corrected one.

Code: `scripts/review_followup4/a1_paired_ci.py` (methods + verification), `a1_coverage_sim.py` (coverage study + pre-specified selection rule), `a1_ci_audit.py` (real-data recomputation).
Tables (this directory): `CI_VALIDITY_AUDIT.csv` (984 rows, old vs new), `A1_JUDGMENT_CHANGES.csv` (36 changes), `A1_OPENVLA_2x2.csv` (165 paired 2x2 tables), `A1_CALLSITES.csv`, `A1_COVERAGE_SIM_iid.csv`, `A1_COVERAGE_SIM_cluster.csv`, `A1_TOST_SIZE_SIM.csv`, `A1_METHOD_SELECTION.json`, `A1_DEGENERACY_DEMO.json`, `a1_audit_meta.json`, `a1_coverage_sim.log`.

Run order (about 5 minutes on CPU):

    cd /root/VLA/autovla_misalignment_poc/scripts/review_followup4
    nice -n 19 /root/miniforge3/envs/autovla/bin/python a1_paired_ci.py        # verification checks
    nice -n 19 /root/miniforge3/envs/autovla/bin/python a1_coverage_sim.py     # writes A1_METHOD_SELECTION.json first
    nice -n 19 /root/miniforge3/envs/autovla/bin/python a1_ci_audit.py         # reads the selection, then the data

## 1. Reconstructed 2x2 tables for the flagged rows

Counts are a11/b10/c01/d00: both succeed / only arm a succeeds / only arm b succeeds / both fail. The unit is one LIBERO-Spatial initial state with one paired episode per condition. All 165 tables are in `A1_OPENVLA_2x2.csv`.

| row | table | estimate | OLD 95% CI (sign-flip inversion) | NEW 95% CI (Newcombe 10) | Tango score 95% CI | exact McNemar p |
|---|---|---|---|---|---|---|
| P0E.succ_task.C-F_d8.t5 | 10/0/0/0 | 0 | [−2.5e−13, +2.5e−13] (zero width) | [−0.278, +0.278] | [−0.278, +0.278] | 1 |
| P0E.succ_task.C-F_d24.t6 | 10/0/0/0 | 0 | [−2.5e−13, +2.5e−13] (zero width) | [−0.278, +0.278] | [−0.278, +0.278] | 1 |
| P0E.succ_task.C-F_d8.t7 (example of a one-sided case) | 7/0/1/2 | −0.10 | [−0.333, +2.5e−13] (upper end pinned at 0) | [−0.364, +0.173] | [−0.404, +0.205] | 1 |

Across the 150 OpenVLA per-task rows:
- 11 have zero-width CIs: all 10 differences are 0 (C-F d8 t5, C-F d24 t6, F-N d8 t2, R-N d8 t1/t6/t9, F-Nrep d8 t9, C-Nrep d8 t2/t3/t8, R-Nrep d8 t2).
- 75 have one end pinned at 0 ± 1e−13. In these rows every discordant state has the same sign, often only one state.

There is also one degenerate row at state level: OpenVLA C−N d8 (6 vs 14 discordant). Its upper end is −7e−14, so the old CI "excluded 0" by a numerical artefact even though p = 0.115.

In SpatialVLA, 37 of the 819 DIRECT_EFFECTS rows have an end pinned at 0. Examples:
- exp35 r1e4 C−F move_near primary: [−29.1, +5e−11] pp, p = 0.058.
- exp35 r1e4 R−N coke: [−15.0, +7e−11] pp, p = 0.079.

## 2. Why the inversion collapses (reproduced; `A1_DEGENERACY_DEMO.json`)

`stats_core.signflip_ci(N, D, θ)` keeps a value δ when the two-sided sign-flip p of the shifted cluster totals T_g = N_g − δ·D_g is above 0.05.

- **All 10 differences are 0.** At δ = 0 every T_g = 0, so p = 1. Once δ passes the 1e−12·max(1, Σ|T|) tie tolerance (|δ| > 2.4958e−13), all 10 shifted totals equal −δ and have the same sign. The observed sum is then the most extreme of the 2^10 patterns, so p = 2/1024 = 0.002 and δ is rejected. Bisection therefore converges to the tolerance, which gives exactly the ±2.4958e−13 stored in numbers.json. This is reproduced bit-for-bit.
- **One discordant state (−1), nine zeros.** Any δ > tol makes all ten shifted totals negative, so p = 0.002. The upper end is pinned at 0, although the exact McNemar p is 1.

The cause is a broken assumption, not a coding slip. The test and its inversion require the shifted totals to be symmetric about 0 under H0: θ = δ. For a paired difference d ∈ {−1, 0, 1} with zeros, that can hold only at δ = 0, because every zero becomes −δ. So:
- the sign-flip test at δ = 0, i.e. the McNemar null of symmetric discordance, remains valid;
- the inversion CI and the shifted sign-flip TOST (`stats_core.tost(method="signflip")`) at δ = ±margin do not.

A valid zero-null McNemar test does **not** make the margin tests or the population risk-difference CI valid.

## 3. Estimand

- **OpenVLA.** The difference in success probability, P(success | arm a) − P(success | arm b), for a new initial state drawn from the same LIBERO-Spatial init-file distribution, including run-to-run execution randomness.
  - That randomness is real: re-running the unperturbed policy changes 20% of outcomes.
  - Caveat: the 10 states per task are the first 10 of the benchmark init file, not a random sample.
- **Fixed-sample alternative.** The difference over these 100 states only. Even here outcomes are random given the state, so a zero-width interval is not valid for this estimand either. Zero width would hold only if outcomes were deterministic, and then the result is a count, not an interval.
- **Status of the old intervals.** The old sign-flip intervals do not target either estimand at the boundary. They are not evidence of equality and must not be cited.
- **SpatialVLA.** The original estimand is kept unchanged.
  - Within a task: the unit-weighted mean paired difference over the 120 (or 40) units, i.e. the difference of success rates on the same units.
  - Pooled: the task-equal mean of the two task estimates.
  - Clusters: repeated instruction-inclusive initial states ("primary": move_near 53 in exp35, 29 in exp34) or physical layouts ("physical": move_near 10).
  - The "episode" clustering is the original unit and is kept for reproduction only.

## 4. Candidate methods (all paired; no independent-proportion formula)

| method | where used | notes |
|---|---|---|
| OLD sign-flip inversion | audited | Exact trinomial re-implementation equals `stats_core.signflip_ci` / `tost` to 3.6e−15 on 120 random inputs |
| Wald (Newcombe method 1) | reference | Degenerates when b = c = 0 |
| **Newcombe 10** (MOVER-Wilson on the two marginals, φ correction) | iid units | Matches R `ratesci` 1.1.0 reference values to 5e−11 (20 table × α cases; values taken from the statsmodels main test suite). statsmodels 0.14.6, the newest version installable for python 3.9, has **no** `confint_proportions_paired`, so it was not installed or used |
| Tango (1998) asymptotic score interval | sensitivity | Closed-form restricted MLE checked against brute-force maximisation (max diff 6e−9) |
| **Rao–Scott design-effect Newcombe 10** | clustered units | The 2×2 table is divided by deff = clip(V_cluster/V_iid of d̄, 1, Σm_g²/n). When no pair is discordant, the ICC = 1 bound Σm_g²/n is used (conservative). Pooled rows: MOVER for the task-equal mean of the two independent task intervals. Interactions: MOVER difference with the cluster-robust correlation of the two paired differences |
| cluster-robust t, cluster bootstrap percentile | clustered comparators | Degenerate when every cluster total is equal |

## 5. Coverage study (`A1_COVERAGE_SIM_*.csv`)

The selection rule was written in `a1_coverage_sim.py` before the real data were recomputed, and the simulation does not read the real outcomes. It was:
1. exclude any method that returns a zero-width interval with positive probability while the true discordance is > 0;
2. among the rest, pick the highest minimum coverage (required ≥ 0.925);
3. break ties by mean |coverage − 0.95|, then by width.

**iid pairs, exact enumeration of all 2×2 tables (8 scenarios × n = 10, 100).** Scenarios cover high and low success, 0–2% discordance including p01 = 0, and an OpenVLA-like case.

| method | min coverage n = 100 | min coverage n = 10 | max P(zero width) |
|---|---|---|---|
| old sign-flip inversion | 0.633 | 0.096 | 0.904 |
| Wald | 0.633 | 0.096 | 0.904 |
| **Newcombe 10** | **0.951** | 0.991 | 0 |
| Tango | 0.940 | 0.987 | 0 |

- The old inversion behaves exactly like the Wald interval.
- **Selected for iid units: Newcombe 10.** Tango is reported as a sensitivity (`tango_ci95` column). Its state-level OpenVLA intervals differ from Newcombe by ≤ 1 pp.

**Clustered (Monte Carlo, R = 1,000, seed 20261017).** Cluster sizes are the real move_near designs (10 physical layouts; 53 primary states) plus 120 singletons. Cluster cell probabilities follow Dirichlet(κπ) with κ = 2 (strong clustering) or 10, at 4 discordance levels. Two interaction scenarios have 4 outcomes per unit.

| design | old inversion | cluster t | cluster bootstrap | **RS-Newcombe** | naive Newcombe (ignores clusters; reference) |
|---|---|---|---|---|---|
| G = 10 (min coverage) | 0.713 | 0.778 | 0.693 | **0.937** | 0.616 |
| G = 53 | 0.927 | 0.936 | 0.902 | **0.948** | 0.857 |
| 120 singletons | 0.873 | 0.947 | 0.919 | **0.960** | 0.960 |
| interaction, G = 10 / 53 | 0.928 / 0.938 | 0.942 / 0.947 | – | MOVER **0.980 / 0.982** | – |

- **Selected for clustered units: RS-Newcombe (MOVER for pooled and interaction rows).** No regime had to be marked NR.
- The cluster-robust t was excluded at G = 53 by rule 1 (zero-width P = 0.021) even though its minimum coverage was 0.936. This follows the pre-specified rule and does not mean the cluster t is badly calibrated there.
- Caveat: at G = 10 the RS-Newcombe coverage is only approximately nominal (0.937–0.993; MC SE 0.007). Interactions are conservative (0.98).

**Equivalence at the margin** (`A1_TOST_SIZE_SIM.csv`), giving P(declare equivalence | true |δ| = m):

| procedure | n = 100, δ = −0.10, m = 0.10 | n = 100, δ = +0.05, m = 0.05 | n = 120, δ = +0.10, low discordance |
|---|---|---|---|
| old sign-flip TOST | 0.056 | **0.079** | **0.076** |
| Newcombe 90% CI inside ±m | 0.048 | 0.042 | 0.050 |
| Tango | 0.048 | 0.024 | 0.042 |

- The old TOST is anti-conservative at the margin when discordance is low (iid).
- Clustered boundary scenarios: RS-Newcombe 0.004–0.063, old 0.010–0.063.

## 6. Old vs new judgments

These are POST-HOC corrections. Two things are unchanged:
- the difference-from-0 tests: the sign-flip / exact McNemar p at δ = 0 is valid and unchanged, and so are the Holm families;
- the margins (±10 pp original / pre-registered, ±5 pp post hoc) and the families.

New TOST p values are obtained by CI inversion of the selected interval. Holm is applied within the same family × task level × clustering as before. All rows: `CI_VALIDITY_AUDIT.csv`. Changed judgments: `A1_JUDGMENT_CHANGES.csv`.

**OpenVLA (100 states; families F1–F3 Holm):**

| row | table | est | OLD CI | NEW CI (Newcombe) | old p_equiv ±10 | new p_equiv ±10 | judgment change |
|---|---|---|---|---|---|---|---|
| C−F d8 | 71/8/11/10 | −3 pp | [−11.8, +5.9] | [−11.9, +5.9] | 0.059 | 0.060 | none (not different; equivalence not established) |
| C−F d24 | 72/7/12/9 | −5 pp | [−13.6, +3.9] | [−13.9, +3.8] | 0.134 | 0.132 | none |
| C−N d8 | 73/6/14/7 | −8 pp | [−16.7, **−7e−14**] | [−17.1, **+1.0**] | 0.335 | 0.329 | **old "95% CI excludes 0" was a tie artefact.** Now includes 0, consistent with p = 0.115 |
| R−Nrep d8 | 74/10/7/9 | +3 pp | [−5.0, +11.1] | [−5.5, +11.5] | 0.049 | 0.052 | **unadjusted ±10 equivalence lost** (Holm 0.20 → 0.21; was never Holm-significant) |
| R−N d8, F−Nrep d8, C−Nrep d8 | – | – | – | – | 0.029 / 0.022 / 0.019 | 0.037 / 0.024 / 0.022 | unadjusted equivalence kept; none survives Holm (old or new) |

- **Per-task OpenVLA intervals (150 rows): old intervals invalid.** The new Newcombe intervals are valid for the new-state estimand but uninformative: 10 states give widths of 0.35–0.79. For the all-concordant tasks the CI is ±0.18 to ±0.28, depending on the margins.
- These rows are **descriptive only**. No per-task conclusion, including "no difference in task 5", may be drawn.

**SpatialVLA.** The P1-D pooled equivalence claims all **survive**:

| row | Holm p_equiv ±10, primary (old → new) | Holm p_equiv ±10, physical (old → new) |
|---|---|---|
| r1e4 δR | 0.0002 → 0.0007 | 0.0016 → 0.0067 |
| r1e4 δC | 0.0059 → 0.0057 | 0.0021 → 0.0086 |
| r4e1 δR | 0.0059 → 0.0057 | 0.010 → 0.011 |
| r4e1 δC | 0.0059 → 0.0046 | 0.014 → 0.011 |

Other results:
- **Per-task ±10 equivalence (Holm, primary): still 3 of 8.** These are r1e4 R−N coke (0.003 → 0.007), r1e4 R−N move_near (0.037 → 0.042) and r1e4 C−F move_near (0.037 → 0.042). ±5 pp holds for none.
- **P1-D S1–S4 (pre-registered secondary family): pooled ±10 equivalence survives in all 4** (Holm 0.009–0.039 new vs 0.014–0.047 old). The per-task S rows are still not equivalent.
- **Changed judgments:**

| row | old | new | detail |
|---|---|---|---|
| **exp34 r1e4/perp_left C−F, coke** (all clusterings; 1 vs 1 discordant of 40 units) | ±10 equivalent (Holm) | unadjusted only | Old p_equiv 0.0003 (Holm 0.001) came from the invalid shifted sign-flip. New 95% CI [−11.3, +11.3], p_equiv 0.037, Holm 0.147. **The F−C mirror row changes in the same way** |
| P1-D r2e1 − r1e1 C−F interaction, pooled, physical | ±10 equivalent (Holm) | not established | Holm 0.0446 → 0.0512 (borderline) |
| P1-D r1e4 C−F move_near, physical (3 informative clusters of 10) | CI [−2.6, +4.3], unadjusted ±5 equivalence (p 0.014) | CI [−6.9, +8.6]; ±5 not established | The old interval was an artefact. This also changes the "r1e4 6.9–21.6 pp" task-CI width statement: new widths are r1e4 13.1–20.1 pp and r4e1 18.2–25.3 pp (old 18.1–29.3) |
| exp35 r1e4 C−F move_near, primary | 95% CI [−29.1, +0.0] | [−29.1, −0.3]: now excludes 0 | **The difference judgment is unchanged** (sign-flip p = 0.058, Holm 0.23). The interval barely excludes 0 while the δ = 0 test does not reject. This is a boundary disagreement between two valid procedures, not evidence of an effect. Report it as "not different from 0 (Holm)" |
| exp35 r2e2 F−N coke; exp35 r4e1 C−N move_near physical; exp35 r4e1 − r1e4 R−N interaction move_near (episode clustering) | old CI end at 0 | new CI excludes 0 | Same remark. Aux / episode-level rows only |
| exp35 r2e1 F−N, pooled, physical | old CI excluded 0 | new CI includes 0 | – |
| Borderline unadjusted flips near p = 0.05 (17 rows, counting mirror and clustering duplicates) | – | – | e.g. P1-D r1e4 C−F coke 0.049 → 0.052 (unadjusted equivalence lost); P1-D r4e1 R−N coke 0.051 → 0.0496; P1-D S4 move_near physical 0.064 → 0.049; exp35 r4e1 − r1e4 R−N interaction coke 0.049 → 0.054. None reaches Holm significance either way |

- **Weighting-sensitivity rows** (`SPATIAL_TASK_EFFECTS_weighting.csv`, different estimands): not recomputed. Their sign-flip CIs share the defect (3 rows pinned at 0), so they are descriptive only.
- **`B4_TRAJECTORY_VS_SUCCESS.csv` success CIs:** 15 zero-width and 19 pinned rows. Use `CI_VALIDITY_AUDIT.csv` for the same contrasts.

**Call sites** (`A1_CALLSITES.csv`). The affected functions are `stats_core.signflip_ci` and `stats_core.tost(method="signflip")`, plus their callers:
- `b_direct_effects.infer` / `infer_light`: SpatialVLA; recomputed;
- `p0_b_spatial_task_effects.infer_ND`: weighting rows; flagged;
- `p0_e_openvla` (MC inversion, TOST, per-task `signflip_ci`): recomputed;
- AutoVLA callers (`a3_cluster_reanalysis`, `p0_d_dependence`, `analyze_p1c`).

The AutoVLA callers were not recomputed because AutoVLA is outside A1's scope; they are flagged instead:
- **1 endpoint-pinned row** in `cluster_reanalysis_long.csv`: P1C DM F unit-weighted sf_ci90 lower end = −0.0.
- **1** in `p1c_subsets_drive_level.csv`: S2 day-vehicle disjoint, G = 7, RN P, [−0.064, 0.0], p = 0.0625 = p_min.
- **4** in `DEPENDENCE_disjoint_subsets.csv`, with 6–7 drives.

In those rows, read the p value (valid at 0), not the CI end. Two further notes:
- The P1-C decision rule "not replicated, below MME ⇔ 90% sign-flip-inversion CI inside ±MME" uses the same construction on scene-aggregated binary outcomes. Re-check it before using any "below MME" verdict.
- The P2 Geo rows are 0 by construction (structural zero, `n_informative_drives = 0`). They are descriptive and are not evidence of equality.

## 7. What changes in the text

1. The OpenVLA per-task CIs (numbers.json `P0E.succ_task.*`, `OPENVLA_REANALYSIS.md` per-task table) must not be cited. If shown, use Newcombe 10 intervals, labelled "descriptive; 10 states per task". A zero-width CI is never evidence that two arms are equal.
2. OpenVLA state level:
   - Replace "sign-flip-inversion CI" with Newcombe 10 (Tango in a footnote).
   - Delete any reading of C−N d8 as "CI excludes 0".
   - The ±10 pp TOST results stay post hoc and unadjusted. None is Holm-significant.
3. SpatialVLA Tables 7 and 10 (sign-flip-inversion CIs):
   - Replace the CIs with the RS-Newcombe values from `CI_VALIDITY_AUDIT.csv`, labelled post-hoc correction.
   - Pooled ±10 pp equivalence and "3 of 8 per task" are unchanged.
   - Change "task-level CIs span up to 18–29 pp (r4e1; r1e4 6.9–21.6 pp)" to "up to 18–25 pp (r4e1; r1e4 13–20 pp)".
   - The exp34 perp_left coke C−F "±10 equivalent (Holm)" row is downgraded to "unadjusted only". The exp34 pooled judgments (not equivalent) are unchanged.
4. Keep the original columns in every table. The corrected columns are an added post-hoc sensitivity, not a silent replacement.
