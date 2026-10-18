# MATCHED_PREFIX_RESULTS — B2, AutoVLA matched-prefix factorial (review_followup4)

**Status: DONE (2026-10-18).**
- Protocol: `MATCHED_PREFIX_PROTOCOL.md`, committed in 552b3ac **before** any main data. No protocol change was made after the data existed.
- Runs:
  - `B2/dev_main` (52.9 min on the RTX 5090);
  - `B2/p1c_main` (≈ 86 min, from 22:48 to 00:14 UTC).
- Analysis: `b2_analyze.py`, unchanged from the commit, giving `B2/{p1c,dev}_analysis/`.
- Combined table: `MATCHED_PREFIX_RESULTS.csv` (column `cohort`).

**How to read these results.**
- Every cell with a GT older prefix or a GT latest block is an **oracle-conditioned diagnostic**. It is not an online recovery method and not a deployment improvement.
- B2 re-runs existing drives with new intervention arms. It is **not external validation**.
- B2 uses one edited prefix that is **kept** at fixed positions for all later steps. It does **not** test removal of the correction ("release").
- Results are never pooled with the repeated Recent-GT / Reverse-N runs.
- NSRC vs CSRC differences are differences of **value** that arise through provenance. They are not a pure provenance effect.

## 0. Execution and integrity checks

**P1-C images.** All 26 shards were re-fetched: 3,831/3,831 kept images, 0 missing (`B2/fetch/`, ledger `fetched_files.txt`).

**Input verification (protocol §2, stop rule 95 %).** The bank Normal tokens equal the stored P1-C H7 `normal` tokens in **2,160 / 2,160 units (100 %)**. PASS (`p1c_main/input_check.json`). The re-fetched images therefore reproduce the original P1-C inputs.

**Identity checks over all main units** (P1-C: 4,160 unit-anchors / 2,160 units; dev: 2,630 / 1,327):

| check | P1-C | dev |
|---|---|---|
| noop suffix = NAT_NSRC suffix | 100 % | 100 % |
| NAT_NSRC = bank Normal | 100 % | 100 % |
| executed tokens before the anchor identical across cells | 100 % | 100 % |
| prefix sha = bank prefix sha | 100 % | 100 % |
| bank sha unchanged | yes | yes |

**Other run facts.**
- Invalid outputs: 0 in every cohort and cell.
- Disk: B outputs total 16 MB; `/root` has 2.4 GB free.
- Gate (phase 1): PASS (`B2/gate_stage0/gate_judgement.json`).

**Denominators.**

| set | scenes | units | drives | day–vehicle |
|---|---|---|---|---|
| P1-C cohort, primary anchor (a = t*+3, t* ≤ 4) | 312 (F 100 / S 212) | 2,160 (F 711 / S 1,449) | 24 (F 18) | 16 (F 10) |
| P1-C cohort, sensitivity anchor (a = t*+4, t* ≤ 3) | 289 (F 95 / S 194) | 2,000 | 24 (F 18) | 16 |
| B2 domain E4 (P estimand target) | 1,775 (F 100 / S 1,675) | – | 24 | – |
| P1-C sample excluded by the anchor rule (t* 5–7) | 41 | 276 | – | – |
| dev cohort, primary anchor (development; descriptive) | 196 (A− 49 / A+ 147) | 1,327 | 19 (A− 13) | 13 |

**Shares used by the pre-fixed restricted analyses** (P1-C, primary anchor, units):
- the bank older value equals GT: F 12.7 %, S 30.2 %. In these units the older factor is empty in value.
- c^F = c^N at a−1: F 24.6 %, S 45.2 %.
- c^N = gt at a−1: F 12.8 %, S 32.3 %.

## 1. Cell levels (descriptive; P1-C, primary anchor; scene-weighted)

| cell | F: severe | F: A+ | P (E4, Hajek): severe | P: A+ |
|---|---|---|---|---|
| NAT_NSRC (= Normal continuation) | 0.384 | 0.533 | 0.060 | 0.922 |
| NAT_REF (one-shot Recent-GT) | 0.129 | 0.799 | 0.020 | 0.950 |
| GT_REF (one-shot Full) | 0.083 | 0.845 | 0.022 | 0.959 |
| GT_NSRC (matched one-shot Reverse) | 0.336 | 0.587 | 0.055 | 0.920 |
| GT_CSRC | 0.178 | 0.762 | 0.033 | 0.944 |
| NAT_CSRC | 0.199 | 0.739 | 0.033 | 0.933 |

## 2. Primary families (severe; primary anchor; drive clusters; exact sign-flip; Holm m = 4)

### B2-F (failure stratum F: 100 scenes / 711 units / 18 drives; α = 0.05)

| contrast | θ_F | boot 95 % CI | sign-flip p | Holm p | decision |
|---|---|---|---|---|---|
| C_GT = GT_NSRC − GT_REF | **+0.253** | [+0.205, +0.293] | 1.5e-5 | 4.6e-5 | **reject H0** |
| C_NAT = NAT_NSRC − NAT_REF | **+0.255** | [+0.216, +0.290] | 7.6e-6 (floor) | 3.1e-5 | **reject H0** |
| INT = C_GT − C_NAT | −0.002 | [−0.038, +0.031] | 0.92 | 0.92 | not rejected |
| C_PROV = GT_CSRC − GT_NSRC | **−0.158** | [−0.201, −0.102] | 7.3e-4 | 1.5e-3 | **reject H0** |

### B2-P (eligible domain E4, Hajek with recomputed weights: 312 scenes / 2,160 units / 24 drives; α = 0.025)

| contrast | θ_P | boot 95 % CI | sign-flip p | Holm p | decision |
|---|---|---|---|---|---|
| C_GT | +0.033 | [+0.010, +0.056] | 0.017 | 0.051 | **not rejected** at the family α (Holm p 0.051 > 0.025) |
| C_NAT | **+0.040** | [+0.027, +0.057] | 1.6e-5 | 6.5e-5 | **reject H0** |
| INT | −0.007 | [−0.022, +0.008] | 0.41 | 0.41 | not rejected |
| C_PROV | −0.022 | [−0.042, −0.002] | 0.066 | 0.13 | not rejected |

**Day–vehicle clustering (sensitivity; F 10 groups, P 16 groups).**
- F: C_GT p = 0.002, C_NAT p = 0.002, C_PROV p = 0.0098, INT p = 0.85.
- P: C_GT p = 0.064, C_NAT p = 0.00098, C_PROV p = 0.10, INT p = 0.41.
- The F conclusions keep their direction and significance at this coarser clustering.

The connected day/vehicle components (G = 6, exact floor 0.031) are not estimated here. That remains an inference limit, as in P0-D.

## 3. Pre-fixed restricted analyses (severe; drive clusters)

**INT, units whose bank older value differs from GT:**
- F: −0.017 [−0.054, +0.019], p = 0.40 (100 scenes / 621 units).
- P: −0.016 [−0.037, +0.012], p = 0.30.
- So INT ≈ 0 is not an artefact of empty older factors.

**C_PROV, units with c^F ≠ c^N:**
- F: −0.192 [−0.256, −0.110], p = 4e-4 (536 units).
- P: −0.028 [−0.057, +0.000], p = 0.083.

## 4. Secondary outcomes (P1-C, primary anchor; F | P; CIs in the tables below)

| contrast | FDE5 (m) | token at a changed | ‖Δ local motion‖ at a (m) | suffix tokens changed | endpoint distance (m) |
|---|---|---|---|---|---|
| C_GT | +2.41 \| +0.99 | 0.76 \| 0.56 | 0.38 \| 0.19 | 0.77 \| 0.54 | 3.16 \| 1.48 |
| C_NAT | +2.35 \| +0.91 | 0.77 \| 0.57 | 0.37 \| 0.20 | 0.78 \| 0.57 | 3.50 \| 1.60 |
| INT | +0.06 \| +0.09 (n.s.) | −0.01 \| −0.01 | +0.01 \| −0.01 | −0.01 \| −0.03 (P p = 1e-4) | −0.33 (p = 0.004) \| −0.13 (p = 0.046) |
| C_PROV | −1.74 \| −0.71 | 0.66 \| 0.48 | 0.36 \| 0.18 | 0.68 \| 0.46 | 2.95 \| 1.43 |

Distance-type metrics (token change, motion change, endpoint distance) are non-negative per unit. For C_* they measure the size of the change, not its direction.

**Older-prefix main effects within each latest-block level** (severe, NAT_x − GT_x):
- F: REF +0.046 (p = 0.013), NSRC +0.048 (p = 7.5e-4), CSRC +0.020 (p = 0.14).
- P: −0.001, +0.005, +0.000 (all n.s.).

**Corrected-source latest block vs GT latest block** (GT_CSRC − GT_REF): F +0.095 (p = 6e-5); P +0.011 (p = 0.07). **NAT_CSRC − NAT_NSRC:** F −0.185, P −0.027 (p = 0.0013).

**Sensitivity anchor (n_old = 2; F 95 scenes / 674 units):**
- F: C_GT +0.265, C_NAT +0.268, INT −0.003 (p = 0.79), C_PROV −0.229 (all others p ≤ 1.5e-5).
- P: C_GT +0.049, C_NAT +0.023, **INT +0.027 (p = 0.0046)**, C_PROV −0.048.
- The P-level INT at this anchor is a family-free sensitivity result. It is not a preregistered test.

## 5. Dev cohort (development set; unweighted; descriptive; drive clusters, 13 A− drives / 19 overall)

| contrast | A− (49 scenes) | all (196 scenes) |
|---|---|---|
| C_GT | +0.285 [+0.204, +0.363] | +0.090 [+0.050, +0.125] |
| C_NAT | +0.284 [+0.192, +0.374] | +0.088 [+0.061, +0.115] |
| INT | +0.001 [−0.041, +0.047], p = 0.97 | +0.002 [−0.024, +0.019], p = 0.87 |
| C_PROV | −0.163 [−0.231, −0.098] | −0.054 [−0.075, −0.027] |

The pattern matches P1-C: equal latest-block effects inside GT and natural older prefixes, INT ≈ 0, and C_PROV < 0. Day–vehicle clustering keeps all four directions (A−: p ≤ 0.014 except INT).

## 6. Interpretation (within the preregistered scope)

1. **The latest context block's value matters, and that effect does not depend on whether the older prefix is GT.**
   - In the failure stratum, putting the frozen Normal-source token into the latest block instead of GT raises severe outcomes by ≈ 25 pp. This holds both behind a GT older prefix (C_GT) and behind the natural older prefix (C_NAT).
   - The interaction is ≈ 0 (−0.2 pp, 95 % CI ±3.5 pp), also in units where the older values truly differ.
   - The older prefix itself adds only ≈ 2–5 pp in F (and ≈ 0 in P).
   - So, for this one-shot, single-anchor schedule, the Reverse-type effect does **not** rest on the presence of an older GT prefix. It tracks the value of the most recent context token.
   - This remains an **oracle-conditioned** diagnostic: the REF level is GT.
2. **Effect sizes by population.**
   - The effect is large in the failure stratum.
   - It is small in the eligible domain E4: C_NAT +4.0 pp, significant; C_GT +3.3 pp, not significant after Holm at α = 0.025.
   - E4 is the t* ≤ 4 domain of the P1-C eligible set, not the original 2,191-scene estimand.
3. **The corrected-branch source (C_PROV)** lowers severe outcomes vs the Normal source by 16 pp in F (19 pp where the values differ), but stays above GT_REF by 9.5 pp. The corrected branch's own token at a−1 carries part, not all, of the GT benefit.
   - This is a **value** difference (the corrected branch emits different tokens). It is not a pure provenance effect.
   - It is not significant in P after Holm.
4. **Endpoint changes are not only verdict changes.** C_GT and C_NAT move the 5 s endpoint by ≈ 3.2–3.5 m in F and ≈ 1.5 m in P. They change the token at a in 56–77 % of units. Most of these changes do not flip the severe verdict, especially in S (A+ scenes). Only the severe/A+ contrasts are in the families.
5. **Coordinate accumulation.** No edited token is executed (identity checked for 100 % of units). All endpoint/FDE5 differences arise through the suffix tokens generated after the edit and their accumulation.
6. **Not shown:**
   - removal ("release") of the correction;
   - repeated-schedule behaviour;
   - external validity beyond these drives;
   - any comparison with Impromptu or other models.

## 7. Claim implications (for the lead)

- **Keep, narrowed:** "In AutoVLA's failure stratum, a single retained edit of the most recent context token changes subsequent generation and severe outcomes by ≈ 25 pp. The size is the same whether the older context is GT or the model's own natural prefix (interaction ≈ 0). This is an oracle-conditioned diagnostic."
- **Population level:** report the small E4 effects (+3–4 pp), with C_GT not significant after Holm.
- **Do not write:**
  - "the effect requires a GT history";
  - "pure provenance";
  - "online correction";
  - "external replication".

---

## Appendix: full tables (generated by `b2_report_tables.py` from the analysis CSVs)

#### p1c | anchor n_old=1 | metric=severe | subset=all | cluster=drive

| contrast | estimand | scenes | units | clusters | theta | boot 95% CI | sign-flip p | Holm p |
|---|---|---|---|---|---|---|---|---|
| C_GT | F | 100 | 711 | 18 | +0.253 | [+0.205, +0.293] | 1.5e-05 | 4.6e-05 |
| C_GT | P | 312 | 2160 | 24 | +0.033 | [+0.010, +0.056] | 0.017 | 0.051 |
| C_GT | S | 212 | 1449 | 24 | +0.020 | [-0.002, +0.043] | 0.11 | – |
| C_NAT | F | 100 | 711 | 18 | +0.255 | [+0.216, +0.290] | 7.6e-06 | 3.1e-05 |
| C_NAT | P | 312 | 2160 | 24 | +0.040 | [+0.027, +0.057] | 1.6e-05 | 6.5e-05 |
| C_NAT | S | 212 | 1449 | 24 | +0.027 | [+0.013, +0.044] | 0.00037 | – |
| INT | F | 100 | 711 | 18 | -0.002 | [-0.038, +0.031] | 0.92 | 0.92 |
| INT | P | 312 | 2160 | 24 | -0.007 | [-0.022, +0.008] | 0.41 | 0.41 |
| INT | S | 212 | 1449 | 24 | -0.007 | [-0.022, +0.009] | 0.4 | – |
| C_PROV | F | 100 | 711 | 18 | -0.158 | [-0.201, -0.102] | 0.00073 | 0.0015 |
| C_PROV | P | 312 | 2160 | 24 | -0.022 | [-0.042, -0.002] | 0.066 | 0.13 |
| C_PROV | S | 212 | 1449 | 24 | -0.014 | [-0.035, +0.007] | 0.24 | – |

#### p1c | anchor n_old=1 | metric=severe | subset=all | cluster=day_vehicle

| contrast | estimand | scenes | units | clusters | theta | boot 95% CI | sign-flip p | Holm p |
|---|---|---|---|---|---|---|---|---|
| C_GT | F | 100 | 711 | 10 | +0.253 | [+0.195, +0.288] | 0.002 | – |
| C_GT | P | 312 | 2160 | 16 | +0.033 | [+0.007, +0.055] | 0.064 | – |
| C_GT | S | 212 | 1449 | 16 | +0.020 | [-0.004, +0.042] | 0.15 | – |
| C_NAT | F | 100 | 711 | 10 | +0.255 | [+0.209, +0.287] | 0.002 | – |
| C_NAT | P | 312 | 2160 | 16 | +0.040 | [+0.024, +0.056] | 0.00098 | – |
| C_NAT | S | 212 | 1449 | 16 | +0.027 | [+0.011, +0.043] | 0.0059 | – |
| INT | F | 100 | 711 | 10 | -0.002 | [-0.030, +0.014] | 0.85 | – |
| INT | P | 312 | 2160 | 16 | -0.007 | [-0.022, +0.008] | 0.41 | – |
| INT | S | 212 | 1449 | 16 | -0.007 | [-0.022, +0.008] | 0.39 | – |
| C_PROV | F | 100 | 711 | 10 | -0.158 | [-0.200, -0.092] | 0.0098 | – |
| C_PROV | P | 312 | 2160 | 16 | -0.022 | [-0.042, -0.000] | 0.1 | – |
| C_PROV | S | 212 | 1449 | 16 | -0.014 | [-0.035, +0.008] | 0.25 | – |

#### p1c | anchor n_old=1 | metric=severe | subset=older_values_differ | cluster=drive

| contrast | estimand | scenes | units | clusters | theta | boot 95% CI | sign-flip p | Holm p |
|---|---|---|---|---|---|---|---|---|
| INT | F | 100 | 621 | 18 | -0.017 | [-0.054, +0.019] | 0.4 | – |
| INT | P | 303 | 1632 | 24 | -0.016 | [-0.037, +0.012] | 0.3 | – |
| INT | S | 203 | 1011 | 24 | -0.016 | [-0.037, +0.014] | 0.33 | – |

#### p1c | anchor n_old=1 | metric=severe | subset=src_values_differ | cluster=drive

| contrast | estimand | scenes | units | clusters | theta | boot 95% CI | sign-flip p | Holm p |
|---|---|---|---|---|---|---|---|---|
| C_PROV | F | 100 | 536 | 18 | -0.192 | [-0.256, -0.110] | 0.0004 | – |
| C_PROV | P | 297 | 1330 | 24 | -0.028 | [-0.057, +0.000] | 0.083 | – |
| C_PROV | S | 197 | 794 | 24 | -0.017 | [-0.048, +0.011] | 0.29 | – |

#### p1c | anchor n_old=1 | metric=fde5 | subset=all | cluster=drive

| contrast | estimand | scenes | units | clusters | theta | boot 95% CI | sign-flip p | Holm p |
|---|---|---|---|---|---|---|---|---|
| C_GT | F | 100 | 711 | 18 | +2.41 | [+1.83, +2.94] | 7.6e-06 | – |
| C_GT | P | 312 | 2160 | 24 | +0.99 | [+0.67, +1.44] | 1.2e-07 | – |
| C_NAT | F | 100 | 711 | 18 | +2.35 | [+1.83, +2.87] | 7.6e-06 | – |
| C_NAT | P | 312 | 2160 | 24 | +0.91 | [+0.57, +1.40] | 2.4e-07 | – |
| INT | F | 100 | 711 | 18 | +0.06 | [-0.13, +0.22] | 0.57 | – |
| INT | P | 312 | 2160 | 24 | +0.09 | [-0.01, +0.19] | 0.11 | – |
| C_PROV | F | 100 | 711 | 18 | -1.74 | [-2.36, -1.19] | 3.1e-05 | – |
| C_PROV | P | 312 | 2160 | 24 | -0.71 | [-1.06, -0.47] | 1.2e-07 | – |

#### p1c | anchor n_old=1 | metric=tok_a_changed | subset=all | cluster=drive

| contrast | estimand | scenes | units | clusters | theta | boot 95% CI | sign-flip p | Holm p |
|---|---|---|---|---|---|---|---|---|
| C_GT | F | 100 | 711 | 18 | +0.758 | [+0.713, +0.808] | 7.6e-06 | – |
| C_GT | P | 312 | 2160 | 24 | +0.561 | [+0.502, +0.627] | 1.2e-07 | – |
| C_NAT | F | 100 | 711 | 18 | +0.772 | [+0.733, +0.810] | 7.6e-06 | – |
| C_NAT | P | 312 | 2160 | 24 | +0.568 | [+0.513, +0.629] | 1.2e-07 | – |
| INT | F | 100 | 711 | 18 | -0.013 | [-0.031, +0.008] | 0.23 | – |
| INT | P | 312 | 2160 | 24 | -0.006 | [-0.033, +0.017] | 0.65 | – |
| C_PROV | F | 100 | 711 | 18 | +0.664 | [+0.627, +0.699] | 7.6e-06 | – |
| C_PROV | P | 312 | 2160 | 24 | +0.479 | [+0.422, +0.544] | 1.2e-07 | – |

#### p1c | anchor n_old=1 | metric=local_disp_change_m | subset=all | cluster=drive

| contrast | estimand | scenes | units | clusters | theta | boot 95% CI | sign-flip p | Holm p |
|---|---|---|---|---|---|---|---|---|
| C_GT | F | 100 | 711 | 18 | +0.38 | [+0.30, +0.45] | 7.6e-06 | – |
| C_GT | P | 312 | 2160 | 24 | +0.19 | [+0.14, +0.25] | 1.2e-07 | – |
| C_NAT | F | 100 | 711 | 18 | +0.37 | [+0.30, +0.43] | 7.6e-06 | – |
| C_NAT | P | 312 | 2160 | 24 | +0.20 | [+0.15, +0.25] | 1.2e-07 | – |
| INT | F | 100 | 711 | 18 | +0.01 | [-0.01, +0.03] | 0.31 | – |
| INT | P | 312 | 2160 | 24 | -0.01 | [-0.02, +0.00] | 0.28 | – |
| C_PROV | F | 100 | 711 | 18 | +0.36 | [+0.28, +0.44] | 7.6e-06 | – |
| C_PROV | P | 312 | 2160 | 24 | +0.18 | [+0.12, +0.24] | 1.2e-07 | – |

#### p1c | anchor n_old=1 | metric=suffix_changed_frac | subset=all | cluster=drive

| contrast | estimand | scenes | units | clusters | theta | boot 95% CI | sign-flip p | Holm p |
|---|---|---|---|---|---|---|---|---|
| C_GT | F | 100 | 711 | 18 | +0.769 | [+0.722, +0.816] | 7.6e-06 | – |
| C_GT | P | 312 | 2160 | 24 | +0.539 | [+0.477, +0.606] | 1.2e-07 | – |
| C_NAT | F | 100 | 711 | 18 | +0.781 | [+0.740, +0.822] | 7.6e-06 | – |
| C_NAT | P | 312 | 2160 | 24 | +0.567 | [+0.508, +0.631] | 1.2e-07 | – |
| INT | F | 100 | 711 | 18 | -0.012 | [-0.022, +0.000] | 0.089 | – |
| INT | P | 312 | 2160 | 24 | -0.029 | [-0.039, -0.017] | 0.00011 | – |
| C_PROV | F | 100 | 711 | 18 | +0.678 | [+0.644, +0.709] | 7.6e-06 | – |
| C_PROV | P | 312 | 2160 | 24 | +0.464 | [+0.408, +0.528] | 1.2e-07 | – |

#### p1c | anchor n_old=1 | metric=endpoint_dist_m | subset=all | cluster=drive

| contrast | estimand | scenes | units | clusters | theta | boot 95% CI | sign-flip p | Holm p |
|---|---|---|---|---|---|---|---|---|
| C_GT | F | 100 | 711 | 18 | +3.16 | [+2.58, +3.74] | 7.6e-06 | – |
| C_GT | P | 312 | 2160 | 24 | +1.48 | [+1.06, +2.03] | 1.2e-07 | – |
| C_NAT | F | 100 | 711 | 18 | +3.50 | [+2.90, +4.11] | 7.6e-06 | – |
| C_NAT | P | 312 | 2160 | 24 | +1.60 | [+1.17, +2.21] | 1.2e-07 | – |
| INT | F | 100 | 711 | 18 | -0.33 | [-0.61, -0.13] | 0.0042 | – |
| INT | P | 312 | 2160 | 24 | -0.13 | [-0.24, -0.02] | 0.046 | – |
| C_PROV | F | 100 | 711 | 18 | +2.95 | [+2.27, +3.59] | 7.6e-06 | – |
| C_PROV | P | 312 | 2160 | 24 | +1.43 | [+0.98, +2.01] | 1.2e-07 | – |

#### p1c | anchor n_old=1 | metric=A_plus | subset=all | cluster=drive

| contrast | estimand | scenes | units | clusters | theta | boot 95% CI | sign-flip p | Holm p |
|---|---|---|---|---|---|---|---|---|
| C_GT | F | 100 | 711 | 18 | -0.258 | [-0.303, -0.205] | 1.5e-05 | – |
| C_GT | P | 312 | 2160 | 24 | -0.039 | [-0.056, -0.020] | 0.0019 | – |
| C_NAT | F | 100 | 711 | 18 | -0.266 | [-0.301, -0.228] | 7.6e-06 | – |
| C_NAT | P | 312 | 2160 | 24 | -0.028 | [-0.053, -0.001] | 0.061 | – |
| INT | F | 100 | 711 | 18 | +0.008 | [-0.026, +0.051] | 0.69 | – |
| INT | P | 312 | 2160 | 24 | -0.011 | [-0.030, +0.011] | 0.38 | – |
| C_PROV | F | 100 | 711 | 18 | +0.175 | [+0.121, +0.220] | 6.1e-05 | – |
| C_PROV | P | 312 | 2160 | 24 | +0.024 | [+0.002, +0.045] | 0.06 | – |

#### p1c | anchor n_old=1 | metric=severe | subset=all | cluster=drive

| contrast | estimand | scenes | units | clusters | theta | boot 95% CI | sign-flip p | Holm p |
|---|---|---|---|---|---|---|---|---|
| O_main_REF | F | 100 | 711 | 18 | +0.046 | [+0.012, +0.080] | 0.013 | – |
| O_main_REF | P | 312 | 2160 | 24 | -0.001 | [-0.013, +0.011] | 0.84 | – |
| O_main_NSRC | F | 100 | 711 | 18 | +0.048 | [+0.028, +0.066] | 0.00075 | – |
| O_main_NSRC | P | 312 | 2160 | 24 | +0.005 | [-0.003, +0.015] | 0.24 | – |
| O_main_CSRC | F | 100 | 711 | 18 | +0.020 | [-0.005, +0.042] | 0.14 | – |
| O_main_CSRC | P | 312 | 2160 | 24 | +0.000 | [-0.011, +0.012] | 0.99 | – |
| C_PROV_NAT | F | 100 | 711 | 18 | -0.185 | [-0.218, -0.148] | 1.5e-05 | – |
| C_PROV_NAT | P | 312 | 2160 | 24 | -0.027 | [-0.044, -0.014] | 0.0013 | – |
| CSRC_vs_REF_GT | F | 100 | 711 | 18 | +0.095 | [+0.065, +0.124] | 6.1e-05 | – |
| CSRC_vs_REF_GT | P | 312 | 2160 | 24 | +0.011 | [+0.001, +0.025] | 0.07 | – |

#### p1c | anchor n_old=2 | metric=severe | subset=all | cluster=drive

| contrast | estimand | scenes | units | clusters | theta | boot 95% CI | sign-flip p | Holm p |
|---|---|---|---|---|---|---|---|---|
| C_GT | F | 95 | 674 | 18 | +0.265 | [+0.219, +0.299] | 1.5e-05 | – |
| C_GT | P | 289 | 2000 | 24 | +0.049 | [+0.036, +0.064] | 1.9e-06 | – |
| C_NAT | F | 95 | 674 | 18 | +0.268 | [+0.231, +0.302] | 7.6e-06 | – |
| C_NAT | P | 289 | 2000 | 24 | +0.023 | [+0.010, +0.037] | 0.0037 | – |
| INT | F | 95 | 674 | 18 | -0.003 | [-0.027, +0.017] | 0.79 | – |
| INT | P | 289 | 2000 | 24 | +0.027 | [+0.011, +0.046] | 0.0046 | – |
| C_PROV | F | 95 | 674 | 18 | -0.229 | [-0.264, -0.186] | 1.5e-05 | – |
| C_PROV | P | 289 | 2000 | 24 | -0.048 | [-0.064, -0.034] | 1.9e-06 | – |

#### p1c | anchor n_old=2 | metric=fde5 | subset=all | cluster=drive

| contrast | estimand | scenes | units | clusters | theta | boot 95% CI | sign-flip p | Holm p |
|---|---|---|---|---|---|---|---|---|
| C_GT | F | 95 | 674 | 18 | +2.36 | [+1.86, +2.87] | 7.6e-06 | – |
| C_GT | P | 289 | 2000 | 24 | +0.97 | [+0.68, +1.30] | 1.2e-07 | – |
| C_NAT | F | 95 | 674 | 18 | +2.37 | [+1.87, +2.87] | 7.6e-06 | – |
| C_NAT | P | 289 | 2000 | 24 | +0.84 | [+0.59, +1.18] | 1.2e-07 | – |
| INT | F | 95 | 674 | 18 | -0.01 | [-0.17, +0.16] | 0.87 | – |
| INT | P | 289 | 2000 | 24 | +0.13 | [-0.04, +0.30] | 0.15 | – |
| C_PROV | F | 95 | 674 | 18 | -2.11 | [-2.59, -1.62] | 7.6e-06 | – |
| C_PROV | P | 289 | 2000 | 24 | -0.90 | [-1.21, -0.64] | 1.2e-07 | – |

#### dev | anchor n_old=1 | metric=severe | subset=all | cluster=drive

| contrast | estimand | scenes | units | clusters | theta | boot 95% CI | sign-flip p | Holm p |
|---|---|---|---|---|---|---|---|---|
| C_GT | F | 49 | 345 | 13 | +0.285 | [+0.204, +0.363] | 0.00024 | 0.00098 |
| C_GT | P | 196 | 1327 | 19 | +0.090 | [+0.050, +0.125] | 0.00067 | 0.002 |
| C_GT | S | 147 | 982 | 19 | +0.025 | [+0.003, +0.041] | 0.069 | – |
| C_NAT | F | 49 | 345 | 13 | +0.284 | [+0.192, +0.374] | 0.00024 | 0.00098 |
| C_NAT | P | 196 | 1327 | 19 | +0.088 | [+0.061, +0.115] | 0.00024 | 0.00098 |
| C_NAT | S | 147 | 982 | 19 | +0.023 | [+0.006, +0.048] | 0.016 | – |
| INT | F | 49 | 345 | 13 | +0.001 | [-0.041, +0.047] | 0.97 | 0.97 |
| INT | P | 196 | 1327 | 19 | +0.002 | [-0.024, +0.019] | 0.87 | 0.87 |
| INT | S | 147 | 982 | 19 | +0.002 | [-0.034, +0.026] | 0.83 | – |
| C_PROV | F | 49 | 345 | 13 | -0.163 | [-0.231, -0.098] | 0.002 | 0.0039 |
| C_PROV | P | 196 | 1327 | 19 | -0.054 | [-0.075, -0.027] | 0.0032 | 0.0065 |
| C_PROV | S | 147 | 982 | 19 | -0.018 | [-0.030, -0.000] | 0.09 | – |

#### dev | anchor n_old=1 | metric=severe | subset=all | cluster=day_vehicle

| contrast | estimand | scenes | units | clusters | theta | boot 95% CI | sign-flip p | Holm p |
|---|---|---|---|---|---|---|---|---|
| C_GT | F | 49 | 345 | 10 | +0.285 | [+0.197, +0.355] | 0.002 | – |
| C_GT | P | 196 | 1327 | 13 | +0.090 | [+0.046, +0.123] | 0.0068 | – |
| C_GT | S | 147 | 982 | 13 | +0.025 | [+0.002, +0.041] | 0.084 | – |
| C_NAT | F | 49 | 345 | 10 | +0.284 | [+0.180, +0.370] | 0.002 | – |
| C_NAT | P | 196 | 1327 | 13 | +0.088 | [+0.059, +0.115] | 0.002 | – |
| C_NAT | S | 147 | 982 | 13 | +0.023 | [+0.007, +0.049] | 0.016 | – |
| INT | F | 49 | 345 | 10 | +0.001 | [-0.038, +0.045] | 0.95 | – |
| INT | P | 196 | 1327 | 13 | +0.002 | [-0.026, +0.018] | 0.88 | – |
| INT | S | 147 | 982 | 13 | +0.002 | [-0.036, +0.025] | 0.83 | – |
| C_PROV | F | 49 | 345 | 10 | -0.163 | [-0.232, -0.085] | 0.014 | – |
| C_PROV | P | 196 | 1327 | 13 | -0.054 | [-0.075, -0.023] | 0.014 | – |
| C_PROV | S | 147 | 982 | 13 | -0.018 | [-0.031, -0.001] | 0.086 | – |

#### dev | anchor n_old=1 | metric=severe | subset=older_values_differ | cluster=drive

| contrast | estimand | scenes | units | clusters | theta | boot 95% CI | sign-flip p | Holm p |
|---|---|---|---|---|---|---|---|---|
| INT | F | 48 | 299 | 13 | +0.000 | [-0.053, +0.067] | 1 | – |
| INT | P | 190 | 986 | 19 | +0.004 | [-0.027, +0.025] | 0.8 | – |
| INT | S | 142 | 687 | 19 | +0.005 | [-0.040, +0.037] | 0.76 | – |

#### dev | anchor n_old=1 | metric=severe | subset=src_values_differ | cluster=drive

| contrast | estimand | scenes | units | clusters | theta | boot 95% CI | sign-flip p | Holm p |
|---|---|---|---|---|---|---|---|---|
| C_PROV | F | 47 | 261 | 13 | -0.205 | [-0.320, -0.107] | 0.0042 | – |
| C_PROV | P | 186 | 831 | 19 | -0.079 | [-0.113, -0.040] | 0.003 | – |
| C_PROV | S | 139 | 570 | 19 | -0.037 | [-0.063, -0.008] | 0.038 | – |

#### dev | anchor n_old=1 | metric=fde5 | subset=all | cluster=drive

| contrast | estimand | scenes | units | clusters | theta | boot 95% CI | sign-flip p | Holm p |
|---|---|---|---|---|---|---|---|---|
| C_GT | F | 49 | 345 | 13 | +3.27 | [+2.49, +3.93] | 0.00024 | – |
| C_GT | P | 196 | 1327 | 19 | +1.48 | [+1.02, +1.83] | 3.8e-06 | – |
| C_NAT | F | 49 | 345 | 13 | +2.89 | [+2.37, +3.41] | 0.00024 | – |
| C_NAT | P | 196 | 1327 | 19 | +1.33 | [+1.06, +1.51] | 7.6e-06 | – |
| INT | F | 49 | 345 | 13 | +0.38 | [-0.22, +0.72] | 0.27 | – |
| INT | P | 196 | 1327 | 19 | +0.16 | [-0.11, +0.36] | 0.35 | – |
| C_PROV | F | 49 | 345 | 13 | -2.19 | [-2.81, -1.61] | 0.00024 | – |
| C_PROV | P | 196 | 1327 | 19 | -1.02 | [-1.28, -0.65] | 6.9e-05 | – |

#### dev | anchor n_old=1 | metric=tok_a_changed | subset=all | cluster=drive

| contrast | estimand | scenes | units | clusters | theta | boot 95% CI | sign-flip p | Holm p |
|---|---|---|---|---|---|---|---|---|
| C_GT | F | 49 | 345 | 13 | +0.760 | [+0.669, +0.829] | 0.00024 | – |
| C_GT | P | 196 | 1327 | 19 | +0.620 | [+0.571, +0.657] | 3.8e-06 | – |
| C_NAT | F | 49 | 345 | 13 | +0.769 | [+0.696, +0.833] | 0.00024 | – |
| C_NAT | P | 196 | 1327 | 19 | +0.632 | [+0.590, +0.662] | 3.8e-06 | – |
| INT | F | 49 | 345 | 13 | -0.009 | [-0.066, +0.026] | 0.69 | – |
| INT | P | 196 | 1327 | 19 | -0.012 | [-0.030, +0.003] | 0.16 | – |
| C_PROV | F | 49 | 345 | 13 | +0.651 | [+0.561, +0.718] | 0.00024 | – |
| C_PROV | P | 196 | 1327 | 19 | +0.518 | [+0.458, +0.564] | 3.8e-06 | – |

#### dev | anchor n_old=1 | metric=local_disp_change_m | subset=all | cluster=drive

| contrast | estimand | scenes | units | clusters | theta | boot 95% CI | sign-flip p | Holm p |
|---|---|---|---|---|---|---|---|---|
| C_GT | F | 49 | 345 | 13 | +0.42 | [+0.35, +0.51] | 0.00024 | – |
| C_GT | P | 196 | 1327 | 19 | +0.24 | [+0.19, +0.28] | 3.8e-06 | – |
| C_NAT | F | 49 | 345 | 13 | +0.41 | [+0.35, +0.50] | 0.00024 | – |
| C_NAT | P | 196 | 1327 | 19 | +0.25 | [+0.20, +0.28] | 3.8e-06 | – |
| INT | F | 49 | 345 | 13 | +0.01 | [-0.04, +0.05] | 0.67 | – |
| INT | P | 196 | 1327 | 19 | -0.01 | [-0.03, +0.01] | 0.42 | – |
| C_PROV | F | 49 | 345 | 13 | +0.38 | [+0.32, +0.49] | 0.00024 | – |
| C_PROV | P | 196 | 1327 | 19 | +0.22 | [+0.18, +0.27] | 3.8e-06 | – |

#### dev | anchor n_old=1 | metric=suffix_changed_frac | subset=all | cluster=drive

| contrast | estimand | scenes | units | clusters | theta | boot 95% CI | sign-flip p | Holm p |
|---|---|---|---|---|---|---|---|---|
| C_GT | F | 49 | 345 | 13 | +0.809 | [+0.747, +0.857] | 0.00024 | – |
| C_GT | P | 196 | 1327 | 19 | +0.622 | [+0.568, +0.665] | 3.8e-06 | – |
| C_NAT | F | 49 | 345 | 13 | +0.811 | [+0.754, +0.859] | 0.00024 | – |
| C_NAT | P | 196 | 1327 | 19 | +0.633 | [+0.581, +0.676] | 3.8e-06 | – |
| INT | F | 49 | 345 | 13 | -0.002 | [-0.021, +0.009] | 0.82 | – |
| INT | P | 196 | 1327 | 19 | -0.011 | [-0.021, -0.001] | 0.068 | – |
| C_PROV | F | 49 | 345 | 13 | +0.689 | [+0.606, +0.756] | 0.00024 | – |
| C_PROV | P | 196 | 1327 | 19 | +0.519 | [+0.461, +0.572] | 3.8e-06 | – |

#### dev | anchor n_old=1 | metric=endpoint_dist_m | subset=all | cluster=drive

| contrast | estimand | scenes | units | clusters | theta | boot 95% CI | sign-flip p | Holm p |
|---|---|---|---|---|---|---|---|---|
| C_GT | F | 49 | 345 | 13 | +4.25 | [+3.43, +4.85] | 0.00024 | – |
| C_GT | P | 196 | 1327 | 19 | +2.16 | [+1.62, +2.54] | 3.8e-06 | – |
| C_NAT | F | 49 | 345 | 13 | +4.12 | [+3.53, +4.67] | 0.00024 | – |
| C_NAT | P | 196 | 1327 | 19 | +2.22 | [+1.76, +2.55] | 3.8e-06 | – |
| INT | F | 49 | 345 | 13 | +0.13 | [-0.39, +0.46] | 0.63 | – |
| INT | P | 196 | 1327 | 19 | -0.06 | [-0.22, +0.07] | 0.43 | – |
| C_PROV | F | 49 | 345 | 13 | +3.58 | [+2.93, +4.25] | 0.00024 | – |
| C_PROV | P | 196 | 1327 | 19 | +1.90 | [+1.45, +2.25] | 3.8e-06 | – |

#### dev | anchor n_old=1 | metric=A_plus | subset=all | cluster=drive

| contrast | estimand | scenes | units | clusters | theta | boot 95% CI | sign-flip p | Holm p |
|---|---|---|---|---|---|---|---|---|
| C_GT | F | 49 | 345 | 13 | -0.288 | [-0.374, -0.187] | 0.00024 | – |
| C_GT | P | 196 | 1327 | 19 | -0.094 | [-0.136, -0.046] | 0.00085 | – |
| C_NAT | F | 49 | 345 | 13 | -0.277 | [-0.360, -0.183] | 0.00024 | – |
| C_NAT | P | 196 | 1327 | 19 | -0.084 | [-0.115, -0.051] | 0.00049 | – |
| INT | F | 49 | 345 | 13 | -0.010 | [-0.047, +0.030] | 0.66 | – |
| INT | P | 196 | 1327 | 19 | -0.010 | [-0.030, +0.018] | 0.53 | – |
| C_PROV | F | 49 | 345 | 13 | +0.153 | [+0.090, +0.208] | 0.0017 | – |
| C_PROV | P | 196 | 1327 | 19 | +0.053 | [+0.024, +0.075] | 0.0034 | – |

#### dev | anchor n_old=1 | metric=severe | subset=all | cluster=drive

| contrast | estimand | scenes | units | clusters | theta | boot 95% CI | sign-flip p | Holm p |
|---|---|---|---|---|---|---|---|---|
| O_main_REF | F | 49 | 345 | 13 | +0.055 | [+0.020, +0.087] | 0.02 | – |
| O_main_REF | P | 196 | 1327 | 19 | +0.016 | [-0.002, +0.030] | 0.16 | – |
| O_main_NSRC | F | 49 | 345 | 13 | +0.054 | [-0.001, +0.103] | 0.11 | – |
| O_main_NSRC | P | 196 | 1327 | 19 | +0.014 | [+0.004, +0.026] | 0.043 | – |
| O_main_CSRC | F | 49 | 345 | 13 | +0.056 | [+0.009, +0.103] | 0.041 | – |
| O_main_CSRC | P | 196 | 1327 | 19 | +0.026 | [+0.013, +0.037] | 0.002 | – |
| C_PROV_NAT | F | 49 | 345 | 13 | -0.161 | [-0.222, -0.104] | 0.00098 | – |
| C_PROV_NAT | P | 196 | 1327 | 19 | -0.043 | [-0.057, -0.025] | 0.0026 | – |
| CSRC_vs_REF_GT | F | 49 | 345 | 13 | +0.122 | [+0.048, +0.183] | 0.0078 | – |
| CSRC_vs_REF_GT | P | 196 | 1327 | 19 | +0.036 | [+0.015, +0.055] | 0.0098 | – |

#### dev | anchor n_old=2 | metric=severe | subset=all | cluster=drive

| contrast | estimand | scenes | units | clusters | theta | boot 95% CI | sign-flip p | Holm p |
|---|---|---|---|---|---|---|---|---|
| C_GT | F | 48 | 339 | 13 | +0.346 | [+0.236, +0.442] | 0.00049 | – |
| C_GT | P | 192 | 1303 | 19 | +0.111 | [+0.063, +0.149] | 0.00049 | – |
| C_NAT | F | 48 | 339 | 13 | +0.317 | [+0.191, +0.431] | 0.00024 | – |
| C_NAT | P | 192 | 1303 | 19 | +0.093 | [+0.055, +0.126] | 0.00049 | – |
| INT | F | 48 | 339 | 13 | +0.029 | [-0.017, +0.094] | 0.3 | – |
| INT | P | 192 | 1303 | 19 | +0.018 | [+0.003, +0.030] | 0.056 | – |
| C_PROV | F | 48 | 339 | 13 | -0.275 | [-0.355, -0.179] | 0.00049 | – |
| C_PROV | P | 192 | 1303 | 19 | -0.094 | [-0.128, -0.052] | 0.00049 | – |

#### dev | anchor n_old=2 | metric=fde5 | subset=all | cluster=drive

| contrast | estimand | scenes | units | clusters | theta | boot 95% CI | sign-flip p | Holm p |
|---|---|---|---|---|---|---|---|---|
| C_GT | F | 48 | 339 | 13 | +3.46 | [+2.76, +4.06] | 0.00024 | – |
| C_GT | P | 192 | 1303 | 19 | +1.48 | [+1.06, +1.76] | 7.6e-06 | – |
| C_NAT | F | 48 | 339 | 13 | +3.20 | [+2.61, +3.69] | 0.00024 | – |
| C_NAT | P | 192 | 1303 | 19 | +1.34 | [+1.00, +1.57] | 7.6e-06 | – |
| INT | F | 48 | 339 | 13 | +0.26 | [-0.02, +0.46] | 0.14 | – |
| INT | P | 192 | 1303 | 19 | +0.13 | [-0.00, +0.24] | 0.089 | – |
| C_PROV | F | 48 | 339 | 13 | -3.04 | [-3.66, -2.29] | 0.00024 | – |
| C_PROV | P | 192 | 1303 | 19 | -1.34 | [-1.62, -0.96] | 7.6e-06 | – |
