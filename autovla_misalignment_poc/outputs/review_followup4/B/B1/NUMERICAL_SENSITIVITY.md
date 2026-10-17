# B1 NUMERICAL_SENSITIVITY — AutoVLA (review_followup4)

**Status:** DONE (2026-10-17).
- Protocol: `NUMERICAL_SENSITIVITY_PROTOCOL.md`, written before the run, plus amendment B1-A1 made before the main run.
- Run: `run/` (one process on the RTX 5090, 18.3 min; peak 25.7 GB during SF).
- Analysis: `scripts/review_followup4/b/b1_analyze.py`, giving `NUMERICAL_SENSITIVITY.csv` and `numerical_sensitivity_rows.json`.

This is a sensitivity analysis of the numerical implementation. It is **not** a new estimate of the main effects and **not** external replication. The normal numeric variation is reported **next to** the effects. It is never subtracted from them.

## Setup

**Manifest.** The Stage-0 rule with take = 20 per t* stratum (late stratum: all 16): 56 dev scenes (A− 15 / A+ 41).
- 56 `alt0` units on H7/H8.
- 56 P1A family-G units.

**Settings.** All use t001 (experiment decoding), seed 0, and the same inputs, dose and pairing.

| id | what changes |
|---|---|
| S0 | reference: bf16, sdpa, experiment batch (H7 B = 1, H8 B = 93, P1A B = 10) |
| S0r | S0 repeated, same process |
| SB | batch shape only: H7 rows taken from the H8 B = 93 batch; H8 rows at B = 1 (Reverse-N by token substitution); P1A at B = 1 |
| SE | eager attention instead of sdpa, LM and vision. Co-change: 4-D mask, fp32 softmax |
| SF | fp32 arithmetic on the same bf16-rounded weights (exact upcast, restored afterwards; parameter checksum equal after restore) |

## 1. Row-level variation vs S0 (row-runs = unit × row; free run)

| setting | row-runs (units) | max \|Δlogit\| (median) | suffix changed | units with any change | endpoint dev. mean / max (m) | endpoint dev. > 0.5 m | severe flips | A− flips | mean \|ΔFDE5\| (m) |
|---|---|---|---|---|---|---|---|---|---|
| S0r | 952 (112) | 0 (0), bitwise | 0 [0, 0.4 %] | 0 | 0 / 0 | 0 | 0 | 0 | 0 |
| SB | 784 (112) | 0.50 (0.25) | 70 = **8.9 %** [7.0, 11.1] | 33 | 0.056 / 8.9 | 18 | 2 = 0.26 % [0.03, 0.92] | 1 = 0.13 % [0.00, 0.71] | 0.040 |
| SE | 952 (112) | 2.0 (0.88) | 251 = **26.4 %** [23.6, 29.3] | 63 | 0.252 / 12.3 | 94 | 11 = 1.2 % [0.6, 2.1] | 16 = 1.7 % [1.0, 2.7] | 0.182 |
| SF | 952 (112) | 2.43 (0.95) | 238 = **25.0 %** [22.3, 27.9] | 61 | 0.192 / 12.3 | 82 | 7 = 0.7 % [0.3, 1.5] | 12 = 1.3 % [0.7, 2.2] | 0.147 |

Per path (CSV):
- SB changes the suffix in 27/224 H8 row-runs and 43/560 P1A row-runs.
- SE/SF change the suffix most often on P1A (188 and 162 of 560 row-runs).

Given a suffix change, the mean endpoint deviation is 0.6–1.0 m. Individual changes reach 8.9–12.3 m.

**Reading.**
- Same-process repetition is bitwise. No hidden nondeterminism was seen.
- **Batch shape alone changes ≈ 9 % of suffixes.** Logit differences are only ≤ 0.5 (bf16 ulps, within T_equiv = 0.75), but they flip tokens wherever the top-2 margin is small. Through accumulation, a few of these changes move the 5 s endpoint by metres.
  - This is the row-level mechanism behind the 88–89 % Normal agreement across harnesses (M-A3). The earlier finding that free-run/full-recompute and batch shape change suffixes is confirmed and kept.
- Changing the attention backend or the arithmetic precision changes ≈ 25 % of suffixes. Logit differences are ≈ 2, larger than T_equiv: these are not equivalent computations in the P1 sense.
- Verdict flips are rare: ≤ 1.2 % of row-runs for severe and ≤ 1.7 % for A−.

## 2. Actual interventions on the same units (S0)

| contrast | units | suffix differs | endpoint shift mean / median (m) | shift > 0.5 m | severe differs | A− differs | mean d_severe | mean d_FDE5 (m) |
|---|---|---|---|---|---|---|---|---|
| RN (Recent-GT − Normal, H7) | 56 | 48 | 2.17 / 0.89 | 33 | 8 | 9 | −0.143 | −1.76 |
| FULL (Full-GT − Normal, H7) | 56 | 48 | 2.24 / 0.78 | 33 | 8 | 9 | −0.143 | −1.93 |
| RV (Reverse-N − Full-GT, H8) | 56 | 48 | 1.85 / 0.70 | 33 | 8 | 9 | +0.143 | +1.53 |
| PERT (OC − U, P1A) | 56 | 52 | 2.15 / 0.95 | 36 | 4 | 8 | −0.036 | +0.11 |
| SELF_R (OC_refR − OC) | 56 | 51 | 1.76 / 0.61 | 32 | 3 | 6 | +0.054 | −0.02 |
| SELF_H (OC_refH − OC) | 56 | 52 | 1.81 / 0.71 | 33 | 5 | 8 | +0.054 | −0.07 |

**Scale comparison.** The interventions change the suffix in 86–93 % of units and the endpoint by ≈ 1.8–2.2 m on average. Normal numeric variation changes the suffix in 9 % (batch) to 26 % (backend/precision) of row-runs, with a mean endpoint deviation of 0.06–0.25 m. Interventions change the severe verdict in 3–8 of 56 units; numeric variation changes it in 0.3–1.2 % of row-runs.

## 3. Paired-contrast stability (unit mean of d; Δ vs S0; decision rule §5 of the protocol)

| contrast | setting | mean d_severe S0 → s | units whose d_severe changed | mean d_FDE5 S0 → s (m) | mean \|unit ΔFDE5\| (m) | material? |
|---|---|---|---|---|---|---|
| RN | SB / SE / SF | −0.143 → −0.143 / −0.125 / −0.143 | 0 / 1 / 0 | −1.76 → −1.61 / −1.78 / −1.73 | 0.18 / 0.09 / 0.06 | **no** |
| RN_mixed (B = 1 vs B = 93 rows) | SB | −0.143 → −0.143 | 0 | −1.76 → −1.63 | 0.17 | no |
| FULL | SB / SE / SF | −0.143 → −0.143 / −0.125 / −0.143 | 0 / 1 / 0 | −1.93 → −1.78 / −1.95 / −1.91 | 0.17 / 0.08 / 0.05 | no |
| RV | SB / SE / SF | +0.143 → +0.125 / +0.107 / +0.125 | 1 / 2 / 1 | +1.53 → +1.56 / +1.53 / +1.54 | 0.05 / 0.10 / 0.07 | **no** (largest \|Δ\| = 0.036 < 0.05) |
| RV_mixed | SB | +0.143 → +0.143 | 0 | +1.53 → +1.54 | 0.02 | no |
| PERT | SB / SE / SF | −0.036 → −0.036 / −0.036 / −0.018 | 0 / 2 / 1 | +0.11 → +0.04 / −0.07 / −0.13 | 0.11 / 0.82 / 0.68 | **yes** (FDE5 sign / > 25 %) |
| SELF_R | SB / SE / SF | +0.054 → +0.054 / +0.071 / +0.071 | 0 / 1 / 1 | −0.02 → +0.01 / +0.15 / +0.16 | 0.10 / 0.49 / 0.52 | **yes** (FDE5 sign) |
| SELF_H | SB / SE / SF | +0.054 → +0.036 / +0.054 / +0.071 | 1 / 2 / 1 | −0.07 → −0.03 / +0.17 / +0.11 | 0.07 / 0.43 / 0.50 | **yes** (FDE5 sign) |

**Reading.**
- **RN, FULL and RV (the main AutoVLA context contrasts) are not material under any setting.**
  - The severe contrast moves by at most 0.036 (2 of 56 units).
  - The FDE5 contrast moves by ≤ 0.16 m against effects of 1.5–1.9 m (≤ 9 %).
  - Pairing rows from different batch compositions (RN_mixed, RV_mixed) does not change these contrasts materially on this manifest.
- **The P1A self-reference contrasts are material:**
  - On these units their S0 FDE5 contrasts are close to 0 (−0.07 to +0.11 m), while the unit-level numeric noise is 0.4–0.8 m (SE/SF) and ≈ 0.1 m (SB). The sign of the mean FDE5 contrast therefore follows the numeric setting.
  - Their severe contrasts move by 1–2 units (≤ 0.018), the same size as the S0 effects themselves (0.036–0.054).
  - This does not show the P1-A results are wrong. It shows that on 56 dev units a *near-zero* FDE5 difference between self-reference rows cannot be distinguished from implementation numerics. Any statement that rests on the sign or size of these small differences must be limited.

## 4. Consequences and minimal rerun scope

- **Keep:** RN / RV / Full-GT contrast directions and magnitudes are robust to batch shape, attention backend and arithmetic precision on this manifest. Report the measured floor with them: batch shape changes 9 % of suffixes, backend/precision ≈ 25 %, and severe flips are ≤ 1.2 % of row-runs.
- **Limit:**
  - P1-A self-reference FDE5 differences near zero (PERT, SELF_R, SELF_H) are "not distinguishable from numeric implementation variation" (SB/SE/SF move unit FDE5 by 0.1–0.8 m on average).
  - Row-level outcomes compared *across* batch compositions (e.g. Normal rates across harnesses) differ in ≈ 9 % of suffixes. This is the existing M-A3 caveat, now quantified at the endpoint/verdict level.
- **Minimal rerun scope if a precision-robust self-reference statement is wanted:** re-run P1-A raw (G family; 23.8 min on the 5090 per run) under SE and SF. That is ≈ 1 GPU h for both, plus analysis. Not done here; the decision is for the lead.
- B2 is designed so that all paired cells run at **B = 1 on one shared prefix**. Batch-composition numerics are therefore removed from its contrasts by construction (MATCHED_PREFIX_PROTOCOL §14).

## 5. Limits

- 56 dev scenes, alt0 / G-family units, t001 only. Units within a scene and rows within a unit are not independent. Row-level CIs (Clopper–Pearson) ignore this clustering; unit counts are given alongside.
- SE co-changes the mask construction and softmax precision. SF keeps the bf16-rounded weights, so a true fp32-weight model was not run.
- No separate-process repeat was run (P1 c5b was already bitwise, 886/886). H9/H32/P1B were not run.
- No p-values are given for "no difference". "Not material" means "below the pre-fixed thresholds on this manifest". It does not mean equivalence.
