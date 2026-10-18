---

# Claim 25. The latest context value matters; no dependence on an older GT prefix in a one-shot schedule (new 2026-10-18; review_followup4 B2, preregistered 552b3ac)
**Evidence** (driving / AutoVLA / open-loop 5 s plan; P1-C cohort, primary anchor a = t*+3, t* ≤ 4; one edited prefix per unit-anchor, edit **retained** at fixed positions; every cell B = 1; severe = A− ∧ FDE5 > 3 m; scene-weighted; drive-cluster bootstrap 95% CI; exact drive sign-flip p; Holm m = 4)
- Failure stratum F (100 scenes / 711 unit-anchors / 18 drives; α = 0.05): **C_GT = GT_NSRC − GT_REF +25.3 pp [+20.5, +29.3]** (Holm 4.6e-5); **C_NAT = NAT_NSRC − NAT_REF +25.5 pp [+21.6, +29.0]** (Holm 3.1e-5); **INT = C_GT − C_NAT −0.2 pp [−3.8, +3.1]** (p = 0.92); C_PROV = GT_CSRC − GT_NSRC −15.8 pp [−20.1, −10.2] (Holm 1.5e-3).
- B2 eligible domain E4 (Hajek, recomputed weights; 312 sampled scenes / 2,160 / 24 drives; α = 0.025): C_GT +3.3 pp [+1.0, +5.6] (Holm 0.051, **not rejected**); C_NAT +4.0 pp [+2.7, +5.7] (Holm 6.5e-5); INT −0.7 pp [−2.2, +0.8]; C_PROV −2.2 pp [−4.2, −0.2] (Holm 0.13).
- INT where the older values truly differ: F −1.7 pp [−5.4, +1.9]. Older-prefix main effects within a latest level: F +2.0 to +4.8 pp, P ≈ 0. Day–vehicle clustering (F 10 groups): C_GT, C_NAT p = 0.002, C_PROV p = 0.0098, INT p = 0.85.
- Development cohort (A− 49 scenes / 13 drives; descriptive): C_GT +28.5, C_NAT +28.4, INT +0.1, C_PROV −16.3 pp.
- Sensitivity anchor (n_old = 2): F INT −0.3 pp; P INT +2.7 pp [+1.1, +4.6], p = 0.0046 (not a preregistered test).
- Integrity: identity checks on 100% of 4,160 (P1-C) and 2,630 (dev) unit-anchors; P1-C inputs reproduce the stored P1-C Normal tokens in 2,160/2,160 units.

**Source:** `O/review_followup4/B/MATCHED_PREFIX_RESULTS.{md,csv}`; `numbers.json` `RF4.B2.*`; Table 12.
**Manuscript-safe:** "In a preregistered matched-prefix diagnostic with a single retained edit, placing the frozen Normal-branch token instead of ground truth in the most recent context position raised severe outcomes in the P1-C failure stratum by 25.3 pp behind a ground-truth older prefix and by 25.5 pp behind the model's own natural older prefix (18 drives); the interaction was −0.2 pp [−3.8, +3.1]. This oracle-conditioned diagnostic indicates that, in a one-shot schedule, the effect tracks the value of the latest context token rather than an older ground-truth prefix; across the B2 eligible domain the effect was small (+3–4 pp)."
**Overclaim:** "the effect requires a GT history"; "pure provenance effect" (NSRC vs CSRC are value differences); "online correction" / "deployment improvement"; "external replication" (existing drives); "release of the correction was tested"; pooling with the repeated Recent-GT / Reverse-N results; quoting the E4 numbers as the original 2,191-scene estimand.

# Claim 26. Numerical sensitivity of the AutoVLA contrasts (new 2026-10-18; review_followup4 B1)
**Evidence** (56 dev scenes, Stage-0 rule take 20; t001; RTX 5090)
- Same-process repeat: bitwise. Batch shape alone changes **8.9%** of free-run suffixes (logit Δ ≤ 0.5); eager attention 26.4%, fp32 arithmetic 25.0% (logit Δ ≈ 2). Severe flips ≤ 1.2% of row-runs.
- RN / Full-GT / RV paired contrasts: not material under any setting (severe Δ ≤ 0.036; FDE5 Δ ≤ 0.16 m vs effects of 1.5–1.9 m).
- P1-A self-reference contrasts (OC − U, OC_refR − OC, OC_refH − OC): material — the near-zero FDE5 contrasts change sign with the numeric setting (unit-level noise 0.4–0.8 m under SE/SF).

**Source:** `O/review_followup4/B/B1/NUMERICAL_SENSITIVITY.{md,csv}`; `RF4.B1.*`; Table 14.
**Manuscript-safe:** "The Recent-GT, Full-GT and Reverse-N contrasts were unchanged in direction and nearly unchanged in size under batch-shape, attention-backend and fp32-arithmetic changes; near-zero differences between ground-truth-free self-reference conditions could not be distinguished from numerical implementation variation."
**Overclaim:** subtracting the numeric floor from effects; "all P1-A contrasts are numerically robust"; calling B1 a replication of the main effects.

# Claim 27. Corrected paired success-rate intervals (new 2026-10-18; review_followup4 A1, post hoc)
**Evidence:** see the 2026-10-18 section above and `O/review_followup4/A1_A4/CI_VALIDITY_AUDIT.md`; `RF4.A1.*`; Table 13.
**Manuscript-safe:** "Paired success-rate differences are reported with Newcombe score intervals (independent initial states) or Rao–Scott design-effect Newcombe intervals (repeated physical states), chosen by a coverage rule fixed before recomputation; difference-from-zero tests are exact sign-flip / McNemar tests."
**Overclaim:** citing any sign-flip-inversion success CI with zero width or an end at 0; "no difference in task k" from 10 OpenVLA states; reading the exp35 r1e4 C − F move_near corrected CI (barely excluding 0) as an effect while the δ = 0 test does not reject.
