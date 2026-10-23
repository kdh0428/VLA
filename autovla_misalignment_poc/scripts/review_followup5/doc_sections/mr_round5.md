
---

## 2026-10-18 revision 2 (review_followup5)

Supersedes conflicting wording above (including the review_followup4 section). Sources:
`autovla_misalignment_poc/outputs/review_followup5/{CLAIM_CHANGES.md, MANUSCRIPT_REVISION.md}`; tables 15–17 in `tables.tex`.

**K1b (replaces the review_followup4 K1b)** — driving / AutoVLA / open loop [`RF4.B2.p1c.a1.severe.*`, `RF5.B2C.*`]
> "In the tested single-anchor, retained-edit condition, correcting the most recent context token to ground truth had a large effect on severe outcomes in the P1-C failure stratum both when the earlier compared context position held ground truth (25.3 pp, cluster-t 95% CI 20.5–30.2) and when it kept the model's own generated value (25.5 pp, 21.6–29.4; 100 scenes, 18 drives). The estimated difference between the two conditions was −0.2 pp (−4.2 to +3.8). The earlier factor is a single context position; the earlier-prefix main effect itself was +2.0 to +4.8 pp. This is a ground-truth-conditioned diagnostic on re-run existing drives, significant at drive and day–vehicle level; with six connected day–vehicle components it is sign-consistent but cannot reach a Holm-adjusted p below 0.125."

**K2' amendment** — replace "−4.1 pp [−6.4, −2.4] and +3.6 pp [+2.2, +5.5]" by "−4.1 pp and +3.6 pp (drive-level sign-flip p = 5.7 × 10⁻⁶ and 9.5 × 10⁻⁷; no interval method reached nominal coverage for this weighted structure)"; replace the B2 E4 intervals by "+4.0 pp (p = 1.6 × 10⁻⁵) and +3.3 pp (p = 0.017, not significant after Holm)".

**K1' amendment** — the intervals become cluster-robust t: 39.8 pp [28.7, 50.9], 28.6 pp [20.1, 37.2] (13 drives); new drives −34.2 [−39.0, −29.4] and +26.8 [+22.1, +31.5].

**K4'' (adds B_NUMERIC)**
> "In failure-selected scenes, the endpoint response to a ground-truth-free context perturbation exceeded coordinate accumulation, four kinematic extrapolations and a learned motion extrapolator under the original computation and under eager attention and fp32 arithmetic (all 12 Holm-adjusted tests p ≤ 0.009; paired setting changes −0.31 m [−0.91, +0.40] and +0.08 m [−0.39, +0.52])."

**K7 amendment** — "+6.2 pp (p = 0.073; cluster-t 95% CI −0.1 to +12.4)".

**K13' (replaces the review_followup4 K13)**
> "In Impromptu VLA 3B (open loop; implementation parity not established), ground-truth context reduced and a reversed last context increased trajectory amplification; because batch and decoding-path numerics alone flipped 5–6% of scene-level verdicts and condition-correlated numerical effects were not tested, these observations are descriptive."

| # | delete / replace | why | replace with |
|---|---|---|---|
| R28 | "the effect does not depend on an older ground-truth prefix" / "tracks the value of the latest token rather than an older ground-truth prefix" / "older GT does not matter" / "the two effects are equal" | INT non-significance is not equality; the older factor is one position; no margin | K1b |
| R29 | "Impromptu effects are robust because they exceed the 5–6% numerical floor" | an observed disagreement rate is not an upper bound on condition-specific bias | K13' |
| R30 | P-level intervals "[−6.4, −2.4]", "[+2.2, +5.5]", B2 E4 bootstrap intervals presented as validated CIs | coverage not established | K2' amendment |
| R31 | "the GT-near token is not equivalent to GT (±5 pp, TOST p = 0.09)" | that TOST was invalid | "a post-hoc cluster-t TOST placed it within ±5 pp (p = 0.027, unadjusted); not a confirmatory equivalence" |
| R32 | P1-C window "win4 − gt_history within ±5 pp" for S | no valid interval | F only |
