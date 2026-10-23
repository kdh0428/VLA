
# Claim 28. The GT-free endpoint excess survives alternative computation settings (new 2026-10-18; review_followup5 B_NUMERIC)
**Evidence** (driving / AutoVLA / open-loop; P1-A family G, 1,453 units / 208 selected scenes, A− 364 / 52 / 13 drives; S0 = original computation, reproduced token-for-token 14,530/14,530; SE = eager attention; SF = fp32 arithmetic; same units, reference, perturbation, batch, CRN)
- A− excess Δ_model − Δ_b (m), S0 → SE / SF: Geo 4.37 → 4.06 / 4.45; CM 2.35 → 2.04 / 2.43; CV0 4.02 → 3.71 / 4.10; CTRV-K3 3.02 → 2.71 / 3.10; CTRA-2 1.43 → 1.12 / 1.51; learned prefix extrapolator 1.97 → 1.65 / 2.04. Primary family (12 tests, Holm): all p ≤ 0.009.
- Paired setting change in A−: SE − S0 −0.31 m [−0.91, +0.40]; SF − S0 +0.08 m [−0.39, +0.52]; not material by the pre-fixed rule, but changes up to ≈ 0.9 m are not excluded.
- A+ and selected-all: unchanged conclusions (no general excess over motion extrapolation).

**Source:** `O/review_followup5/B_NUMERIC/B_NUMERIC_REPORT.md`, `B_NUMERIC_RESULTS.csv`; `numbers.json` `RF5.BN.*`; Table 17.
**Manuscript-safe:** "In failure-selected scenes, the endpoint response to a ground-truth-free context perturbation exceeded coordinate accumulation, four kinematic extrapolations and a learned motion extrapolator under the original computation and under eager attention and fp32 arithmetic (all 12 Holm-adjusted tests p ≤ 0.009); the paired change between settings was −0.31 m [−0.91, +0.40] and +0.08 m [−0.39, +0.52]."
**Overclaim:** "numerically identical" (17–20% of suffix tokens change); "equivalent across settings"; extending it to E / GF families or to the near-zero self-reference FDE contrasts; calling the procedure GT-free online (t* and strata use the GT label).
