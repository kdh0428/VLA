# REVIEW_COVERAGE — FINAL (review_followup3, 2026-10-15)

**State after the CP2 decisions** (`CP2_DECISION.md`, 444299f; executed in fae513c):
- A5 SpatialVLA split-fingerprint re-verification: PASS 20/20.
- O0 AutoVLA gate extension (H32, P1B): PASS.
- **Tier 3 was not run**: P3 and P4 by decision; P5 because no disjoint logs exist.

Every item that depends on Tier-3 work is **"unresolved — addressed by narrowing"**, never "resolved".

"Required limitation sentence" means text that must stay in the manuscript even where the work succeeded. It comes from the prompt's "남는 한계" column; additions from this review round are marked (+). Row IDs refer to `paper_quantitative_package/numbers.json`.

| item | work | verdict | evidence (rows / files) | required limitation sentence (manuscript) |
|---|---|---|---|---|
| **M1** failure-stratum conditional effects, population uncertainty, A− frequency | P0-A done; P5 NOT_RUN (no disjoint logs) | **partially resolved.** Population inference beyond the evaluation set: **unresolved — addressed by narrowing** | Selection flow 2,747 → 52 A− (all A−, π = 1) / 156 A+; per-contrast denominators; RN/RV/DM for F, S and P with CIs; prevalence with drive / day–vehicle CIs; verified P = F + S decomposition. Rows `P0A.FLOW.*`, `P0A.P1C.*`, `P0A.DEV.*`, `P0A.PREV.*`, `P0A.DECOMP.*`; Table 8 (`tab:rf3_population`). | "The A− frequency is a frequency in the evaluation set, not in the deployment distribution." (+) "The −4.1/+3.6 pp effects are evaluation-population estimands over the 2,191 eligible scenes of the P1-C logs." |
| **M2** the Reverse treatment is not matched across models | P3 NOT_RUN (CP2 decision) | **unresolved — addressed by narrowing** | Specification only (`P1/INTERVENTION_SPEC.md` M-A1, §5). AutoVLA Reverse-N = Normal-branch token injection; Impromptu = own-branch waypoint, dynamic. P3 protocol kept, not run. | "Effect sizes cannot be compared across models." "Without the Impromptu run, the Impromptu Reverse remains unmatched." (+) "Interventions were matched in spirit rather than identical across models." |
| **M3** the CM baseline is too narrow | P2-A, P2-B done | **resolved** (the claim is narrowed to A−) | CM = body-frame SE(2) repeat. Added CV0, CTRV-K3, CTRA-2 and a learned non-AR motion extrapolator (2 inputs × 3 seeds, drive-disjoint training). A− exceeds all; A+ / all do not. Rows `P2A.*`, `P2B.*`, `P2.verdict.*`; Table 9 (`tab:rf3_motion`). | "The extrapolator is not a non-AR version of the VLA." (+) "The excess over motion extrapolation holds only in failure-selected (A−) scenes." |
| **M4** task-level evidence | P0-B done; P4 NOT_RUN (CP2 decision; ±10 pp per task approved if ever run) | **partially resolved.** The new-physical-state part: **unresolved — addressed by narrowing** | Full task × config × contrast table: exp 35 and P1-D separate, both clusterings, ±10/±5 TOST, equal-task means, heterogeneity, weighting. Rows `P0B.*`; Table 10 (`tab:rf3_spatial`). P4 protocol and power plan kept. | "Only two tasks." (+) "move_near has 10 physical layouts; per-task success effects remain uncertain (±10 pp equivalence for 3 of 8 P1-D task-level contrasts; ±5 pp for none)." |
| **M5** small number of clusters, higher-level dependence | P0-D done; P5 NOT_RUN — blocked | **partially resolved** (dependence sensitivity). External replication: **unresolved — addressed by narrowing** | Drive / day–vehicle / vehicle / day / component axes; exact floors; leave-one-group-out (no sign flips). Inventory: 0 day-and-vehicle-disjoint logs. Rows `P0D-*`, `P0D.DISJ.*`. | "Without day/vehicle-disjoint logs there is no external validation." (+) "Under day × vehicle component clustering (6 groups), the exact test cannot reach Holm < .05 for the new-drive contrasts." |
| **M6** implementation specification, no-op | P1 done for AutoVLA (6 paths) and SpatialVLA; Impromptu / OpenVLA spec only | **partially resolved.** The Impromptu / OpenVLA part: **unresolved — addressed by narrowing** (their sections stay descriptive or unverified) | Spec table `P1/INTERVENTION_SPEC.md`. AutoVLA gate PASS: H7/H8/H9/P1A (Stage 0) + H32/P1B (O0), 0 failures. SpatialVLA restore audit PASS, with C2 settled by the split-fingerprint re-verification (strict 0/20 and post-hoc A5 20/20 kept on record). Rows `P1.AV.*`, `P1.SV.*`; Table 11 (`tab:rf3_verification`). | "Past runs outside the verification manifest are not retro-validated." (+) "The Impromptu and OpenVLA intervention paths were not parity-verified." |
| **M7** missing numbers | P0-A/B/C done | **resolved** for every row with unit-level data; **NR kept** where none exists | +4.37 m fixed to one authoritative row ([+3.34, +5.17], Holm 7.3e-4; the post-hoc V3 row is kept separate). Rounding and label errors corrected in EXPERIMENT_SUMMARY, PAPER_NUMBERS, tables.tex and MANUSCRIPT_REVISION (`cp2_doc_fixes.py`). All AutoVLA rows are now on verified paths. | "Rows without unit-level data stay NR" (exp 5, exp 10). |
| **m1** weighting of eq. (8) | P0-C | **resolved** | Scene-weighted recomputation, joint numerator/denominator resampling. Old unit-weighted values kept side by side. Rows `P0C.R9.rec.*`. | — |
| **m2** Figure 2 n | P0-C | **resolved** | 365 units / 52 scenes / 13 drives / 10 day–vehicles, identical for every window (`P0C.R9.counts.*`). | — |
| **m4** OpenVLA comparison | P0-E | **resolved as a re-analysis; the claim stays descriptive** | Initial-state clusters, task strata, equal-task weights. Episode-level intervals kept. 92.6 % definition fixed from code. Physical-unit translation done; gripper binarisation recorded. Rows `P0E.*`. | "Precision is low with 10 initial states per task." |

## Summary

| verdict | items |
|---|---|
| resolved | M3, M7 (rows with unit-level data), m1, m2, m4 (descriptive re-analysis) |
| partially resolved | M1, M4, M5, M6 |
| unresolved — addressed by narrowing | M2; plus the P4 part of M4, the P5 part of M1/M5, and the Impromptu/OpenVLA part of M6 |
