# STATUS — review_followup4 (final, 2026-10-18)

**Basis.** AUTOVLA_NEXT_STEPS_20261018.md (§3 A1–A5, §4 B1–B3, §5 C, §6 feasibility, §7 deliverables). AutoVLA is the main model; SpatialVLA and OpenVLA are complementary evidence at different intervention / execution levels.

**History is preserved.** The CP2 decision of review_followup3 not to run P3 and P4 (`/root/VLA/review_followup3/CP2_DECISION.md`, commit 444299f) stays recorded as **NOT_RUN (historical)**. B2 below is a **new round** with its own protocol (552b3ac) and results (c7d1bba). It is **not** a retroactive DONE of P3.

**Status words:** DONE / PARTIAL / BLOCKED / NOT_RUN. **Compute:** CPU = re-analysis of stored data only; GPU-verify = model runs that verify code paths or numerics; GPU-effect = a new effect-estimating experiment. All GPU work ran on the RTX 5090, bf16.

## Task table

| item | status | compute | result (one line) | claim impact | commit / files |
|---|---|---|---|---|---|
| **A1** degenerate success-rate CIs | **DONE** (post-hoc correction) | CPU | Sign-flip inversion collapses for paired binary data with zero / one-signed discordance (±2.4958e−13 reproduced bit for bit). Newcombe 10 (iid) and Rao–Scott Newcombe / MOVER (clustered) selected by a coverage rule fixed before the real-data recomputation. 984 rows recomputed; 36 judgment changes. | OpenVLA per-task CIs not citable; C − N "excludes 0" deleted; exp34 coke C − F → unadjusted only; P1-D pooled ±10 pp and per-task 3/8 survive; CI widths 18–25 pp | 2a31ad7; `A1_A4/CI_VALIDITY_AUDIT.{md,csv}`, `A1_JUDGMENT_CHANGES.csv` |
| **A2** correction-window semantics | **DONE** | CPU | GT persists at fixed positions in every later context; nothing is released (4,136/4,136 retained transitions, 0 removals; 23,936/23,936 stored rows follow the rule). | 90% sentence narrowed (retained GT); all release language deleted | 5523ce3; `A2_A3/WINDOW_SEMANTICS_AUDIT.md` |
| **A3** Impromptu parity cause | **DONE** (cause identified; path not gated) | CPU + GPU-verify (≈ 3 GPU-min, 4 + 1 scenes) | Same path and batch reproduce bit for bit (108/108); batch size, padding, prefill split or decoder implementation alone change outputs; 5–6% scene-level verdict flips. | Impromptu downgraded to descriptive / unverified; large contrasts kept with the floor | 5523ce3; `A2_A3/IMPROMPTU_PARITY_DIAGNOSIS.md` |
| **A4** denominators / metrics / scope | **DONE** | CPU | 8 sets, 14 numbers, 3 definitions asserted from source tables; 64 file:line wording violations with replacements. | Applied to EXPERIMENT_SUMMARY and the paper package (57 of 64 applied; 7 in historical review_followup3 reports listed, not edited) | 2a31ad7; `A1_A4/COHORT_ESTIMAND_MAP.{md,csv}`, `A4_WORDING_VIOLATIONS.csv`; this round `DOC_EDITS_APPLIED.csv` |
| **A5** reproducibility package | **DONE** | CPU | `reproducibility_minimal.zip` (≈ 72 MB; code, protocols, manifests, judge logs, unit-level tables, B2 cells / bank, pinned revisions and checksums, MISSING list). Regeneration actually run from an unzipped copy: 6 targets (A1 coverage selection, A1 corrected CIs, B2 P1-C, B2 dev, P0-A population effects, P2 motion baselines), 22/22 files reproduced (20 byte-identical, 2 equal in value: run time / absolute input path only). | Third-party recomputation of the core tables possible from the ZIP alone | this round; `REPRO_CHECK.{md,csv}`, `SOURCE_MANIFEST.csv` |
| **B1** AutoVLA numerical sensitivity | **DONE** | GPU-verify (18.3 min) | RN / Full-GT / RV contrasts not material under batch shape, eager attention or fp32 arithmetic; P1-A self-reference near-zero FDE contrasts material. | K8 kept; self-reference FDE claims limited | 552b3ac; `B/B1/NUMERICAL_SENSITIVITY.{md,csv}` |
| **B2** AutoVLA matched-prefix factorial | **DONE** (new effect experiment, new round) | GPU-effect (gate + pilot; dev main 52.9 min; P1-C main ≈ 86 min) | F: C_GT +25.3, C_NAT +25.5, INT −0.2 [−3.8, +3.1], C_PROV −15.8 pp (Holm). E4: C_NAT +4.0 (Holm sig.), C_GT +3.3 (Holm 0.051, n.s. at α 0.025), INT −0.7. Identity checks 100% of 4,160 + 2,630 unit-anchors. | Claim 1b / 2 (new); "requires GT history" deleted; oracle-conditioned, one-shot, retained edit, existing drives | protocol 552b3ac, results c7d1bba; `B/MATCHED_PREFIX_{PROTOCOL,RESULTS}.md`, `.csv` |
| **B3** OpenVLA minimal path gate | **NOT_RUN** (user decision 2026-10-17) | – | Checkpoint not on disk; ≈ 2.3–3 GB free. | OpenVLA descriptive / unverified, excluded from core evidence | `OPENVLA_GATE.{md,csv}` |
| **C** claims, manuscript text, tables | **DONE** (no manuscript `.tex` in the repo) | CPU | CLAIM_CHANGES (keep / narrow / delete); MANUSCRIPT_REVISION (64 A4 + A1/A2/A3/B1/B2 changes); numbers.json +1,532 RF4 rows; tables.tex Tables 8–11 captions corrected, new Tables 12–14; EXPERIMENT_SUMMARY §6g; paper package round sections. | – | this round (uncommitted) |
| **§6** π₀-FAST feasibility (+ WorldVLA note) | **DONE** (feasibility only) | CPU (code / metadata reading) | Feasible with conditions, not feasible on this machine today (disk ≈ 20–25 GB; JAX on sm_120 unverified); cross-step action-context interventions NA; WorldVLA matched mask pair unverified. | No π₀-FAST claim | 28f1bfd; `FEASIBILITY.md` |
| P3 (review_followup3 CP2) | **NOT_RUN (historical)** | – | CP2 decision 444299f | superseded in content by B2 for AutoVLA; record unchanged | `/root/VLA/review_followup3/STATUS.md` |
| P4 (review_followup3 CP2) | **NOT_RUN (historical)** | – | CP2 decision 444299f; still not run | SpatialVLA per-task new-state part stays narrowed | same |
| P5 | **NOT_RUN — BLOCKED** (data, unchanged) | – | 0 day-and-vehicle-disjoint logs | no external validation | same |

## Counts for this round

- **New GPU runs:** A3 parity (4 scenes + 1 unmodified exp_cde scene, ≈ 3 GPU-min); B1 (56 scenes × 4 settings + repeat, 18.3 min); B2 gate (8 units / 16 unit-anchors), throughput pilot, dev main (196 scenes / 1,327 units / 2,630 unit-anchors, 52.9 min) and P1-C main (312 scenes / 2,160 units / 4,160 unit-anchors, ≈ 86 min). One new effect experiment (B2). No OpenVLA, SpatialVLA or π₀-FAST inference.
- **CPU-only:** A1, A2, A4, A5, C, §6, all analyses.

## Independent units (denominators)

| data | units |
|---|---|
| AutoVLA dev A− mechanism stratum | 52 scenes / 365 units / 13 drives / 10 day–vehicles |
| P1-C failure stratum F | 105 scenes / 748 units / 18 drives / 10 day–vehicles |
| P1-C evaluation frame (P) | 2,191 eligible scenes; 353 sampled scenes / 2,436 units / 24 drives / 16 day–vehicles; 6 connected day–vehicle components |
| B2 P1-C primary anchor | F 100 scenes / 711 unit-anchors / 18 drives; E4 1,775 eligible, 312 sampled scenes / 2,160 / 24 drives |
| B2 dev | 196 scenes / 1,327 / 19 drives (A− 49 scenes / 13 drives) |
| B1 | 56 dev scenes (A− 15 / A+ 41), 112 units |
| SpatialVLA P1-D | coke 120 physical states; move_near 120 instruction-inclusive states from 10 physical layouts |
| OpenVLA | 100 initial states (10 per task), 1 episode per state and condition (descriptive) |

## Limits that remain in the conclusions

- No external (day/vehicle-disjoint) validation; P1-C and B2 re-use drives that share days and vehicles with development.
- B2 is single-anchor and one-shot with a retained edit; it does not test release or repeated schedules, and GT cells make it an oracle-conditioned diagnostic.
- B1 covers 56 development scenes and four paths; H9 / H32 / P1B were not re-run under alternative numerics.
- Impromptu and OpenVLA intervention paths are not parity-gated (B3 not run).
- SpatialVLA per-task success effects remain uncertain (two tasks; move_near 10 layouts; P4 not run).
- No manuscript `.tex` exists in the repository: sentence-level replacements and tables are provided, the manuscript itself was not edited.
- Files written in this round are **not committed** (instruction: no commits).
