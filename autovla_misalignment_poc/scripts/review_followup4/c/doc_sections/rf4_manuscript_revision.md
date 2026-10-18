# MANUSCRIPT_REVISION — review_followup4 round (2026-10-18)

**No manuscript `.tex` exists in the repository.** `rg --files -g '*.tex' /root/VLA` finds only `paper_quantitative_package/{tables.tex, paper_cross_model_table.tex, paper_detection_table.tex, paper_navhard_table.tex}` (and third-party LaTeX in vendored packages). So no manuscript was edited. This file gives (1) sentence-level replacements for every A4 wording violation, (2) the A1/A2/A3/B1/B2/B3-driven sentence changes, and (3) the machine-generated tables. The paper-ready KEEP / DELETE text is in `paper_quantitative_package/MANUSCRIPT_REVISION.md`, section "2026-10-18 revision (review_followup4)"; the verdicts are in `CLAIM_CHANGES.md`.

**What was edited in the repository** (manuscript-facing documents only; result files under `outputs/review_followup*/` untouched):
- `/root/VLA/EXPERIMENT_SUMMARY.md`: new §6g (Korean) and in-place corrections, each line marked "(2026-10-18 수정)".
- `/root/VLA/paper_quantitative_package/`: `PAPER_NUMBERS.md` (2026-10-18 section, Claims 25–27, in-place fixes), `MANUSCRIPT_REVISION.md` (banner + 2026-10-18 revision section, in-place fixes), `README.md` (note), `PAPER_EVIDENCE.md`, `EXP35_RESULTS.md`, `CONSISTENCY_AUDIT.md`, `SOURCE_MAP.md` (in-place fixes), `tables.tex` (hand-written lines: end-of-line comment "corrected 2026-10-18"; machine block regenerated), `numbers.json` / `numbers_sources.csv` (+1,532 RF4 rows).
- Not edited: `/root/VLA/review_followup3/*.md` (historical round reports; 7 A4 items listed below with their replacements, superseded by `CLAIM_CHANGES.md`), `CONCLUSIONS.md` (outside the allowed set; its window sentences l. 213–214, 217, 374 are covered by A2 rule R18 below), code docstrings.
- Every replaced text is logged in `DOC_EDITS_APPLIED.csv` (file, line, old, new, source).

## 1. A4 wording violations (64 items; file:line refers to the files before this round's edits)

Status column: **applied** = replaced in place (line marked); **c_tables** = applied by the table generator; **not edited** = historical report, use the replacement when quoting.

<!-- A4_TABLE -->

## 2. Sentence changes driven by A1 / A2 / A3 / B1 / B2 / B3

### A1 — paired success-rate intervals (post-hoc correction; originals kept)

| replace | with | rows |
|---|---|---|
| "OpenVLA C − N (8 bins): 95% CI [−16.7, 0.0] excludes 0" | "C − N 8 bins −8 pp, Newcombe 95% CI [−17.1, +1.0] (p = 0.115); not different from 0" | `RF4.A1.P0E.succ.C-N_d8` |
| any OpenVLA per-task CI, incl. "[0, 0]" / "no difference in task 5" | (delete) or "descriptive; 10 states per task" with the Newcombe interval | `RF4.A1.P0E.succ_task.*` |
| "R − N_rep (8 bins) equivalent within ±10 pp (unadjusted)" | "equivalence within ±10 pp not established (p = 0.052)" | `RF4.A1.P0E.succ.R-Nrep_d8` |
| "sign-flip-inversion 95% CI" (SpatialVLA Tables 7 / 10) | "post-hoc corrected Rao–Scott design-effect Newcombe 95% CI (MOVER for the two-task mean); original sign-flip-inversion CI in the supplement" | Table 13 |
| "task-level CIs span up to 18–29 pp (r4e1; r1e4 6.9–21.6 pp)" | "task-level CIs span up to 18–25 pp (r4e1; r1e4 13–20 pp)" | `RF4.A1.P0B.P1D.*` |
| "exp34 r1e4/perp_left coke C − F: equivalent within ±10 pp (Holm)" | "equivalent within ±10 pp only before multiplicity adjustment (corrected 95% CI [−11.3, +11.3], Holm p 0.15)" | `RF4.A1.chg.*` |
| "P1-D r2e1 − r1e1 C − F interaction (pooled, physical) equivalent within ±10 pp (Holm)" | "equivalence not established (Holm p 0.051)" | `RF4.A1.chg.*` |
| "P1-D r1e4 C − F move_near (physical) within ±5 pp; CI [−2.6, +4.3]" | (delete); corrected CI [−6.9, +8.6] | `RF4.A1.P0B.P1D.r1e4.C-F.move_near.physical` |
| "exp35 r1e4 C − F move_near: CI [−29.1, 0.0]" | "not different from 0 after Holm (p = 0.058); the corrected interval [−29.1, −0.3] barely excludes 0 — a boundary disagreement, not evidence of an effect" | `RF4.A1.P0B.exp35.r1e4.C-F.move_near.primary` |
| Methods: "p and CI by cluster sign-flip (inverted CI)" | "p by cluster sign-flip at δ = 0 (exact); CIs by Newcombe 10 (independent states) or Rao–Scott design-effect Newcombe (repeated states), chosen by a coverage rule fixed before recomputation (post hoc)" | – |

### A2 — correction window (GT persists; no release)

| replace | with |
|---|---|
| "Correcting three consecutive context steps recovered about 90% …" | "Three fixed context positions replaced with ground truth and retained in later contexts recovered about 90% of the full-history correction effect." + qualifiers (scene-weighted 90.1% [81.2, 98.9]; 365 units / 52 scenes / 13 drives; denominator Normal − GT-history = 41.7 pp; 2.96 GT positions per unit, 1.90 differing from own; mean 3.8 steps after the last added GT position) |
| "post-release (free) horizon", "free post-release steps", "after release", "고정 해제 후 horizon" | "steps after the last added GT position (GT retained)" |
| "교정을 멈추면 서서히 재발산", "해제 후 감쇠", "해제 후 재발산 25–29%", "re-divergence after release" | "re-divergence while the w corrected GT positions remain in the context (exp 10, summary only)" |
| "after a 3-step correction the GT information is removed and the effect persists"; "later generation reads only the model's own tokens"; "a short transient correction demonstrated a lasting recovery" | (delete) |
| Impromptu "교정 해제 후 재발산은 거의 없었습니다" | (delete) |
| (optional) | "When each of the three ground-truth tokens was shown only once, as the most recent context position, and then replaced by the model's own token, the recovery was 78% (95% CI 64–87; post-hoc comparison, 326 units, 12 drives)." |

### A3 — Impromptu

| replace | with |
|---|---|
| "Impromptu VLA strongly replicates the token-level temporal pattern" / "Strong replication" / "the three models replicate strongly" | "In Impromptu VLA 3B (open loop, descriptive; implementation parity not established), ground-truth context reduced and a reversed last context increased trajectory amplification by margins well above the 5–6% scene-level flip rate caused by batch and decoding-path numerics alone." |
| "no-op parity 23/300 / 237/2,728"; "comparisons are like for like" | "23/300 compares two decoding implementations and batches; 237/2,728 compares batch compositions; the same path and batch reproduce bit for bit (108/108)" |
| win1 −15.4 pp; direction − magnitude +15.6 pp; random − direction-ok +17.3 pp called "replicated" | "descriptive only (near the numerical floor; condition-correlated batching untested)" |

### B1 — numerical sensitivity

| add / replace | text |
|---|---|
| add (Methods / caption) | "Batch shape alone changed 9% of AutoVLA free-run suffixes and attention backend or fp32 arithmetic about 25%; the Recent-GT, Full-GT and Reverse-N contrasts were unchanged in direction and nearly unchanged in size." |
| replace any P1-A self-reference conclusion that rests on the sign of a near-zero FDE difference | "not distinguishable from numerical implementation variation (B1)" |

### B2 — matched prefix (new round)

| add / replace | text |
|---|---|
| add (Results) | K1b and the B2 part of K2' (`paper_quantitative_package/MANUSCRIPT_REVISION.md`, 2026-10-18 section); Table 12 |
| delete | "the effect requires a ground-truth history"; "pure provenance"; "online correction / deployment improvement"; "external replication"; "the correction was released"; pooling with the repeated runs |
| add (Limitations) | "The matched-prefix factorial uses one retained edit at a single anchor on existing drives; it is an oracle-conditioned diagnostic and does not test release or repeated schedules." |

### B3 — OpenVLA

| replace | with |
|---|---|
| "OpenVLA replicates …" / OpenVLA as core or verified evidence | "descriptive; intervention path not parity-verified" (K10') |
| "effects within the 20% rerun noise" used as a threshold | "no detected task-success effect (−3 pp [−11.9, +5.9]; equivalence not established); re-running alone changed 20% of outcomes" |

## 3. Machine-generated tables

- `paper_quantitative_package/numbers.json` — new rows `RF4.*` (1,532): `RF4.B2.*` (1,176: B2 contrasts and denominators, status `VERIFIED_IN_RUN_ALL_UNITS`, analysis tag preregistered / oracle-conditioned), `RF4.B1.*` (47: materiality), `RF4.A1.*` (213 corrected CI rows + 36 judgment changes; status NOT_VERIFIED for OpenVLA and exp 35, VERIFIED for P1-D), `RF4.A4.*` (25: cohort map, AUDIT), `RF4.A2.*`, `RF4.A3.*` (Impromptu, NOT_VERIFIED), `RF4.B3.OPENVLA_GATE` (NOT_RUN), `RF4.FEAS.pi0fast` (FEASIBILITY_ONLY). 781 existing rows carry `rf4_note`.
- `paper_quantitative_package/tables.tex`
  - between `% >>> BEGIN machine-generated review_followup3 (cp2_tables.py) >>>` and its END: Tables 8–11 regenerated with corrected captions / labels (P = evaluation frame from 353 sampled scenes, F = 18 drives; Table 9 "all (208 selected)", learned extrapolator renamed L; Table 10 marked "original sign-flip-inversion CI … corrected CIs in Table 13"; Table 11 B1 floor). Values unchanged.
  - new block `% >>> BEGIN machine-generated review_followup4 (c_tables.py) >>>`: **Table 12** B2 matched-prefix factorial (F / E4 / dev), **Table 13** post-hoc corrected CIs (SpatialVLA P1-D per task, OpenVLA state level, next to the original intervals), **Table 14** B1 numerical sensitivity.
- Regenerate: `scripts/review_followup4/c/c_numbers.py`, then `c_tables.py`.
