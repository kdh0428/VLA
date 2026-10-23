# MANUSCRIPT_REVISION — paper-ready text after the 2026-10-13 review (review_followup2)

**2026-10-18 (review_followup4):** the section "2026-10-18 revision (review_followup4)" at the end of this file supersedes conflicting wording in all earlier sections (abstract, contributions, conclusion, K1–K12, R1–R17). Lines corrected in place carry "(2026-10-18 수정)"; the replaced text is in `autovla_misalignment_poc/outputs/review_followup4/DOC_EDITS_APPLIED.csv`. Still no manuscript `.tex` exists in the repository; nothing here edits a manuscript.

**Why this file exists.** The repository contains no manuscript `.tex` (checked with `rg --files -g '*.tex' /root/VLA`: the only `.tex` files are `paper_quantitative_package/tables.tex`, `paper_cross_model_table.tex`, `paper_detection_table.tex`, `paper_navhard_table.tex`). The text below is a drop-in replacement for the abstract, contribution list and conclusion, plus a list of sentences to remove from any existing draft.

**Rules followed.** Every sentence uses only claims marked keep / shrink in `autovla_misalignment_poc/outputs/review_followup2/CLAIM_EVIDENCE_MATRIX.md`, with its allowed wording. Numbers come from `PAPER_NUMBERS.md` (2026-10-13). Independent units: drives for AutoVLA/Impromptu, initial-state clusters for SpatialVLA. p values are drive-level exact sign-flip with their floors; no unit-level p is used. Re-analyses of existing data are labelled post hoc in the Methods; preregistered judgments (P1-A–D, Exps. 33–35) are reported as registered.

---

## Abstract (revised)

Autoregressive vision-language-action (VLA) models generate each action conditioned on the actions they have just produced. We ask whether this self-generated action context causally shapes later actions and failures. In AutoVLA on NAVSIM, repeatedly replacing the most recent action context with ground truth at every generation step reduced trajectory amplification in failure-stratum (A−) scenes by 40 percentage points (pp) (95% CI 30–49; 52 scenes, 13 independent drives; exact drive-level sign-flip p = 2.4 × 10⁻⁴, the minimum attainable), and injecting the separate Normal branch's previous token into a ground-truth context restored it (+29 pp). Both effects replicated in a preregistered test on new drives (failure stratum: 105 scenes, 18 drives; no drive shared with the development set, but days and vehicles partly shared). Without any ground-truth information, a single perturbed context token changed subsequent local actions and produced endpoint deviations 4.4 m beyond pure coordinate accumulation; this response exceeded constant-motion extrapolation only in failure-selected scenes. Correcting three consecutive context steps recovered about 90% of the full-history correction effect (95% CI 81–99%), and we found no evidence that corrections at the first deviating steps are more effective than equally long later corrections. Analogous context interventions, matched in spirit but not identical, changed subsequent action tokens in Impromptu VLA, OpenVLA and SpatialVLA. In SpatialVLA closed-loop manipulation, context perturbations changed generated actions and the physical end-effector trajectory, and longer replanning intervals made these deviations more persistent; their effect on task success, averaged over two tasks, was bounded within ±10 pp in the tested configurations, while task-specific effects remain uncertain. Deployable mitigation came from selecting among the model's own candidates and from a current-frame safety filter rather than from conditioning on external references. (2026-10-18 수정)

*(≈ 250 words. If space is tight, drop the mitigation sentence first.)*

## Contributions (revised)

1. **Causal evidence that self-generated action context drives amplification in a driving VLA.** Repeated last-position correction lowers amplification by 39.8 pp [−49.0, −29.9] and Normal-branch re-insertion restores it by 28.6 pp [+20.9, +36.1] (13 drives, p = 2.4 × 10⁻⁴, Holm 7.3 × 10⁻⁴; leave-one-drive-out: no sign change). Repeated last-position correction recovers 93% (92.9% [89.4, 99.1]) of the full-history correction effect. A preregistered replication on new drives (days/vehicles partly shared) reproduces both contrasts: failure stratum (105 scenes, 18 drives) −34.2 / +26.8 pp; evaluation-frame estimand over 2,191 eligible scenes (353 sampled, weighted; 24 drives) −4.1 / +3.6 pp, about 60% of the RN effect coming from the non-failure (A+) stratum. (2026-10-18 수정)
2. **A ground-truth-free control (AutoVLA).** Perturbing only a context token, with the model's own unperturbed rollout as reference, changes later local actions (+0.57 m/segment) and moves the 5 s endpoint 4.37 m [3.32, 5.19] beyond pure coordinate accumulation (post-hoc excess metric; the predeclared P1A-2 row is 4.37 m [3.34, 5.17], Holm 7.3 × 10⁻⁴ — see the 2026-10-15 section). Against constant-motion extrapolation the excess is present only in failure-selected scenes (+2.35 m [1.27, 3.19]); across the 208 selected mechanism scenes (52 A− + 156 A+; not the natural mix) no difference from extrapolation was detected (+0.20 m [−0.32, +0.56], p = 0.51; equivalence not tested). (2026-10-18 수정)
3. **Correction length rather than an early critical window.** Three fixed context positions replaced with ground truth and retained in later contexts recover about 90% (81–99%) of the full-history effect (1/2/3/4 steps: 47/77/90/96%); in a preregistered start-position test, early minus late corrections of equal length were +10.7, +4.3 and +0.5 pp for 1–3 steps, none significant. (2026-10-18 수정)
4. **Cross-model and closed-loop scope.** The token-level context effect recurs in three further temporally or within-step autoregressive VLAs under model-specific interventions whose effect sizes we do not compare across models. In SpatialVLA, a preregistered dose-controlled experiment (4,800 episodes, state restoration verified) shows that longer replanning increases the persistence of context-induced trajectory deviations at a fixed context-perturbation budget (+214 cm·step [171, 255]; a composite configuration effect), that temporal ensembling does not absorb context effects once executed dose is matched, and that the pooled two-task success direct effects lie within ±10 pp (initial-state clusters, Holm), with task-level effects uncertain.
5. **Statistical and data audit.** We report all mechanism contrasts at the level of independent drives (AutoVLA, Impromptu) or physical initial states (SpatialVLA: move_near has only 10 physical layouts), disclose log-level but not drive-level separation of earlier held-out sets, and document protocol amendments and failed runs.

*(If the paper keeps the mitigation results: "6. Candidate selection by decoding confidence improves preregistered held-out PDMS by +0.0095 [0.0046, 0.0148] on held-out logs (not drive-disjoint); on the reactive navhard benchmark most of the gain comes from a current-frame safety filter (+0.105 of +0.118 EPDMS).")*

## Conclusion (revised paragraph)

Self-generated action context is a causal input to subsequent action generation in autoregressive VLAs. In AutoVLA, repeatedly correcting the most recent context position removes most of the amplification that follows a small first deviation, re-inserting the model's own previous action restores it, and both effects replicate on new drives; a ground-truth-free perturbation shows that the context alone changes later local actions beyond coordinate accumulation. Two stronger readings did not survive scrutiny: the response exceeds constant-motion extrapolation only in failure-selected scenes, and the first steps after a deviation are not demonstrably more important than later steps of equal length—what matters is how many consecutive context steps are corrected. Direction errors were more harmful than magnitude errors in the development drives, but this was inconclusive on new drives. Across models, interventions were matched in spirit rather than identical, so we compare directions, not magnitudes. In closed-loop manipulation with short chunks and frequent replanning, context perturbations change generated actions and the physical trajectory, but their effect on success, averaged over two tasks, was bounded within ±10 pp in the tested configurations; task-specific and smaller effects remain open, and the limited number of distinct physical initial states in the benchmark constrains how precisely they can be measured.

## Limitations paragraph (suggested additions)

- Mechanism experiments in AutoVLA were exploratory (not preregistered); the independent sample is 13 drives (10 day-vehicles), which bounds the smallest attainable exact p at 2.4 × 10⁻⁴. The one preregistered new-data replication (24 drives) shares day-vehicles with the development set.
- Recent-GT is a repeated intervention; the 93% figure is relative to the full GT-history effect. The ground-truth-free control exists only for AutoVLA, and its perturbation position and strata inherit the GT-defined failure label.
- Correction length and the number of steps after the last added ground-truth position change together (the ground-truth positions stay in the context; no release experiment exists); absolute pose and that horizon cannot both be equalized within the 10-token plan. (2026-10-18 수정)
- SpatialVLA: two tasks; move_near has 10 physical layouts shared by all experiments; ±5 pp equivalence and task-level bounds were not established. Experiments 34/35 used history-dependent resets without restore hashes; Experiment 34 recorded no end-effector trajectory.
- Held-out selection results and navhard halves are log-disjoint but not drive-disjoint.
- P1-C: the pilot agreement criterion was relaxed before evaluation data existed (amendment 1); after an out-of-memory failure of one block, that block was re-run with the same command after two other contrasts had been viewed (amendment 2; judgments unchanged and the incomplete analysis preserved).

---

## Sentences to remove or replace

| Remove (as written in earlier drafts / summaries) | Why | Replace with |
|---|---|---|
| "Replacing only the most recent self-generated action token with ground truth … removes 93% of the effect" / "a single token correction" | Recent-GT is applied at every step; 93% is a share of the GT-history effect (85% of the Normal rate) | "Last-position correction applied at every step recovered 93% of the full-history correction effect." |
| Any unit-level p: "p = 8 × 10⁻⁴²", "6.7 × 10⁻³¹", "2 × 10⁻³¹", "7 × 10⁻²⁶¹", "p = 0.0 (underflow)", "p = 6.1 × 10⁻⁴" for direction | Units within scenes/logs/drives are not independent | Drive-level exact sign-flip p with its floor (e.g. "p = 2.4 × 10⁻⁴, the minimum attainable with 13 drives") |
| "Correcting the 2–4 context steps that follow the first deviation recovers 76–96%" / "a short 2–4 step critical window" / "genuine short early window" | 2 steps = 77% (CI ≤ 88%); no early-position advantage (P1-B) | "Three fixed context positions replaced with ground truth and retained in later contexts recovered about 90% of the full-history correction effect (95% CI 81–99%)." + "We found no evidence that corrections at the first deviating steps are more effective than equally long later corrections." (2026-10-18 수정) |
| "A different token with GT-like motion is as effective as the GT token" / "identity does not matter" | ±5 pp equivalence not established (TOST p = 0.09) | "A token with GT-like motion produced a small, non-significant difference from the GT token (equivalence not established)." |
| "Direction errors are more harmful than magnitude errors" (unqualified) | Inconclusive on new drives (p = 0.073) | "… in the development drives; on new drives the effect was in the same direction but inconclusive." |
| "Self-amplification beyond extrapolation" / "the response is indistinguishable from constant-motion extrapolation in failing scenes" | The old comparison mixed in the baseline's own extrapolation error; the excess holds only in A− | "In failure-selected scenes the response exceeded constant-motion extrapolation; across the 208 selected mechanism scenes no difference from extrapolation was detected (equivalence not tested)." (2026-10-18 수정) |
| "GT-free correction fixes failures" / "reference-based correction reduces deviation by 3.5 m" presented as task improvement | It restores reference consistency but worsens GT amplification (+19.7 pp in A−) | "Reference-based correction restored consistency with the model's own rollout but did not reduce ground-truth failures." |
| "4-step replanning makes reverse increase failures (−6.25 pp, p = 0.04)" / "a small task effect appears only for reverse" | p = 0.14–0.26 after initial-state clustering; not replicated (+1.25 pp) | (delete) |
| "Ensembling protects against / absorbs feedback" | Not supported once executed dose is matched | "Temporal ensembling did not absorb context effects once executed dose was matched." |
| "Task success changes by at most 6 pp" / "the context effect on task success is indistinguishable from zero" / "no task-level effect" | Clustered CIs are ±8–14 pp (Exps. 34/35); equivalence only for pooled P1-D at ±10 pp | "… their effect on success, averaged over two tasks, was bounded within ±10 pp in the tested configurations; task-specific effects remain uncertain." |
| "Context-only perturbation moves the trajectory 0.21–0.23 m (vs rerun 0.14 m)" / "+0.116 m trajectory divergence" | Cumulative commanded action, not the actual TCP | Actual TCP: "2.7–6.0 cm mean end-effector distance (P1-D)" |
| "Replanning and ensembling cause the absorption" / "replanning interval alone controls persistence" | Ensemble half deleted; replanning effect is composite (observation refresh, context reset, execution exposure) | "Lengthening the replanning interval increased the persistence of context-induced trajectory deviations even at a fixed context-perturbation budget." |
| Cross-model row "Reverse effect: ✓ ✓ ✓" without definitions; comparing +29 vs +82 pp | Different treatments and amplification definitions | "Interventions were matched in spirit rather than identical across models; effect sizes are not comparable across columns." |
| "Pre-registered held-out" / "new held-out drives" for Exps. 21–26 | Held-out shares 22 drives (85% of scenes) with dev; navhard halves share 16 drives | "held-out logs (not drive-disjoint)" |
| "Replicated on new logs" for P1-C | Understates it: drives are disjoint | "replicated on new drives (failure stratum 105 scenes / 18 drives; evaluation frame 24 drives; days and vehicles partly shared)" (2026-10-18 수정) |
| "SpatialVLA move_near: 120 distinct configurations" / "seed 0–119 = 53 independent states" as independent samples | Only 10 physical layouts, shared across experiments | "120 distinct initial states including the instruction, from 10 physical layouts" |
| "R8R9 confirms Experiments 8–9" | Recovery run with known targets; a reproducibility audit | "The original values were reproduced exactly (220/220) in a recovery run." |
| "Pre-deviation entropy is identical between selected and discarded candidates" | Difference is small but its CI excludes 0 | "Pre-deviation entropy differed only slightly (−0.016) compared with after the deviation (−0.40)." |
| "Attention masks have no effect" | Non-detection, equivalence not tested | "Masking attention to the history produced no detectable change (−1.4 and −3.3 pp)." |

---

## 2026-10-15 revision (review_followup3): exact sentences to KEEP and to DELETE/REPLACE

**Scope rule for this round.** The manuscript is a **driving-VLA** study:
- The main evidence is AutoVLA, open-loop 5 s plan evaluation on navtest.
- SpatialVLA (closed loop, SimplerEnv) and OpenVLA (closed loop, LIBERO-Spatial) are complementary task-level evidence, not replications.
- Every result sentence names its domain, model and evaluation mode.
- No sentence speaks about "VLAs in general" or "autoregressive VLAs".

**Sources.**
- Numbers: `numbers.json` (row IDs in brackets).
- Verdicts: `review_followup3/CLAIM_EVIDENCE_MATRIX.md` (final).
- **This section supersedes conflicting wording above.** In particular, the abstract's opening sentence and the conclusion's first sentence are replaced (R1, R2).

### A. KEEP (use as written)

**K1** — driving / AutoVLA / open loop [`P0A.DEV.RN.Am`, `P0A.DEV.RV.Am`, `P0A.P1C.RN.F`, `P0A.P1C.RV.F`]

> "In AutoVLA (open-loop 5 s plans on NAVSIM navtest), repeatedly replacing the most recent action-context token with ground truth at every generation step reduced amplification in failure-stratum scenes by 39.8 pp [29.9, 49.0] (13 drives), and injecting the separate Normal branch's previous token into a ground-truth context restored it by 28.6 pp [20.9, 36.1]; both effects replicated in a preregistered test on new drives (failure stratum, 105 scenes / 18 drives; −34.2 and +26.8 pp; days and vehicles partly shared)." (2026-10-18 수정)

**K2** — driving / AutoVLA / open loop [`P0A.P1C.RN.P`, `P0A.P1C.RV.P`, `P0A.PREV.DEV.Aminus.2747.drive`]

> "On the evaluation frame of the new-drive logs (2,191 eligible scenes; estimated from 353 sampled scenes with design weights, 24 drives), the corresponding effects were −4.1 pp [−6.4, −2.4] and +3.6 pp [+2.2, +5.5] (about 60% of the RN effect from non-failure scenes); the failure (A−) category occurred in 1.9% of development evaluation scenes, a frequency in the evaluation set rather than in the deployment distribution." (2026-10-18 수정)

**K3** — driving / AutoVLA / open loop, GT-free [`P0C.plus437.AUTHORITATIVE`]

> "Without ground-truth information, perturbing a single context token moved the 5 s endpoint 4.37 m [3.34, 5.17] beyond pure coordinate accumulation (13 drives; Holm-adjusted p = 7.3 × 10⁻⁴)."

If the sentence instead describes the post-hoc excess metric, quote [3.32, 5.19] with "post hoc" and Holm 9.8 × 10⁻⁴. Never mix the two rows.

**K4** — driving / AutoVLA / open loop [`P2A.excess.G.*.end.*`, `P2B.excess.G.*.end.*.primary`, `P2.verdict.*`]

> "In failure-selected scenes, this response exceeded four kinematic extrapolations — including the original constant-motion baseline, which repeats the perturbed token's body-frame motion including its yaw — and a small learned non-autoregressive motion extrapolator (e.g. +2.35 m [1.27, 3.19] over constant motion; +1.97 m [1.06, 2.87] over the learned extrapolator); in non-failure scenes and across the 208 selected mechanism scenes no excess over motion extrapolation was detected (equivalence not tested)." (2026-10-18 수정)

**K5** — driving / AutoVLA / open loop [`P0C.R9.rec.win3.NEW_scene_drive`, `P0C.R9.counts.*`]

> "Three fixed context positions replaced with ground truth and retained in later contexts recovered about 90% (scene-weighted; 95% CI 81–99%) of the full-history correction effect (365 units, 52 scenes, 13 drives for every window length); the number of corrected positions and the number of steps generated after the last added ground-truth position change together." (2026-10-18 수정)

**K6** — driving / AutoVLA / open loop [`P0C.P1B.*`]

> "We found no evidence that corrections at the first deviating steps are more effective than equally long later corrections."

**K7** — driving / AutoVLA / open loop [`P0A.DEV.DM.Am`, `P0A.P1C.DM.F`]

> "Direction errors in the last context token were more harmful than magnitude errors in the development drives (+9.9 pp [3.9, 21.0]); on new drives the effect had the same sign but was inconclusive (+6.2 pp, p = 0.073)."

**K8** — manipulation / SpatialVLA-4B / closed loop, SimplerEnv [`P0B.P1D.*`]

> "In SpatialVLA closed-loop manipulation (SimplerEnv Google Robot; two tasks), context perturbations changed generated actions and the physical end-effector trajectory, while their effect on task success, averaged over the two tasks, was bounded within ±10 pp in the tested configurations; per task, ±10 pp equivalence was established for 3 of 8 contrasts and ±5 pp for none, so task-specific effects remain uncertain."

**K9** — manipulation / SpatialVLA-4B / closed loop

> "Lengthening the replanning interval increased the persistence of context-induced trajectory deviations even at a fixed context-perturbation budget (a composite configuration effect); temporal ensembling did not absorb context effects once executed dose was matched."

**K10** — manipulation / OpenVLA-7B / closed loop, LIBERO-Spatial [`P0E.succ.C-F_d8`, `P0E.succ.C-F_d24`, `P0E.phaseC.d0_|8|_feedback`]

> "In OpenVLA-7B on LIBERO-Spatial (100 initial states, one episode per state and condition), correcting the context after an x-token perturbation changed task success by −3 pp (95% CI [−12, +6], initial-state clusters; 8 bins) and −5 pp ([−14, +3]; 24 bins), while re-running the unperturbed policy alone changed 20% of outcomes; within a single step, an 8-bin x-token shift an 8-bin x-token shift (≈ 2.6 mm commanded) changed at least one later component (y, z, roll, pitch, yaw; gripper excluded) by at least the effective injected number of bins in 92.6% of 4,741 units (initial-state-cluster CI [90.8, 94.0]; translation components alone 72%). These results are descriptive." (2026-10-18 수정)

**K11** — verification [`P1.*`]

> "No-op, self-patch, cache-invalidation and full-recompute parity of the intervention code was verified on fixed verification manifests (AutoVLA: all six intervention code paths; SpatialVLA: state restoration on 20 units); earlier runs outside these manifests are not retro-validated, and the Impromptu VLA and OpenVLA intervention paths were not parity-verified."

**K12** — cross-model

> "Interventions were matched in spirit rather than identical across models; effect sizes are not comparable across models."

### B. DELETE or REPLACE

| # | delete / replace (as written in drafts or summaries) | why | replace with |
|---|---|---|---|
| R1 | Abstract opening: "Autoregressive vision-language-action (VLA) models generate each action conditioned on the actions they have just produced." | General VLA claim | "In driving vision-language-action models that generate a plan token by token, each action is conditioned on the actions just produced; we test whether this self-generated context causally shapes later actions in AutoVLA." |
| R2 | Conclusion: "Self-generated action context is a causal input to subsequent action generation in autoregressive VLAs." | General VLA claim | "In AutoVLA (open-loop driving), self-generated action context is a causal input to subsequent action generation." |
| R3 | "Analogous context interventions … changed subsequent action tokens in Impromptu VLA, OpenVLA and SpatialVLA" when presented as replications of the driving result | Not matched; Impromptu/OpenVLA machinery not parity-verified | K12 plus "Model-specific context interventions also changed subsequent action tokens in Impromptu VLA (open-loop driving), OpenVLA and SpatialVLA (manipulation); these are complementary observations, not replications." |
| R4 | "+4.37 m [3.34, 5.18]" / "+4.37 m [3.32, 5.19], Holm 7.3 × 10⁻⁴" | 5.18 is a double-rounding error; the predeclared Holm was attached to the post-hoc CI | K3 |
| R5 | "+0.57 m/seg [0.44, 0.68]" | Double rounding | "+0.57 m/seg [0.43, 0.68]" |
| R6 | "52/2,747 (1.9%) natural failure rate" / "population effect −4.1/+3.6 pp" (unqualified) | An evaluation-set frequency and an evaluation-population estimand | K2 |
| R7 | "The constant-motion baseline ignores curvature" (reviewer premise) / adding "CTRV" as a curvature fix of CM | CM is already a body-frame SE(2) repeat with yaw (constant turn rate) | Rebuttal: "The constant-motion baseline repeats the perturbed token's body-frame translation and yaw increment, i.e. it is already a constant-turn-rate extrapolation; we additionally report zero-turn, multi-step turn-rate, turn-rate-plus-acceleration and learned non-autoregressive extrapolators (Table 9)." |
| R8 | "Beyond extrapolation" / "self-amplification beyond motion extrapolation" (unrestricted) | Holds only in A− | K4 |
| R9 | SpatialVLA exp 34 docs: "the intervention arms are identical to HF `generate`" / "generate-identical decoding" | Batch-8 chunks equal HF `generate` in only 16/20 audit units (B = 1 decode: 17/20) | "All arms decode with the same fixed batch of 8 and are therefore internally consistent; outputs are not identical to HF `generate` (bf16 batch-shape numerics)." |
| R10 | "+1.2 pp [−5.6, +8.0]" (P1-D r4e1 δR) / "−6.2 pp" (exp 35 r4e1) | Exact half-even ties | "+1.25 pp [−5.6, +8.0]" / "−6.25 pp" (the exp 35 claim stays deleted) |
| R11 | "Task-level CIs span 18–29 pp" (unscoped) | Applies to P1-D r4e1 only | "Task-level CIs span up to 18–29 pp (r4e1)" |
| R12 | "92.6% [89.8, 94.4]" without its cluster unit | That CI uses task clusters | "92.6% (task-cluster CI [89.8, 94.4]; initial-state-cluster CI [90.8, 94.0])" |
| R13 | "Recovery 47/76/90/96%" and "47/77/90/96%" used interchangeably | Unit- and scene-weighted estimands | Name the weighting; scene-weighted is primary (K5) |
| R14 | "Validated on held-out / external data" / "day- and vehicle-independent" for P1-C | No day/vehicle-disjoint logs exist; P1-C shares days and vehicles | "replicated on new drives (failure stratum 105 scenes / 18 drives; days and vehicles partly shared); no external (day/vehicle-disjoint) validation was possible" (2026-10-18 수정) |
| R15 | "Reverse is the same treatment in all models" / "matched Reverse" | P3 not run | K12; AutoVLA arm = "Normal-branch token injection" |
| R16 | "New physical initial states confirm the SpatialVLA result" / "±5 pp per task" | P4 not run | (delete) |
| R17 | "All past runs were verified" | Manifest scope only | K11 |

### C. Limitation sentences that must stay (prompt review table + this round)

1. (M1) "The A− frequency is a frequency in the evaluation set, not in the deployment distribution."
2. (M2) "Effect sizes cannot be compared across models; the Impromptu Reverse remains unmatched because the matched Reverse experiment was not run."
3. (M3) "The learned extrapolator is a motion model, not a non-autoregressive version of the VLA."
4. (M4) "Task-level evidence comes from only two SpatialVLA tasks; move_near has 10 physical layouts, and no new physical states were sampled."
5. (M5) "Without day/vehicle-disjoint logs there is no external validation; under day × vehicle clustering the new-drive contrasts cannot reach Holm-adjusted p < .05 with six independent groups."
6. (M6) "Past runs outside the verification manifests are not retro-validated; the Impromptu and OpenVLA intervention paths were not parity-verified."
7. (M7) "Rows without unit-level data (Experiments 5 and 10) are reported as summaries only."
8. (m4) "OpenVLA precision is low with 10 initial states per task."

---

## 2026-10-18 revision (review_followup4)

**Scope.** Driving / AutoVLA / open-loop 5 s plans is the main evidence. SpatialVLA (closed loop) is complementary task-level evidence. Impromptu VLA and OpenVLA are descriptive and unverified (A3; B3 not run) and are not counted as replications. No sentence speaks about VLAs in general. Sources: `autovla_misalignment_poc/outputs/review_followup4/CLAIM_CHANGES.md` (verdicts), `MANUSCRIPT_REVISION.md` in the same directory (all 64 A4 line-level replacements and every A1/A2/A3/B1/B2 change), `numbers.json` rows `RF4.*`, `tables.tex` Tables 12–14. **This section supersedes conflicting wording above.**

### A. KEEP (replaces K1, K2, K4, K5, K8, K10 above; K3, K6, K7, K9, K11, K12 stay as written)

**K1'** — driving / AutoVLA / open loop [`P0A.DEV.RN.Am`, `P0A.DEV.RV.Am`, `P0A.P1C.RN.F`, `P0A.P1C.RV.F`]
> "In AutoVLA (open-loop 5 s plans on NAVSIM navtest), repeatedly replacing the most recent action-context token with ground truth at every generation step reduced amplification in the failure stratum by 39.8 pp [29.9, 49.0] (52 scenes, 13 drives), and injecting the separate Normal branch's previous token into a ground-truth context restored it by 28.6 pp [20.9, 36.1]. In a preregistered test on new drives (failure stratum: 105 scenes, 18 drives; no drive shared with development, days and vehicles partly shared) the effects were −34.2 pp [−38.3, −29.5] and +26.8 pp [+21.9, +30.7]."

**K1b (new, B2)** — driving / AutoVLA / open loop [`RF4.B2.p1c.a1.severe.*`]
> "In a preregistered matched-prefix diagnostic with a single retained edit, placing the frozen Normal-branch token instead of ground truth in the most recent context position raised severe outcomes in the P1-C failure stratum by 25.3 pp [20.5, 29.3] behind a ground-truth older prefix and by 25.5 pp [21.6, 29.0] behind the model's own natural older prefix (100 scenes, 18 drives); the interaction was −0.2 pp [−3.8, +3.1], and the earlier compared position is a single context position (difference estimate reported with its uncertainty; not equality). Because the reference cells contain ground truth, this is an oracle-conditioned diagnostic, not an online correction." (2026-10-18 수정, review_followup5)

**K2'** — driving / AutoVLA / open loop [`P0A.P1C.RN.P`, `P0A.P1C.RV.P`, `P0A.DECOMP.*`, `RF4.B2.p1c.a1.severe.C_*.all.P.drive`]
> "On the evaluation frame of the new-drive logs (2,191 eligible scenes, estimated from 353 sampled scenes with design weights; 24 drives), the corresponding effects were −4.1 pp [−6.4, −2.4] and +3.6 pp [+2.2, +5.5]; about 60% of the Recent-GT effect came from non-failure scenes. In the matched-prefix eligible domain (t* ≤ 4; 312 sampled scenes) the latest-token effect was small (+4.0 pp [+2.7, +5.7] behind the natural prefix; +3.3 pp [+1.0, +5.6] behind a ground-truth prefix, not significant after Holm). The failure category occurred in 1.9% of development evaluation scenes, an evaluation-set frequency rather than a deployment rate."

**K4'** — driving / AutoVLA / open loop [`P2A.excess.G.*.end.*`, `P2B.excess.G.*.end.*.primary`, `P2.verdict.*`, `RF4.B1.stab.*`]
> "In failure-selected scenes, this response exceeded four kinematic extrapolations — including the original constant-motion baseline, which repeats the perturbed token's body-frame motion including its yaw — and a small learned non-autoregressive motion extrapolator (+2.35 m [1.27, 3.19] over constant motion; +1.97 m [1.06, 2.87] over the learned extrapolator); in non-failure scenes and across the 208 selected mechanism scenes no excess over motion extrapolation was detected (equivalence not tested)."
> (B1, Methods / caption) "The Recent-GT, Full-GT and Reverse-N contrasts were unchanged in direction and nearly unchanged in size under batch-shape, attention-backend and fp32-arithmetic changes (batch shape alone changed 9% of free-run suffixes); near-zero differences between ground-truth-free self-reference conditions could not be distinguished from numerical implementation variation."

**K5'** — driving / AutoVLA / open loop [`P0C.R9.rec.win3.NEW_scene_drive`, `RF4.A2.*`]
> "Three fixed context positions replaced with ground truth and retained in later contexts recovered about 90% of the full-history correction effect (scene-weighted 90.1%, 95% CI 81–99; 365 units, 52 scenes, 13 drives; 2.96 corrected positions per unit, 1.90 of them differing from the model's own token); the number of corrected positions and the number of steps generated after the last added ground-truth position change together."

**K8'** — manipulation / SpatialVLA-4B / closed loop [`RF4.A1.P0B.P1D.*`]
> "In SpatialVLA closed-loop manipulation (SimplerEnv Google Robot; two tasks), context perturbations changed generated actions and the physical end-effector trajectory, while their effect on task success, averaged over the two tasks, was bounded within ±10 pp in the tested configurations (post-hoc corrected Rao–Scott Newcombe intervals, initial-state clusters, Holm); per task, ±10 pp equivalence held for 3 of 8 contrasts and ±5 pp for none, with task-level 95% CIs up to 18–25 pp wide, so task-specific effects remain uncertain."

**K10'** — manipulation / OpenVLA-7B / LIBERO-Spatial [`RF4.A1.P0E.succ.*`, `P0E.phaseC.d0_|8|_feedback`]
> "In OpenVLA-7B on LIBERO-Spatial (100 initial states; intervention path not parity-verified; descriptive), correcting the context after an x-token perturbation produced no detected change in task success (−3 pp, Newcombe 95% CI [−11.9, +5.9]; equivalence not established), while re-running the unperturbed policy alone changed 20% of outcomes; within a single step, an 8-bin x-token shift (about 2.6 mm commanded) changed at least one later component by at least the effective injected number of bins in 92.6% of 4,741 units (initial-state-cluster CI [90.8, 94.0]; translation components alone 72%)."

**K11 addition** (append to K11): "Batch shape alone changed 9% of AutoVLA free-run suffixes; all paired contrasts are therefore computed within one harness and batch, and the B2 cells run at batch size 1 on one shared prefix."

**K13 (new, Impromptu)** — driving / Impromptu VLA 3B / open loop
> "In Impromptu VLA 3B (open loop, descriptive; implementation parity not established), ground-truth context reduced and a reversed last context increased trajectory amplification (descriptive: batch and decoding-path numerics alone flipped 5–6% of scene-level verdicts, which is not an upper bound on condition-specific numerical bias)." (2026-10-18 수정, review_followup5)

### B. DELETE or REPLACE (in addition to R1–R17)

| # | delete / replace | why | replace with |
|---|---|---|---|
| R18 | "Correcting three consecutive context steps recovered about 90%" / "a short (transient) correction has a lasting effect" / "after release the model conditions on its own tokens only" / "post-release horizon" / "re-divergence after release" | A2: GT positions are retained in every later context; no release exists | K5'; "steps after the last added ground-truth position"; "re-divergence while the corrected ground-truth positions remain in the context (exp 10, summary only)" |
| R19 | "Impromptu VLA strongly replicates …" / "the three models replicate strongly" / "no-op parity 23/300, 237/2,728" | A3: numerical floor 5–6%, path not gated; the pairs were not same-condition repeats | K13 |
| R20 | any OpenVLA per-task success CI, "no difference in task k", "C − N 95% CI excludes 0" | A1: sign-flip inversion degenerate; per-task n = 10 | K10'; per-task rows descriptive |
| R21 | "exp34 perp_left coke C − F equivalent within ±10 pp (Holm)"; "task-level CIs span up to 18–29 pp (r4e1; r1e4 6.9–21.6 pp)"; "P1-D r1e4 C − F move_near within ±5 pp" | A1 corrected intervals | "unadjusted only"; "up to 18–25 pp (r4e1; r1e4 13–20 pp)"; (delete) |
| R22 | "the Reverse effect requires a ground-truth history" / "pure provenance effect" / "the matched-prefix result is an external replication or an online correction" | B2: INT ≈ 0; value differences; existing drives; oracle cells | K1b |
| R23 | "replicated on 24 new drives" for the failure-stratum effects; "population effect −4.1 / +3.6 pp"; "across all scenes" for the 208 selected scenes | A4 denominators | K1', K2', K4' |
| R24 | "92.6% of later dimensions amplified" / "93% of the remaining dimensions move further" | A4: at least one component; bins; commanded | K10' |
| R25 | P1-A self-reference conclusions that rest on the sign of a near-zero FDE difference | B1: material under numeric settings | (delete) or "not distinguishable from numerical implementation variation" |
| R26 | "OpenVLA path verified" / OpenVLA as core evidence | B3 not run | K10' (descriptive) |
| R27 | "π₀-FAST vs π₀ shows an architecture effect"; π₀-FAST cross-step context effect "0" | not run; NA by design | (delete) |

### C. Limitation sentences added this round
1. "Interval estimates for paired success rates were corrected post hoc (Newcombe / Rao–Scott Newcombe); the original sign-flip-inversion intervals are kept in the supplement; difference-from-zero tests are unchanged."
2. "The correction-window experiments retain the corrected ground-truth positions in all later contexts; no experiment removes them."
3. "The matched-prefix factorial uses one retained edit at a single anchor on existing drives; it is an oracle-conditioned diagnostic and does not test release or repeated schedules."
4. "Numerical implementation choices (batch shape, attention backend, precision) change 9–26% of free-run suffixes without changing the main contrasts."
5. "The Impromptu VLA and OpenVLA intervention paths were not parity-verified; their results are descriptive."


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
