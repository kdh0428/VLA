# CLAIM_CHANGES — review_followup4 (final, 2026-10-18)

**What this file does.** It states, sentence by sentence, which manuscript claims are **kept**, **narrowed** or **deleted** after review_followup4 (AUTOVLA_NEXT_STEPS_20261018 §5). It supersedes conflicting wording in `/root/VLA/review_followup3/CLAIM_EVIDENCE_MATRIX.md` (2026-10-15) and in the earlier sections of `paper_quantitative_package/MANUSCRIPT_REVISION.md`. Those files are kept as history; review_followup3 reports were not edited.

**Evidence used (all committed):** A1/A4 `A1_A4/` (2a31ad7), A2/A3 `A2_A3/` (5523ce3), B1 `B/B1/` and the B2 protocol (552b3ac), FEASIBILITY (28f1bfd), B2 results `B/MATCHED_PREFIX_RESULTS.*` (c7d1bba), B3 `OPENVLA_GATE.md` (NOT_RUN, this round). Row ids refer to `paper_quantitative_package/numbers.json` (`RF4.*` rows were added by `scripts/review_followup4/c/c_numbers.py`).

**Tags.** Every claim carries domain / model / evaluation mode:

| tag | meaning |
|---|---|
| **D-AV-OL** | driving / AutoVLA (Qwen2.5-VL-3B, 2,048 action tokens) / open-loop 5 s plan, NAVSIM navtest |
| **D-IM-OL** | driving / Impromptu VLA 3B / open-loop 5 s waypoints |
| **M-SV-CL** | manipulation / SpatialVLA-4B / closed loop, SimplerEnv Google Robot (pick_coke_can, move_near) |
| **M-OV-CL / M-OV-OFF** | manipulation / OpenVLA-7B / closed loop LIBERO-Spatial, or offline within-step decode |

No sentence may speak about "VLAs in general", "autoregressive VLAs" or a universal cause of VLA errors.

**Verdict words.** keep = use as written; narrow = use only the narrowed sentence given; delete = do not use; downgrade = keep only as descriptive / unverified.

**Machine status vs statistical validity.** A `VERIFIED_*` status certifies an intervention code path on its manifest or run. It does not certify the estimand or the interval (A1). Non-significance is never equivalence.

---

## 1. Section 5 central claims

### Claim 1 — In AutoVLA's defined failure stratum, action-context interventions strongly change later generation and trajectories. **KEEP, narrowed** (D-AV-OL)

**1a. Repeated sliding correction and Normal-branch injection** (unchanged numbers; wording narrowed by A4).

> "In AutoVLA (open-loop 5 s plans on NAVSIM navtest), repeatedly replacing the most recent action-context token with ground truth at every generation step reduced amplification in the failure stratum by 39.8 pp [29.9, 49.0] (52 scenes, 13 drives), and injecting the separate Normal branch's previous token into a ground-truth context restored it by 28.6 pp [20.9, 36.1]. In a preregistered test on new drives (failure stratum: 105 scenes, 18 drives; no drive shared with development, days and vehicles partly shared) the effects were −34.2 pp [−38.3, −29.5] and +26.8 pp [+21.9, +30.7]."

Rows: `P0A.DEV.RN.Am`, `P0A.DEV.RV.Am`, `P0A.P1C.RN.F`, `P0A.P1C.RV.F`. Limits that stay: Recent-GT is applied at every step (8 − t* applications); Reverse-N is "Normal-branch token injection", never "the model's own error re-inserted"; "new drives", never "external" or "24 independent failure drives".

**1b. New (B2): the latest context value matters; no dependence on an older GT prefix in a one-shot schedule.** Oracle-conditioned diagnostic.

> "In a preregistered matched-prefix diagnostic with a single retained edit, placing the frozen Normal-branch token instead of ground truth in the most recent context position raised severe outcomes in the P1-C failure stratum by 25.3 pp [20.5, 29.3] behind a ground-truth older prefix and by 25.5 pp [21.6, 29.0] behind the model's own natural older prefix (100 scenes, 711 unit-anchors, 18 drives; Holm-adjusted p ≤ 4.6 × 10⁻⁵). The interaction was −0.2 pp [−3.8, +3.1] (−1.7 pp [−5.4, +1.9] where the older values truly differ), so in this one-shot schedule the effect tracks the value of the most recent context token rather than the presence of an older ground-truth prefix."

Required qualifiers (caption or next sentence): ground-truth cells make this an **oracle-conditioned diagnostic**, not an online or deployable correction; the edited prefix is **kept** at fixed positions (not a release test); existing P1-C drives, **not external validation**; the result is not pooled with the repeated Recent-GT / Reverse-N runs; the corrected-source arm (C_PROV −15.8 pp [−20.1, −10.2]) is a **value** difference, not a pure provenance effect. Rows: `RF4.B2.p1c.a1.severe.*`.

### Claim 2 — Part of the effect is visible in the evaluation-set average, with a different size, denominator and uncertainty. **KEEP, narrowed** (D-AV-OL)

> "On the evaluation frame of the new-drive logs (2,191 eligible scenes, estimated from 353 sampled scenes with design weights; 24 drives), the corresponding effects were −4.1 pp [−6.4, −2.4] and +3.6 pp [+2.2, +5.5]; about 60% of the Recent-GT effect came from non-failure (A+) scenes, so this average is not a recovery of originally failing scenes."

> (B2) "In the B2 eligible domain (t* ≤ 4; 312 sampled scenes, 24 drives; weights recomputed for this domain), the latest-token effect was small: +4.0 pp [+2.7, +5.7] behind the natural prefix (Holm-adjusted p = 6.5 × 10⁻⁵) and +3.3 pp [+1.0, +5.6] behind a ground-truth prefix (Holm-adjusted p = 0.051, not significant at the family α of 0.025)."

> "The failure-category (A−) frequency in the development evaluation set was 52/2,747 = 1.89% [1.16, 2.64], an evaluation-set frequency, not a deployment failure rate."

Rows: `P0A.P1C.RN.P`, `P0A.P1C.RV.P`, `P0A.DECOMP.*`, `RF4.B2.p1c.a1.severe.C_*.all.P.drive`, `P0A.PREV.DEV.Aminus.2747.drive`. Never call these "population" or "deployment" effects; never multiply a frequency by a stratum effect; the B2 domain E4 is not the original 2,191-scene estimand.

### Claim 3 — In the failure stratum, the endpoint response exceeds the kinematic and learned extrapolation controls evaluated here. **KEEP, narrowed** (D-AV-OL)

> "In failure-selected scenes, AutoVLA's endpoint response to a ground-truth-free context perturbation exceeded four kinematic extrapolations — including the original constant-motion baseline, which repeats the perturbed token's body-frame motion including its yaw — and a small learned non-autoregressive motion extrapolator (+2.35 m [1.27, 3.19] over constant motion; +1.97 m [1.06, 2.87] over the learned extrapolator); in non-failure scenes and across the 208 selected mechanism scenes no excess over motion extrapolation was detected (equivalence not tested)."

> "Without ground-truth information, perturbing a single context token moved the 5 s endpoint 4.37 m [3.34, 5.17] beyond pure coordinate accumulation (13 drives; Holm-adjusted p = 7.3 × 10⁻⁴)."

**B1 limit (new):** "Numerical sensitivity checks (batch shape, attention backend, fp32 arithmetic) left the Recent-GT, Full-GT and Reverse-N contrasts unchanged in direction and nearly unchanged in size (severe within 0.036, FDE within 0.16 m against effects of 1.5–1.9 m on 56 development scenes), but near-zero FDE differences between ground-truth-free self-reference rows could not be distinguished from implementation numerics." Rows: `RF4.B1.stab.*`. Any statement resting on the sign or size of a P1-A self-reference FDE difference near zero (OC − U, OC_refR − OC, OC_refH − OC on these units) is **deleted**. The +4.37 m contrast and the A− excess are far above the measured floor and stay.

The learned extrapolator is a motion model, not a non-autoregressive version of the VLA. "all" always means the 208 selected scenes (52 A− + 156 A+).

### Claim 4 — In manipulation models, generation change, actual trajectory change and task success must be judged separately. **KEEP** (M-SV-CL; OpenVLA descriptive)

> "In SpatialVLA closed-loop manipulation (SimplerEnv Google Robot; two tasks), context perturbations changed generated action tokens (50–58% of suffix tokens) and the actual end-effector trajectory (2.7–6.0 cm mean distance), while their effect on task success, averaged over the two tasks, was bounded within ±10 pp in the tested configurations (post-hoc corrected intervals, initial-state clusters, Holm); per task, ±10 pp equivalence held for 3 of 8 contrasts and ±5 pp for none, with task-level 95% CIs up to 18–25 pp wide, so task-specific effects remain uncertain."

> "In OpenVLA-7B on LIBERO-Spatial (100 initial states; intervention path not parity-verified; descriptive), no task-success effect was detected (correction minus feedback −3 pp [−11.9, +5.9]; equivalence not established), while re-running the unperturbed policy alone changed 20% of outcomes."

Commanded action deltas (SpatialVLA exp 34/35 metre-scale numbers; OpenVLA bins) are never written as realised TCP motion.

---

## 2. Items lowered from the central contribution (§5 list)

| # | item | verdict | exact replacement / rule |
|---|---|---|---|
| L1 | persistent GT window "90%" as a short transient correction with a lasting effect | **narrow** | "Three fixed context positions replaced with ground truth and retained in later contexts recovered about 90% of the full-history correction effect." Qualifiers: scene-weighted 90.1% [81.2, 98.9], 365 A− units / 52 scenes / 13 drives; denominator = Normal − GT-history in the same run (41.7 pp, severe at 5 s); 2.96 GT positions per unit, 1.90 differing from the own token; the number of steps after the last added GT position (mean 3.8 for w = 3) changes with w. Optional: transient once-shown GT 78% [64, 87] (post hoc, 326 units, 12 drives; mixes retention with exposure). |
| L2 | Impromptu and OpenVLA as strong independent replications | **downgrade** | Impromptu: "In Impromptu VLA 3B (open loop, descriptive; implementation parity not established), ground-truth context reduced and a reversed last context increased trajectory amplification by margins well above the 5–6% scene-level flip rate caused by batch and decoding-path numerics alone." OpenVLA: descriptive, excluded from core evidence (B3 NOT_RUN). |
| L3 | two-task mean equivalence generalised to "no effect per task" | **delete** the generalisation | Claim 4 sentence; per-task CIs reported in full (Table 13). |
| L4 | structure ranking from raw effects of different treatments across models | **delete** | "Interventions were matched in spirit rather than identical across models; effect sizes are not comparable across models." |
| L5 | a universal cause of VLA errors | **delete** | Scope: AutoVLA (driving, open loop) main; manipulation complementary. |
| L6 | π₀ vs π₀-FAST difference as a pure architecture effect | **delete** (not run) | FEASIBILITY.md: π₀-FAST feasible with conditions, not run; it has no action history in context, so cross-step action-context interventions are NA, never "measured 0". |

---

## 3. Changes driven by each work item

### B2 (new effect experiment; new round, not a retroactive P3)
- **Keep (new):** Claim 1b and the B2 sentence of Claim 2.
- **Narrow:** "The Reverse-type effect requires a GT history" is **deleted**; INT ≈ 0 says it does not, for this one-shot schedule. The P-level INT at the sensitivity anchor (+2.7 pp [+1.1, +4.6], p = 0.0046) is a family-free sensitivity result and may only be reported as such.
- **Delete:** "pure provenance effect", "online correction", "external replication", "release/removal of the correction was tested", any pooling with repeated Recent-GT / Reverse-N results.
- CP2's P3/P4 NOT_RUN stays as the historical decision; B2 is reported as the review_followup4 matched-prefix round (protocol 552b3ac, results c7d1bba).

### B1 (numerical sensitivity)
- **Keep:** RN / Full-GT / RV directions and magnitudes, reported with the floor: batch shape alone changes 8.9% of free-run suffixes, attention backend / fp32 arithmetic ≈ 25%; severe flips ≤ 1.2% of row-runs.
- **Narrow:** P1-A self-reference near-zero FDE contrasts → "not distinguishable from numerical implementation variation".
- **Keep the existing caveat** that row-level outcomes compared across batch compositions differ in ≈ 9% of suffixes (M-A3).

### A1 (paired success-rate intervals; post-hoc correction)
| claim | old | new |
|---|---|---|
| OpenVLA C − N 8-bin "95% CI excludes 0" | [−16.7, −7e−14] | **delete**: tie artefact; corrected [−17.1, +1.0], consistent with p = 0.115 |
| OpenVLA R − N_rep 8-bin ±10 pp equivalence (unadjusted) | p_eq 0.049 | **delete**: 0.052 (never Holm-significant) |
| OpenVLA per-task CIs (`P0E.succ_task.*`, incl. zero-width rows) | sign-flip inversion | **delete as evidence**; Newcombe-10 intervals (±0.18–0.28) descriptive only; "no difference in task 5" is not allowed |
| SpatialVLA exp34 r1e4/perp_left coke C − F (and F − C) "±10 pp equivalent (Holm)" | Holm 0.001 | **narrow** to "unadjusted only" (corrected CI [−11.3, +11.3], Holm 0.147) |
| P1-D r2e1 − r1e1 C − F interaction, pooled, physical, "±10 equivalent (Holm)" | Holm 0.045 | **narrow**: not established (0.051) |
| P1-D r1e4 C − F move_near physical "±5 pp equivalent (unadjusted)" and CI [−2.6, +4.3] | artefact | **delete**; corrected [−6.9, +8.6] |
| "task-level CIs span up to 18–29 pp (r4e1; r1e4 6.9–21.6 pp)" | old widths | **replace**: "up to 18–25 pp (r4e1; r1e4 13–20 pp)" |
| exp35 r1e4 C − F move_near primary CI | [−29.1, +0.0] | report as "not different from 0 (Holm)"; corrected CI [−29.1, −0.3] barely excludes 0 while the δ = 0 test does not reject — boundary disagreement, **not** evidence of an effect |
| P1-D pooled ±10 pp equivalence (4 contrasts, both clusterings), per-task 3 of 8, S1–S4 pooled ±10 | – | **keep** (survive; Holm p updated in Table 13 / `RF4.A1.*`) |
| difference-from-0 p values and Holm families | – | **keep** (sign-flip / McNemar at δ = 0 is valid) |
| AutoVLA P1-C rule "not replicated, below MME ⇔ 90% sign-flip-inversion CI inside ±MME" | same construction | **flag**: re-check before using any "below MME" verdict; read p (valid at 0), not the pinned CI end, in the 6 flagged AutoVLA rows |

All other judgment changes (36 rows incl. mirrors and clusterings) are in `A1_A4/A1_JUDGMENT_CHANGES.csv`; none of the borderline unadjusted flips reaches Holm significance either way.

### A2 (window semantics)
- **Delete:** "after a 3-step correction the GT information is fully removed and the effect is retained"; "later generation reads only the model's own tokens"; "a short transient correction alone demonstrated a lasting recovery"; "post-release (free) horizon", "after release", "교정을 멈추면 서서히 재발산", "해제 후 감쇠", "해제 후 재발산 25–29%" (→ "re-divergence while the w corrected GT positions remain in the context (exp 10, summary only)"); Impromptu "교정 해제 후 재발산은 거의 없었습니다".
- **Keep:** L1 allowed sentence with qualifiers. No true release result exists and none was created.

### A3 (Impromptu)
- **Downgrade** exp 33 "Strong replication" → "descriptive, parity not established; large correction and reverse effects (−63.4 pp, +81.9 pp) exceed the measured 5–6% numerical floor". Small contrasts (win1 −15.4 pp; direction − magnitude +15.6 pp; random − direction-ok +17.3 pp) are **not** called replicated.
- **Delete:** "Impromptu VLA strongly replicates the token-level temporal pattern", "the three models replicate strongly", "the exp 33 comparisons are like-for-like no-ops", "23/300 and 237/2,728 are no-op parity".

### A4 (denominators, metrics, scope)
| rule | required wording |
|---|---|
| sets | dev natural 2,747; dev mechanism 208 selected (A− share 25% vs 1.89% natural); P1-C target E 2,191 eligible; P1-C intervention sample 353 selected, weighted; P1-C failure stratum F 105 scenes / 18 drives; motion-baseline "all" = 208 selected |
| RN ≈ 60% from A+ | keep the decomposition; never explain the P effect as recovery of failing scenes |
| OpenVLA 92.6% | "at least one later component (y, z, roll, pitch, yaw; gripper excluded) changed by at least the effective injected number of bins" in 4,741 (frame, sign) units; bins mix m and rad; translation-only 72%; commanded, not realised, not task failure |
| physical units | 8 bins = x 2.64 / y 2.41 / z 2.93 mm; roll 3.31 / pitch 6.02 / yaw 5.18 mrad; gripper binary |
| commanded vs TCP | SpatialVLA metre-scale numbers are cumulative commanded actions; actual TCP is centimetre-scale (P1-D 2.67 / 5.95 cm) |
| heterogeneity | opposite per-task point estimates do not establish heterogeneity or cancellation (e.g. P1-D r4e1 C − F −4.2 / +4.2 pp, heterogeneity −8.3 pp [−21.7, +5.0]) |
| drive vs day/vehicle | P1-C is drive-disjoint, not day/vehicle-disjoint (10 day–vehicles shared; 6 components; Holm floor 0.094) |
| A− / severe | A− = NOT `pra_labels.label_A` (longitudinal and lateral class rules); severe = A− ∧ FDE5 > 3 m at pose 10; thresholds documented for a 3 s horizon are applied to the 5 s plan (original definition preserved) |
| statistics | non-significance ≠ equivalence; a verified path ≠ a validated estimand / interval |

The 64 file:line violations and their replacements are applied or listed in `MANUSCRIPT_REVISION.md` (this directory).

---

## 4. Final claim list (one line each)

| # | claim | tag | verdict |
|---|---|---|---|
| K1 | Repeated Recent-GT / Normal-branch injection effects in the failure stratum (dev 13 drives; P1-C F 18 drives) | D-AV-OL | keep (narrowed wording) |
| K2 | B2: latest-token value effect ≈ 25 pp in F, INT ≈ 0 (one-shot, retained edit, oracle-conditioned) | D-AV-OL | keep (new) |
| K3 | Evaluation-frame effects −4.1 / +3.6 pp (353 sampled, weighted; 60% RN from A+); B2 E4 +4.0 / +3.3 pp (C_GT n.s. after Holm) | D-AV-OL | keep (narrowed) |
| K4 | GT-free +4.37 m beyond coordinate accumulation; beyond motion extrapolation only in A− | D-AV-OL | keep (narrowed) |
| K5 | 90% with three retained GT positions | D-AV-OL | narrow (L1) |
| K6 | No evidence for a special early window | D-AV-OL | keep |
| K7 | Direction > magnitude: development only; inconclusive on new drives | D-AV-OL | keep (narrowed) |
| K8 | Main contrasts robust to batch / backend / precision; self-reference near-zero FDE limited | D-AV-OL | keep (B1, new) |
| K9 | SpatialVLA generation and TCP change; two-task mean within ±10 pp; per task uncertain (corrected CIs) | M-SV-CL | keep (A1 numbers) |
| K10 | Replanning interval increases persistence (composite); ensembling does not absorb at matched dose | M-SV-CL | keep |
| K11 | Impromptu correction / reverse | D-IM-OL | downgrade (descriptive, unverified) |
| K12 | OpenVLA within-step 92.6% and no detected task effect | M-OV-OFF / M-OV-CL | downgrade (descriptive, unverified, not core) |
| D1 | release / removal / transient-correction claims | – | delete |
| D2 | Reverse requires a GT history; pure provenance; online correction; external validation | – | delete |
| D3 | OpenVLA per-task conclusions; C − N "CI excludes 0"; zero-width CI as equality | – | delete |
| D4 | "VLA in general", universal error cause, cross-model structure ranking, π₀ vs π₀-FAST architecture effect | – | delete |
