# A2. WINDOW_SEMANTICS_AUDIT: what the AutoVLA correction window does to the context

Round: review_followup4, A2. Written 2026-10-17.

- Compute: CPU only. No GPU run was needed for A2 (see §2: traces of the actual forwards already exist).
- No existing file was modified. Every number below was recomputed from stored raw data by the scripts listed in §8.
- Scope: AutoVLA, driving, open loop. Experiments covered: exp 9 (R9 raw recovery), exp 10 (code only, no per-unit raw), P1-B, and the review_followup3 P1 Stage-0 gate (path H9).

## 0. Verdict

**GT persists. Nothing is released.**

- Window `win_w` writes GT into the fixed absolute context positions t*+1 … t*+w.
- Those positions keep their GT tokens in the context of every later generation step, up to the last step a_9.
- After the window ends, the model stops receiving *new* GT positions: positions t*+w+1 onward hold its own tokens. It never loses the GT positions it already has.
- So "after the window the model conditions on its own tokens only" is false for exp 9, exp 10, P1-B persistent rows, and the Impromptu exp 33 windows.

Evidence (detailed in §2):

| layer | source | result |
|---|---|---|
| code | `temporal_feedback_window.py:188-197` | `use_gt = start <= o < start+length` with o = j − t*. The test depends only on the context position, never on the step k. The whole post-t* span is rebuilt and recomputed at every step (`cache.crop(P)` at :204), so no stale KV state is involved. |
| actual forwards (gate) | review_followup3 P1 `gate_stage0/run{1,2}/decodes.jsonl`, path H9 | 9 units × 4 settings × 2 runs = 792 row decodes, 4,928 step forwards. Fed span = rule in 4,928/4,928 steps. Source tags = rule in 4,928/4,928. A GT position stayed GT in the next step in **4,136/4,136** step transitions. **0** removal events. |
| actual forwards (P1-B) | `P1B_start_position/run_20261008/raw/intervention_trace.jsonl` (24 units) | Persistent rows `P_gt_*`: 576/576 GT positions stayed GT in every later step. Transient rows `T_gt_*`: 576/576 GT positions appeared only once (as the most recent position) and were then replaced by own tokens. |
| link to exp 9 data | gate H9 (T = 0.01, experiment batch B = 17) vs stored exp 9 records | Executed tokens identical in 153/153 row comparisons in each of run1 and run2. The gate traces are therefore traces of the exp 9 computation itself. |
| exp 9 stored records | `R8R9_raw_recovery/run_20261008/raw/exp9/records.jsonl` (1,408 units × 17 rows) | Stored final-step GT offsets = rule in 23,936/23,936 rows. GT flag constant over steps in 23,936/23,936. |

## 1. Per-step context, exactly as executed

Notation:
- Positions 0 … t*−1 hold the model's natural prefix. Position t* holds the forced token f.
- At step k the model reads the context span [f, c_{t*+1}, …, c_{k−1}] and produces a_k.
- The executed (output) token at every position is always the row's own sample. Only the context copy is overwritten.

### 1.1 Gate trace (actual forward), unit 434f74e0c3e25cdd|alt0, t* = 1, row win3

Each context entry below reads: context token (source tag, own executed token, GT token).

| step k | full action prefix fed (positions 0 … k−1) | context positions j = t*+1 … k−1 |
|---|---|---|
| 2 | 82 731 | (none; a_{t*+1} sees only f, so no window can change it) |
| 3 | 82 731 242 | j2: 242 (gt, own 242, gt 242) |
| 4 | 82 731 242 119 | j2: 242 (gt) · **j3: 119 (gt, own 43 → overwritten)** |
| 5 | 82 731 242 119 865 | j2 gt · j3 **119 gt** · j4: 865 (gt) |
| 6 | … 865 42 | j2–j4 **still gt** · j5: 42 (own) |
| 7 | … 42 105 | j2–j4 **still gt** · j5–j6 own |
| 8 | … 105 208 | j2–j4 **still gt** · j5–j7 own |
| 9 | … 208 1034 | j2–j4 **still gt** · j5–j8 own |

- Position j3 holds GT 119 in the contexts of k = 4 … 9.
- The model's own token 43 at j3 is executed (it appears in the trajectory) but is never read back.

### 1.2 Persistent vs transient: the same P1-B unit (013fbdcd9db35b43|original, t* = 1; actual trace)

| k | P_gt_s1_w3 (exp 9 rule) | T_gt_s1_w3 (GT only while most recent) |
|---|---|---|
| 3 | j2 768 gt | j2 768 gt |
| 4 | j2 768 gt · j3 1525 gt | j2 **1770 own** · j3 1525 gt |
| 5 | j2–j4 768/1525/950 gt | j2 1770 own · j3 **1437 own** · j4 950 gt |
| 6–9 | j2–j4 **gt retained** · j5+ own | **no GT anywhere** · all own |

Full per-step tables are in `a2_step_trace_examples.csv`. Columns: source, unit, row, step, full prefix, position, executed own token, context token, GT token, source tag, is_gt, overwritten. The file covers gate traces, P1-B traces, and two exp 9 units reconstructed from stored tokens.

### 1.3 Two different things that are easy to confuse

| | exp 9 `win_w` / exp 10 / P1-B `P_*` | P1-B `T_*` | exp 7 / P1-C Recent-GT |
|---|---|---|---|
| new GT positions added | only at steps k = t*+2 … t*+w+1 | same | at every step |
| existing GT removed from context later | **no** | **yes** (after one forward) | yes (each GT is replaced by own at the next step) |
| GT in context at steps after the window | **yes, w positions** | no | (no window; GT at the last position at every step) |

"Stopped adding new GT" (true for `win_w`) is not "removed the GT from the context" (false for `win_w`, true only for `T_*` and the sliding Recent-GT).

## 2. What "post-release" counts

- **Analyzer** (`analyze_temporal_feedback_window.py:70-79`): `rel = range(t* + start + length + 1, 10)`. For start = 1 these are steps k = t*+w+2 … 9.
  - These are the steps whose context also contains a self-generated token at a position *after* the window (t*+w+1 …).
  - The count is max(0, 8 − t* − w), which `WINDOW_WEIGHT_AUDIT.md` §2 reports as "free post-release steps" (win3: mean 3.84, distribution {0: 20, 1: 6, 2: 13, 3: 49, 4: 161, 5: 116}).
- **Exp 10** (`horizon_controlled_window.py:21-27, 197`): "release pose r_w = t*+w+1" and "N free steps after release". GT is retained the same way, through all 20 generated steps.
- **P1-B** (`p1b_start_position.py`, `release_pose = t* + s + w`; E_post in `WINDOW_WEIGHT_AUDIT.md` §4): same definition.
- **What it counts from:** the last step at which a *new* GT position enters the context (k = t*+w+1). It does **not** count time since GT left the context, because GT never leaves.
- **Correct name:** "steps after the last added GT position" (or "after the window ends"), not "after release" or "free autoregression".
- Consequences:
  - `self_sustain_gap` ("0 = as good as continued correction") compares win_w with GT-history at steps where win_w still has w GT positions in its context.
  - Exp 10's "re-divergence after release 25–29%" and "decays after release" are re-divergence **with GT retained** at t*+1 … t*+w.

## 3. Reconciling the two review_followup3 documents

| document | statement | verdict |
|---|---|---|
| `review_followup3/P1/INTERVENTION_SPEC.md` M-A2 (line 40) and §1.2 table row "Window win_w" | GT at fixed absolute positions t*+1..t*+w "**stays** in the context for every later step … Nothing is 'released'" | **Correct.** Confirmed by code, gate traces (0 removals in 4,136 transitions), P1-B traces and exp 9 records. |
| `review_followup3/P1/INTERVENTION_SPEC.md` line 84 (Impromptu) | "Windows are persistent GT (no release)" | **Correct** (`exp_cde.py:52`, `v = g if j <= 1 + w`). |
| `review_followup3/P0/WINDOW_WEIGHT_AUDIT.md` §2 (line 44) | "the last corrected context position is t* + w, and generation steps t* + w + 2 … 9 **condition on own tokens only**" | **Wrong wording; the count is right.** Steps t*+w+2 … 9 condition on f, on the **w GT tokens at t*+1 … t*+w**, and on own tokens at t*+w+1 … k−1. The step count max(0, 8 − t* − w) is correct as "steps after the last added GT position". |

Replacement text for `WINDOW_WEIGHT_AUDIT.md` §2 (to be applied by the document owner; this round does not edit it):

> Steps after the window = max(0, 8 − t* − w): the last GT position added to the context is t* + w (entering at step t* + w + 1). Generation steps t* + w + 2 … 9 read the w GT tokens, which stay at positions t* + 1 … t* + w, plus the model's own tokens at t* + w + 1 … k − 1. GT is never removed from the context, so this is not a release horizon.

## 4. The 90% number: definition to report alongside it

| item | value (exp 9 / R9 raw, A−) |
|---|---|
| units / scenes / logs / drives | 365 (scene, forced-token) units / 52 scenes / 16 logs / 13 drives (10 day–vehicles) |
| t* distribution (units) | {0: 116, 1: 161, 2: 49, 3: 13, 4: 6, 5: 12, 7: 8} |
| corrected context positions (win3) | min(3, 8 − t*): 3 for 357 units, 1 for 8 units (t* = 7); mean **2.96** |
| positions where GT actually differed from the row's own executed token | mean **1.90** per unit; distribution {0: 11, 1: 115, 2: 139, 3: 100} |
| GT retention | each corrected position o stays GT in the contexts of steps t*+o+1 … 9. P1-B, same rule, all A− units: mean 17.4 corrected (step, position) pairs per unit, 11.7 with GT ≠ own; the transient schedule has 2.96 pairs. |
| generation after t* | a_{t*+1} … a_9 = 9 − t* steps (mean **7.8**). a_{t*+1} cannot be changed by any window, so 8 − t* steps can be changed (mean **6.8**). |
| steps after the last added GT position (win3) | 8 − t* − 3, floored at 0: mean **3.84**. 20 units have 0 (win3 ≡ GT-history when t* ≥ 5). |
| outcome | amplification = A− (P/R/A accept rule, 5 s) and FDE at pose 9 (5 s) > 3 m; binary per unit |
| estimand (primary, review_followup3) | scene-weighted ratio of sums f = Σ_s (N_s − W3_s) / Σ_s (N_s − F_s). Units are averaged within a scene first (weight 1/n_s); the 52 scenes count equally. |
| numerator / denominator | 37.5 pp / **41.7 pp** (scene-weighted levels: Normal 45.1%, win3 7.6%, Full GT-history 3.4%; same run, same units, same batch B = 17) |
| value | **90.1%** (95% CI 81.2–98.9, drive-cluster bootstrap with joint numerator/denominator resampling, 13 drives). Recomputed here: 90.05%. |
| alternative estimand | unit-weighted 89.7% (counts 170 / 31 / 15 of 365 amplified; 38.1 / 42.5 pp). This is the original exp 9 value; it is a different estimand, not a discrepancy. |
| continuous metric | FDE5 fraction, unit-weighted: 92.5% (6.45 → 2.15 m vs Full 1.80 m) |
| replication inside P1-B (B = 53) | P_gt_s1_w3: 89.8% (all A− units) / 89.0% [79.5, 98.0] (t* ≤ 2: 326 units, 46 scenes, 12 drives) |

## 5. Existing evidence on removing GT (descriptive, post-hoc)

P1-B already ran a schedule with GT removal: `T_gt_s1_w3`. The GT token at t*+1 … t*+3 is shown once, as the most recent position, and then replaced by the model's own token, so no GT is in the context from step t*+5 on.

It is **not** "the persistent 3-step window, then released":
- GT is never present at two positions at once.
- Exposure is 3 (step, position) pairs instead of about 18.

Same run, same units and seeds as `P_gt_s1_w3`, scene-weighted, drive-cluster bootstrap, 10,000 replicates (`a2_p1b_transient_vs_persistent.json`):

| A− subset | persistent P_gt_s1_w3 | transient T_gt_s1_w3 | T − P amplification | T − P FDE5 |
|---|---|---|---|---|
| t* ≤ 2 (326 units, 46 scenes, 12 drives; predeclared P1-B support) | 89.0% [79.5, 98.0] | 78.3% [64.0, 86.6] | +4.8 pp [+2.9, +7.8]; discordant units 22 vs 5 | +0.78 m [+0.44, +1.08] |
| all (365 / 52 / 13) | 89.8% [81.2, 98.3] | 79.1% [65.8, 86.9] | +4.5 pp [+2.9, +7.2] | +0.71 m [+0.42, +0.98] |

The P1-B report gives the transient fraction as 78% (`analysis.md` §3.3). The T − P contrast and its CI are new **post-hoc** numbers. They show that retention matters: three GT tokens that are later removed recover less than three GT tokens that are retained. The comparison mixes retention with exposure count, so it is not a pure release effect.

No experiment exists that keeps the exp 9 window for three steps and then deletes those three GT positions. This round does not create one.

## 6. Allowed sentence and sentences to delete

**Allowed (main text):**

> Three fixed context positions replaced with ground truth and retained in later contexts recovered about 90% of the full-history correction effect.

**Required qualifiers** (caption or next sentence):
- scene-weighted 90.1%, 95% CI 81–99, 13 drives
- 365 A− units / 52 scenes
- denominator = Normal − GT-history in the same run (41.7 pp of amplification, FDE5 > 3 m at 5 s)
- 2.96 corrected positions per unit on average, of which 1.90 differed from the model's own token
- of 7.8 generated steps, the correction was retained through the last; the window length and the number of steps after the last added GT position (mean 3.8 for three positions) change together.

**Optional, if the transient row is used:**

> When each of the three ground-truth tokens was shown only once, as the most recent context position, and then replaced by the model's own token, the recovery was 78% (95% CI 64–87; post-hoc comparison, 326 units, 12 drives).

**Delete or replace:**

| location | sentence | action |
|---|---|---|
| `review_followup3/P0/WINDOW_WEIGHT_AUDIT.md` §2 | "generation steps t* + w + 2 … 9 condition on own tokens only" | replace (§3 text) |
| `paper_quantitative_package/PAPER_NUMBERS.md` Claim 5 "Manuscript-safe"; `MANUSCRIPT_REVISION.md` item 3 and K5; `review_followup3/CLAIM_EVIDENCE_MATRIX.md` D6; `EXPERIMENT_SUMMARY.md` l. 28, 127 | "Correcting three consecutive context steps recovered about 90% …" | replace with the allowed sentence. "Correcting three steps" reads as a transient correction. |
| `PAPER_NUMBERS.md` l. 116, 118; `MANUSCRIPT_REVISION.md` l. 33, K5; `EXPERIMENT_SUMMARY.md` l. 435-436; `WINDOW_WEIGHT_AUDIT.md` §2/§4 | "post-release (free) horizon", "free post-release steps", "after release", "고정 해제 후 horizon" | rename to "steps after the last added GT position" (GT retained) |
| `EXPERIMENT_SUMMARY.md` l. 78, 128, 132, 546; `CONCLUSIONS.md` l. 213-214, 374; `PAPER_NUMBERS.md` l. 119 | "교정을 멈추면 서서히 재발산", "해제 후 감쇠", "해제 후 재발산 25–29%", "re-divergence after release" | delete, or reword as "re-divergence while the w corrected GT positions remain in the context (exp 10, summary only)" |
| `CONCLUSIONS.md` l. 217 (already withdrawn) | "…it decays after release, so the window delays rather than permanently prevents re-divergence" | keep withdrawn; also wrong on semantics |
| `EXPERIMENT_SUMMARY.md` l. 248 (Impromptu) | "교정 해제 후 재발산은 … 거의 없었습니다" | delete; Impromptu windows are also persistent (`exp_cde.py:52`) |
| exp 9 report `TEMPORAL_FEEDBACK_WINDOW.md` §6 heading and caption; fig3 title | "Window 해제 후 자기 유지 여부", "0이면 교정을 멈춰도 계속 교정한 것과 같음", "self-sustain gap after the window is released" | do not cite. The gap is measured while GT is retained. |
| code docstrings `temporal_feedback_window.py:22-23`, `horizon_controlled_window.py:21-24` | "After the window the model conditions on its own outputs again (free autoregression)" | historical code, not edited. Cite this audit instead. |

Any claim of the following kinds is deleted:
- after a 3-step correction the GT information is fully removed and the effect is retained;
- later generation reads only the model's own tokens;
- a short transient correction alone demonstrated a lasting recovery.

## 7. Limits

- Gate traces cover 9 exp 9 units; P1-B traces cover the first 24 units of the main run. The remaining units rest on the code rule and on the stored final-step offsets, which match the rule in 23,936/23,936 rows.
- Exp 10 has no per-unit raw. Its semantics come from code (`horizon_controlled_window.py:197`, same rule).
- The transient-vs-persistent contrast is post-hoc and confounds retention with exposure.

## 8. Files and commands (all CPU, `nice`)

- `scripts/review_followup4/a2_window_semantics.py`. Writes `a2_checks.json`, `a2_step_trace_examples.csv`, `a2_exp9_unit_window.csv` (per A− unit: t*, corrected positions, effective overwrites, steps after the window, for win1–4).
- `scripts/review_followup4/a2_p1b_transient_vs_persistent.py` (conda env `autovla`, pandas). Writes `a2_p1b_transient_vs_persistent.json`.
- Run: `nice python3 scripts/review_followup4/a2_window_semantics.py`, then `nice /root/miniforge3/envs/autovla/bin/python scripts/review_followup4/a2_p1b_transient_vs_persistent.py`.
