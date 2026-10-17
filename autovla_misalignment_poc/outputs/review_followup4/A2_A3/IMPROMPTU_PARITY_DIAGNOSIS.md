# A3. IMPROMPTU_PARITY_DIAGNOSIS: why Impromptu "no-op" comparisons rarely match exactly

Round: review_followup4, A3. Written 2026-10-17.

- Model: Impromptu VLA 3B (`/root/VLA/impromptu/3B_AD`, bf16, sdpa). Data: exp 33 (`outputs/cross_vla_temporal_replication/`).
- Compute: CPU analysis of the stored raw data, plus two small RTX 5090 runs (§4), 4 + 1 scenes, about 3 GPU-minutes in total.
- No existing file was modified. Table: `IMPROMPTU_PARITY_DIAGNOSIS.csv`, one row per comparison.

## 0. Verdict

**Cause confirmed: numerical path and batch sensitivity, amplified by greedy digit decoding and self-conditioning.** No semantic bug was found.

1. **Same path and same batch reproduce bit for bit.**
   - Natural decode B = 1, twice: 4/4 identical.
   - Rerun vs the stored `natural`: 4/4.
   - Exact exp_ab 18-row batch vs stored `natural_rep` and `pert_m0`: 4/4 each.
   - Unmodified `exp_cde.py --limit 1` rerun vs the stored record: **108/108 rows identical**.
   - The stored records are therefore deterministic and reproducible.
2. **Changing only the computation shape changes the output, with identical tokens.**
   - Each of these alone diverges from B = 1 in 4/4 scenes: 18 identical rows with no padding (batch size only); left padding at B = 2; re-encoding w1 as a prefill instead of an incremental step (B = 1); the FastWaypointDecoder (B = 1).
   - Teacher-forced, single step, exact natural prefix, B = 1: the HF re-encoded path reproduces the natural next waypoint in only 23/36 steps, the fast decoder in 21/36.
   - The mismatches are 0.01 m (last digit) in 11/13 and 12/15 cases.
3. **Ruled out:**
   - re-tokenization: generated ids = `tokenizer(prefix_text(parsed waypoints))` in 36/36 prefixes, and the rebuilt text equals the generated text;
   - 0.01 m formatting (python `round`, no trailing zeros, no −0.0);
   - inputs and coordinate transform (cross-process bitwise reproduction);
   - RNG (greedy, no RNG consumed on these rows);
   - source branch (not involved in `normal` / `natural_rep`).
4. **Divergence and amplification.**
   - The first divergence is usually a 0.01 m last-digit flip: 61% (S1), 64% (S4).
   - It is often at the first free waypoint (S1: 692 of 2,491 divergent scenes at w1).
   - The model then conditions on the flipped digit and amplifies it, like any other context perturbation. Median final displacement is 0.42–0.55 m; A−/amplification flips in 5–6% of scenes.
5. **The documented "no-op" pairs are not same-condition repeats.**
   - 23/300 compares two different decoding implementations and batches.
   - 237/2,728 compares the same input and decoding rule at different batch composition and padding.
   - The exp 33 RESULTS correctly calls natural_rep "batch composition only". `INTERVENTION_SPEC.md` M-I1 labels both pairs "no-op parity", which overstates what they test.
   - Exp 33 PROTOCOL §6 says "same path for all rows, so comparisons are like for like". That is also too strong: within one exp_cde step, rows with byte-identical context still disagree (S5).

**Recommendation:**
- Keep Impromptu as **descriptive supporting evidence with a measured numerical floor**. Do not count it as a strong or verified independent replication.
- The cause is now identified. The path, however, has not passed a per-path no-op / same-token gate like the AutoVLA H7–H9 gates.
- Large contrasts far above the floor can be described with the floor reported next to them: correction −63 pp, reverse vs GT-history +82 pp amplification, floor 5–6% scene-level flips.
- Contrasts near the floor should not be called replicated. This does not justify new large Impromptu runs.

## 1. What each documented comparison holds fixed

| id | comparison (doc number) | input | prefix tokens | decoding rule | batch | forward path | exact match |
|---|---|---|---|---|---|---|---|
| S1 | exp_ab `natural` vs `natural_rep` (237/2,728) | same | same (empty) | same (greedy, rep 1.05, HF generate) | **B = 1 vs B = 18, left-padded** | same | **237/2,728 (8.7%)**, reproduced exactly with the `analyze_ab` filter (all rows ok) |
| S2 | exp_ab `natural` vs `pert_m0` | same | w1 identical, but re-encoded as text prefix | same | B = 1 vs 18 | **prefill vs incremental** | 274/2,728 |
| S3 | exp_ab `natural_rep` vs `pert_m0` | same | different (empty vs w1 text) | same | same batch, different rows | different | 231/2,728 |
| S4 | exp_cde delta-0 unit `normal` vs `natural` (23/300) | same | w1 identical; later waypoints re-tokenized from text every step | greedy + rep 1.05, **re-implemented** (`FastWaypointDecoder`) | **B = 1 vs suffix-length groups, chunks ≤ 32** | **prompt KV + suffix prefill + incremental vs one HF generate** | **23/300 (7.7%)**, reproduced |
| S5 | exp_cde, same run: the 12 rows of a unit at k = 2 (context byte-identical: [w1]) | same | same | same | rows sit in different chunk positions or chunks | same | 2,504/2,700 units have all 12 rows equal; 784/32,400 rows differ from `normal` |

S1–S3 detail:

| | first divergent waypoint (1-based), scene counts | size of first divergence = 0.01 m | median final shift | A− flips | amplification flips |
|---|---|---|---|---|---|
| S1 | {1: 692, 2: 744, 3: 489, 4: 235, 5: 127, 6: 83, 7: 54, 8: 30, 9: 22, 10: 15} | 60.6% | 0.52 m | 5.9% | 5.9% |
| S2 | {2: 1,083, 3: 621, 4: 332, …} (w1 forced, so never 1) | 63.0% | 0.42 m | 5.2% | 5.2% |
| S4 | {2: 110, 3: 87, 4: 36, 5: 13, 6: 14, 7: 2, 8: 3, 9: 6, 10: 6} | 63.5% | 0.43 m | 5.3% | 5.7% |

Neither S1 nor S4 is a no-op. S4 is a test of implementation parity (fast decoder vs HF generate). S1 is a test of batch-composition sensitivity. The only true same-condition no-op found in the stored data is S5. Even there, 7.3% of units show a disagreement at the very first free step, caused only by the row's position or chunk inside the step.

## 2. Code facts (`scripts/cross_vla_temporal/`)

- **exp_ab** (`exp_ab.py:52-62`): `natural` = `continue_batch(q, imgs, [""])`, B = 1. `natural_rep` = the same empty prefix as row 0 of an 18-row `continue_batch` (chunk 24). `padding_side = "left"`, so it is left-padded to the longest perturbed prefix. Same `generate(do_sample=False, repetition_penalty=1.05, max_new_tokens=200)`.
- **exp_cde** (`exp_cde.py:105-128`; `impromptu_core.py:146-224`):
  - `natural` = HF generate, B = 1.
  - Every row, including the delta-0 `normal` row, is decoded waypoint by waypoint by `FastWaypointDecoder`: the scene prompt is prefilled once into a KV cache; at each k each row's whole assistant suffix is re-tokenized from text (`prefix_text` → `fmt_wp`, python `round(.,2)`) and fed as one forward; then greedy decoding with a re-implemented repetition penalty runs until a token containing "]".
  - Rows are grouped by suffix token length and processed in chunks of ≤ 32. Batch membership therefore depends on the row's text length, which differs between GT rows and own rows.
- **Decoding:** greedy (`generation_config` top_k = 1 is ignored under `do_sample=False`). No random numbers are consumed for `natural`, `natural_rep`, `pert_m0` or `normal`. The substitution rows use `rng_for(token, dir, row, k)`, which does not touch other rows.
- **Rounding:** the context copy of a forced w1 is rounded to 0.01 m while the executed w1 is not (≤ 0.005 m). For the delta-0 unit, w1 = natural w1, which is already a 2-decimal value, so there is no rounding difference.

## 3. Same-condition runs on the RTX 5090 (`a3_parity_run.py`, 4 scenes)

Scenes 00137a5ce92d591c, 003a1fdca9505c09, 008844070af456e3, 00abc794f8ce5195 (the first 4 exp_cde scenes; all are also in exp_ab). Peak memory 17.7 GB. The free memory on the 5090 was checked before the run (32 GB free).

| id | condition (only this changes) | exact match | first divergent waypoint per scene |
|---|---|---|---|
| R1 | natural B = 1, repeated in one process | 4/4 | – |
| R2 | natural B = 1 vs stored natural (other process, 13 days earlier) | 4/4 | – |
| R3 / R4 | exact exp_ab 18-row batch vs stored natural_rep / pert_m0 | 4/4 / 4/4 | – |
| R5 | natural_rep (18-row batch) vs natural B = 1 (S1 reproduced) | 0/4 | 1, 3, 4, 8 |
| R6 | 18 identical empty rows (**batch size only**, no padding) vs B = 1 | 0/4 (the 18 rows agree with each other) | 1, 3, 4, 3 |
| R7 | B = 2, empty row **left-padded** vs B = 1 | 0/4 | 1, 3, 1, 8 |
| R8 | w1 **re-encoded** as prefix, B = 1 vs natural | 0/4 | 2, 4, 4, 8 |
| R9 | FastWaypointDecoder delta-0 `normal`, B = 1 vs natural | 0/4 | 2, 3, 4, 3 |
| R10 | FastWaypointDecoder B = 1 vs stored exp_cde `normal` (batched) | 0/4 | 2, 3, 4, 3 |
| T1 | teacher-forced, exact natural prefix, HF one-waypoint re-encoded path, B = 1 | 23/36 steps | mismatch size: 0.01 m × 11, 0.02 m × 2 |
| T2 | same, FastWaypointDecoder | 21/36 | 0.01 m × 12, 0.02 m × 2, 0.04 m × 1 |
| T3 | fast vs HF path, same prefix | 26/36 | |
| K1 | re-tokenization: generated ids = tokenizer(rebuilt text) and text identical, k = 1..9 | 36/36 | the model emits `Ġ[`, digits, `,`, `Ġ-`/`Ġ`, `],` exactly as `prefix_text` re-encodes them |
| R11 | **unmodified** `exp_cde.py --limit 1` rerun (00137a5ce92d591c) vs stored | **108/108 rows** | – |

Interpretation:
- The model is deterministic for a fixed computation graph.
- Any change of batch size, padding, or prefill/incremental split changes bf16 logits slightly.
- Greedy argmax over digit tokens with near ties then flips the last digit of a waypoint in 36–42% of single steps (T1 13/36, T2 15/36).
- Because each waypoint conditions the next, a 0.01 m flip propagates: this is the very context-feedback phenomenon under study.

This was confirmed on 4 scenes and 36 single steps. Rates outside these scenes come from the stored-data rows S1–S5, not from the reruns.

## 4. Consequences for the Impromptu claims

- **Noise floor.** Scene-level A−/amplification flips from batch or path alone are 5–6% (S1, S4). The A− rate difference natural_rep − natural is +0.18 pp (McNemar 83 vs 78, p = 0.75; exp 33 RESULTS §4). The noise is non-directional at the population level: it inflates per-unit variance and does not obviously bias mean contrasts.
- **Untested.** exp_cde groups rows by suffix length, so batch composition is correlated with the condition. No same-token reinsertion or self-patch gate exists for this path, so a condition-correlated numerical component of row contrasts cannot be excluded.
- **Large contrasts are robust to a 5–6% floor:**
  - gt_history − normal amplification −63.4 pp [−66.7, −60.6];
  - reverse − gt_history +81.9 pp [+80.4, +84.2];
  - recent_gt − normal −62.2 pp.
- **Smaller contrasts are not called replicated:** win1 −15.4 pp; dir_wrong_mag_ok − dir_ok_mag_wrong +15.6 pp [+9.0, +24.2]; random_mag_matched − dir_ok_mag_wrong +17.3 pp. Their CIs exclude 0, but the floor and the untested condition-correlated batching make them descriptive only.
- **Impromptu windows are persistent GT** (`exp_cde.py:52`, `v = g if j <= 1 + w`). Absolute GT waypoints in the context also leak trajectory information. The A2 wording rules apply: no "release" language.
- **Wording:**
  - Allowed: "In Impromptu VLA 3B (open loop, descriptive; implementation parity not established), ground-truth context reduced and a reversed last context increased trajectory amplification by margins well above the 5–6% scene-level flip rate caused by batch and decoding-path numerics alone."
  - Delete: "Impromptu VLA strongly replicates the token-level temporal pattern" and "the three models replicate strongly", as well as any claim that the exp 33 comparisons are like-for-like no-ops.

## 5. Corrections to existing documents (for the document owner; not edited here)

| document | text | correction |
|---|---|---|
| `review_followup3/P1/INTERVENTION_SPEC.md` l. 83 and §3 "Parity data that already exists (M-I1)" | "The no-op parity actually observed … 23/300 … 237/2,728" | Neither pair is a same-condition no-op. 23/300 = fast decoder vs HF generate (different implementation and batch). 237/2,728 = batch composition and padding. Same path and batch reproduce 108/108 and 4/4 bitwise (review_followup4 A3). |
| exp 33 `PROTOCOL.md` §6 | "only last-digit (0.01–0.05 m) bf16 differences … all rows and the δ = 0 unit use the same path, so comparisons are like for like" | Single-step differences are last-digit, but they cascade (median final shift 0.42 m). Rows in one step use different batch groupings, and identical-context rows disagree in 7.3% of units (S5). |
| exp 33 `RESULTS.md` final verdict "Strong replication"; `EXPERIMENT_SUMMARY.md` §Impromptu | "Strong replication" | "Descriptive, parity not established; large correction and reverse effects exceed the measured numerical floor." |

## 6. Files

- `scripts/review_followup4/a3_parity_run.py`: GPU, 4 scenes. Calls the unmodified `impromptu_core`; writes `a3_parity_run/parity_records.jsonl` and `run_meta.json`.
- Unmodified `scripts/cross_vla_temporal/exp_cde.py <out> --limit 1` → `a3_exp_cde_rerun_limit1/`.
- `scripts/review_followup4/a3_parity_diagnosis.py`: CPU; writes `IMPROMPTU_PARITY_DIAGNOSIS.csv` and `a3_parity_summary.json`.
- Commands:
  - `CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 nice /root/miniforge3/envs/autovla/bin/python scripts/review_followup4/a3_parity_run.py outputs/review_followup4/A2_A3/a3_parity_run --tokens 00137a5ce92d591c,003a1fdca9505c09,008844070af456e3,00abc794f8ce5195`
  - `(cd scripts/cross_vla_temporal && CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 nice /root/miniforge3/envs/autovla/bin/python exp_cde.py ../../outputs/review_followup4/A2_A3/a3_exp_cde_rerun_limit1 --limit 1)`
  - `nice /root/miniforge3/envs/autovla/bin/python scripts/review_followup4/a3_parity_diagnosis.py`
