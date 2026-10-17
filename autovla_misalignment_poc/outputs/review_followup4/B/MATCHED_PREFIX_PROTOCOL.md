# MATCHED_PREFIX_PROTOCOL — B2, AutoVLA matched-prefix factorial (review_followup4)

**Status: DRAFT for the lead's commit (2026-10-17).**
- Run so far:
  - the no-op / cache gate of the new arm code on the 9-scene Stage-0 manifest (§9);
  - a throughput pilot (§11) that writes no cell outcomes.
- No main B2 data exist.
- The file is frozen when the lead commits it. Any change after main data exist is an amendment that states what had been seen.

**History kept as history.** At review_followup3 CP2 (444299f) P3 was *not run*. `review_followup3/P3/MATCHED_REVERSE_*` keep their NOT_RUN status. B2 is a **new follow-up round** and reuses the P3 draft (`outputs/review_followup3/P3/MATCHED_REVERSE_PROTOCOL.md`) with the changes listed in §14.

**Question.** Does the value of the **latest context block** matter, or does the effect depend largely on the presence of an **older GT prefix**?

**Scope.**
- Driving, AutoVLA (`AutoVLA_PDMS_89.ckpt`), open-loop 5 s plan.
- Within-model identification only.
- GT-conditioned cells are **diagnostic**. They are not an online recovery method and not a deployment improvement.
- Existing drives re-run with new intervention arms. This is **not external validation**.
- The existing repeated Recent-GT / Reverse-N results (exp 7/8, P1-C H7/H8) are a different treatment schedule. They are **never pooled** with B2.

## 1. Unit, anchor, cells

**Unit.**
- Unit = (scene, perturbation), forced token f at t*, exactly as in exp 7 / P1-C: `original` = pred[t*], `alt<j>` = the distance-matched alternatives.
- Positions 0 … t*−1 are the natural tokens (= GT, since t* is the first mismatch). Position t* is f, which is executed and shared by every cell.

**Anchor (fixed rule, not chosen by outcome).**
- **Primary:** n_old = 1, so a = t*+3.
  - Older prefix = position t*+1.
  - Latest block = position t*+2 = a−1.
  - Generated suffix = positions a … 9 = 7 − t* tokens.
- **Eligibility:** t* ≤ 4, so at least **3 tokens (1.5 s) are generated after the edit** and the older-prefix factor has one position.
  - Units with t* ∈ {5, 6, 7} are excluded before any outcome is seen. They are reported in the denominators (§3).
- **Sensitivity anchor:** n_old = 2, a = t*+4 (older = t*+1, t*+2; latest = t*+3), for units with t* ≤ 3 (suffix ≥ 3 tokens).
  - Reported separately (family-free).

**Factors.**

| factor | level | value at the edited position(s) |
|---|---|---|
| O, older prefix (positions t*+1 … a−2) | GT | gt[j] (the existing GT→token projection) |
| | NAT | the **matched natural prefix**: the frozen bank Normal token c^N_j, same positions, same length |
| L, latest block (position a−1) | REF | gt[a−1] |
| | NSRC | frozen **Normal-source** token c^N_{a−1} |
| | CSRC | frozen **corrected-source** token c^F_{a−1}: the token the Full-GT-history branch (GT at every context position, every step) sampled at position a−1 |

**One-shot, single edited prefix (the only block run in B2).**
- **Executed tokens** (identical for every cell before a): pred[:t*] + [f] + c^N_{t*+1 … a−1} + own[a … 9].
  The edited context tokens are **never executed**.
- **Context at step k ≥ a:** [f] + O-values[t*+1 … a−2] + L-value[a−1] + own[a … k−1].
- **The edited prefix is KEPT** at the same absolute positions for every later step (fixed-position override, as in exp 9 windows). It is never removed.
  - B2 therefore measures "one edited prefix, retained". It does **not** measure "correction information removed afterwards". No release arm is run.
- Steps before a are not re-decoded. The pre-anchor context equals the Normal continuation by construction (bank tokens). Every cell shares the same executed tokens up to a−1.

**Cells and rows (all decoded alone, B = 1, on one shared prefix cache per unit).**

| cell | reads as |
|---|---|
| GT_REF | one-shot Full (GT prefix through a−1) |
| GT_NSRC | **matched one-shot Reverse**: GT older + frozen Normal latest |
| GT_CSRC | GT older + corrected-branch latest |
| NAT_REF | one-shot Recent-GT: natural older + GT latest |
| NAT_NSRC | identical in value to the Normal continuation (value-identity check, §8) |
| NAT_CSRC | natural older + corrected-branch latest |
| noop | no override at all (context = executed tokens). Identical values to NAT_NSRC, built by a different code route |

**Fixed across cells:**
- observation (scene video), instruction, stub, t*, f;
- non-target positions;
- the source bank;
- decoding: the experiments' T = 0.01 sampling with common random numbers (key = the unit's perturbation, re-seeded per row per step, as in H7/H8);
- batch shape (B = 1 for every row);
- horizon: 10 poses / 5 s.

Greedy decoding is not run.

**Value vs provenance.** When two cells' edited values coincide (e.g. c^F = c^N, or c^N = gt), the cells are the same input and must give identical output. Differences between NSRC and CSRC are differences of **value** that arise from different provenance. They are **never called a pure provenance effect**.

## 2. Source bank (frozen before any cell)

**Phase A**, for every eligible unit of the cohort, **before** phase B starts:
- the P1-verified H7 path (`p1b_engine.decode`, B = 1, t001) runs rows `normal` and `gt_history`.

The bank (`bank.jsonl`) stores per unit:
- the Normal and Full-GT own tokens;
- each token's decoded local motion (forward m, left m, dyaw rad, from the codebook geometry);
- the per-step span hashes;
- the prefix-ids and pixel hashes;
- the Normal and Full-GT executed plans.

**Freezing.**
- `bank_manifest.json` records the sha256 of `bank.jsonl` and the unit → scene → log → drive → day–vehicle map.
- Phase B asserts the sha before it starts.
- No source value is re-selected from any cell outcome.

**P1-C input verification.** The bank's Normal tokens are compared with the stored P1-C H7 `normal` tokens of the same unit (same path, seed and batch). This checks that the re-fetched images reproduce the original inputs.
- **Stop rule:** agreement < 95 % of units → stop and report. Do not analyse.
- The dev pilot floor for this comparison is 100 % (P1 c5d: 18/18 rows).

## 3. Cohort, estimands, denominators (fixed now; built by `b2_cohort.py` without outcomes)

| set | scenes | units | drives | role |
|---|---|---|---|---|
| P1-C eligible E (t* ≤ 7) | 2,191 (F 105 / S 2,086) | – | 24 | original P1-C target. **Not** the B2 estimand. |
| **B2 domain E4** = E with t* ≤ 4 | **1,775 (F 100 / S 1,675)** | – | 24 | B2 target (P estimand) |
| P1-C intervention sample ∩ E4 = **B2 primary cohort** | **312 (F 100 / S 212)** | **2,160 (F 711 / S 1,449)** | **24 (F 18)**; 16 day–vehicle | analysed |
| excluded by anchor rule (t* 5–7) | 41 (F 5 / S 36) | 276 | – | reported only |
| dev mechanism set, t* ≤ 4 (**secondary**) | 196 of 208 (A− 49) | 1,327 | 19 (A− 13) | development, descriptive |

**Strata and weights (domain E4).**
- **F** = natural A−, census: π = 1.
- **S** = natural A+: 4 per log were sampled by sha order in P1-C. Weights are **recomputed for the domain**: w = N′_S,log / m′_S,log, where
  - N′ = S-eligible scenes of the log with t* ≤ 4;
  - m′ = sampled S scenes of the log with t* ≤ 4.
- Coverage gap (N′ > 0, m′ = 0): 0 logs. Σw_S = 1,675 = N′_S.
- The original 2,191-scene estimand and the P1-C weights are **not reused**.
- **P** = F ∪ S, Hajek over E4 ("all-eligible, t* ≤ 4").
- The failure stratum F is reported separately.

**Denominators reported:** task → scene → unit → drive → day–vehicle, per estimand. Per-unit invalid outputs are counted per cell pair.

**Dev cohort.**
- Unweighted. F = A−, S = A+ by the dev label.
- These are development scenes, used to design the original experiments. They are **not** a population estimate.

**Feasibility facts (checked 2026-10-17).**
- P1-C camera images were deleted after P1-C (0/64 eval logs present in `sensor_blobs/test`). Scene JSONs (`navtest_ext`), stubs (`stage1/records.jsonl`) and unit records (`units/records.jsonl`) are present.
- Re-fetch: only the P1-C keep-lists (3,831 jpg ≈ 0.36 GB) are fetched, by `b2_fetch_p1c_images.sh`, streaming the public OpenScene shards (same URL as P1-C).
  - Tested on shard 26: 39/39 files, 3.5 min, ≈ 6.2 GB streamed per shard, < 5 MB written.
  - Full re-fetch: 26 shards ≈ 90 min of network, ≈ 0.36 GB of disk.
- Dev images (front, left, right cameras) are present for all 208 dev scenes. The unused back/side-front images are absent, as before.
- Disk had ≈ 3.1 GB free at writing time. Each script stops below 1 GB (1.5 GB for the fetch).

## 4. Outcomes

Per cell (executed 10-pose plan, same definitions as P1-C, `analyze_action_history.unit_metrics` unchanged):
- **severe** = (not A+) ∧ FDE5 > 3 m (the P1-C `amplification` outcome; A+ = `label_A` P/R/A accept rule at 5 s);
- **A+**;
- **FDE5** (m).

Secondary per contrast:
- **local action change at a:** token at a differs (0/1); ‖Δ(forward, left)‖ of the token's local motion (m); |Δdyaw| (rad);
- **suffix change:** the fraction of positions a … 9 whose tokens differ;
- **5 s endpoint distance** ‖X9^a − X9^b‖ (m).

**Relation to coordinate accumulation.**
- AutoVLA tokens are relative motions chained from (0, 0, 0). In B2 no edited token is executed.
- So every endpoint/FDE5 difference between cells arises **only** through the tokens generated at positions ≥ a and their accumulation. It contains no direct accumulation of the edited values.
- This differs from the forced token f (executed, shared) and from the Geo/CM baselines of P2. The executed-prefix identity is checked per unit (§8).

Invalid output (fewer than 10 actions / no trajectory): severe = 0, FDE5 missing (dev rule). A unit enters a contrast only if both cells are valid. Missing ≠ 0.

## 5. Contrasts

All are unit-paired inside one unit and one anchor, with d = Y(cell_a) − Y(cell_b).

**Primary family B2-F** (F stratum, primary anchor, severe; Holm m = 4, α = 0.05 two-sided):

| id | contrast | question |
|---|---|---|
| C_GT | GT_NSRC − GT_REF | latest-block source effect inside a GT older prefix (matched one-shot Reverse − one-shot Full) |
| C_NAT | NAT_NSRC − NAT_REF | latest-block source effect inside a natural older prefix (Normal − one-shot Recent-GT) |
| INT | C_GT − C_NAT | does the older-prefix source change the latest-block effect? (older × latest interaction) |
| C_PROV | GT_CSRC − GT_NSRC | corrected-branch source vs Normal source (value via provenance) inside a GT older prefix |

**Family B2-P:** the same 4 contrasts on the P estimand (Hajek, E4), Holm m = 4, α = 0.025 (the P1-C convention).

**Secondary** (CIs; no family claim):
- the same contrasts on FDE5, local action change, suffix change and endpoint distance;
- the S stratum;
- older-prefix main effects within each L level (NAT_x − GT_x);
- NAT_CSRC − NAT_NSRC;
- GT_CSRC − GT_REF;
- the sensitivity anchor (n_old = 2);
- the dev cohort.

**Pre-fixed restricted analyses** (sensitivity; subgroup defined by bank values only, never by cell outcomes):
- INT on units whose bank older value differs from GT (c^N_{t*+1} ≠ gt_{t*+1}). Otherwise the O factor is empty in value. The share is reported.
- C_PROV on units with c^F_{a−1} ≠ c^N_{a−1}. Report the share and the |Δmotion| distribution.

**Interpretation rules.**
- A non-significant contrast is reported with its CI and is never called "no effect". No equivalence margin is preregistered.
- If C_NAT ≈ 0 while C_GT ≠ 0, the latest-block value matters mainly when an older GT prefix is present. The claim then narrows to an **oracle-conditioned diagnostic**.
- If cells differ only through normal compensation (e.g. endpoint changes without severe changes), the "error/amplification" naming is lowered accordingly (§5 of AUTOVLA_NEXT_STEPS).

## 6. Estimators and inference

- The P1-C estimators are imported unchanged (`analyze_p1c.estimate`). Scene effect d_s = mean of the unit d over the scene's units.
  - θ_F = mean over F scenes.
  - θ_S and θ_P = Hajek with the recomputed weights.
  - Dev: unweighted.
- **Primary clustering: drive.** Exact two-sided sign-flip on drive totals (F: 18 drives, floor 7.6e-6; P: 24 drives). Drive block bootstrap B = 10,000 (seed 20261017, percentile 95 / 90 %). Cluster-robust t.
- **Sensitivity clustering: day–vehicle** (P: 16 groups).
  - The day/vehicle-connected components (P0-D: G = 6, exact floor 0.031) are reported as an inference limit only.
- The Holm-adjusted p of each family is reported with the decision "reject at family α".
- Code: `scripts/review_followup4/b/b2_analyze.py` (written with this protocol; tested only in `--mask` mode on gate output).

**Precision (planning, from the P3 draft; repeated-schedule P1-C F effects).**
- SE of one paired contrast ≈ 2.2–2.5 pp in F.
- MDE at 80 % power under Holm worst case (α/4) ≈ 8–9 pp for C_GT, C_NAT and C_PROV, and ≈ 12 pp for INT.
- One-shot effects are expected to be smaller than repeated-schedule effects.

## 7. Sample size and stopping rule

- **Sample = the frame:** all 2,160 eligible units of the 312 P1-C cohort scenes, plus the 1,327 dev units (secondary).
- No adding of units. No early stopping on any outcome.
- **Order:** the P1-C cohort first (F units, then S units, each in sha256("B2-20261017|" + unit_id) order), then dev. The runner (`b2_matched_prefix.py`, main mode) sorts the eligible units by (F first, key) before phase A.
- **Stop only for:** a gate failure (§9), the P1-C input-verification stop (§2), disk < 1 GB, or a harness error rate > 5 % of units.
- A stopped run is reported as stopped. A partial cohort is never presented as the frame estimate.
- **Budget truncation:** if GPU time runs out, the analysis is restricted, before reading outcomes, to the completed **strata** in processing order: F complete → B2-F only. The truncation is reported.

## 8. Leak / identity checks (every unit in the gate; in main runs the cheap ones are recorded per unit)

1. **Value identity:**
   - `noop` vs NAT_NSRC: bitwise (identical spans, B = 1);
   - any two cells with identical spans: bitwise.
2. **NAT_NSRC vs bank Normal suffix:** bitwise (same B = 1 computation, same prefix cache content, same spans, same CRN).
3. **Pre-anchor executed identity:** every cell's executed tokens before a equal the bank Normal tokens.
4. **Prefix hash:** the cell pass prefix-ids sha = the bank prefix sha.
5. **Bank immutability:** the sha is asserted before phase B.

## 9. Gate for the new arm code (run before the main runs; result in §13)

**Manifest:** the 9-scene Stage-0 manifest (`review_followup3/P1/stage0_manifest.json`), unit `alt0`.
- The anchor is clipped to ≤ 9. The t* = 7 scene has no valid anchor and is reported as not testable.

**Decoding:** t001. **Tolerances:** unchanged from P1 (`pilot_c5_20261014/tolerances.json`: T_exact = 0, T_equiv = 0.75).

| check | comparison | class |
|---|---|---|
| G1 value identity | noop vs NAT_NSRC; coinciding spans | exact |
| G2 bank identity | NAT_NSRC vs bank Normal (phase A, H7 path) | exact |
| G3 c5a repeat | every cell, same process | exact |
| G4 full recompute (C4) | cached-edit path vs no-cache forward from position 0 of the final edited sequence, B = 1, teacher-forced; position ids equal | equivalence (≤ 0.75, no unexplained flip) |
| G5 incremental (C3a) | span re-encode vs incremental KV decoding (cells are append-only after a), teacher-forced | equivalence |
| G6 pre-anchor / prefix | executed tokens before a; prefix sha | exact |

**Pass** = every check on every testable unit and anchor. Any failure → stop, report, no main run.

## 10. Saved per unit × anchor × cell (compact, < 0.3 GB for both cohorts)

- Edited context tokens with their source tags and decoded motions; the token at a and its motion.
- Generated tokens and executed actions; the decoded 10-pose plan (1e-5 m).
- Per-step span hashes; prefix sha; bank sha.
- severe / A+ / FDE5; the endpoint.
- The bank holds the source tokens, motions and hashes.
- No logits are stored (gate comparisons are summaries).

## 11. Throughput pilot and cost (5090)

The pilot command is in §12. It ran on alt0 units of the B1 dev manifest (t* ≤ 4) with `--timing-only` (no cell outcomes written). Measured rates (filled from `pilot_throughput/run_meta_*.json`):

| phase | measured (44 units: dev alt0, t* ≤ 4, 4 with one anchor and 40 with two) | per unit |
|---|---|---|
| A (bank: H7 normal + gt_history, B = 1) | mean 1.07 s | 1.07 s |
| B (7 rows × 1–2 anchors, B = 1, shared prefix) | mean 1.24 s (0.68 s with one anchor, 1.30 s with two) | 1.24 s |
| total | 1.7 min for 44 units, peak 10.9 GB | ≈ 2.3 s |

The full-frame cost is in §13.

## 12. Launch (after the lead commits this protocol)

Common environment:
```
source /root/miniforge3/etc/profile.d/conda.sh && conda activate autovla
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1
cd /root/VLA/autovla_misalignment_poc
```

1. Re-fetch the P1-C images (CPU/network, ≈ 90 min, ≈ 0.36 GB; can run during the dev run):
   ```
   bash scripts/review_followup4/b/b2_fetch_p1c_images.sh all
   ```
2. The main runs, via `scripts/review_followup4/b/run_b2.sh` (waits for ≥ 14 GB free GPU memory; nice; one model load at a time):
   ```
   bash scripts/review_followup4/b/run_b2.sh p1c   # outputs/review_followup4/B/B2/p1c_main
   bash scripts/review_followup4/b/run_b2.sh dev   # outputs/review_followup4/B/B2/dev_main
   ```
3. Analysis (CPU):
   ```
   python scripts/review_followup4/b/b2_analyze.py --run outputs/review_followup4/B/B2/p1c_main --cohort p1c \
          --cohort-manifest outputs/review_followup4/B/B2/p1c_cohort_t4.json --out outputs/review_followup4/B/B2/p1c_analysis
   python scripts/review_followup4/b/b2_analyze.py --run outputs/review_followup4/B/B2/dev_main --cohort dev --out outputs/review_followup4/B/B2/dev_analysis
   ```

The results go to `MATCHED_PREFIX_RESULTS.md/.csv`.

## 13. Gate result, pilot throughput and cost (filled 2026-10-17, phase 1)

**Gate (`B2/gate_stage0/`, `gate_judgement.json`): PASS.**
- Coverage: 8 testable units (the t* = 7 scene has no valid anchor), 16 unit-anchors, 112 cell-runs, t001.

| check | comparisons | failures | max \|Δlogit\| |
|---|---|---|---|
| G1 noop vs NAT_NSRC | 16 | 0 | 0 (bitwise) |
| G1 value identity (cells with coinciding spans) | 174 pairs | 0 | 0 (bitwise) |
| G2 NAT_NSRC vs bank Normal (H7 path) | 16 | 0 | 0 (bitwise, suffix equal) |
| G3 c5a repeat | 112 | 0 | 0 |
| G4 full recompute, teacher-forced, positions equal | 112 | 0 | 0.375 (≤ 0.75) |
| G5 incremental KV, teacher-forced | 112 | 0 | 0.375 |
| G6 pre-anchor executed identity / prefix sha | 16 / 16 | 0 | – |

The gate output contains cell outcomes for these 8 dev units. They were **not summarised**: the analysis code was tested on them only in `--mask` mode.

**Value-coincidence note.** 174 of the 336 within-unit cell pairs had identical spans: early t* units often have c^N = gt or c^F = c^N. This is why §5 pre-fixes the restricted analyses (older values differ; source values differ) and reports their shares.

**Cost (5090; pilot rate 2.3 s per unit; ×1.3 margin for the P1-C t* mix, which has more early-t* units and so longer suffixes):**

| run | units | GPU time |
|---|---|---|
| P1-C main (312 scenes) | 2,160 | ≈ 1.4 h (1.8 h with margin) + 3 min load |
| dev main (196 scenes) | 1,327 | ≈ 0.85 h (1.1 h with margin) |
| total | 3,487 | **≈ 2.3–3.0 GPU h**, plus waiting for ≥ 14 GB free (another agent's runs hold up to 29 GB) |

Other costs:
- Image re-fetch: ≈ 90 min of network and ≈ 0.36 GB of disk (CPU only). It can overlap the dev run.
- Output disk: ≈ 7.6 kB per unit-anchor → ≈ 50 MB for both cohorts.


## 14. Changes from the P3 draft

| P3 draft | B2 | why |
|---|---|---|
| Cells decoded in one batch from step t*+1; pre-anchor tokens "equal Normal under CRN" | Every cell B = 1 from step a on one shared prefix; pre-anchor executed tokens = bank Normal **by construction** | Removes batch-composition numerics from every paired contrast (B1 / M-A3). NAT_NSRC becomes a bitwise check against the bank. |
| Anchor a = t*+3 with only a > 9 excluded | t* ≤ 4 (≥ 3 generated tokens after the edit); sensitivity a = t*+4 with t* ≤ 3 | "enough suffix remains" (AUTOVLA_NEXT_STEPS B2) |
| P1-C 353 scenes, 2,191-scene estimand | 312 scenes / 2,160 units, domain E4 = 1,775 scenes, recomputed weights | the anchor restricts the eligible set |
| Repeated-schedule secondary block | not run | B2 is single-anchor. Repeated results stay separate. |
| Contrast C2 = (NAT,REF) − (NAT,NSRC) | C_NAT = NAT_NSRC − NAT_REF (same sign convention as C_GT) | INT = C_GT − C_NAT reads directly |
| Impromptu add-on | not part of B2 | AutoVLA is the main model. Impromptu parity is A3. |
