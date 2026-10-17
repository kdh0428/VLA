# B1 NUMERICAL_SENSITIVITY_PROTOCOL — AutoVLA numerical sensitivity (review_followup4)

Written 2026-10-17 **before any B1 output exists** (repo HEAD a41b64a). Changes after B1 data exist are listed as amendments at the end of this file, with what had been seen.

**Kind of study:** a sensitivity analysis of the numerical implementation. It is **not** a new estimate of any main effect and **not** external replication. The question is: how much do *normal* numerical variations of the same computation move the decoded suffixes, the 5 s endpoint and the severe/A− verdicts, compared with the actual interventions, and how much do the paired contrasts change?

What is kept from before:
- PROTOCOL_P1 c5 / gate (review_followup3): same process and separate process repeats are bitwise. Experiment batch vs B = 1 changes logits by ≤ 0.375 (bf16 ulps). Free-run suffixes were equal on the 9-scene manifest.
- M-A3: Normal agreement across harnesses is 88–89 % at P1-C scale. Free-run/full-recompute and batch shape *can* change suffixes.
These results are kept as they are. B1 does not re-run the gate.

## 1. Manifest (fixed rule, no outcome read)

Built by `scripts/review_followup4/b/b1_manifest.py` → `B1/b1_manifest.json`.

- Rule = the Stage-0 manifest rule (`p1_stage0_manifest.py`), with a larger take:
  - population: the 208 dev equal-distance scenes;
  - strata by t*: early {0, 1}, middle {2, 3}, late {4–7};
  - order: sha256("P1-STAGE0-20261014|" + token) within each stratum;
  - **take: first 8 per stratum → 24 scenes.** The first 3 per stratum are the 9 Stage-0 scenes.
- Units per scene:
  - `alt0` (first distance-matched alternative) on H7 and H8;
  - P1A family G at t_p = t*, lowest available sector (the Stage-0 rule). No GF3.
- Size: 24 H7/H8 units + 24 P1A units. The A−/A+ mix is recorded, not balanced.
- Why 8 per stratum: this fits the budget (≤ 1.5 GPU h, §5). It was chosen before any run.

## 2. Settings compared

All settings use:
- t001 decoding (the experiments' T = 0.01 sampling, seed 0, CRN key as in the experiments);
- RTX 5090; checkpoint `AutoVLA_PDMS_89.ckpt` loaded into the bf16 model (as in all experiments);
- the same inputs, units, rows, forced token (dose) and pairing.

Only the listed factor changes.

| id | what changes | H7 (Recent-GT, Full-GT, Normal) | H8 (Reverse-N = `reverse@emb`; Normal, Full-GT, Recent-GT) | P1A (self-reference rows) |
|---|---|---|---|---|
| **S0** reference | nothing: bf16, sdpa, experiment batch | B = 1 per row | B = 93, experiment hooks | B = 10 |
| **S0r** no-edit repeat | the S0 computation repeated in the same process | same | same | same |
| **SB** batch shape | batch composition only | rows taken from the H8 B = 93 batch (same context rule, same CRN key) | every key row at B = 1: Normal / Full-GT / Recent-GT = the H7 B = 1 rows; Reverse-N at B = 1 by token substitution of the B = 1 Normal row's token (the C4-verified equivalent of the layer-0 emb patch) | all 10 rows at B = 1 |
| **SE** attention backend | eager attention instead of sdpa (LM and vision attention classes swapped in place; `config._attn_implementation = "eager"`), bf16, experiment batch | B = 1 | B = 93 | B = 10 |
| **SF** compute precision | fp32 compute: the bf16 parameters are upcast exactly to fp32 (buffers unchanged), sdpa, experiment batch | B = 1 | B = 93 | B = 10 |

Recorded co-changes:
- **SB:** for H7 the "alternative" rows come from a different code path (H8). Its context rule and CRN key are identical (verified in P1). The H8 hooks are installed, but they only edit the patch rows.
- **SE:** eager attention also changes how the causal mask is built (`_update_causal_mask` returns a 4-D mask instead of None). It also changes the softmax precision (eager upcasts to fp32).
- **SF:** the weights stay the bf16-rounded values. Only the arithmetic precision changes. A true fp32-weight model (the fp32 checkpoint without bf16 rounding) is a *different parameter set*. It is not run here (§7).
- If SF does not fit in GPU memory at the experiment batch (needs ≈ 26 GB free), SF is skipped and recorded. It is **not** run with a smaller batch, because that would confound precision with batch shape.

## 3. Rows and contrasts

Rows kept per unit:
- H7: normal, recent_gt, gt_history.
- H8: normal, gt_history, recent_gt, reverse@emb.
- P1A: all 10 rows.

Interventions, as paired contrasts inside one unit (cond − ref):

| id | cond − ref | path in S0 |
|---|---|---|
| RN | recent_gt − normal | H7 (B = 1) |
| FULL | gt_history − normal | H7 |
| RV | reverse@emb − gt_history | H8 (B = 93) |
| PERT | OC − U | P1A |
| SELF_R | OC_refR − OC | P1A |
| SELF_H | OC_refH − OC | P1A |

Mixed-batch pairing (SB only, reported separately): RN_mixed = recent_gt (H7, B = 1) − normal (H8, B = 93). This mimics pairing rows that come from different batch compositions.

## 4. Metrics

**Per (unit, row, setting) vs S0:**
- Suffix equal (all generated tokens), first divergence step.
- Logits: max |Δlogit| over the 2,048 action rows on the steps whose input span is identical in both runs (free run). For the substituted Reverse-N the effective input is used. Only summaries are stored, no logit arrays.
- Decoded 5 s plan: endpoint deviation ‖X9 − X9^S0‖ (m) and max pose deviation.
- Outcome flips vs S0:
  - `severe` = amplification = (not A+) ∧ FDE5 > 3 m;
  - `A−` = not A+ (A+ = `label_A` P/R/A accept rule, 5 s);
  - FDE5 change (m).
  These are the P1-C unit outcomes (`analyze_action_history.unit_metrics`, unchanged).

**Reference scale (actual interventions, S0 only, same units):**
- endpoint shift ‖X9^cond − X9^ref‖;
- severe / A− change rate for each contrast of §3.

**Paired-contrast stability (per contrast, per setting s):**
- the unit mean of d_s = Y_s(cond) − Y_s(ref) for severe, A− and FDE5;
- Δ = mean d_s − mean d_S0;
- the number of units whose binary d changes;
- mean |d_s − d_S0| for FDE5.

The numeric variation is **reported next to** the effects. It is **never subtracted** from them.

## 5. Decision rule (fixed now)

A setting is called **material for a contrast** if any of these holds:
- (a) the sign of mean d (severe or FDE5) differs from S0 while |mean d_S0| ≥ 1 unit/24;
- (b) |Δ mean d_severe| > 0.05 (half of the P1-C F-stratum MME of 0.10);
- (c) |Δ mean d_FDE5| > 25 % of |mean d_FDE5,S0|.

S0r must be bitwise identical. Otherwise this is a harness problem and it is reported first.

If a setting is material for RN or RV:
- the corresponding claim is limited ("sensitive to <factor>");
- the minimal rerun scope is estimated from the P1-C run times (PROTOCOL_P1 §7 table).

Otherwise the claim stays, with the measured floor reported.

Limits:
- 24 units is small. Flip rates are reported with exact (Clopper–Pearson) 95 % CIs over row-runs.
- Row-runs within a unit are not independent. Unit-level counts are also reported.
- No p-values for "no difference": non-significance is not equivalence.

## 6. Cost and safety

- One model load. The settings run sequentially in one process: S0 → S0r → SB → SE → SF.
- Expected ≈ 50–60 min on the 5090. The cap is 1.5 GPU h: stop after the current setting if exceeded.
- GPU checks:
  - ≥ 14 GB free before the load;
  - ≥ 12 GB additional free (`torch.cuda.mem_get_info`) before SF, else wait up to 15 min, then skip SF.
- `nice -n 10`; `CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1`.
- Disk: per-row summaries and tokens only (< 20 MB). Stop if `/root` has < 1 GB free.

## 7. Not run / out of scope

- fp32-weight model (different parameters).
- Greedy decoding (the gate already covers it).
- Separate-process repeats (already bitwise in P1 c5b, 886/886).
- The H9 / H32 / P1B paths.
- Any P1-C-scale rerun.

## 8. Outputs

`B1/`:
- `b1_manifest.json`;
- `records.jsonl` (per unit × setting × row: tokens, plan, metrics);
- `logit_cmp.jsonl`;
- `run_meta.json`;
- `NUMERICAL_SENSITIVITY.csv` / `.md` (by `b1_analyze.py`, CPU).

Code: `scripts/review_followup4/b/b1_manifest.py`, `b1_runner.py`, `b1_analyze.py`.

## Amendments

### B1-A1 (2026-10-17, before the main run) — larger take, SF memory check, OOM retry
**What had been seen.** One smoke run (scratchpad, not part of the results) on the first manifest scene (c05f89f6…, 2 units) over all 5 settings:
- S0r was bitwise;
- SB / SE / SF max |Δlogit| were 0.33 / 0.92 / 1.26;
- all suffixes were equal;
- elapsed 0.4 min, peak 25.7 GB with SF at B = 93.

No contrast or outcome summary was computed.

**Changes.**
1. The take is raised from 8 to **20 per stratum**. The late stratum has only 16 scenes, so all 16 are taken: 56 scenes, 112 units, in `b1_manifest_take20.json`. Same rule, same order. The 24-scene manifest is the first 8 per stratum of it and is kept unchanged.
   - Why: the measured cost (≈ 0.4 min per scene for all settings) is far below the 1.5 GPU h cap, and flip rates need more row-runs.
2. The free-memory requirement before SF is raised from 12 to **17 GB** additional (measured peak 25.7 GB in total).
3. An OOM during one unit is retried after 30 s, up to 20 times. A unit that still fails is listed as skipped and is not imputed. Duplicate partial records are resolved by keeping the last record per (setting, unit, path, row).

Unchanged: settings, rows, contrasts, metrics, decision rule.
