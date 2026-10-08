# P1-C leakage check (design stage, 2026-10-08)

CPU only. No model inference, no download, no existing file changed. Labels: **[확인한 사실]** read from files; **[계산한 결과]** computed here; **[미확인]** not checkable; **[가설]** interpretation.
Scripts: `scripts/scan_logs.py`, `scripts/scan_tokens.py`, `scripts/build_split.py`. Intermediate files are in `work/`. The split itself is in `split.csv`.

## 0. Summary

| Check | Result | Status |
|---|---|---|
| Eval logs used in any perturbation/mechanism experiment (exp 4–19, 32, 33 inputs, full_extract, frame_index, P0-A/B) | 0 of 64 | pass |
| Eval drive (`date.time_veh`) shared with a dev (PoC) log | 0 of 24 drives, by construction (42 adjacent logs moved to `excluded_drive_adjacent_to_dev`) | pass |
| Pilot drive shared with an eval drive | 0 | pass |
| Derived PoC frame scenes (`navtest_poc_frames`, 2,260 scenes, exp 12–17) in eval logs | 0 | pass |
| Synthetic navhard scenes used as P1-C units | none (P1-C uses original navtest scenes only) | pass |
| Eval logs that are *sources* of navhard synthetic scenes | 34 of 64 | **flag**: sensitivity analysis required (§3) |
| Same day + same vehicle as a dev log | 41 of 64 eval logs, 10 of 16 eval day-vehicles; 69 of 113 eval A− frame-label scenes | **flag**: cannot be avoided without losing most of the A− stratum (§4) |
| Same vehicle (any day) | 11 of 14 eval vehicles also appear in dev | reported only |
| Natural (unperturbed) labels of eval scenes already seen | yes. Exp 20–24 decoded every eval scene and reported natural failure rates. | **flag**: these are not P1-C outcomes (§5) |

## 1. What has been used before [확인한 사실 + 계산한 결과]

- navtest has 136 logs (`navtest.yaml`). All 136 have been used somewhere.
  - 28 PoC logs (shards 0–5) were used for every mechanism experiment.
  - 56 logs in shards 6–17 (dev) and 52 logs in shards 18–31 (held-out) were used only in selection, detection, and PDM experiments. No scene-token, log-name, or code path puts them in a perturbation experiment.
  - The 108 logs from shards 6–31 have no overlap with each other or with PoC (0/0/0).
- Method:
  - Scanned every text file under `outputs/` (excluding `full_extract/tensors`) for log names (`scan_logs.py`).
  - Mapped 12,125 scene tokens to logs and scanned for the tokens as well (`scan_tokens.py`).
  - Per-log directory lists are in `split.csv:output_dirs_using`.
- The union of directories that touch eval logs:
  - `expanded_best_of_n(_5090)`, `heldout_best_of_n(_5090)`, `candidate_count_curve`, `candidate_oracle`, `pdm_score_best_of_n`, `instability_detection_baselines`, `safety_filter*`, `navhard_eval`, `navhard_full_validation`.
  - None of them runs an action-token perturbation or context correction.
- 11 test logs are outside navtest. Nothing has used them (role `reserve_non_navtest_not_used`).
  - All of them are in Boston.
  - 1 shares a drive with dev. 7 share a day-vehicle with dev. 1 shares a drive with eval.
  - Why NAVSIM excludes them is [미확인]. They are **not** in the eval set.
- Every navtest log has been used, so no unused navtest log exists. The eval set is therefore the "least-used" option: logs never used in any mechanism experiment.
  - Logs outside the test split (navtrain/trainval) were rejected because AutoVLA was trained on navtrain [가설: trainval contamination cannot be excluded]. Using them would also need a multi-hundred-GB download.

## 2. Adjacent scenes and drives [계산한 결과]

- A navsim "log" is one segment (`<date.time>_veh-NN_<start>_<end>`) of a longer drive.
  - 28 PoC segments come from 20 drives.
  - The 108 non-PoC segments come from 38 drives.
  - **44 of the 108 non-PoC segments belong to a drive that also has a PoC segment.** Scenes in those segments can sit seconds away from dev scenes.
- Rule applied: any segment whose drive has a dev segment is excluded from eval.
  - 42 such segments are `excluded_drive_adjacent_to_dev`.
  - 2 such segments are `pilot_harness_check` (harness check only, never analysed).
- Result: eval = **64 segments, 24 drives, 16 day-vehicles, 5,671 navtest scenes, shards 6–31**.
- The independent cluster in the eval analysis is the **drive**, not the segment. Segments of one drive are not independent.
  - P0-B clustered at segment level. 16 A− segments there = 13 drives [계산한 결과 from split.csv; P0-B did not check this].

## 3. Derived and synthetic scenes [확인한 사실]

- `navtest_poc_frames` (prev/next-frame scenes for exp 12–17) contains only PoC logs.
- navhard two-stage has 5,462 synthetic scenes from 76 source logs: 13 PoC, 30 dev-shard, 33 held-out-shard.
  - 34 eval logs are navhard sources.
  - P1-C never uses synthetic scenes.
  - The navhard experiments tuned selection/filter rules, not any mechanism setting. So nothing learned from those scenes enters the P1-C settings [가설, from the experiment list].
- Required sensitivity analysis (protocol §8): the primary contrasts restricted to the 30 non-navhard eval logs (13 drives, 51 A− frame-label scenes).

## 4. Same vehicle / same day [계산한 결과]

- Dev day-vehicles: 14. Eval day-vehicles: 16. Shared: 10.
  - Shared: 2021.05.25 veh-25/35, 2021.06.03 veh-35, 2021.06.28 veh-14/26/38, 2021.09.16 veh-42/45, 2021.09.29 veh-28, 2021.10.06 veh-52.
  - Eval-only: 2021.05.25 veh-30, 2021.06.28 veh-16, 2021.08.16 veh-45, 2021.09.09 veh-40/48, 2021.09.16 veh-08.
- Excluding shared day-vehicles leaves 23 logs, 7 drives, and 44 A− frame-label scenes. That is too few clusters for the failure-conditional estimand. The primary set therefore keeps them.
- Prespecified mitigations:
  - (a) Sensitivity analysis on the 23 day-vehicle-disjoint logs.
  - (b) Sensitivity analysis clustering at day-vehicle level (16 clusters).
  - (c) The paper must state that the replication is drive-disjoint but not day/vehicle-disjoint.
- [가설] Same-day, same-vehicle drives share weather, route region, and sensor calibration. Effects that depend on those would replicate too easily, so this limits generalisation claims.
- City mix (from `navsim_logs` `map_location`):
  - dev logs: Las Vegas 14, Boston 9, Pittsburgh 4, Singapore 1.
  - eval logs: Las Vegas 24, Boston 22, Pittsburgh 12, Singapore 6.
  - Eval is more varied, which is good for generalisation. Report city as a descriptive stratum.

## 5. Information already seen about eval scenes

- Exp 20–24 decoded every eval scene with the selection harness (row 0 = natural plan at T = 0.01). Natural failure rates are therefore known.
  - Frame labels from the 5090 rows: 113 A− scenes (t* ≤ 8) / 111 eligible (t* ≤ 7) in eval logs.
  - Planning only: `work/frame_labels_5090.json`.
- These labels are a **baseline covariate** (unperturbed outcome), not the P1-C outcome. No perturbation, context correction, reverse, or window condition has ever been run on these scenes.
- Use in this design:
  - Expected stratum sizes for the power simulation.
  - Shard budgeting.
- Not used for:
  - Defining strata. The protocol redefines strata from a fresh natural pass in the evaluation harness (stage 1). Planning labels are only compared with them.
- Expected disagreement between the two label sources [계산한 결과]:
  - 3080 Ti vs 5090 rows of the same scenes: 3,260 pairs. A− agreement 99.7% overall, but only 41 of 50 A− scenes in either run were A− in both (≈18% boundary flips).
  - The harness changes (batch 1, own decoding loop) will cause similar flips.

## 6. Remaining [미확인]

- Whether same-day drives of one vehicle overlap spatially (same streets). Not checked; it would need map routes.
- The compressed size of each camera shard (no network access was used). The estimate is ≈3.8 GB per shard, from the 23 GB for shards 0–5.
