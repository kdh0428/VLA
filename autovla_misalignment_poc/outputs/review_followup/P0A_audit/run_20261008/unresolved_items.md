# unresolved_items — P0-A

Items that could not be verified. None of them was filled in by estimation.

## AutoVLA / NAVSIM

| # | 항목 | 상태 | 이유 | 해결에 필요한 것 |
|---|---|---|---|---|
| U1 | Unit-level results of experiment 8, the main run (records.jsonl) | 미확인 | Not on disk. Only `summary.json`, `layer_results.json` and `sanity` exist. Claims 3 (reverse 33.4%, 68%) and 4 (3.8–7.4%) can be checked only against the summary. | Full re-run of experiment 8 (about 20 min on the 5090). This audit re-ran only 3 units (`rerun_tiny/exp8`) to check the intervention structure. |
| U2 | Unit-level results of experiment 9, the main run | 미확인 | Not on disk. Claim 5 (47/76/90/96%) and the critical window can be checked only against the summary. | Full re-run of experiment 9 (about 6.4 min). This audit re-ran only 3 units (`rerun_tiny/exp9`). |
| U3 | Experiment 5 records and logits | 미확인 | Not kept (`first_mismatch_causal/` holds only the summary). Recovery 17.3 → 73.1% is summary-only. | Re-run of experiment 5 (about 16 min). |
| U4 | Experiment 10 records and `errors.jsonl` | 미확인 | Not on disk. Not in scope for this audit, but the Claim 5 horizon numbers are summary-only. | Re-run. |
| U5 | Whether the original experiment 7 run (from before the 2026-09-26 server move) matches the current raw | Partly confirmed | The current `records.jsonl` is the 2026-09-27 re-run. REPRODUCTION.md says it "matches entirely", but the original raw is gone, so this is a document claim. Experiments 8 and 9 were run against the original experiment 7 records, so their "reproduction rate 88.6/89.2%" refers to the original raw. | Not recoverable. |
| U6 | Pre-registration of the AutoVLA mechanism experiments 4–11 | Unverifiable | Scripts and results were committed together (1541f30). | Only a new pre-registered run can fix this (P1). |
| U7 | Upstream base commit of the vendored NAVSIM v1.1 (`autovla/navsim`) | 미확인 | It has no git metadata of its own; the AutoVLA repo is at ba34eed. It has no human filter, so the 2025-09-29 fix does not apply. | Compare file hashes against upstream tags (needs network). |
| U8 | Actual import path of the navhard scoring `pdm_score` | Inferred | The import path is not logged. v2 is inferred from the run script path (navsim_v2) and from v1.1 not being able to run that script. | Re-score once with an import-path log. |
| U9 | Whether the tiny re-runs equal the original runs | Expected not to match exactly | The re-runs use the same script and GPU, but batch size and numeric path make Normal differ from the experiment 7 raw in 1 of 3 units (experiment 9 rerun `original`). The traces show only the intervention structure; they do not validate the original numbers. | – |

## SpatialVLA
| # | 항목 | 상태 | 이유 |
|---|---|---|---|
| S1 | Actual TCP trajectory divergence in experiment 34 | 미확인 | The TCP pose was not saved. Every experiment 34 trajectory claim in metres is a commanded-action proxy. |
| S2 | Experiment 35 batch-size check (batch changes in 4/40 chunks) and executor-equivalence check (5e-6) | 미확인 | The outputs of these checks were not saved. Reading the code, the two executors are logically equivalent. |
| S3 | Frame of `tcp.pose` and whether the robot base stays fixed | 미확인 | Paired distances do not depend on the frame as long as the base is fixed. |
| S4 | Whether experiment 34 natural_mix actually shared a batch with intervention rows | 가설 | The code runs one condition at a time, so natural_mix probably mixed with intervention rows only at block boundaries. |

## Out of scope for this audit, noted only
- Statistical re-analysis (log clusters, leave-one-log-out) → P0-B.
- Experiment 35 analysis with seeds fixed in common → P0-C.
