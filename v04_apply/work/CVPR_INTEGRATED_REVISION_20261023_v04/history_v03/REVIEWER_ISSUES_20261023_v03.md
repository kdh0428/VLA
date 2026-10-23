# REVIEWER_ISSUES_20261023_v03 — how each issue of the integrated prompt (v02) was handled

| Issue (prompt section) | Handling in v03 | Where |
| --- | --- | --- |
| R3.1 Do not treat the 0.070 secondary result as primary | Kept as "discovery, all with donor" (secondary, exploratory). The v03 primary is the new pre-registered follow-up (0.048 [0.026, 0.072]); no heading says "dominates" | §5.4, Table 5, Supp. P |
| R3.1 Distance confounding | Follow-up uses a pre-registered distance-matched rule: mean d_L 0.054 vs d_O 0.053. Discovery confounding stated | §5.4, Supp. P–Q |
| R3.2 O fixes the intermediate action | Stated in §3.4, in Supp. S, and in the result text ("intermediate action held fixed; not total propagation") | §3.4, Supp. S |
| R3.3 Fresh-cache evaluation path | Described as "official model and processor with a fresh generation cache per call". The 7/105 and 4/105 first-pilot failures and the 3 unresolved cases are disclosed; no bf16 attribution; adapter logit differences up to 0.125 vs R 0.0 (drawer ≤ 0.5) | §5.4, Supp. P |
| R3.4 Separate from cross-step success | New within-call results are command level only. The ±10 pp bound remains attached to the cross-step two-task means; the driving cluster limits are unchanged | §5.4, §7, Limitations |
| R4 Reselection from the old pool is insufficient | Reported (16 obs / 15 layouts, 2 move_near), and given as the motivation for the new pre-registered rule, not as new effect samples | Supp. P |
| E6 Reproducibility snapshot | Config, tokenizer, generation config, model code, transformers cache/generation sources and harness sources snapshotted with hashes. Checkpoint config files are byte-identical to the v01 fingerprint; harness/transformers files are identified by mtime (unchanged since 2026-10-05, before v01). A standalone NumPy verifier reproduces all v01 numbers | `snapshot/`, `verifier/` |
| E7 Sample fixed before results | coke 4000–4099 survey → 25 layouts (4000–4024); move_near reused 10; calls 2/6/10; protocol and code hashed before any v02 output | `svla_v02/PROTOCOL_*`, `PROTOCOL_LOCK_v02.txt` |
| E8 New donor rule | First joint-valid and distance-matched sample among ≤ 64, seeds as specified; MATCHED_SOURCE_MISSING otherwise (36/105); every sample stored with its rejection reason | `svla_v02/run/main/records.jsonl` |
| E9 Path checks | Smoke on coke and move_near; per-observation official = collection = adapter, R tokens and EOS, decoded equality: 105/105; no padding or clipping | Supp. Q, Table S13 |
| E10 Analysis and denominators | Layout → task → two-task, layout bootstrap (seed 20261023). Full denominator chain, source reasons, ratio distribution, secondary analyses; coverage target met (25 coke and 9 move_near matched layouts vs targets 15 and 8) | Supp. Q, `analysis/summary.json` |
| C Extra task (at most 1) | open drawer (first compatible; selection by compatibility only), 20-layout panel. Collection was interrupted (slow rendering), so 44 recovered observations were used by amendment A1 before any intervention output. Result reported separately: 0.145 [0.091, 0.210] | Supp. R, `svla_drawer/` |
| D Alternative models | pi0-FAST, OpenVLA, OFT, WorldVLA compared from official sources with revisions and local checks. No GPU experiments; none counted as experimental | Supp. T, `MODEL_SELECTION_20261023_v03.md` |
| §5 Cross-policy comparison | Table of edited position, fixed prefix, donor, response window, execution, metric and unit for AutoVLA / VLA-0 / SpatialVLA. AutoVLA's contrast is not treated as the same test | Supp. S |
| §5 "Observed in another backbone" vs "backbone effect isolated" | Abstract and Limitations say observed, not isolated | Abstract, §8 |
| §5 8-page main, full supplement | Main 8 content pages (references on p. 9). Three figures moved to the new Supp. U with their numbers kept in the text; supplement A–U, 18 content pages; no font or style change | `BUILD_QA_20261023_v03.json` |
| §5 Preserve VLA-0 D(OL,L), motion intervals, SpatialVLA 8 rows, PDMS, cluster limits | Unchanged; the v02 verifier passes 668/668 on the v03 TeX | `qa/` |
| §6 Budget | B 16.1 (smoke 0.8 + main 15.3) ≤ 60; C 26.6 (20.8 interrupted + 5.8 recovered) ≤ 30; total with v01 = 62.8 ≤ 150 model-GPU min | `gpu_ledger_v03.json` |
