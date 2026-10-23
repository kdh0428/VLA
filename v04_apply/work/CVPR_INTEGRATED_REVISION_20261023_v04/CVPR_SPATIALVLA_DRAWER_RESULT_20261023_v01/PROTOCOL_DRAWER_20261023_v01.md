# PROTOCOL_DRAWER_20261023_v01 — supplementary task (C): google_robot_open_drawer

- **Status.** Pre-registered before any drawer intervention output (`PROTOCOL_LOCK_drawer.txt`). This is a supplementary analysis: it is never mixed into the two-task estimand of B, and the three tasks are not treated as a sample from a task population.

## Task selection (compatibility only)
- **Order checked** (per the request): open drawer → close drawer.
- **Selected task.** `google_robot_open_drawer`, i.e. ManiSkill2-real2sim `OpenDrawerCustomInScene-v0` with its prepackaged config, from the local `simpler_env` registry.
- **Compatibility.**
  - Same Google-robot embodiment and overhead camera path (`get_image_from_maniskill2_obs_dict`).
  - Same official SimplerEnv SpatialVLA adapter and the same fractal checkpoint and unnormalization statistics.
  - Drawer tasks appear in SpatialVLA's own SimplerEnv evaluation table (`simplerenv.md`).
  - No training is needed.
- Selection did not use success rates or effects. Close drawer was not needed.

## Sample
- **Layouts.** Seeds 5000–5099 were surveyed in order on CPU (`manifest/survey_open_drawer_5000_5099.json`). The cabinet pose is fixed; the robot base pose and the target drawer vary. The layout fingerprint covers robot and articulation poses, actor poses and the instruction. The panel is the first 20 unique layouts: seeds 5000–5018 and 5020 (seed 5019 repeats an earlier layout). The survey found 69 unique layouts in total.
- **Observations.** Natural generation calls 2, 6 and 10, so at most 60 observations (`manifest/drawer_layouts.json`).

## Donors, arms, checks, analysis
Identical to B (`PROTOCOL_20261023_v02.md`) except where noted:
- Up to 64 sources; first joint-valid **and** distance-matched donor.
- Arms N, R, O, L, OL, with per-observation parity checks.
- Δ is the within-task layout-weighted mean, with a layout bootstrap of 10,000 resamples, seed 20261023, and a 95% percentile interval. The interval is NA if ≤ 1 layout.
- Reported separately, with the full denominator chain.

## Budget
At most **30** model-GPU minutes, within the global 150-minute cap.

## Amendments
- **A1 (2026-10-23; before any drawer intervention output).**
  - **What happened.** Attempt 1 (`run/main_attempt1_collection_interrupted`, 20.8 model-GPU min) spent its whole time collecting observations. The drawer scene renders slowly in software, at about 2.2 s per simulator step (CPU timing) against well under 1 s for coke/move_near. The process ended at 02:03 before collection finished: 44 observation PNGs from 16 layouts were saved, and 0 intervention records were produced.
  - **Budget.** C's remaining budget (9.2 min) cannot pay for a new collection.
  - **Change.** The pilot uses the **44 recovered observations**, ordered by the frozen manifest (`manifest/recovered_observations.json`):
    - seeds 5000–5011 with calls 2/6/10, and seeds 5012–5015 with calls 2/6;
    - the episodes were cut off by the interruption, not by any outcome;
    - images are lossless PNGs, with input hashes recomputed from them.
  - **Lost check.** The collection-time `predict_action` outputs were held only in memory, so the official-vs-collection check is **NA** for these observations. Every other per-observation check is unchanged: adapter natural = official, R a3–a4 tokens and EOS, decoded actions. Code changes: phase `recovered` in `wc_run_v02.py`; NA accepted in `analyze_v02.py` and `verify_v02_standalone.py` (the B results are byte-identical after this change).
  - **Unchanged.** Donor rule, arms and analysis are as pre-registered. Phase cap 8.8 min; observations not reached are reported as not run.
