# LEARNED_EXTRAPOLATOR_PROTOCOL (P2-B, review_followup3)

- **Written:** 2026-10-14, before any training data was extracted and before any extrapolator was trained or evaluated. The sha256 and UTC time of this file are recorded in `extrap_protocol_lock.json`. The training and evaluation scripts refuse to run if this file has changed.
- **What had been seen when this was written:**
  - the C1 reproduction and CM definition check (P2-A steps 1–2; identical to C1 / v1);
  - a validation of the numpy GT→token port: 208 navtest_poc scenes, 2,080/2,080 tokens equal to the stored `gt` tokens, and decoded poses equal to the running matched poses within 9e-6 m;
  - a check of the ego-state fields: 439 test-split frames, navtest_poc JSON velocity, acceleration and instruction equal the OpenScene pickle fields; GT future poses within 6 cm.
- **Not yet seen:** no P2-A new-baseline (CV0/CTRV-K3/CTRA-2) results and no extrapolator results.
- **Status:** POST-HOC control analysis on existing P1-A rollouts. All AutoVLA-derived numbers are **PROVISIONAL** until the AutoVLA P1 no-op/cache parity gate is judged (`outputs/review_followup3/P1/NOOP_CACHE_AUDIT.*`).
- **Scope:** this is a small **motion** model. It is not a non-autoregressive version of the VLA and must never be called one.

## 1. Question
Is AutoVLA's suffix response to a single perturbed executed action token larger than what a learned prefix-only motion model would extrapolate from the same perturbed prefix?
- Model response: P1-A OC vs U.
- Comparison: `Δ_model − Δ_extrap`, where Δ = φ(perturbed) − φ(unperturbed), computed for each method separately.
- The same φ, units, horizon, scene weights and drive clusters are used as in P2-A (MOTION_BASELINE_SPEC).

## 2. Inputs (fixed)
**Primary input (E-prefix)** contains the same information as CM: the decoded plan poses 0..j and the position j.
- Plan poses are in the decoder frame, i.e. the ego frame at plan time.
- At evaluation, j = t_p:
  - perturbed prefix = decode(r[:t] + [p]);
  - unperturbed prefix = decode(r[:t] + [r_t]).
- The feature vector is `p2_extrap_common.prefix_features`:
  - Origin and poses 0..j, expressed in the frame of pose j: 10 slots × (x/10 m, y/10 m, heading rad).
  - Body increments of steps 0..j: 10 slots × (fwd/10, left/10, dyaw).
  - Slot mask: 10.
  - One-hot j: 9.
  - Total 79 dimensions. Slots after j are zero and masked.
  - The function reads nothing after pose j. This is enforced by a unit test with garbage after j.

**Variant (E-prefix+ego):** the primary input plus the ego state that AutoVLA actually receives in its prompt (`models/autovla.py`).
- Inputs:
  - speed |v|/10 (m/s);
  - |a|/2 (m/s², magnitude, as in the prompt);
  - driving command one-hot (left / straight / right).
- Total 84 dimensions.
- It has more information than CM and is reported as a separate row.
- At evaluation the values come from the same scene JSON the model received (`autovla/dataset/nuplan/navtest_poc/<token>.json`). In training they come from the OpenScene `ego_dynamic_state` and `driving_command` of the plan frame. Field equality was verified on test frames.

**Not used:** images, maps, agents, route, future GT at evaluation, model outputs, the evaluation suffix r[t+1:], and any development/evaluation scene.

## 3. Output (non-AR)
- One forward pass predicts every suffix pose j+1..9.
- Output: 9 slots × (x/10, y/10, heading), in the frame of pose j. Slots beyond 9−j are masked.
- The network is not autoregressive and its outputs do not feed back into it.
- Predicted poses are mapped back to the decoder frame (`assemble`). Plan = prefix poses 0..j plus predicted poses j+1..9.

## 4. Architecture and training (fixed, no tuning except the grid in 4.3)
**4.1 Network:**
- MLP: in → H → H → H → 27, with GELU activations, no dropout, no normalisation layers.
- Inputs are standardised with the training-split mean/std (std floor 1e-3).

**4.2 Optimisation:**
- AdamW, weight decay 1e-4, batch 2048.
- Cosine learning-rate decay to 0 over a fixed 15 epochs, no warm-up, no early stopping; the final-epoch weights are used.
- Loss: masked MSE over the valid suffix slots on (x/10, y/10, heading).
- CPU only, torch threads ≤ 8, `nice -n 19`.

**4.3 Hyperparameter selection (validation drives only):**
- Grid: H ∈ {256, 512} × peak lr ∈ {1e-3, 3e-4}. Four runs, primary input, seed 0.
- Criterion: masked mean position error (m) on validation drives, averaged over all valid suffix poses of all validation rows.
- The selected (H, lr) is used for **both** input variants and all seeds.
- Selection never touches evaluation (P1-A) data.

**4.4 Seeds:**
- Final models are trained with seeds {0, 1, 2} for each variant on the training drives only. Validation drives are held out from the weights.
- Seeds control initialisation and batch order.

## 5. Data (training split only, drive-disjoint)
**Source:** OpenScene v1.1 **trainval** metadata, which is the AutoVLA checkpoint's training split (P0: `DISJOINT_LOG_INVENTORY_meta.json`).
- Trainval trajectories are not on disk. The public tarball (`openscene_metadata_trainval.tgz`, 7.05 GB, dataset commit a76f840b) is **streamed**, the same way P0 streamed it. Only per-frame ego fields are kept, as a compact parquet file of about 40 MB: log, token, frame index, timestamp, x, y, yaw, v, a, command.
- The tarball is never written to disk.

**Drive exclusion (drive = `<date.time>_veh-NN`):**
- Any trainval drive that appears in **any** OpenScene test-split drive is removed. The test split contains the development set (20 drives, incl. the 13 A− drives), P1-C (26 drives incl. pilot; 24 evaluation drives) and all P5 candidate drives.
- Known facts: 0 shared logs, 0 shared drives, 1 shared day (2021.10.06).
- The separation check (`extrap_drive_separation.json`) lists the counts. It is asserted in code before training and again before evaluation.
- Day and vehicle overlap with the evaluation drives is reported descriptively. It is not required to be zero; the requirement is drive-disjointness.

**Validation drives:** 15% of the eligible drives, drawn by `numpy.random.default_rng(20261014).permutation` on sorted drive ids, rounded up. The remaining drives are training drives.

**Samples:**
- A plan frame i is used when frame_idx is even (stride 2, to limit near-duplicates) and frames i+1..i+10 exist in the same log with each Δt within [0.45, 0.55] s.
- GT future poses are computed relative to ego pose i (yaw from the quaternion). Samples with any GT step > 15 m per 0.5 s are dropped.
- Each GT future is tokenized with the AutoVLA GT→token projection (numpy port, exact on the 208 navtest_poc scenes) and decoded. The token-domain poses are both the prefix input and the target, so prefixes come from the same quantised representation that the P1-A prefixes use.
- Each plan frame yields 9 rows, j = 0..8.

## 6. Evaluation (same as P2-A)
**Units and contrast:**
- P1-A units (families G, E, GF1, GF3; p ≠ r_t), strata A−, A+ and all. These are the same 5,553 units as C1/P2-A.
- Per unit, per seed: Δ_E = φ(E(prefix_p)) − φ(E(prefix_0)) for φ ∈ {end, sfx, dref, dgt, area}.
- The φ definitions are exactly MOTION_BASELINE_SPEC 2.2. The body increments for `sfx` come from each method's own poses.

**Primary estimate:**
- The per-unit Δ_E is averaged over the 3 seeds, then Δ_model − mean_seed(Δ_E) is computed.
- Inference uses `c_baseline_window.drive_row`: scene-weighted, drive-cluster bootstrap with B = 10,000, exact drive sign-flip.

**Seed variance:**
- The same contrast is computed with each seed's Δ_E separately (three rows).
- The seed SD and range of the point estimate are reported.

**Multiplicity:**
- Holm with m = 2 over {E-prefix, E-prefix+ego} within (family, stratum, metric).
- The P2-A kinematic Holm family (m = 4) is unchanged.

**Seeds for the bootstrap:** base 20261015, +11 per row in a fixed loop order.

## 7. Coverage of perturbed prefixes (out-of-distribution check)
- **Feature space:** the primary input features without the mask and one-hot (60 dimensions), standardised with the training statistics.
- **Reference set:** for each j, a uniform random sample of up to 100,000 training rows with that j (rng 20261016) is indexed with a cKDTree.
- **Reference distribution:** nearest-neighbour (NN) distances of up to 20,000 validation rows with that j.
- **Evaluation:** NN distance of every evaluation prefix (perturbed and unperturbed separately). Report the quantiles 50/90/99/max, and their ratio to the validation quantiles for the same j, per stratum.
- **Out of range:** the prefix NN distance exceeds the 99th percentile of the validation NN distance for the same j. A unit is out of range if its perturbed **or** unperturbed prefix is out of range.
- **Sensitivity:** the primary contrast recomputed on in-range units only, reported separately. The primary analysis keeps all units.

## 8. Interpretation rule (fixed)
- **Primary cell:** family G, metric M1 (`end`), in each stratum S ∈ {A−, A+, all}.
- "Beyond motion extrapolation" may be stated for S only if all three hold:
  1. Δ_model − Δ_b > 0 for **all four** kinematic controls (CM, CV0, CTRV-K3, CTRA-2). The criterion is the P2-A locked rule: Holm m = 4 p < 0.05 and the drive 95% CI excludes 0.
  2. Δ_model − Δ_E > 0 for **both** learned extrapolator variants, with Holm m = 2 p < 0.05 and the drive 95% CI excluding 0. Each uses its primary seed-averaged estimate.
  3. The in-range sensitivity analysis does not reverse the sign.
- **Any failure:** if any one comparator accounts for the model's response in S (estimate ≤ 0, or not detected), the section 6.2 statement for S is narrowed to "not distinguished from motion extrapolation by <comparator>".
- **Language rules:**
  - "Not detected" is not equivalence.
  - A− is GT-defined and failure-selected, so it is not generalised.
  - All results carry domain, model and evaluation tags: driving, AutoVLA, open-loop 5 s plan.
  - Results are PROVISIONAL until the P1 gate.
- **Secondary metrics:** M2–M5 are reported but do not override M1.

## 9. Deliverables
- `LEARNED_EXTRAPOLATOR_RESULTS.csv/.md`
- `extrap_drive_separation.json`
- `extrap_training_log.csv` (per run: epoch, train and validation loss)
- `extrap_hparam_selection.csv`
- `extrap_coverage.csv`
- `extrap_units.parquet`
- model weights in `P2/_extrap_models/`
- Code: `scripts/review_followup3/p2_extrap_common.py`, `p2_extrap_data.py`, `p2_extrap_train.py`, `p2_extrap_eval.py`
