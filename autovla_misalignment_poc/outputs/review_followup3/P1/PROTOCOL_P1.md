# PROTOCOL_P1 — intervention machinery verification (P1-B / P1-C), Stage-0 gate first

Status (2026-10-14, phase 1):
- Done:
  - spec (`INTERVENTION_SPEC.md`)
  - fixed manifest (`stage0_manifest.json`)
  - harness (`scripts/review_followup3/p1/`)
  - **control-5 pilot only** (`pilot_c5_20261014/`)
- Not yet run: controls 1–4. **No parity verdict exists yet.** Every AutoVLA number in P0 stays `PROVISIONAL` until the gate in §7 is run and judged.
- This file is frozen before the gate runs. A change after the gate output exists is a CP3 amendment.

## 1. Questions and hypotheses (pre-specified)

Each control is a parity hypothesis H0: "the manipulation that must not change the computation does not change it". A control *passes* when its pass rule (§6) holds on every manifest unit, row, step and decoding.

| Control | What is compared | Expected | Class |
|---|---|---|---|
| C1 hook no-op | untouched vs experiment hooks installed, no edit (H7: embedding hook + 36 attention wrappers; H8: layer-0 pre-hook + 36 post-hooks with an empty plan) | identical computation | exact |
| C2-hook self-patch | untouched vs hooks that read and write back the same value. H7: attention mask cloned and written back, all span embeddings re-written. H8: **every** patch row sourced from itself, at its own layer and alpha, through the experiment's `apply_patch` arithmetic (`h + α(h − h)`, float32 → bf16) | identical | exact |
| C2-token self-override | each row's override rule (GT / reference / Normal-branch slot) filled with the row's own token, vs the untouched all-own row (`normal` / `OC` / `O1`) | identical inputs; cross-row in batched paths | exact for H7 (B = 1); equivalence for batched paths |
| C3a incremental | experiment span path (crop to P, re-encode span) vs standard incremental KV decoding of append-only rows, B = 1, teacher-forced on the same tokens | equivalent computation, different kernel shapes | equivalence |
| C3b deep invalidation | B = 1 span path vs: crop the KV to the stub start, re-encode stub + prefix actions with unchanged tokens, then the span path | equivalence | equivalence |
| C4 full recompute | the cached-edit path actually used (experiment batch, experiment hooks) vs a no-cache forward from position 0 of the same final edited sequence (B = 1), teacher-forced. H8 `reverse@emb` / `patch_full@emb` are evaluated by token substitution (a layer-0 input patch ≡ token substitution). Position ids must be identical (`get_rope_index` of the full sequence vs `cache_position + rope_deltas`). Also a free-run full-recompute suffix (H7, H9, P1A). | equivalence | equivalence |
| C5 self-reproducibility | c5a same process, same batch; c5b separate process; c5c experiment batch vs B = 1 (batch-shape floor); c5d vs the stored raw of the original runs | c5a/c5b exact; c5c defines T_equiv | pilot (fixes tolerances) |

Decodings are run separately:
- `greedy`: argmax over the 2,048 action rows, first index on ties.
- `t001`: the experiments' T = 0.01 sampling with the experiment seeds (common random numbers; the generator is re-seeded per row).

## 2. Code paths and manifest (fixed rule; not chosen by outcome)

Code paths (details in `INTERVENTION_SPEC.md` §1):

| Path | Experiment | Interventions it carries | Experiment batch |
|---|---|---|---|
| H7 | exp 7, P1-C H7 | Recent-GT, Full-GT | B = 1 |
| H8 | exp 8, P1-C H8 | Reverse-N = `reverse@emb`, plus all 93 rows | B = 93 |
| H9 | exp 9, P1-C H9 | windows, Full-GT | B = 17 |
| P1A | P1-A | self-reference rows U/OC/O1/C1/OC_ref1/OC_refR/OC_refH/O_refH/OC_gtR/OC_gtH | B = 10 |

The P1A path also covers coordinate accumulation (fixed-position GF3, whose prefix includes reference-continuation tokens) and the **rollout-saving path that is the C1 input of §6.2 / Fig 1**: the stored `traj_xyh` is re-decoded from the stored tokens, and the numpy rollout is compared with the decoder.

Manifest: `stage0_manifest.json`, built by `p1_stage0_manifest.py`.

| Item | Rule |
|---|---|
| Population | 208 dev equal-distance scenes |
| Strata | early t* ∈ {0, 1}, middle {2, 3}, late {4–7} |
| Ordering | sha256("P1-STAGE0-20261014\|" + token) |
| Selection | first 3 per stratum |
| Units per scene | `alt0` on H7/H8/H9; family G at t_p = t*, lowest available sector, on P1A; GF3 (t_p = 3) for the 3 early scenes |
| Size | 9 scenes, 21 path-units |
| Group mix | A− 3 / A+ 6 (recorded, not balanced) |
| Positions covered | t* ∈ {1, 2, 4, 5, 7}, t_p ∈ {1, 2, 3, 4, 5, 7}; prefix lengths vary with stub length and t_p (pilot: P = 1,240 for t* = 1) |
| Pilot subset | the first scene of each stratum (c05f89f6… t* 1, a7fb000f… t* 2, ac2da3da… t* 7), i.e. 7 path-units |

Not in the manifest:
- The H32 (DM, exp 32) and P1-B start-position scripts use the same span machinery but are not verified by this gate.
- Units outside the manifest and the original full runs are **not retro-validated** (M6 remaining limit).

## 3. Metrics saved per (control, path, unit, decoding, row, step)

All from `p1b_harness.py` → `comparisons.jsonl`, `decodes.jsonl`, `logits_keyrows.npz`:
- **Hashes:** prefix-ids sha, pixel sha, span input-ids sha per step, position-id sha per step, logits sha per step.
- **Positions:** `pos_full_eq_cached` (C4); attention mask (`ones(B, P+m)`).
- **Logits:** max and mean |Δlogit| over the 2,048 action rows (float64 of the bf16 forward). Only steps whose input and position hashes are equal are compared, so a diverged free run is never compared on different inputs.
- **Tokens:** argmax agreement; sampled-pick agreement; full-suffix agreement; first divergence step; the reference run's top-2 margin at every flip.
- **Decoded output:** max |Δ pose| of the decoded trajectory (m / rad).
- **Overrides:** override positions and *effective* override positions (value ≠ own) per step. For C4 H8, the substituted (target, source) tokens.
- **Hook bookkeeping:** call counts, and a check that no module hook or attention wrapper remains after uninstall.

## 4. Unit of inference

This is a verification, not an effect estimate. The decision rule is "all units pass". No p-values. The cluster structure is irrelevant for pass/fail. The counts of units, rows and steps checked are reported.

## 5. Control-5 pilot (run 2026-10-14) and the tolerances it fixes

Run setup:
- Pilot command: §9.1.
- Two separate processes (`run1`, `run2`).
- 3 pilot scenes, 4 paths, greedy + t001, 5090.
- Environment: bf16, sdpa, deterministic algorithms off, TF32 off, bf16 reduced-precision reduction on.

Tolerance rule (written before the pilot output was read; `p1b_pilot_tolerances.py`):
- **T_exact** = max |Δlogit| over c5a and c5b.
- **T_equiv** = 2 × max |Δlogit| over c5c_tf, across paths.
- **Flip rule:** a flip is explained iff |Δlogit| at that step ≤ the applicable tolerance and the reference top-2 margin ≤ that tolerance.

Pilot results (filled from `pilot_c5_20261014/{pilot_summary.csv, tolerances.json, c5b_cross_process.jsonl, c5d_r8r9_run1.jsonl}`):

Coverage: 7 path-units (3 H7, 3 H8, 3 H9; P1A 3 G + 1 GF3) × 2 decodings × 2 processes. Elapsed 3.7 min per process; peak 13.0 GB.

| Control | Paths | Rows compared | Max \|Δlogit\| | Tokens |
|---|---|---|---|---|
| c5a: same process, same batch | H7, H8, H9, P1A | all rows (H8 93, H9 17, P1A 10, H7 3) | **0.0 (bitwise)** | identical; no flip |
| c5b: separate process | all | 886 row-runs | **0.0**; logits sha equal 886/886 | suffix equal 886/886 |
| c5c_tf: experiment batch vs B = 1, teacher-forced | H8 / H9 / P1A | 18 / 30 / 80 rows; 102 / 170 / 460 steps | 0.375 / 0.375 / 0.328 (≈ 1–3 bf16 ulps); mean 0.045–0.056 | no flip |
| c5c_free: same, free run | H8 / H9 / P1A | as above | as above | suffix equal 18/18, 30/30, 80/80 |
| c5d: vs stored raw | H7 vs exp 7 raw | 18/18 rows | – | equal |
| c5d: vs stored raw | P1A vs P1-A raw (`raw`, `raw_fixedpos`) | 80/80 | – | equal |
| c5d: vs stored raw | H8/H9 vs the R8R9 dev re-runs of exp 8/9 | 60/60 (normal, gt_history, recent_gt for H8; all 17 rows for H9) | – | equal |
| Rollout-saving path (C1 input) | P1A | 10 rows × 4 units | stored `traj_xyh` vs re-decoded ≤ 5e-7 m (1e-6 rounding); numpy rollout vs decoder ≤ 1.5e-6 (max over both 3e-6) | – |

Environment facts recorded:
- `get_prompt` has no `second_per_grid_ts`.
- rope_deltas = −675; prompt_len = 1,209; P = 1,240 at t* = 1.

**Fixed tolerances** (`pilot_c5_20261014/tolerances.json`):
- **T_exact = 0** (bitwise).
- **T_equiv = 0.75** = 2 × 0.375.
- Units: absolute logit over the 2,048 action rows.

Caveat fixed now, before any C1–C4 data:
- Observed top-2 margins go down to 0.0625. Steps whose margin is ≤ 0.75 can flip under an equivalent computation, and such a flip counts as *explained*.
- The pilot observed no such flip in 732 batch-shape-varied steps.
- The gate reports the number of explained flips per control. The free-run suffix agreement is compared with the pilot floor (100% here). A drop in agreement is reported at CP1 together with any logit/hash failure (§6).
- **c5c is a property of the experiments themselves.** The same input gives ulp-level different logits at a different batch size. This is why Normal agreement across harnesses is 88–89%, not 100% (M-A3): any pair of rows from different batch compositions is not exactly paired.

## 6. Pass / fail rules (per control × path × decoding)

**Exact class** (C1, C2-hook, C2-token on H7). Every one of these must hold:
- every compared step has |Δlogit| ≤ T_exact (= 0, i.e. bitwise);
- identical argmax and sampled tokens, identical full suffix and decoded poses;
- all input, position, prefix and pixel hashes equal;
- no effective override in a self-patch;
- hooks fully removed afterwards.

**Equivalence class** (C2-token on H8/H9/P1A, C3a, C3b, C4-tf). Every one of these must hold:
- every compared step has |Δlogit| ≤ T_equiv;
- no unexplained flip;
- the required hashes are equal;
- for C4: `pos_full_eq_cached` is true at every step, and the substituted source token of `reverse@emb` equals the Normal row's c_{k−1}.

Equivalence class, reported but not a failure on its own:
- free-run suffix disagreement (C4-free, c5c_free). It is compared with the c5c_free floor.
- A disagreement rate above the floor on a path is escalated to CP1 **only if** it is accompanied by any logit/hash failure above.

**Path verdict:**
- PASS = every control of the path passes on every manifest unit and both decodings.
- FAIL = any failure.
- Results go to `NOOP_CACHE_AUDIT.csv/.md`.

## 7. CP1 stop rule

If any path FAILS a no-op control (C1, C2, C3, C4):
1. Stop. Do not start large follow-up runs of that intervention, and do not fix the code before approval. A small diagnostic reproduction on the manifest units is allowed.
2. Report (`CP1_REPORT.md`):
   - the failing control, path, rows and steps;
   - the magnitude against T_exact / T_equiv;
   - a cause hypothesis;
   - the affected runs and the manuscript parts withheld (table below);
   - a proposed fix and its rerun cost (table below).
3. After approval: fix, re-evaluate the original main contrasts of the affected runs **on the same manifest** with the changed code, and keep old and new results side by side. A few verified traces are never claimed to validate all earlier runs.

| Failing path | Affected runs (raw location) | Manuscript parts withheld until re-verified | Rerun cost on 5090 (measured run time) |
|---|---|---|---|
| H7 (Recent-GT / Full-GT) | exp 7 `outputs/action_history_causal` (19.5 min); P1-C H7 `P1C…/run_20261008/mech/action_history_causal` (66.9 min); exp 32 / P1-C H32 share the loop logic (5.4 / 18.8 min) | §5 entire, Table 4, App. B, Figure 2 | ≈ 1.8 h |
| H8 (Reverse-N) | exp 8 dev re-run `R8R9_raw_recovery/run_20261008/raw/exp8` (43.1 min); P1-C H8 `mech/prev_action_state_patching` (63.3 min) | §5 entire, Table 4, App. B, Figure 2 | ≈ 1.8 h |
| H9 (windows / Full-GT) | exp 9 dev re-run `R8R9…/raw/exp9` (18.4 min); P1-C H9 `mech/temporal_feedback_window` (27.6 min) | §5, Figure 2 (R9) | ≈ 0.8 h |
| P1A self-reference / coordinate accumulation | P1-A `P1A…/run_20261008/{reference (6.3), raw (23.8), raw_fixedpos (27.7 min)}`; P1-B start position (38.1 min, same span machinery) | §6.1, §6.3 | ≈ 1.6 h |
| P1A rollout-saving path (C1 input) | P1-A raw `traj_xyh` | §6.2, Figure 1 | CPU re-decode only (minutes) unless the tokens themselves are wrong |

If an exact-class control fails only by batch-composition numerics, i.e. it is within T_equiv: this is still reported at CP1. The proposed fix is "disclose and re-pair within the same batch". No result is changed until approval.

## 8. SpatialVLA P1-B / P1-C (Tier 1; protocol fixed here, code not yet written)

Model order: AutoVLA first. SpatialVLA runs after the AutoVLA gate verdict, so a CP1 stop on AutoVLA does not block it. Machinery: P1-D runner (`p1d_runner.py`: reconfigure resets, `decode_fixed` with batch 8, `Exec2`).

### 8.1 Manifest
| Item | Rule |
|---|---|
| Tasks × configs | 2 tasks (pick_coke_can, move_near) × 2 configs (r1e4, r4e1) |
| Physical initial states | 5 per task-config, chosen by sha256("P1-SVLA-20261014\|" + task + episode_id/seed) order from the P1-D unit list (`build_units.py`); never by outcome |
| Size | 20 units |
| Pilot (control 5 only, fixes tolerances) | first 2 units per task, each N twice in-process + once in a fresh process |

### 8.2 Arms per unit
All arms start from an identical full state. Fingerprints are taken at t = 0, T0 = 8 and every step.

| Arm | Control | Definition |
|---|---|---|
| U0 | C5 | untouched natural from reset |
| U0′ | C5 | U0 rerun in-process |
| U0″ | C5 | U0 rerun in a fresh process |
| U1 | C1 | override hook installed, returns None |
| U2 | C2 | null (g, g) through the intervention path (exists in P1-D) |
| U3 | C3 / P1-C | N branched at T0 via restore (replay + executor deepcopy) and continued |
| U4 | C4 analogue | decoded chunks at T0 re-generated by B = 1 `decode_chunks` and by HF `generate` on the identical observation (token level) |

Then the four main arms N/F/C/R from the identical branch-point state. The pass requirement is identical fingerprints for all arms up to T0.

### 8.3 Restore-scope table
| State item | exp 34/35 | P1-D | Gate: restore method | Fingerprint access (to be verified at implementation) | Tolerance |
|---|---|---|---|---|---|
| Robot qpos / qvel / qacc | reset(seed), no reconfigure | reset + replay; in `get_state()` hash | reset(reconfigure) + replay; or `set_state` if replay is not exact | `u.agent.robot.get_qpos/get_qvel/get_qacc()`, `u.get_state()` | from SVLA pilot (expected bitwise) |
| Object poses / velocities | as above (reset drift ≤ 0.26 mm) | in `get_state()` hash | as above | `u.get_state()` (actors [p, q, v, ω]) | pilot |
| Controller targets / integrator | not restored | implicit (re-simulation), not hashed | replay; **hash added** | `u.agent.controller.controllers[*]._target_qpos, _start_qpos, _last_drive_qpos_targets, _target_pose, _interpolation_path(_vel), _integral, _step`; `get_drive_target()` | bitwise |
| Gripper | not restored | implicit | replay + sticky state in executor copy | `u.agent.get_gripper_closedness()`, drive targets; executor `sticky_on, repeat, sticky_action, prev_grip` | bitwise |
| Physics time / step counter | – | implicit (replay length) | replay; **hash added** | `u._elapsed_steps` (+ scene time if exposed) | exact |
| Env RNG | reset(seed) | reset(seed), not hashed | reset(seed); **hash added** | `u._episode_seed, u._main_seed, u._episode_rng.get_state(), u._main_rng.get_state()` | exact |
| Policy RNG | none (greedy) | none | none; recorded | – | – |
| Policy / ensemble / action buffer | fresh per episode | deepcopy of executor at T0 | deepcopy; **hash added** | executor `chunks` (t_gen, 4 × 7 float64), cf dict | bitwise |
| Render state | lavapipe, not hashed | 224-px image sha per step | same + camera pose | obs image sha before and after resize; `u._cameras[*].camera.get_model_matrix()`; overlay / episode_id / urdf_version | bitwise |
| Actual TCP | exp 35 only | logged | logged every step for every arm | `u.tcp.pose.p, .q` | bitwise to tolerance |

### 8.4 Rules and cost
- **Pass:** U0 ≡ U0′ ≡ U1 ≡ U2 (float64 actions, every fingerprint, image sha, TCP) over all 80 steps.
- **U3:** fingerprints equal to U0 at every step after T0. The four main arms are identical up to T0.
- A tolerance > 0 is allowed only where the SVLA pilot shows nonzero rerun noise, at 2 × the pilot maximum.
- **Failure:** CP1 (withholds §7.3, §7.4, Table 5, App. D).
- **Cost:** about 20 × 9 + 12 pilot ≈ 190 episodes. At the P1-D throughput (≈ 120 episodes/h, 8 workers, CPU-simulator bound, ≈ 10 GB RAM, ≥ 13 GB free on the 5090) this is **≈ 1.6–2 h** wall, plus about half a day to write the fingerprint/runner extension.

## 9. Run commands and GPU time

Common environment for all commands:
```
source /root/miniforge3/etc/profile.d/conda.sh && conda activate autovla
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1        # RTX 5090 only
```
Before each model load, wait until ≥ 14 GB is free on GPU 1 (`run_c5_pilot.sh` `wait_gpu`). Never run two model loads concurrently. Use `nice -n 10`.

### 9.1 Control-5 pilot (done)
```
bash scripts/review_followup3/p1/run_c5_pilot.sh        # run1 + run2 + c5b compare
python scripts/review_followup3/p1/p1b_pilot_tolerances.py outputs/review_followup3/P1/pilot_c5_20261014
python scripts/review_followup3/p1/p1b_c5d_r8r9.py outputs/review_followup3/P1/pilot_c5_20261014/run1 outputs/review_followup3/P1/pilot_c5_20261014/c5d_r8r9_run1.jsonl
```

### 9.2 (a) AutoVLA Stage-0 gate (after this protocol is frozen)
```
S=scripts/review_followup3/p1; O=outputs/review_followup3/P1/gate_stage0
nice -n 10 python $S/p1b_harness.py --manifest outputs/review_followup3/P1/stage0_manifest.json --out $O/run1 \
     --controls c1,c2,c3,c4,c5 --paths H7,H8,H9,P1A --decodings greedy,t001
nice -n 10 python $S/p1b_harness.py --manifest outputs/review_followup3/P1/stage0_manifest.json --out $O/run2 \
     --controls c5 --paths H7,H8,H9,P1A --decodings greedy,t001
python $S/p1b_compare_runs.py $O/run1 $O/run2 $O/c5b_cross_process.jsonl
```
Then a judge script (to be written before the run) applies §6 with `pilot_c5_20261014/tolerances.json` and writes `NOOP_CACHE_AUDIT.csv/.md`.

Estimated GPU time, from pilot timing:

| Item | Time |
|---|---|
| Control 5 alone | H7 ≈ 4 s, H8 ≈ 6 s, H9 ≈ 8 s, P1A ≈ 13 s per unit × decoding |
| Gate, all controls | ≈ 4–6× the C5 cost |
| run1 | 9 × (H7 + H8 + H9) + 12 × P1A, × 2 decodings ≈ 60–80 min |
| run2 | ≈ 12 min |
| Model loads | 2 × ≈ 3 min |
| **Total** | **≈ 1.3–1.8 h on the 5090** (longer if the external process holds the GPU) |

### 9.3 (b) SpatialVLA P1-B / P1-C
Commands are to be added once the fingerprint extension is written. They are based on `outputs/review_followup/P1D_dose_control/run_20261008/launch.sh`:
```
source /root/VLA/simpler/env.sh; nice -n 10 python scripts/review_followup3/p1/svla_noop_restore.py --units-file <manifest> \
     --configs r1e4,r4e1 --arms U0,U0rep,U1,U2,U3,N,F,C,R --workers 8 --out outputs/review_followup3/P1/svla_gate
```
Estimate ≈ 1.6–2 h wall (§8.4).

### 9.4 (c) Impromptu and OpenVLA
**Spec only** (`INTERVENTION_SPEC.md` §3–4). No P1-B run unless P3 (Impromptu matched Reverse) or a new OpenVLA run is approved.

If they are run, the minimal harness is:
- Impromptu: a self-override row writing the own waypoint text back; the token-id round-trip check; B = 1 vs batch. Estimate ≈ 1 h GPU for 20 scenes.
- OpenVLA: δ = 0 and window rewrite with pb = g; B = 1 vs batch. Estimate ≈ 0.5 h GPU.

Their existing parity is poor in places (`INTERVENTION_SPEC.md` M-I1, M-O1). Until a run happens, §7.1 and §7.2 rely on unverified machinery.

## 10. What a PASS will and will not mean
- **It will mean:** for the 21 manifest path-units at HEAD 817d166 on the 5090 in bf16, the hooks, overrides, cache invalidation and cached-edit path do not change the computation beyond the measured numeric floor. In particular:
  - Recent-GT / Full-GT / Reverse-N / self-reference differences are differences of the intended context, not artefacts.
  - Reverse-N equals its token-substitution definition.
- **It will not:**
  - retro-validate runs outside the manifest, H32 or the P1-B start-position runs;
  - make cross-harness "Normal" rates interchangeable (batch composition, M-A3);
  - make AutoVLA, Impromptu and SpatialVLA "Reverse" the same treatment (M-A1, `INTERVENTION_SPEC.md` §5).
