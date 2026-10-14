# SPATIAL_STATE_EXPANSION_PROTOCOL — P4 (DRAFT for CP2; not frozen, not run)

**Status: DRAFT.**
- Nothing has been sampled in a simulator and no policy has been run.
- The only computation so far is a pure-geometry pre-check (`planning/layout_geometry.csv`) and the power simulation (`POWER_PLAN.md`).
- After CP2 approval, this file and the state-sampling code are committed **before** any P4 state or rollout exists.

**Preconditions:**
- the SpatialVLA P1 decision on A5 (CP2);
- CP2 approval of the precision target (§7);
- the feasibility pilot (§6) kept separate from the confirmatory data.

**Scope.**
- Tag: manipulation / **SpatialVLA-4B** / closed loop / SimplerEnv Google Robot.
- Complementary task-level evidence; not a replication of the driving result.
- Two existing tasks only: `google_robot_pick_coke_can` and `google_robot_move_near`. No new task and no new model.

**Name of the result.** All results are **"SimplerEnv variant states"**, not the original visual-matching benchmark. The real-to-sim correspondence that SimplerEnv validated for its benchmark layouts is **not validated for the new layouts**, and the manuscript must say so.

## 1. Why new physical states

- In exp 34, exp 35 and P1-D, move_near uses only **10 physical layouts**: 5 triplets × 2 xy configurations of `MoveNearGoogleBakedTexInSceneEnvV1`. The 60 "episodes" are those 10 layouts × 6 instructions, and the P1-D 120 units span 10 layouts × instruction × overlay.
- Re-analysing instructions or seeds cannot add independent physical states; P0-B has finished that analysis.
- pick_coke_can already draws a continuous xy per seed, so new coke states only need new seeds.

## 2. Validity definition (fixed before sampling)

### 2.1 Sampling region and pose range: where they come from

All ranges are read from SimplerEnv code. None is invented.

**pick_coke_can** (`GraspSingleOpenedCokeCanInScene-v0`, prepackaged config):
- **Position:** the env's own sampler, `GraspSingleInSceneEnv._initialize_actors`, draws xy ~ U([−0.35, −0.02], [−0.12, 0.42]). This is the same rectangle the visual-matching evaluation covers with its 5 × 5 grid (`scripts/rt1_pick_coke_can_visual_matching.sh`: obj-init-x −0.35…−0.12, obj-init-y −0.02…0.42).
- **Orientation:** drawn from the env's `orientations_dict` {upright, laid_vertically, lr_switch}.
- **Overlay:** `urdf_version` from the 4 visual-matching variants (`_additional_prepackaged_config_reset`).
- **Robot base:** (0.35, 0.20), identity rotation.
- The generator is identical to P1-D's coke states; only the seeds are new (§3).

**move_near** (`MoveNearGoogleBakedTexInScene-v1`, prepackaged config):
- **Kept from the benchmark:**
  - the object sets = the 5 benchmark triplets;
  - each object's benchmark initial quaternion (`obj_init_quat_dict`);
  - the instruction set = the 6 ordered (source, target) pairs per triplet;
  - the overlay set = the 4 `urdf_version` variants;
  - the robot pose: (0.35, 0.21), yaw −0.09.
- **Changed:** the three object xy positions, each drawn uniformly in **R_simpler** = [−0.35, −0.12] × [−0.02, 0.42]. This is SimplerEnv's object-sampling rectangle for the same scene (`google_pick_coke_can_1_v4`) and the same table (see coke above).
- **Why not the benchmark box:** the axis-aligned box of the 6 benchmark positions, R_bench = [−0.33, −0.13] × [0.04, 0.34], cannot host a new layout with benchmark-like spacing. The geometry pre-check accepted 0 of 200,000 candidates per triplet (`planning/layout_geometry.csv`).
- **Implementation:** `MoveNearGoogleInSceneEnv.reset` overwrites `init_xys` from `episode_id`. P4 therefore needs a thin subclass whose `reset` passes the sampled `init_xys` / `init_rot_quats` to `MoveNearInSceneEnv._initialize_actors`, which already accepts them. That is new code, and the reason these are "variant" states.

### 2.2 Geometric acceptance (before any physics; pure function of the candidate)

**move_near:**
- Pairwise object distance ≥ 0.20 m.
- **Every** one of the 3 object pairs must start farther apart than SimplerEnv's own `near_tgt_obj` threshold, which is half_xy_diag(src) + half_xy_diag(tgt) + 0.10 m (bbox rotated by the init quat, as in `evaluate()`). The extra margin must be at least **0.032 m**. That margin is the smallest margin present in the benchmark layouts themselves (computed in `layout_geometry_meta.json`), so no instruction can be satisfied, or nearly satisfied, at reset.
- New vs every benchmark layout: some object moves ≥ 0.03 m. That is 100× the 0.26 mm reset drift seen in exp 34/35, and **reset jitter is never counted as a new state**.

**pick_coke_can:** the can xy must differ by ≥ 0.03 m from every coke state used in exp 34, exp 35, P1-D and the P1 audit with the same orientation; otherwise it is a duplicate.

**Feasibility of the supply** (`planning/layout_geometry.csv`):
- Acceptance in R_simpler is 0.14–0.56 % per triplet.
- From 200,000 candidates per triplet, ≥ 2,556 mutually distinct move_near layouts are available across the 5 triplets (greedy count, all pairs ≥ 0.03 m apart).
- This is a lower bound and is enough for the 400–600 states per task in §7.

### 2.3 Collision and instability exclusion (physics, no policy)

Each accepted candidate is reset once on the CPU-side simulator, with the env's own settle routine:
1. lock rotation about x and y, settle 0.5 s;
2. unlock, settle 0.5 s;
3. a further 1.5 s if the objects are still moving.

The candidate is **rejected** if any object meets any of these:
- xy displacement from its spawn > 0.02 m;
- settled height outside ±0.01 m of its resting height on the table, or below the table;
- tilt (non-yaw rotation) change > 10°;
- residual linear velocity > 1e-3 m/s or angular velocity > 1e-2 rad/s;
- for move_near, any object–object contact at the end of settle.

Rejected candidates and their reasons are logged. Settled poses are the recorded geometry.

### 2.4 Task feasibility

- **Geometric feasibility** (§2.2): inside the SimplerEnv object region the robot is evaluated on, not satisfied at reset, objects on the table and stable.
- **No policy-outcome filtering.** A state where SpatialVLA fails is a valid state. Excluding states by any rollout outcome (any arm, pilot included) is forbidden.
- **Physical solvability.** A scripted-oracle check is **not** used: there is no oracle controller for move_near in the repo, and building one is out of scope. As a proxy, the separate pilot (§6) reports the natural (N) success rate on its own states. That is descriptive and is not used to select confirmatory states.

### 2.5 Sample source and duplicate rule (fixed)

- **RNG:** `numpy.random.default_rng(20261016)`, one independent stream per task.
- **Candidate generation:** in a fixed order; each candidate is accepted or rejected by §2.2–2.3 in that order until n states per task are accepted.
- **move_near draw per accepted layout:** triplet (uniform over 5), overlay (uniform over 4), and **one** instruction (uniform over 6). So **one unit per physical state**, and every state is its own cluster.
- **Freezing:** the accepted manifest (task, objects, settled poses, quats, overlay, instruction, seed, state fingerprint) is written and sha256-frozen before any confirmatory rollout.
- **Feasibility-pilot states** use a separate stream (seed 20261017) and are never in the confirmatory manifest.

## 3. Arms, restore and fingerprints

**Arms per state, all from one identical restored state:**
- {N, F, C, R} × {r1e4, r4e1}, with the P1-D definitions unchanged;
- injection chunks at steps 8, 12, 16, 20; T0 = 8 branch point (`scripts/review_followup/p1d/p1d_runner.py`);
- 8 episodes per state.

**Runner.** The P1-audited runner `scripts/review_followup3/p1/svla_noop_restore.py`:
- reset with reconfigure, then replay;
- executor deepcopy at T0;
- per-step fingerprints over the full restore scope of `P1/STATE_RESTORE_AUDIT.md`;
- `decode_fixed` batch 8.

If A5 is decided as "re-run", the fingerprint must hash the policy state and the shadow-cf dict separately before P4.

**Controls.** On a fixed 5 % of states (the first 5 % in manifest order), additionally run U0′ (rerun) and U2 (null arm). A failure stops the run and is reported (CP1 rule).

**Saved per episode:**
- state fingerprint at t = 0 and T0, every-step hashes;
- actual TCP;
- executed actions;
- success at step 80 (intention-to-treat);
- geometry metadata.

## 4. Estimands and analysis

**Primary.** Per task (coke, move_near) × config (r1e4, r4e1) × contrast (δR = S(R) − S(N); δC = S(C) − S(F)):
- the state-level paired difference, every state its own cluster;
- point estimate and 95 % CI (exact sign-flip inversion; bootstrap as a check);
- raw p and Holm p (two-sided difference test);
- **TOST equivalence at the approved margin** (§7).

**Primary family:** Holm over the 8 per-task equivalence tests (m = 8).

**Secondary (each reported separately, never replacing the per-task results):**
- task × intervention difference (coke − move_near) per contrast with its 95 % CI and ±10 pp TOST (unadjusted);
- the equal-task mean, in its own table;
- ±5 / ±7.5 pp CI-based precision statements (descriptive).

**Never:**
- pool the 8 per-task contrasts into 4 pooled contrasts;
- call a non-significant result "effect absent";
- add states after looking at results.

## 5. Stopping and missing data

- Fixed n.
- No interim look at effects. Throughput and parity are monitored.
- Stop only for:
  - a parity/restore failure (→ CP1 report);
  - an infrastructure failure.
- Incomplete states stay in the common cohort with a recorded reason. Missing ≠ failure. A contrast uses the states where both arms exist, and the count is reported.

## 6. Feasibility pilot (separate; not confirmatory)

- 16 states per task from the pilot stream × 8 arms = 256 episodes.
- **Purpose:**
  - measure throughput (episodes/min with 4 workers; A4 showed that 8 workers thrash the page cache);
  - measure rejection rates of §2.2–2.3;
  - check the variant subclass;
  - fix the worker count.
- Pilot outcomes are not analysed for effects and are not reused.

## 7. Sample size, precision target and cost (details in `POWER_PLAN.md`)

**What ±5 pp per task would cost.** Under a true effect of 0, with the planning discordance from P1-D, ±5 pp per task with Holm m = 8 at 80 % power needs **≈ 1,500 states per task**:
- 24,556 episodes;
- 75 GPU h at the P1-audit rate (5.46 episodes/min, 4 workers) up to 209 GPU h at the P1-D rate (1.96 episodes/min, 8 workers, GPU shared).

And it is **unattainable** when the true effect is ≥ 2.5 pp: power 0.28 even at 3,000 states.

**This exceeds a 48-GPU-h budget, so the narrowed target is proposed for human approval at CP2, before any data:**
- **Confirmatory:** ±10 pp per task, Holm m = 8, **n = 500 states per task**.
  - Power (all 8 equivalent): 0.95 (true 0), 0.80 (heterogeneous ±10 pp state effects), 0.73 (discordance 0.35).
  - 8,356 episodes incl. 5 % controls and the 2 × 16-state pilot.
  - **≈ 25.5 GPU h (P1-audit rate) – 71 GPU h (P1-D rate).**
- **Alternatives**, chosen by the human, not by data:
  - n = 400: power 0.85 under true 0; 20.6–57.4 GPU h;
  - n = 600: 0.99 under true 0, 0.89 under stress; 30.5–85 GPU h.
- **Throughput rule (not outcome-dependent):** if the pilot-measured throughput implies more than 48 GPU h for the approved n (i.e. < 2.9 episodes/min for n = 500), stop and ask the human. Do not shrink n silently.
- **±7.5 / ±5 pp** are reported only as CI half-widths. Expected 95 % half-widths at n = 500: 3.0–4.9 pp per contrast.
- **Disk:** < 0.5 GB expected (P1-D: 92 MB for 4,800 episodes). The container has ≈ 13 GB free.

## 8. Remaining limits (must stay in the manuscript)

- Only two tasks.
- Variant states are not validated against real-robot outcomes.
- An equivalence result bounds the effect only at the approved margin. It says nothing about effects inside the margin, or about other tasks or models.
