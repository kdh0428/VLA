# STATE_RESTORE_AUDIT — SpatialVLA P1-B / P1-C (PROTOCOL_P1.md §8)

Tolerance (pilot): bitwise = True, T_action = 0.0, T_tcp = 0.0 m.

| control | pass / n |
|---|---|
| C1 | 20/20 |
| C2 | 20/20 |
| C3_restore | 20/20 |
| C5a | 20/20 |
| C5b | 20/20 |
| P1C_branch_restore | 60/60 |

Strict pre-fixed rule (executor fingerprint incl. the shadow cf dict): 20 arm comparisons fail (['C2']); see AMENDMENT_P1_01.md A5.

C4 analogue at T0 (report only): B=1 decode == batch-8 chunk 17/20; HF generate == batch-8 chunk 16/20.

Failures: 0


## Notes (added by hand; numbers from `svla_gate/`)

**Run setup**
- Units: `svla_manifest.json`, built by `svla_manifest.py`. 2 tasks × 5 physical initial states (hash order; move_near distinct episode_id) × 2 configs (r1e4, r4e1) = 20 units.
- Episodes:

| Run | Episodes | Arms | Wall time |
|---|---|---|---|
| run1 | 160 | U0 natural, U0′ natrep, U1 hooknone, U2 nullfull, U3 branchN, F, C, R | 29.3 min |
| run2 (separate process) | 20 | U0″ | 5.9 min |
| Pilot | 12 | – | 5.8 + 3.9 min |

- Settings: RTX 5090, 4 SAPIEN workers (A4), `decode_fixed` batch 8.
- Pilot tolerance: bitwise. Reruns in the same process and in a separate process were bytes-equal in actions and every fingerprint component over 80 steps, 8/8 pairs.

**Results per arm** (all compared with U0 on the same unit; "all components" = every fingerprint component in the scope table below; "TCP diff" = max |Δ TCP|)

| Arm | Control | Steps compared | Result | Units |
|---|---|---|---|---|
| U0′ | C5a | 80 | bitwise equal: actions (float64 bytes), all components, executor; TCP diff 0 | 20/20 |
| U0″ | C5b | 80 | bitwise equal, as above | 20/20 |
| U1 (hook installed, no edit) | C1 | 80 | bitwise equal, as above | 20/20 |
| U2 (null (g, g) + shadow row) | C2 | 80 | actions, all simulator components and exec/ctx tokens bitwise equal; TCP diff 0 | 20/20 |
| U3 (restore by reset + replay + executor deepcopy at T0 = 8) | C3 / P1-C | 72 (t ≥ 8) | restore ok; bitwise equal over all 72 steps | 20/20 |
| F / C / R branches | P1-C | prefix up to T0 | identical full-state fingerprint at every replayed step and identical executor state at T0 | 60/60 |

**U2 caveat (A5):** the executor fingerprint differs from t = 8 because it includes the shadow-cf dict that null mode fills by design. Under the strict pre-fixed rule this is **0/20**. Under the A5 policy-state check it is 20/20.

**C4 analogue at T0** (report only; token level, same observation, compared with the experiment's batch-8 chunk):
- `decode_chunks` at B = 1: equal 17/20.
- HF `generate` (predict_action): equal 16/20.
- Mismatches: coke 200 r4e1, coke 214 r4e1, move_near 1043 r4e1 (both B = 1 and generate), and move_near 1092 r1e4 (generate only).
- Interpretation: this is the batch-shape numeric floor, the same phenomenon as AutoVLA c5c. Closed-loop arms are internally consistent because every arm decodes with the same fixed 8-row batch. Equality with `generate` / other batch sizes does not hold in general, so the "identical to generate" wording in the exp 34 docs remains wrong (INTERVENTION_SPEC M-S1/M-S2).

**Restore scope verified** (component hashes per step):
- sim state (`get_state`, all actors and articulations), `agent.get_state`;
- robot qpos / qvel / qacc / drive targets;
- arm and gripper controllers: `_target_qpos`, `_start_qpos`, `_last_drive_qpos_targets`, `_target_pose`, `_interpolation_path(_vel)`, `_step`, `_sim_steps`;
- TCP pose, gripper closedness;
- env and TimeLimit elapsed steps;
- episode/main seeds and both RandomState states;
- scene timestep, camera model matrices;
- raw and 224-px images;
- executor (chunks, cf, sticky state).

Not covered: the per-arm physical layout beyond these units. Scope: 20 units at HEAD 817d166 + these scripts; it does not retro-validate exp 34/35, whose resets had no reconfigure and showed reset drift (review_followup2).

## C2 re-verification with the split executor fingerprint (AMENDMENT_P1_02, CP2 decision A5)

Run `svla_gate/a5_rerun/` (natural + nullfull, same 20 units, 4 workers, RTX 5090). Pass rule fixed before the run: executed actions bytes-equal + every simulator fingerprint component equal + policy-state hash equal at all 80 steps (tolerance 0); shadow-cf bookkeeping hash reported only.

| control | pass / n |
|---|---|
| C2_A5_split | 20/20 |

- actions bytes-equal: 20/20; simulator components all equal: 20/20; policy-state hash equal at every step: 20/20; max TCP diff 0.0 m.
- bookkeeping (reported only): shadow-cf hash differs in 20/20 units, first at t = [8].
- Earlier rows are kept unchanged above: strict pre-fixed rule 0/20, A5 post-hoc policy-state check 20/20.
- Failures: 0
