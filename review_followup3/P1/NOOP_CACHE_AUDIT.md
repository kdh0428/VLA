# NOOP_CACHE_AUDIT (Stage-0 gate)

Tolerances: T_exact = 0.0, T_equiv = 0.75 (from /root/VLA/autovla_misalignment_poc/outputs/review_followup3/P1/pilot_c5_20261014/tolerances.json).

| path | verdict |
|---|---|
| H7 | PASS |
| H8 | PASS |
| H9 | PASS |
| P1A | PASS |

| control | path | decoding | class | rows | fail | max abs dlogit | flips (all explained unless failed) | suffix-equal rows |
|---|---|---|---|---|---|---|---|---|
| c1 | H7 | greedy | exact | 27 | 0 | 0 | 0 | 27 |
| c1 | H7 | t001 | exact | 27 | 0 | 0 | 0 | 27 |
| c1 | H8 | greedy | exact | 837 | 0 | 0 | 0 | 837 |
| c1 | H8 | t001 | exact | 837 | 0 | 0 | 0 | 837 |
| c2_hook | H7 | greedy | exact | 27 | 0 | 0 | 0 | 27 |
| c2_hook | H7 | t001 | exact | 27 | 0 | 0 | 0 | 27 |
| c2_hook | H8 | greedy | exact | 837 | 0 | 0 | 0 | 837 |
| c2_hook | H8 | t001 | exact | 837 | 0 | 0 | 0 | 837 |
| c2_token | H7 | greedy | exact | 27 | 0 | 0 | 0 | 27 |
| c2_token | H7 | t001 | exact | 27 | 0 | 0 | 0 | 27 |
| c2_token | H8 | greedy | equiv | 837 | 0 | 0 | 0 | 837 |
| c2_token | H8 | t001 | equiv | 837 | 0 | 0 | 0 | 837 |
| c2_token | H9 | greedy | equiv | 153 | 0 | 0 | 0 | 153 |
| c2_token | H9 | t001 | equiv | 153 | 0 | 0 | 0 | 153 |
| c2_token | P1A | greedy | equiv | 120 | 0 | 0 | 0 | 120 |
| c2_token | P1A | t001 | equiv | 120 | 0 | 0 | 0 | 120 |
| c3a_incremental | H7 | greedy | equiv | 18 | 0 | 0.4062 | 0 | 18 |
| c3a_incremental | H7 | t001 | equiv | 18 | 0 | 0.4062 | 0 | 18 |
| c3a_incremental | H8 | greedy | equiv | 18 | 0 | 0.4062 | 0 | 18 |
| c3a_incremental | H8 | t001 | equiv | 18 | 0 | 0.4062 | 0 | 18 |
| c3a_incremental | H9 | greedy | equiv | 45 | 0 | 0.4062 | 0 | 45 |
| c3a_incremental | H9 | t001 | equiv | 45 | 0 | 0.4062 | 0 | 45 |
| c3a_incremental | P1A | greedy | equiv | 96 | 0 | 0.4375 | 1 | 96 |
| c3a_incremental | P1A | t001 | equiv | 96 | 0 | 0.375 | 2 | 96 |
| c3b_deep_invalidate | H7 | greedy | equiv | 27 | 0 | 0.375 | 1 | 27 |
| c3b_deep_invalidate | H7 | t001 | equiv | 27 | 0 | 0.375 | 1 | 27 |
| c3b_deep_invalidate | H8 | greedy | equiv | 27 | 0 | 0.375 | 1 | 27 |
| c3b_deep_invalidate | H8 | t001 | equiv | 27 | 0 | 0.375 | 0 | 27 |
| c3b_deep_invalidate | H9 | greedy | equiv | 45 | 0 | 0.3359 | 3 | 45 |
| c3b_deep_invalidate | H9 | t001 | equiv | 45 | 0 | 0.3359 | 3 | 45 |
| c3b_deep_invalidate | P1A | greedy | equiv | 120 | 0 | 0.4375 | 5 | 120 |
| c3b_deep_invalidate | P1A | t001 | equiv | 120 | 0 | 0.4375 | 6 | 120 |
| c4_free | H7 | greedy | report | 27 | 0 | 0.3828 | 4 | 23 |
| c4_free | H7 | t001 | report | 27 | 0 | 0.3828 | 3 | 24 |
| c4_free | H9 | greedy | report | 45 | 0 | 0.4375 | 8 | 37 |
| c4_free | H9 | t001 | report | 45 | 0 | 0.4375 | 5 | 36 |
| c4_free | P1A | greedy | report | 120 | 0 | 0.5 | 11 | 110 |
| c4_free | P1A | t001 | report | 120 | 0 | 0.5 | 11 | 108 |
| c4_tf | H7 | greedy | equiv | 27 | 0 | 0.3828 | 4 | 27 |
| c4_tf | H7 | t001 | equiv | 27 | 0 | 0.3828 | 4 | 27 |
| c4_tf | H8 | greedy | equiv | 54 | 0 | 0.3906 | 6 | 54 |
| c4_tf | H8 | t001 | equiv | 54 | 0 | 0.4375 | 6 | 54 |
| c4_tf | H9 | greedy | equiv | 45 | 0 | 0.4375 | 8 | 45 |
| c4_tf | H9 | t001 | equiv | 45 | 0 | 0.4375 | 8 | 45 |
| c4_tf | P1A | greedy | equiv | 120 | 0 | 0.5 | 11 | 120 |
| c4_tf | P1A | t001 | equiv | 120 | 0 | 0.5 | 11 | 120 |
| c5a | H7 | greedy | exact | 27 | 0 | 0 | 0 | 27 |
| c5a | H7 | t001 | exact | 27 | 0 | 0 | 0 | 27 |
| c5a | H8 | greedy | exact | 837 | 0 | 0 | 0 | 837 |
| c5a | H8 | t001 | exact | 837 | 0 | 0 | 0 | 837 |
| c5a | H9 | greedy | exact | 153 | 0 | 0 | 0 | 153 |
| c5a | H9 | t001 | exact | 153 | 0 | 0 | 0 | 153 |
| c5a | P1A | greedy | exact | 120 | 0 | 0 | 0 | 120 |
| c5a | P1A | t001 | exact | 120 | 0 | 0 | 0 | 120 |
| c5b | H7 | greedy | exact | 27 | 0 | 0 | 0 | 0 |
| c5b | H7 | t001 | exact | 27 | 0 | 0 | 0 | 0 |
| c5b | H8 | greedy | exact | 864 | 0 | 0 | 0 | 0 |
| c5b | H8 | t001 | exact | 864 | 0 | 0 | 0 | 0 |
| c5b | H9 | greedy | exact | 198 | 0 | 0 | 0 | 0 |
| c5b | H9 | t001 | exact | 198 | 0 | 0 | 0 | 0 |
| c5b | P1A | greedy | exact | 240 | 0 | 0 | 0 | 0 |
| c5b | P1A | t001 | exact | 240 | 0 | 0 | 0 | 0 |
| c5c_free | H8 | greedy | report | 27 | 0 | 0.4375 | 1 | 26 |
| c5c_free | H8 | t001 | report | 27 | 0 | 0.375 | 0 | 21 |
| c5c_free | H9 | greedy | report | 45 | 0 | 0.375 | 0 | 45 |
| c5c_free | H9 | t001 | report | 45 | 0 | 0.375 | 0 | 41 |
| c5c_free | P1A | greedy | report | 120 | 0 | 0.4688 | 2 | 118 |
| c5c_free | P1A | t001 | report | 120 | 0 | 0.4688 | 2 | 118 |
| c5c_tf | H8 | greedy | equiv | 27 | 0 | 0.4375 | 1 | 27 |
| c5c_tf | H8 | t001 | equiv | 27 | 0 | 0.375 | 0 | 27 |
| c5c_tf | H9 | greedy | equiv | 45 | 0 | 0.375 | 0 | 45 |
| c5c_tf | H9 | t001 | equiv | 45 | 0 | 0.375 | 0 | 45 |
| c5c_tf | P1A | greedy | equiv | 120 | 0 | 0.4688 | 2 | 120 |
| c5c_tf | P1A | t001 | equiv | 120 | 0 | 0.4688 | 2 | 120 |
| c5d | H7 | t001 | report | 27 | 0 | 0 | 0 | 0 |
| c5d | P1A | t001 | report | 120 | 0 | 0 | 0 | 0 |
| save_path | P1A | None | report | 120 | 0 | 0 | 0 | 0 |

Failures (0):


## Notes (added by hand; numbers are from `gate_stage0/`)

**Run facts**
- Gate run1: 21 path-units × 2 decodings, all of c1–c5, on the RTX 5090. 50.2 min, peak 13.0 GB.
- run2 (separate process, c5 only): 15 min.
- Harness/judge logic fixes made before or while the gate ran, with no rule or tolerance change: `AMENDMENT_P1_01.md` (A1–A3).

**c5b:** the judge table shows `suffix-equal rows` = 0 for c5b because the judge does not count suffixes for c5b. That is a display artefact. `compare.log` gives logits sha equal 2,658/2,658, suffix equal 2,658/2,658, max |Δlogit| 0.0.

**C4 checks (run1)**
- Token substitution for H8 `reverse@emb` / `patch_full@emb` verified at 82/82 substituted steps.
- All 36 H8 emb-patch row comparisons were compared at every step, with max |Δlogit| ≤ 0.4375.
- `pos_full_eq_cached` was true for all 492 C4 rows: the M-RoPE ids from `get_rope_index` on the full sequence equal `cache_position + rope_deltas` at every step.

**Hooks:** after every c1/c2 uninstall, no module hook and no attention wrapper remained.

**Stored-raw diagnostics** (c5d, T = 0.01 tokens):

| Comparison | Equal rows |
|---|---|
| H7 vs exp 7 raw | 27/27 |
| P1A vs P1-A raw (`raw`, `raw_fixedpos`) | 120/120 |
| H8/H9 vs the R8R9 dev re-runs of exp 8/9 | 180/180 |

Rollout-saving path (C1 input of §6.2 / Fig 1): stored `traj_xyh` vs re-decoded, and numpy rollout vs decoder, differ by at most 6.2e-6 m (stored values are rounded to 1e-6).

**Batch composition (report-only rows)**
- The same inputs at B = 1 vs the experiment batch change the free-run suffix of 0–22% of rows. Under T = 0.01: H8 21/27, H9 41/45 and P1A ≈ 108–118/120 rows equal.
- The full-recompute free runs (c4_free) diverge at a similar rate: H7 23–24/27, H9 36–37/45, P1A 108–110/120.
- These divergences come from ulp-level logit differences (≤ 0.5) at near-tie steps. They are not caused by the intervention machinery.
- This is the mechanism behind the 88–89% cross-harness Normal agreement (`INTERVENTION_SPEC.md` M-A3). It is why every effect must be paired within one harness and one batch composition, as RN (B = 1) and RV (B = 93) are.

**Scope of the PASS:** the 21 manifest path-units at HEAD 817d166 + harness 0ef9334 on the 5090 in bf16. The PASS does not cover H32 (DM) or the P1-B start-position script, and it does not retro-validate units outside the manifest (PROTOCOL §10).

## SpatialVLA rows (from STATE_RESTORE_AUDIT; `svla_gate/`)

| control | arm | units | verdict |
|---|---|---|---|
| C5a same process | natrep | 20 | 20/20 bitwise over 80 steps |
| C5b separate process | natural (run2) | 20 | 20/20 bitwise |
| C1 hook installed, no edit | hooknone | 20 | 20/20 bitwise |
| C2 self-patch (null) | nullfull | 20 | strict pre-fixed rule 0/20 (executor fingerprint incl. shadow cf dict); actions / simulator / exec tokens bitwise 20/20 (A5) |
| C3 / P1-C restore | branchN | 20 | 20/20 bitwise for t ≥ 8, restore ok |
| P1-C branch restore | F / C / R | 60 | 60/60 identical full state up to T0 |
| C4 analogue (report) | T0 chunk | 20 | B = 1 == batch-8: 17/20; generate == batch-8: 16/20 |

## O0 extension: exp 32 motion-semantics (H32) and P1-B start-position (P1B) paths (AMENDMENT_P1_03, CP2 Stage 3)

Run `gate_O0/`: run1 = c1-c5, run2 = c5 in a separate process; 24.8 min + run2; peak 10.9 GB; RTX 5090. Manifest `stage0_manifest_O0.json` = the 9 Stage-0 alt0 units (same fixed hash rule, no outcome use) on both paths. Tolerances and rules unchanged (T_exact 0, T_equiv 0.75). Neither script installs hooks, so c1 / c2_hook are not applicable (as H9 / P1A).

| path | verdict |
|---|---|
| H32 | PASS |
| P1B | PASS |

| control | path | decoding | class | rows | fail | max abs dlogit | flips (all explained unless failed) | suffix-equal rows |
|---|---|---|---|---|---|---|---|---|
| c2_token | H32 | greedy | equiv | 54 | 0 | 0 | 0 | 54 |
| c2_token | H32 | t001 | equiv | 54 | 0 | 0 | 0 | 54 |
| c2_token | P1B | greedy | equiv | 477 | 0 | 0 | 0 | 477 |
| c2_token | P1B | t001 | equiv | 477 | 0 | 0 | 0 | 477 |
| c3a_incremental | H32 | greedy | equiv | 9 | 0 | 0.4062 | 0 | 9 |
| c3a_incremental | H32 | t001 | equiv | 9 | 0 | 0.4062 | 0 | 9 |
| c3a_incremental | P1B | greedy | equiv | 54 | 0 | 0.4062 | 4 | 54 |
| c3a_incremental | P1B | t001 | equiv | 54 | 0 | 0.4062 | 4 | 54 |
| c3b_deep_invalidate | H32 | greedy | equiv | 54 | 0 | 0.625 | 5 | 54 |
| c3b_deep_invalidate | H32 | t001 | equiv | 54 | 0 | 0.625 | 5 | 54 |
| c3b_deep_invalidate | P1B | greedy | equiv | 90 | 0 | 0.375 | 11 | 90 |
| c3b_deep_invalidate | P1B | t001 | equiv | 90 | 0 | 0.375 | 8 | 90 |
| c4_free | H32 | greedy | report | 54 | 0 | 0.4375 | 8 | 46 |
| c4_free | H32 | t001 | report | 54 | 0 | 0.4375 | 8 | 49 |
| c4_free | P1B | greedy | report | 90 | 0 | 0.5312 | 16 | 74 |
| c4_free | P1B | t001 | report | 90 | 0 | 0.5312 | 13 | 86 |
| c4_tf | H32 | greedy | equiv | 54 | 0 | 0.4375 | 9 | 54 |
| c4_tf | H32 | t001 | equiv | 54 | 0 | 0.4375 | 9 | 54 |
| c4_tf | P1B | greedy | equiv | 90 | 0 | 0.5312 | 16 | 90 |
| c4_tf | P1B | t001 | equiv | 90 | 0 | 0.5312 | 13 | 90 |
| c5a | H32 | greedy | exact | 54 | 0 | 0 | 0 | 54 |
| c5a | H32 | t001 | exact | 54 | 0 | 0 | 0 | 54 |
| c5a | P1B | greedy | exact | 477 | 0 | 0 | 0 | 477 |
| c5a | P1B | t001 | exact | 477 | 0 | 0 | 0 | 477 |
| c5b | H32 | greedy | exact | 108 | 0 | 0 | 0 | 0 |
| c5b | H32 | t001 | exact | 108 | 0 | 0 | 0 | 0 |
| c5b | P1B | greedy | exact | 567 | 0 | 0 | 0 | 0 |
| c5b | P1B | t001 | exact | 567 | 0 | 0 | 0 | 0 |
| c5c_free | H32 | greedy | report | 54 | 0 | 0.5 | 3 | 51 |
| c5c_free | H32 | t001 | report | 54 | 0 | 0.5 | 4 | 48 |
| c5c_free | P1B | greedy | report | 90 | 0 | 0.5312 | 9 | 82 |
| c5c_free | P1B | t001 | report | 90 | 0 | 0.5312 | 9 | 78 |
| c5c_tf | H32 | greedy | equiv | 54 | 0 | 0.5 | 4 | 54 |
| c5c_tf | H32 | t001 | equiv | 54 | 0 | 0.5 | 6 | 54 |
| c5c_tf | P1B | greedy | equiv | 90 | 0 | 0.5312 | 10 | 90 |
| c5c_tf | P1B | t001 | equiv | 90 | 0 | 0.5312 | 10 | 90 |
| c5d | H32 | t001 | report | 54 | 0 | 0 | 0 | 0 |
| c5d | P1B | t001 | report | 477 | 0 | 0 | 0 | 0 |

Failures (0):


Notes: c5b cross-process 1,350/1,350 rows with identical logits sha and suffix, max |dlogit| 0.0 (the c5b `suffix-equal rows` column is a display artefact, as above). c5d stored raw (T = 0.01): H32 vs exp 32 raw 54/54 rows equal; P1B vs P1-B raw 477/477 rows equal. Max |dlogit| over the equivalence controls 0.625 (H32 c3b) < 0.75; every flip explained. Free-run suffix agreement at B = 1 vs the experiment batch (c5c_free, report only): H32 48-51/54, P1B 78-82/90; full recompute free runs (c4_free): H32 46-49/54, P1B 74-86/90 - the batch-shape numeric floor, not the intervention machinery (M-A3).
