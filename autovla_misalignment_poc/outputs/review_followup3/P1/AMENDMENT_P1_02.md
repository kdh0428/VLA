# AMENDMENT P1-02: SpatialVLA null-arm (C2) re-verification with a split executor fingerprint (2026-10-15)

Decided at CP2 (commit 444299f, `review_followup3/CP2_DECISION.md`, item A5). Written **before** the re-run.

## Why
- In the first SpatialVLA audit, the null arm (U2, `nullfull`) **failed** the strict pre-fixed rule: 0/20.
- The reason: the executor fingerprint `exec_fp` hashed the policy state together with `ex.cf`. `ex.cf` is the shadow counterfactual-chunk dict that P1-D's null mode fills by design.
- The post hoc A5 check passed 20/20. Both results stay recorded unchanged in `STATE_RESTORE_AUDIT.md/.csv`.

## Change: split fingerprint (`svla_noop_restore.py`)
The fingerprint is recorded after every executed step. The old combined `exfp` is still recorded, for continuity.

| Field | What it hashes |
|---|---|
| `exfp_policy` | Everything that determines the executed action: the chunk buffer `ex.chunks` (t_gen + float64 bytes of each 4 × 7 chunk), the sticky-gripper state (`sticky_on`, `repeat`, `sticky_action`, `prev_grip`) and K, E. Excludes `cf`. |
| `exfp_cf` | The shadow-cf bookkeeping dict `ex.cf` (t_gen + float64 bytes), alone. |

## Run
- **Units:** the same 20 units as the first audit (`svla_manifest.json`: 2 tasks × 5 states × r1e4/r4e1).
- **Arms:** `natural` (the untouched comparison U0) and `nullfull` (U2), from reset. 40 episodes.
- **Hardware and settings:** RTX 5090, 4 workers, `decode_fixed` batch 8. Code is otherwise unchanged.
- **Output:** `svla_gate/a5_rerun/`.

## Pass rule
Tolerance 0, the same as the pre-fixed rule. U2 passes on a unit iff, over all 80 steps:
- the executed float64 actions are bytes-equal;
- every simulator fingerprint component is equal (the same component list as before);
- `exfp_policy` is equal.

`exfp_cf` is **reported only**. It is expected to differ from t = 8, because null mode fills the dict and natural does not.

This adds a new verdict row, `C2_A5_split`. It does not replace the strict 0/20 or the A5 20/20 rows.
