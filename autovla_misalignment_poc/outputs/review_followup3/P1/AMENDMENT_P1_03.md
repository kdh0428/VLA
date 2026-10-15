# AMENDMENT P1-03: Stage-3 O0, two code paths added to the AutoVLA parity gate (2026-10-15)

Decided at CP2 (commit 444299f, `review_followup3/CP2_DECISION.md`, Stage 3 = O0). Written **before** any run of the new paths.

## Paths added

| Path | Script (HEAD 817d166) | Used by | Rows / batch | Hooks |
|---|---|---|---|---|
| **H32** | `scripts/motion_semantics_ablation.py` | exp 32 and P1-C H32: DM = dir_wrong_mag_ok − dir_ok_mag_wrong | ROWS (6), B = 6 | none |
| **P1B** | `scripts/review_followup/p1ab/p1b_start_position.py` | P1-B start position | `build_rows()` (53 rows), B = 53 | none |

**H32 context rule** (`motion_semantics_ablation.py:234-251`):
- For k ≥ t*+2, the last context position of each non-normal row is a substitute.
- recent_gt uses gt[k−1].
- The other rows use `motion_sub(row, gt[k−1], own[k−1], token_motion(codebook), trained_ids, rng)`, with `rng = default_rng(sha256(f"{seed}:{token}:{pert}:{k}:rand"))`.
- The sampling seed key is the perturbation, as in the script.

**P1B context rule** (`p1b_start_position.py:50-68`, `ctx_tokens` is imported):
- persistent / transient / full / recent rows.
- The source is GT or the fixed P1-A self-reference (`P1A…/reference/references.jsonl`).
- The seed key is the perturbation.

Neither script installs hooks. As for H9/P1A, the hook controls (c1, c2_hook) are therefore **not applicable**.

## Manifest extension rule
No new selection and no outcome use. The units are the existing Stage-0 manifest's `alt0` units:
- the 9 scenes chosen by sha256("P1-STAGE0-20261014|" + token), first 3 per t* stratum (early, middle, late);
- unit = (scene, perturbation `alt0`);
- run on both new paths;
- P1B units carry that scene's reference tokens (the same file P1-B used).

This gives 9 units × 2 paths = 18 path-units, written to `stage0_manifest_O0.json` by `p1_o0_manifest.py`. The original manifest is not modified.

## Controls, tolerances and pass rules: unchanged
- Controls: PROTOCOL_P1 §1 c2_token, c3a (append-only rows), c3b, c4_tf / c4_free, and c5a/c5b/c5c/c5d.
- c2_token mapping: every row → `normal`. With every override filled by the row's own token, each row's context is all-own.
- Append-only rows for c3a:
  - H32: normal.
  - P1B: normal, full_gt, full_self and every persistent `P_*` row.
- Tolerances: T_exact = 0, T_equiv = 0.75 (`pilot_c5_20261014/tolerances.json`). No new pilot.
- Pass/fail rules: PROTOCOL §6 via `p1b_judge.py`.
- The c2_token class follows the batch (exact only for B = 1). Both paths are batched, so c2_token is in the equivalence class.
- c5d diagnostics:
  - H32 vs exp 32 raw (`outputs/motion_semantics_ablation/records.jsonl`, alt0);
  - P1B vs P1-B raw (`outputs/review_followup/P1B_start_position/run_20261008/raw/records.jsonl`).

## Code changes (additive; existing paths unchanged)
- `p1b_engine.py`: the paths H32 and P1B in `path_rows`, `KEY_ROWS`, `context` and `decode`, plus the reference for P1B.
- `p1b_harness.py`: APPEND_ONLY entries for the new paths and c5d stored-raw loaders.

## Run and outputs
- Commands: run1 `--controls c1,c2,c3,c4,c5 --paths H32,P1B` and a separate-process run2 `--controls c5`; then `p1b_compare_runs.py` and `p1b_judge.py`.
- Outputs: `gate_O0/`.
- New rows are appended to `NOOP_CACHE_AUDIT.md` / `.csv` as separate O0 sections. The existing rows are not touched.
