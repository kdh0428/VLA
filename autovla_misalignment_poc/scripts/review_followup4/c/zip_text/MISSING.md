# MISSING.md — what this ZIP does not contain, and why (2026-10-18)

## A. Truly missing (does not exist anywhere in the repository or on disk)

| item | consequence |
|---|---|
| Per-unit raw output of the original exp 5 (first-mismatch 1-token correction) and exp 10 (horizon-controlled window) main runs | summary-level only; no cluster re-test possible (stated in the reports) |
| Per-unit raw output of the original exp 8 / exp 9 main runs | replaced by the R8R9 recovery run (220/220 rates reproduced); the recovery raw is on disk but not shipped (B.3) |
| Manuscript `.tex` | none in the repository; sentence-level replacements in `MANUSCRIPT_REVISION.md` |
| B3 OpenVLA gate outputs | not run (user decision 2026-10-17; checkpoint not on disk) — `OPENVLA_GATE.md` |
| P3 / P4 results (review_followup3) | not run (CP2 decision, historical); B2 is a separate new round |
| P5 results | blocked: 0 day-and-vehicle-disjoint logs exist |
| OpenVLA LIBERO-Spatial checkpoint file checksums | weights deleted during the exp 34 disk cleanup; only the HF revision `962318c…` is pinned (`PINS.csv`) |
| SpatialVLA / Impromptu checkpoint file checksums | never recorded; HF revisions pinned (`PINS.csv`) |
| Checksums of the OpenScene / NAVSIM / nuPlan downloads | the download script uses unpinned `resolve/main` / S3 URLs; no checksum was recorded. The B2 P1-C image re-fetch was verified functionally (2,160/2,160 Normal tokens equal the stored P1-C outputs) |

## B. Exists but deliberately not shipped (size / policy)

| item | size | how to obtain / effect on regeneration |
|---|---|---|
| Model checkpoints (AutoVLA 16.3 GB, Qwen2.5-VL-3B, Impromptu, SpatialVLA) | > 20 GB | public; revisions and sha256 in `PINS.csv`. Needed only for GPU re-runs, not for the tables |
| OpenScene camera images / sensor blobs, nuPlan maps | > 100 GB | `tools/download_autovla_assets.sh`, `scripts/review_followup4/b/b2_fetch_p1c_images.sh` |
| Raw logits of the P1 gate / pilot (`logits_keyrows.npz`) | 12–24 MB each | on disk in `outputs/review_followup3/P1/*/run*/`; the judge logs and comparison files are shipped |
| P1-C mechanism raw (`run_20261008/mech/`) except the H7 (Recent-GT / Normal) records `mech/action_history_causal/records.jsonl`, which are shipped | 274 MB | on disk; the frozen unit table `analysis/units.parquet` (used by P0-A) is shipped, but the original P1-C decision analysis from all raw blocks (`analyze_p1c.py`, needs H8 / H32 / window raw) cannot be re-run from the ZIP alone |
| R8R9 raw (`R8R9_raw_recovery/run_20261008/raw/`), P1-D rollouts, P1-B `raw/records.jsonl` | 226 / 312 / 65 MB | on disk; unit tables (`units.parquet`) and intervention traces are shipped. A2's trace audit (`a2_window_semantics.py`) needs the R8R9 exp9 raw and is therefore not in `regen.sh` |
| OpenVLA rollouts (`cross_vla_replication/rollouts/`, 263 MB) and token-feedback jsonl (26 MB) | 289 MB | on disk; the episode records CSV and phase-C unit table derived from them are shipped |
| P2-B training data cache and trained extrapolator weights (`P2/_extrap_data/`, `_extrap_models/`) | 145 + 20 MB | re-streamable from public OpenScene trainval metadata (`p2_extrap_data.py`, 7 GB network); unit-level predictions (`extrap_units.parquet`) and results are shipped |
| Figures (png / pdf) | – | not needed for numbers |
