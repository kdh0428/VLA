# MISSING.md — reproducibility_followup5.zip (2026-10-18)

This ZIP is a small add-on for review_followup5. The full review_followup4 package
(`outputs/review_followup4/reproducibility_minimal.zip`, its `MISSING.md` and `PINS.csv`) still applies to everything
earlier.

| item | status |
|---|---|
| Model checkpoint (AutoVLA 16.3 GB) and camera images | not shipped; revision / sha256 in `PINS.csv`; needed only to re-run the GPU part (B_NUMERIC) |
| B_NUMERIC GPU raw | **shipped** (`outputs/review_followup5/B_NUMERIC/run/records.jsonl`, `logit_cmp.jsonl`, `run_meta.json`, smoke report); no logit arrays were ever written |
| Learned extrapolator weights / training cache | not shipped (review_followup3 P2-B); the frozen per-unit predictions `extrap_units.parquet` are shipped and are what B_NUMERIC uses |
| OpenScene private_test_hard metadata | not shipped (streamed during C_DISJOINT, never written to disk; sha256 of the stream in `C_DISJOINT/overlap_audit.json`) |
| navhard two-stage metadata (2,731 pickles) | not shipped (on disk under `/root/VLA/navhard/navhard_two_stage/openscene_meta_datas`); `C_DISJOINT/DATA_INVENTORY.csv` keeps the per-scene day / vehicle / city used |
| Manuscript `.tex` / PDF | do not exist in the repository |
| D_RELEASE results | NOT_RUN (optional item; protocol draft only) |
| Independent day-and-vehicle-disjoint replication | BLOCKED_NO_DISJOINT_DATA |
