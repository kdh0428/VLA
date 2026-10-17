# RUN_COMMANDS — review_followup3 (FINAL, 2026-10-15)

**Paths and machine.**
- Repo `/root/VLA`, branch `autovla-experiments-11-19`.
- `POC=/root/VLA/autovla_misalignment_poc`.

**How runs are made.**
- CPU re-analyses: `nice -n 19` with `CUDA_VISIBLE_DEVICES=""`.
- GPU runs: **RTX 5090 only**, via `CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1`. Without PCI_BUS_ID, index 1 resolves to the 3080 Ti.
- Load one model at a time, and only once ≥ 14 GB are free on GPU 1; an external process holds 13–26 GB.
- RAM is limited to 14.4 GB, so checkpoints load with `torch.load(..., mmap=True)` (already in the code).

## Commits

| commit | content |
|---|---|
| 0ef9334 | P1 protocol, INTERVENTION_SPEC, Stage-0 manifest, C5 pilot tolerances |
| e9907fc | P0 (CPU re-analyses) |
| 5b158f2 | P2-A / P2-B |
| 47efb95 | P1 results: AutoVLA Stage-0 gate PASS; SpatialVLA audit; amendments A1–A5 |
| d70d19c | CP2 package (drafts, Stage-3 protocols, power and cost plans) |
| 444299f | CP2 decision (A5 re-verify; Stage 3 = O0 only; P3/P4 not run) |
| fae513c | A5 split-fingerprint re-verification PASS (AMENDMENT_P1_02); O0 gate H32 + P1B PASS (AMENDMENT_P1_03) |
| (uncommitted) | this finalisation: numbers.json regenerated; final STATUS / CLAIM_EVIDENCE_MATRIX / REVIEW_COVERAGE; document fixes; Tables 8–11; MANUSCRIPT_REVISION 2026-10-15 section; EXPERIMENT_SUMMARY §6f |

## Environments, checkpoints, seeds

| item | value |
|---|---|
| AutoVLA env | `source /root/miniforge3/etc/profile.d/conda.sh && conda activate autovla` (Python 3.9, torch 2.8.0+cu128, transformers 4.49.0, bf16, sdpa) |
| AutoVLA code | `/root/VLA/autovla` @ ba34eed |
| AutoVLA checkpoint | `/root/VLA/autovla_dl/AutoVLA/AutoVLA_PDMS_89.ckpt` |
| AutoVLA config | `autovla/config/training/qwen2.5-vl-3B-nuplan-grpo-cot.yaml` |
| AutoVLA decoding | T = 0.01, common random numbers re-seeded per row: `sha256(f"{seed}:{token}:{key}:{k}")`, seed 0. Greedy decoding is used only as a gate control. |
| SpatialVLA env | `source /root/VLA/simpler/env.sh` (autovla env + SimplerEnv / ManiSkill2_real2sim, SAPIEN 2.2.2, Mesa lavapipe Vulkan, GPU 1) |
| SpatialVLA checkpoint | `/root/VLA/spatialvla/spatialvla-4b-224-sft-fractal` |
| SpatialVLA run settings | `decode_fixed` batch 8; 4 SAPIEN workers (amendment A4) |

**Statistics seeds.**
- P1-C frozen analysis: 20261008.
- review_followup2: 20261013 + offsets.
- P0-C authoritative row: 20261014.
- P2: drive bootstrap with B = 10,000; extrapolator training seeds 0, 1, 2.
- P4 power simulation: 20261014.
- The P3 / P4 protocol seeds (20261015–17) were reserved but never used.

## A. CPU-only re-analyses and document generation (no model inference)

```bash
bash $POC/scripts/review_followup3/p0_regen.sh       # P0 tables, numbers_p0.json, reports (asserts reproduction of P1-C, DIRECT_EFFECTS, OpenVLA analysis)
bash $POC/scripts/review_followup3/p2_regen.sh       # P2-A + P2-B (sha256-locked spec/protocol; extrapolator data step streams 7.05 GB, network)

PY="nice -n 19 /root/miniforge3/envs/autovla/bin/python"; C=$POC/scripts/review_followup3/cp2
$PY $C/stage3_cost_rates.py        # STAGE3_COST_RATES.csv (rates parsed from run logs)
$PY $C/p4_planning_data.py         # P4/planning/ (planning only)
$PY $C/p4_layout_geometry.py       # P4/planning/layout_geometry.csv (pure geometry)
$PY $C/p4_power_sim.py --validate && $PY $C/p4_power_sim.py   # P4/power/ (~10 min)
$PY $C/stage3_cost_plan.py         # STAGE3_COST_PLAN.csv
$PY $C/cp2_numbers.py              # paper_quantitative_package/numbers.json + numbers_sources.csv (gate verdicts read from judge logs)
$PY $C/cp2_doc_fixes.py            # verified rounding/label fixes in EXPERIMENT_SUMMARY.md, PAPER_NUMBERS.md, tables.tex, MANUSCRIPT_REVISION.md (idempotent)
$PY $C/cp2_tables.py               # Tables 8-11 in tables.tex, between the machine-generated markers, from numbers.json
```

**Order.** Run the P0, P1 and P2 outputs first. Then: `cp2_numbers.py` → `cp2_doc_fixes.py` → `cp2_tables.py`.

**Hand-written parts.** The following were written by hand from the files above:
- the §6f section of EXPERIMENT_SUMMARY;
- Claims 22–24 of PAPER_NUMBERS;
- the 2026-10-15 section of MANUSCRIPT_REVISION.

## B. GPU runs (all done; RTX 5090)

```bash
# AutoVLA control-5 pilot -> tolerances T_exact = 0, T_equiv = 0.75 (pilot_c5_20261014/tolerances.json)
bash $POC/scripts/review_followup3/p1/run_c5_pilot.sh
# AutoVLA Stage-0 gate, H7/H8/H9/P1A, 21 path-units (run1 c1-c5 50.2 min; run2 c5 13.6 min) -> P1/NOOP_CACHE_AUDIT.*
bash $POC/scripts/review_followup3/p1/run_gate_stage0.sh
# O0 gate extension, H32 + P1B, 18 path-units (manifest from p1_o0_manifest.py -> P1/stage0_manifest_O0.json)
#   = p1b_harness.py --controls c1,c2,c3,c4,c5 --paths H32,P1B --decodings greedy,t001 (+ run2 c5, compare, judge) -> P1/gate_O0/
bash $POC/scripts/review_followup3/p1/run_gate_O0.sh

# SpatialVLA restore audit, 20 units (pilot, run1 160 episodes 29.3 min, run2 20 episodes) -> P1/STATE_RESTORE_AUDIT.*
bash $POC/scripts/review_followup3/p1/run_svla_audit.sh
# A5 split-fingerprint re-verification (40 episodes; AMENDMENT_P1_02)
source /root/VLA/simpler/env.sh
cd $POC && nice -n 10 python scripts/review_followup3/p1/svla_noop_restore.py --out outputs/review_followup3/P1/svla_gate/a5_rerun \
    --manifest outputs/review_followup3/P1/svla_manifest.json --units all --modes natural,nullfull --workers 4
python scripts/review_followup3/p1/svla_audit.py a5split <a5_rerun dir> <STATE_RESTORE_AUDIT.csv> <STATE_RESTORE_AUDIT.md>
```

## C. Not run (by CP2 decision or for lack of data)

- **P3** matched Reverse (AutoVLA; Impromptu add-on): `P3/MATCHED_REVERSE_PROTOCOL.md`. Estimated cost 2.3–6.0 GPU h for AutoVLA; 11–22 GPU h for Impromptu.
- **P4** SpatialVLA variant states: `P4/SPATIAL_STATE_EXPANSION_PROTOCOL.md`, `P4/POWER_PLAN.md`. The approved target if ever run is ±10 pp per task with n = 500 states per task, ≈ 25.5–71 GPU h.
- **P5** day/vehicle-disjoint replication: blocked; 0 disjoint logs.
