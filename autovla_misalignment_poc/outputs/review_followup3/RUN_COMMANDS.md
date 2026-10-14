# RUN_COMMANDS — review_followup3 (as of CP2, 2026-10-14)

**Conventions.**
- Repo: `/root/VLA`, branch `autovla-experiments-11-19`.
- `POC=/root/VLA/autovla_misalignment_poc`.
- CPU re-analyses run under `nice -n 19` with `CUDA_VISIBLE_DEVICES=""`.
- GPU runs use the **RTX 5090 only**: `CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1`. Without `PCI_BUS_ID`, torch's fastest-first ordering would map GPU 1 to the 3080 Ti.
- Never run two model loads at once. Wait until ≥ 14 GB are free on GPU 1: an external process holds 13–26 GB.
- RAM is limited to 14.4 GB, so checkpoints load with `torch.load(..., mmap=True)` (already in the code).

**Commits.**

| commit | content |
|---|---|
| 0ef9334 | P1 protocol, intervention spec, manifest, C5 pilot tolerances |
| e9907fc | P0 |
| 5b158f2 | P2 |
| 47efb95 | P1 results (AutoVLA gate, SpatialVLA audit, amendments A1–A5) |
| (uncommitted) | the CP2 files of this round |

## Environments, checkpoints, seeds

| item | value |
|---|---|
| AutoVLA env | `source /root/miniforge3/etc/profile.d/conda.sh && conda activate autovla`. Python 3.9, torch 2.8.0+cu128, transformers 4.49.0, bf16, sdpa. AutoVLA code `/root/VLA/autovla` @ ba34eed |
| AutoVLA checkpoint / config | `/root/VLA/autovla_dl/AutoVLA/AutoVLA_PDMS_89.ckpt`; `autovla/config/training/qwen2.5-vl-3B-nuplan-grpo-cot.yaml` |
| AutoVLA decoding | T = 0.01, CRN re-seeded per row: `sha256(f"{seed}:{token}:{key}:{k}")`, seed 0; greedy only in the P1 gate |
| SpatialVLA env | `source /root/VLA/simpler/env.sh` (autovla env + SimplerEnv / ManiSkill2_real2sim, SAPIEN 2.2.2, Mesa lavapipe Vulkan, GPU 1) |
| SpatialVLA checkpoint | `/root/VLA/spatialvla/spatialvla-4b-224-sft-fractal` (IPEC-COMMUNITY/spatialvla-4b-224-sft-fractal); `decode_fixed` batch 8 |
| Statistics seeds | P0 / P1-C bootstrap 20261008 (frozen P1-C analysis); review_followup2 13-drive analyses 20261013 + offsets; P0-C AUTH row 20261014; P2 B = 10,000 drive bootstrap; P2-B training seeds 0, 1, 2; P4 power simulation 20261014 |

## A. CPU-only re-analyses (no model inference)

```bash
# P0 (all P0 tables, numbers_p0.json, reports). Asserts reproduction of P1-C decisions.json,
# DIRECT_EFFECTS.csv and analysis.json. P0_REFETCH=1 re-streams the OpenScene metadata.
bash $POC/scripts/review_followup3/p0_regen.sh

# P2-A + P2-B (spec/protocol locks are asserted by sha256).
# p2_extrap_data.py stream needs the network (7.05 GB streamed, ~56 MB kept).
bash $POC/scripts/review_followup3/p2_regen.sh

# CP2 (this round)
PY="nice -n 19 /root/miniforge3/envs/autovla/bin/python"; C=$POC/scripts/review_followup3/cp2
$PY $C/stage3_cost_rates.py        # -> outputs/review_followup3/STAGE3_COST_RATES.csv (rates parsed from run logs)
$PY $C/p4_planning_data.py         # -> P4/planning/ (P1-D / exp 35 planning data)
$PY $C/p4_layout_geometry.py       # -> P4/planning/layout_geometry.csv (pure geometry, no simulator)
$PY $C/p4_power_sim.py --validate  # -> P4/power/power_validation.csv
$PY $C/p4_power_sim.py             # -> P4/power/power_{grid,required_n,cost,budget48h,analytic_approx}.csv (~10 min)
$PY $C/stage3_cost_plan.py         # -> outputs/review_followup3/STAGE3_COST_PLAN.csv
$PY $C/cp2_numbers.py              # -> /root/VLA/paper_quantitative_package/numbers.json + numbers_sources.csv
```

Run order matters:
- `stage3_cost_rates.py` before `p4_power_sim.py`;
- `p4_power_sim.py` before `stage3_cost_plan.py`;
- the P0 / P1 / P2 outputs before `cp2_numbers.py`.

## B. GPU runs already done (reference; do not re-run without reason)

```bash
# P1 AutoVLA control-5 pilot (fixes tolerances: T_exact = 0, T_equiv = 0.75)
bash $POC/scripts/review_followup3/p1/run_c5_pilot.sh
python $POC/scripts/review_followup3/p1/p1b_pilot_tolerances.py $POC/outputs/review_followup3/P1/pilot_c5_20261014

# Stage-0 AutoVLA gate. run1: c1-c5, 4 paths, greedy + t001, 50.2 min; run2: c5, 13.6 min; then compare + judge.
bash $POC/scripts/review_followup3/p1/run_gate_stage0.sh
#   = p1b_harness.py --manifest P1/stage0_manifest.json --controls c1,c2,c3,c4,c5 --paths H7,H8,H9,P1A --decodings greedy,t001
#   + p1b_compare_runs.py + p1b_judge.py -> P1/NOOP_CACHE_AUDIT.{csv,md}

# SpatialVLA P1-B/P1-C audit. 4 SAPIEN workers (amendment A4); run1 160 episodes in 29.3 min; run2 20 episodes.
bash $POC/scripts/review_followup3/p1/run_svla_audit.sh
#   = svla_noop_restore.py --manifest P1/svla_manifest.json --units all --c4 --workers 4  (+ run2 --modes natural --tag "#proc2")
#   + svla_audit.py gate ... -> P1/STATE_RESTORE_AUDIT.{csv,md}
```

## C. GPU runs proposed for Stage 3 (NOT run; need CP2 approval; scripts and flags marked "new" do not exist yet, so the commands are indicative)

```bash
source /root/miniforge3/etc/profile.d/conda.sh && conda activate autovla
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1

# O0 - gate extension to H32 (DM) and the P1-B start-position path (harness paths to be added: new)
nice -n 10 python $POC/scripts/review_followup3/p1/p1b_harness.py --manifest $POC/outputs/review_followup3/P1/stage0_manifest.json \
   --out $POC/outputs/review_followup3/P1/gate_ext/run1 --controls c1,c2,c3,c4,c5 --paths H32,P1B --decodings greedy,t001

# A5(b) - SpatialVLA null arm with policy-state and cf hashed separately (svla_noop_restore.py change: new)
source /root/VLA/simpler/env.sh
nice -n 10 python $POC/scripts/review_followup3/p1/svla_noop_restore.py --manifest $POC/outputs/review_followup3/P1/svla_manifest.json \
   --units all --modes natural,nullfull --workers 4 --out $POC/outputs/review_followup3/P1/svla_gate_A5b

# O1 - P3 AutoVLA (scripts new; protocol P3/MATCHED_REVERSE_PROTOCOL.md must be committed first)
#   p3_gate (Stage-0 manifest, path P3) -> p3_bank (Normal + Full-GT rows, frozen sha256)
#   -> p3_factorial --block oneshot --anchor 3 -> p3_factorial --block repeated   (P1-C 353 scenes, then dev 208 scenes)

# O3 - P4 SpatialVLA variant states (scripts new; protocol P4/SPATIAL_STATE_EXPANSION_PROTOCOL.md committed first)
#   p4_sample_states (CPU-side reset + settle checks, seed 20261016; pilot stream 20261017) -> frozen manifest
#   -> pilot 16 states/task (throughput) -> svla_noop_restore.py-based runner, 4 workers, n states/task x 8 arms
```
