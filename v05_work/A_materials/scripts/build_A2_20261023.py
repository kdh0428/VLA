#!/usr/bin/env python
import json, pandas as pd
C = json.load(open("/root/VLA/v05_work/A_materials/A2_cost_call_counts_20261023.json"))
P = "/root/VLA/autovla_misalignment_poc/outputs"
L1 = json.load(open("/root/VLA/svla_wc_work/run/gpu_ledger.json"))["attempts"]
L3 = json.load(open("/root/VLA/integrated_v03_work/gpu_ledger_v03.json"))["attempts"]
LV = json.load(open("/root/VLA/vla0_cl_work/run_gpu_ledger.json"))["attempts"]
dur = lambda a: round((a["last"] - a["start"]) / 60, 2)
G5 = "NVIDIA GeForce RTX 5090 (CUDA_VISIBLE_DEVICES=1, PCI_BUS_ID)"
rows = []
def r(**k): rows.append(k)
# ---------------- AutoVLA
r(experiment="AutoVLA matched-prefix (B2), P1-C main", used_in_v04="yes (Table tab:matched, F and E4)", hardware=G5 + "; bf16 weights, sdpa, T=0.01",
  model_calls="31,280 decodes = 2,160 source-bank units (phase A) + 29,120 cell decodes (phase B)", call_definition="one cell = one batch-1 suffix decode from the shared cached prefix with a forced context copy",
  sampling_or_forcing_calls="29,120 forced-context cell decodes (n_old=1: 7 cells x 2,160 = 15,120 incl. noop; n_old=2 secondary: 7 x 2,000 = 14,000); sampling T=0.01 with CRN seeds",
  wall_time_min=84.1, gpu_occupancy_min="84.1 (run log; model held throughout)", peak_vram_gb=10.86, renderer="none (stored sensor inputs)",
  source=f"{P}/review_followup4/B/B2/p1c_main.log ('[done] 84.1 min'); {P}/review_followup4/B/B2/p1c_main/run_meta_20261018T001424.json (gpu, peak_mem_gb 10.860356608); cells.jsonl / bank.jsonl line counts",
  derivation="counts = line counts of cells.jsonl (29,120) and bank.jsonl (2,160)")
r(experiment="AutoVLA matched-prefix (B2), gate + throughput pilot + development cohort", used_in_v04="gate/pilot: yes (gate precedes main run); dev_main: not used for v04 estimates", hardware=G5,
  model_calls="UNKNOWN (not tallied)", call_definition="as above", sampling_or_forcing_calls="UNKNOWN", wall_time_min="gate 1.0; pilot 1.7; dev_main 52.9",
  gpu_occupancy_min="same as wall (run logs)", peak_vram_gb="10.86 (each run_meta peak_mem_gb 10.860356608)", renderer="none",
  source=f"{P}/review_followup4/B/B2/gate_stage0.log, pilot_throughput.log, dev_main.log; gate_stage0/, pilot_throughput/, dev_main/ run_meta_*.json (elapsed_min 1.02 / 1.69 / 52.95)", derivation="[done] lines and run_meta elapsed_min")
r(experiment="AutoVLA repeated correction, development F (RN = exp 7 action_history_causal)", used_in_v04="yes (RN dev F)", hardware="NVIDIA GeForce RTX 5090 (run_meta 'gpu')",
  model_calls="9,856 plan decodes = 1,408 units x 7 conditions (208 scenes; F uses 52 scenes / 365 units)", call_definition="one condition = one 10-token plan decode with the stated context rule",
  sampling_or_forcing_calls="each non-normal condition forces GT/masked context at every later step; T=0.01", wall_time_min=19.48, gpu_occupancy_min="19.48 (elapsed_min)", peak_vram_gb="UNKNOWN (not logged)", renderer="none",
  source=f"{P}/action_history_causal/run_meta.json (elapsed_min 19.48, n_perturbations 1408, 7 conditions)", derivation="units x conditions")
r(experiment="AutoVLA repeated correction, development F (RV = R8 raw-recovery rerun, prev_action_state_patching)", used_in_v04="yes (RV dev F)", hardware="NVIDIA GeForce RTX 5090 (run_meta 'gpu')",
  model_calls="130,944 row decodes = 1,408 units x 93 rows (RV uses reverse@emb vs gt_history only)", call_definition="one row = one plan decode", sampling_or_forcing_calls="all rows are forced/patched context decodes except 'normal'",
  wall_time_min=43.1, gpu_occupancy_min="43.1 (elapsed_min)", peak_vram_gb="UNKNOWN (not logged)", renderer="none",
  source=f"{P}/review_followup/R8R9_raw_recovery/run_20261008/raw/exp8/run_meta.json; records.jsonl row count", derivation="sum of len(rows) over records")
r(experiment="AutoVLA repeated correction, new drives P1-C (stage 1 natural plans + H7 RN + H8 RV)", used_in_v04="yes (RN/RV new-drive F and E)", hardware=G5,
  model_calls="stage 1: 5,671 natural plans; H7: 17,052 plan decodes (2,436 units x 7); H8: 226,548 row decodes (2,436 x 93)", call_definition="one plan/row = one 10-token plan decode",
  sampling_or_forcing_calls="H7 recent_gt and H8 reverse@emb/gt_history are forced-context decodes; T=0.01", wall_time_min="stage1 281.5 (26 shards); units build 26.7; H7 66.9; H8 63.3 (run_meta) [gpu_time.jsonl lists 10.6 min for the H8 step]",
  gpu_occupancy_min="stage1+H7+H8 = 411.7 using run_meta for H7/H8 (H9/H32 blocks of the same run add 28.3 + 19.5 min, not needed for RN/RV)", peak_vram_gb="UNKNOWN (not logged)", renderer="none",
  source=f"{P}/review_followup/P1C_new_log_replication/run_20261008/gpu_time.jsonl; mech/action_history_causal/run_meta.json; mech/prev_action_state_patching/run_meta.json",
  derivation="stage1 = sum of 26 stage1_shard seconds (16,892 s); H8 discrepancy between run_meta elapsed_min (63.3) and gpu_time.jsonl step (637 s) unresolved -- report both")
r(experiment="AutoVLA motion response (P1-A family G+E generation)", used_in_v04="yes (S0 rows = stored P1-A raw; F 364 units, selected 1,453 units)", hardware="NVIDIA GeForce RTX 5090 (gpu_checked)",
  model_calls="28,610 row decodes = 2,861 units x 10 rows (batch_rows 10; family G = 1,453 units)", call_definition="one unit = one batch of 10 rows (U, OC, O1, C1, OC_ref1, OC_refR, OC_refH, O_refH, OC_gtR, OC_gtH)",
  sampling_or_forcing_calls="perturbed token forced at t*; T=0.01", wall_time_min=23.78, gpu_occupancy_min="23.78 (elapsed_min)", peak_vram_gb=10.86, renderer="none",
  source=f"{P}/review_followup/P1A_gt_free_reference/run_20261008/raw/run_meta.json (elapsed_min, peak_mem_gb 10.860356608, n_units 2861, batch_rows 10)",
  derivation="units x batch rows; note peak_mem_gb equals the B2 value to all digits (as logged)")
r(experiment="AutoVLA motion response numerical follow-up (B_NUMERIC S0/SE/SF)", used_in_v04="yes (supp. tab:numeric)", hardware=G5 + "; bf16, sdpa (SE: eager attention; SF: fp32 arithmetic)",
  model_calls="43,590 row decodes = 1,453 units x 10 rows x 3 settings", call_definition="as P1-A", sampling_or_forcing_calls="same forced perturbations; T=0.01",
  wall_time_min="51.8 (S0 15.8, SE 14.6, SF 21.3) + smoke ~3", gpu_occupancy_min="51.8 + ~3 (STATUS.md)", peak_vram_gb=16.42, renderer="none",
  source=f"{P}/review_followup5/B_NUMERIC/run/run_meta.json (setting_min, peak_mem_gb 16.424900608); {P}/review_followup5/STATUS.md", derivation="supplement Sec. D states 14,530 rows per setting, 43,590 total")
r(experiment="AutoVLA learned non-AR motion extrapolator (training + eval)", used_in_v04="yes (learned control)", hardware="CPU only (torch threads <= 8, nice 19)",
  model_calls="10 training runs (4 grid + 6 final), 15 epochs each; eval 5,553 units", call_definition="MLP training, not VLA inference", sampling_or_forcing_calls="NA",
  wall_time_min="data stream 2.0; final runs 14.4-16.0 each; eval 5.3", gpu_occupancy_min=0, peak_vram_gb=0, renderer="none",
  source=f"{P}/review_followup3/P2/_extrap_models/final_*_log.csv (elapsed_s 864-957); _extrap_data/trainval_ego_frames_provenance.json (122.7 s); extrap_run_meta.json (elapsed_s 320.4)", derivation="elapsed_s / 60; grid-run times in grid_*_log.csv")
# ---------------- VLA-0
v = C["vla0_pilot"]
r(experiment="VLA-0 within-call open-loop pilot (150 obs; 149 analyzed)", used_in_v04="yes (Table tab:vla0)", hardware="NVIDIA GeForce RTX 5090 (GPU 1); bf16, B=1",
  model_calls=f"{v['generate_calls']:,} generate calls over 150 observations (reference + one T=0.7 source call returning 4 + 6 forced arms R/O/L/OL/H/HL)",
  call_definition="one HF generate() call", sampling_or_forcing_calls=f"150 source-sampling calls ({v['source_returns']} returned samples); forced-decoding arm calls = remainder",
  wall_time_min="78.1 (final run) + ~8 failed validation attempts", gpu_occupancy_min="78.10 (gpu_time.json gpu_minutes_total)", peak_vram_gb=round(v["peak_vram_gb_max"], 2),
  renderer="observations pre-rendered offline on CPU-side LIBERO/robosuite with Mesa EGL (renderer string not logged)",
  source="/root/VLA/vla0_work/pkg/run/gpu_time.json; /root/VLA/vla0_work/pkg/run/pilot/records.jsonl (n_generate_calls, peak_vram_gb); /root/VLA/vla0_work/pkg/README.md",
  derivation="sum of per-record n_generate_calls; max per-record peak_vram_gb; sum per-record latency_s = 74.3 min")
v = C["vla0_closed_loop"]
r(experiment="VLA-0 paired closed-loop rollouts (4 units x 4 arms x 48 steps + 16-step smoke)", used_in_v04="yes (Table tab:vla0cl)", hardware="NVIDIA GeForce RTX 5090; bf16",
  model_calls=f"{v['policy_calls']} policy calls (replans); {v['generation_calls_total']:,} generation calls", call_definition="policy call = one replan; generation call = natural decode, T=0.7 source call (4 returns) or forced continuation",
  sampling_or_forcing_calls=f"natural {v['gen_natural']}, source sampling {v['gen_sources']}, forced {v['gen_forced']} (318 APPLIED + 160 SELFPATCH_OK)",
  wall_time_min=round(LV[0]["last"] / 60 - LV[0]["start"] / 60, 1), gpu_occupancy_min=f"{dur(LV[0])} (run_gpu_ledger.json; includes load, smoke, all episodes, CPU replays)", peak_vram_gb=8.36,
  renderer="MuJoCo EGL configured (MUJOCO_GL=egl, Mesa vendor dir); post-hoc check returned Mesa llvmpipe; runtime renderer string not logged",
  source="/root/VLA/vla0_cl_work/run_gpu_ledger.json; run/attempt1/STATUS.json (peak_vram_gb per arm, max 8.3595); run/attempt1/*/*/calls.jsonl; env_cl.sh; supplement Sec. L",
  derivation="generation calls counted by presence of t_natural_s / t_sources_s / t_forced_s per call record")
# ---------------- SpatialVLA
d = C["svla_discovery_pilot_attempt2"]; dv = C["svla_discovery_validation_attempt3"]
r(experiment="SpatialVLA within-call discovery (validation 12 + pilot 105 obs)", used_in_v04="yes (Table tab:svla discovery rows; Supp. P)", hardware="NVIDIA GeForce RTX 5090; bf16, eager attention",
  model_calls=f"final runs: pilot {d['model_calls_total']} + validation {dv['model_calls_total']} model calls", call_definition="official natural generate, adapter natural forced pass, R self-patch, each T=0.7 source sample, each forced arm",
  sampling_or_forcing_calls=f"pilot: {d['source_sampling_generate']} source samples (4/obs), {d['R_selfpatch_forced'] + d['arm_forced'] + d['adapter_natural_forced']} forced passes",
  wall_time_min="pilot attempt 2: 6.76; feasibility attempt 3: 1.63 (phase_min)", gpu_occupancy_min="20.1 total across all 7 ledger attempts (feasibility x3, pilot x2, diagnostics x2)",
  peak_vram_gb="UNKNOWN (not logged)", renderer="SimplerEnv / SAPIEN 2.2.2 svulkan2 with software Vulkan (Mesa lavapipe, VK_ICD_FILENAMES) per DEPENDENCIES; runtime device string not logged",
  source="/root/VLA/svla_wc_work/run/gpu_ledger.json; run/pilot/STATUS.json; run/feasibility/STATUS.json; RESULTS_20261022_v01.md 'GPU time'; DEPENDENCIES_20261022_v01.txt; records.jsonl",
  derivation="call counts from records.jsonl; panel-collection rollout calls (to reach calls 0/4/8) not separately logged -> UNKNOWN, included in GPU minutes")
d = C["svla_followup_v02_main"]
r(experiment="SpatialVLA within-call matched follow-up v02 (105 obs; 69 analyzed)", used_in_v04="yes (primary SpatialVLA row; Supp. Q)", hardware="NVIDIA GeForce RTX 5090; bf16, eager attention",
  model_calls=f"main {d['model_calls_total']:,} + smoke {C['svla_followup_v02_smoke']['model_calls_total']}", call_definition="as discovery",
  sampling_or_forcing_calls=f"{d['source_sampling_generate']:,} sequential T=0.7 source samples (max 64/obs); {d['R_selfpatch_forced'] + d['arm_forced'] + d['adapter_natural_forced']} forced passes",
  wall_time_min="main 15.26; smoke 0.83 (phase_min)", gpu_occupancy_min=f"{dur(L3[1])} main + {dur(L3[0])} smoke (gpu_ledger_v03.json) = 16.09 (ledger_min_this_work)", peak_vram_gb="UNKNOWN (not logged)",
  renderer="SAPIEN svulkan2 via /root/VLA/simpler/env.sh (VK_ICD_FILENAMES = Mesa lavapipe); runtime device string not logged",
  source="/root/VLA/integrated_v03_work/svla_v02/run/main/STATUS.json; run/smoke/STATUS.json; /root/VLA/integrated_v03_work/gpu_ledger_v03.json; records.jsonl; /root/VLA/simpler/env.sh (current content)",
  derivation="as discovery; collection-rollout calls not separately logged")
d = C["svla_drawer_recovered"]
r(experiment="SpatialVLA open drawer (supplementary; 44 recovered obs, 35 analyzed)", used_in_v04="yes (Supp. R)", hardware="NVIDIA GeForce RTX 5090; bf16, eager attention",
  model_calls=f"recovered run {d['model_calls_total']:,}; interrupted attempt 1: 0 completed intervention records", call_definition="as discovery",
  sampling_or_forcing_calls=f"{d['source_sampling_generate']:,} source samples; {d['R_selfpatch_forced'] + d['arm_forced'] + d['adapter_natural_forced']} forced passes",
  wall_time_min="recovered 5.80 (phase_min); attempt 1 (collection interrupted) 20.82 (ledger)", gpu_occupancy_min=f"{dur(L3[2])} + {dur(L3[3])} = 26.6 (gpu_ledger_v03.json; ledger_min_this_work 42.71 includes v02 16.09)",
  peak_vram_gb="UNKNOWN (not logged)", renderer="as v02 (SAPIEN svulkan2, lavapipe configured; not logged at runtime)",
  source="/root/VLA/integrated_v03_work/svla_drawer/run/recovered/STATUS.json; /root/VLA/integrated_v03_work/gpu_ledger_v03.json; records.jsonl", derivation="as discovery")
r(experiment="SpatialVLA cross-step P1-D (4,800 rollouts) -- for reference", used_in_v04="yes (Level ii/iii) but not requested", hardware="UNKNOWN in this audit", model_calls="UNKNOWN", call_definition="",
  sampling_or_forcing_calls="UNKNOWN", wall_time_min="UNKNOWN", gpu_occupancy_min="UNKNOWN", peak_vram_gb="UNKNOWN", renderer="UNKNOWN", source="not searched (outside requested list)", derivation="")
pd.DataFrame(rows).to_csv("/root/VLA/v05_work/A_materials/A2_compute_cost_20261023.csv", index=False)
print(pd.DataFrame(rows)[["experiment", "wall_time_min", "gpu_occupancy_min", "peak_vram_gb"]].to_string())
