# Source map — cross-model replication

Root: `/root/VLA/autovla_misalignment_poc/` (paths below relative to it). Commits from `git -C /root/VLA log --oneline -- <path>`.
All model runs used RTX 5090 (per each RESULTS/PROTOCOL; AutoVLA exps 6–9 "GPU1" = RTX 5090 per TEMPORAL_FEEDBACK_WINDOW.md §2).
"recomputed" = reproduced in this package from raw data with the original script's functions/method.

## Common to each experiment

| Exp | Name | Model | Benchmark | RESULTS | Raw data | Prereg commit | Analysis/results commit | Seeds | Script |
|---|---|---|---|---|---|---|---|---|---|
| 6 | Equal-distance perturbation | AutoVLA (Qwen2.5-VL-3B, natural fast) | NAVSIM navtest PoC | `outputs/equal_distance_perturbation/EQUAL_DISTANCE.md` | `rows.jsonl`, `records.jsonl` | none (discovery) | 1541f30 | sampling T 0.01, fixed seeds | (PoC scripts) |
| 7 | Action-history causal | AutoVLA | navtest PoC | `outputs/action_history_causal/ACTION_HISTORY_CAUSAL.md` | `units.jsonl`, `records.jsonl` | none | 1541f30 | seed = sha256(token:pert:k) | — |
| 8 | Prev-action state patching | AutoVLA | navtest PoC | `outputs/prev_action_state_patching/PREV_ACTION_STATE_PATCHING.md` | `layer_results.json`, `summary.json` | none | 1541f30 | same harness | — |
| 9 | Temporal feedback window | AutoVLA | navtest PoC | `outputs/temporal_feedback_window/TEMPORAL_FEEDBACK_WINDOW.md` | `summary.json` | none | 1541f30 | same harness | — |
| 11 | Prev-action identity | AutoVLA | navtest PoC | `outputs/prev_action_identity_decomposition/PREV_ACTION_IDENTITY.md` | `units.jsonl` | none | bf15767 | — | — |
| 32 | Motion semantics | AutoVLA | navtest PoC | `outputs/motion_semantics_ablation/RESULTS.md` | `records.jsonl`, `row_contrasts.json` | af2eb10 (protocol) | aee41e5 | — | — |
| 28 | Cross-VLA (P1) | OpenVLA-7B `openvla/openvla-7b-finetuned-libero-spatial` (bf16) | LIBERO-Spatial 10 tasks × inits 0–9 | `outputs/cross_vla_replication/RESULTS.md` | `rollouts/phaseA,phaseA_rep,phaseB/episodes.jsonl` (100/100/600 lines; untracked in git), `token_feedback/token_feedback.jsonl` (2,462 lines) | 6b9e8ad; amendment 6d0375a (after part of Phase B, before final analysis) | 69402c6 | init states 0–9; greedy; bootstrap seed 0 | `scripts/cross_vla/analyze_cross_vla.py` |
| 33 | Temporal action-chunk replication | Impromptu VLA 3B `aaaaaap/ImpromptuVLAModel/3B_AD` (Qwen2.5-VL-3B) | navtest PoC, 28 logs, 2,748 scenes (2,728 parsable) | `outputs/cross_vla_temporal_replication/RESULTS.md` | `exp_ab/records.jsonl`, `exp_cde/records.jsonl` (+`selected_scenes.json`) | 038a6d3 | 698148a | greedy (+rep. penalty 1.05); C–E scene sample seed 0; bootstrap seed 0 | `scripts/cross_vla_temporal/analyze_temporal.py` |
| 34 | Cross-domain temporal replication | SpatialVLA-4B `IPEC-COMMUNITY/spatialvla-4b-224-sft-fractal` | SimplerEnv Google Robot pick_coke_can, move_near; env seeds 0–39 | `outputs/cross_domain_temporal_replication/RESULTS.md` | `closed_loop/episodes.jsonl` (720 episodes = 9 conds × 80), `token_level/units.jsonl` (1,600 frames) | f578eae | f4cf886 | env seeds 0–39; greedy; bootstrap seed 0 | `scripts/cross_domain_temporal/analyze_svla.py` |
| 35 (record only) | Feedback-protection ablation | SpatialVLA-4B | SimplerEnv | (running; not used) | — | 8fc20bf (PROTOCOL.md records ensemble-weight correction) | — | — | — |

## Statistical methods (read from scripts)

| Exp | CI | Cluster unit | Binary test | Continuous test | Notes |
|---|---|---|---|---|---|
| AutoVLA 6–11, 32 | log-cluster bootstrap 95%, 2,000 | log | McNemar | Wilcoxon | as stated in each report |
| 28 closed loop | task-cluster bootstrap, 2,000, `random.Random(0)`, percentile | task (10) | McNemar exact (binomtest) over (task, init) | — | + exact task-level sign-flip permutation (2^10 enumerations; protocol said 20,000) |
| 28 Phase C | task-cluster bootstrap, **1,000** | task | — | — | means over (frame, sign) units; units with clipped δ_eff = 0 dropped |
| 33 | log-cluster bootstrap, 2,000, `random.Random(0)`, order-statistic | log (28) | McNemar exact over units | Wilcoxon over units | window fraction: (normal − row)/(normal − gt_history), log bootstrap |
| 34 closed loop | episode-cluster bootstrap, 2,000, `random.Random(0)` re-seeded per call, **unstratified** (docstring says stratified — see conflicts m5) | episode (task, seed); 80 | McNemar exact | Wilcoxon | divergence = ‖Σ executed world_vector endpoint(cond) − (natural)‖ |
| 34 token | same bootstrap over units clustered by episode (task, seed) | episode | McNemar | Wilcoxon | |

## Key results → source

| # | Result | Value | N | Metric definition | Test / cluster | Source path + JSON key |
|---|---|---|---|---|---|---|
| O1 | OpenVLA natural success | 87/100 = 87% | 100 eps | LIBERO success | — | `outputs/cross_vla_replication/analysis.json` `natural_success.rate` |
| O2 | OpenVLA official success | 84.7% | — | — | — | `/root/VLA/EXPERIMENT_SUMMARY.md` §6 only (not in exp 28 files) |
| O3 | OpenVLA rerun | 81/100; −6 pp [−14, +2]; disagreement 20/100; McNemar p 0.263; perm p 0.305 | 100 | — | McNemar / task boot | `natural_rep_vs_natural` |
| O4 | OpenVLA within-step amp, x, 8 bins | 92.58% [89.81, 94.35]; D 87.73 bins; recovered 1.5% | 4,741 units / 2,462 frames | max_j \|Δbin_j\| ≥ \|δ\|, j=d+1..5 | task boot 1,000 | `phaseC["d0_\|8\|_feedback"]` |
| O5 | x, 24 bins | 81.16% [77.16, 83.95] | 4,741 | same | same | `phaseC["d0_\|24\|_feedback"]` |
| O6 | y, 8 / 24 bins | 87.37% [85.36, 89.15] / 75.33% [72.38, 77.82] | 4,869 | same | same | `phaseC["d1_\|8\|_feedback"]`, `["d1_\|24\|_feedback"]` |
| O7 | corrected / reverse token | 0% (rec 100%) / identical to feedback | 4,741 | same | — | `phaseC["d0_\|8\|_corrected"]`, `["d0_\|8\|_reverse"]` |
| O8 | Dimension "window" 1–4 (x, 8) | amp 84.5 / 66.9 / 53.4 / 26.8% | 4,741 | same | task boot | `phaseC["d0_\|8\|_window_k"]` |
| O9 | Task: feedback − corrected | +3 pp [−4, +9] p 0.648 (8); +5 pp [−4, +14] p 0.359 (24) | 100 | success | McNemar / task boot / perm 0.594, 0.426 | `feedback_vs_corrected_d8`, `_d24` |
| O10 | Task: reverse − natural | −3 pp [−9, +3] p 0.581 (8); −8 pp [−15, 0] p 0.134 (24) | 100 | success | same; perm 0.563, 0.141 | `reverse_vs_natural_d8`, `_d24` |
| O11 | Task: each condition − natural | −3 … −8 pp, min McNemar p 0.115 (corrected_d8) | 100 | success | same | `vs_natural.*` |
| O12 | Fail given natural success | 9.2–18.4% vs rerun 15% (13/87) | 87 | — | — | `fail_given_natural_success`; rerun share from `natural_rep_vs_natural.b_only`/87 |
| I1 | Impromptu natural | A− 33.36% [26.20, 38.04]; amp 31.49%; FDE5 5.748 m [5.174, 6.273] | 2,728 scenes / 28 logs | P/R/A A−; amp = A− ∧ FDE5 > 3 m | log boot | `AB.natural.*` |
| I2 | Rerun disagreement | A− 5.90% [5.20, 6.85]; amp 5.87%; final wp shift 0.899 m [0.842, 0.956]; A−-rate diff +0.18 pp, McNemar 83 vs 78 p 0.753 | 2,728 | — | log boot | `AB.rerun_variability.*` |
| I3 | Rerun A+→A− | 4.57% [3.74, 5.52] | 1,818 | — | log boot | `AB.A.baseline_flip_natural_ok_to_rep_a_minus` |
| I4 | 0.2 m: A+→A− | 49.49% [46.91, 52.50] | 14,424 units | given pert_m0 A+ | log boot | `AB.A.m0.2.flip_to_a_minus_given_m0_ok` |
| I5 | 0.2 m: paired A− vs pert_m0 | +24.78 pp [+19.44, +32.16], McNemar 7,138 vs 1,731 p≈0 | 21,824 | — | McNemar / log boot | `AB.A.m0.2.vs_m0_a_minus` |
| I6 | 0.2 m: r≥3 / r≤1; median r | 97.39% [96.91, 97.89] / 0.45% [0.34, 0.60]; 27.56 | 21,824 | r = D10/\|δ\| vs pert_m0 | log boot | `AB.A.m0.2.r_ge_3`, `r_le_1`, `r_median` |
| I7 | Prereg coexistence (crit. 3) | 2.24% [1.54, 3.21] (0.2), 3.15% [2.36, 3.88] (0.5) < 5.9% → FAIL | 2,728 | stable (A+ ∧ r≤1) & unstable (amp ∨ r≥3) same scene | log boot | `AB.B.m0.2.coexist_share`, `m0.5` |
| I8 | Correction | normal 63.43% [60.59, 66.67] → recent_gt 1.25% [0.81, 1.77]; Δ −62.18 pp [−65.57, −59.13]; gt_history 0.0% | 2,398 units / 300 scenes / 28 logs | amp | McNemar p 0.0 (underflow) / log boot | `CDE.all_units.means.*.amp`, `vs_normal.recent_gt.amp` |
| I9 | Window 1/2/3/4 (frac of full, amp) | 24.3 [15.6, 33.8] / 38.7 [29.7, 47.8] / 90.5 [87.0, 95.4] / 94.9 [91.8, 98.6]; recent_gt 98.0 | 2,398 | (normal−row)/(normal−gt_history) | log boot | `CDE.all_units.window_frac_of_full.*.amp` |
| I10 | Reverse | 0.0 → 81.90% [80.37, 84.19]; Δamp +81.9 pp (McNemar 1,964 vs 0); ΔD10 +40.52 m [+37.32, +43.86] Wilcoxon p≈0 | 2,398 | — | — | `CDE.all_units.reverse_vs_gt_history.amp`, `.d10` |
| I11 | near_gt − recent_gt | −0.08 pp [−0.60, +0.30], p 0.875 | 2,398 | amp | McNemar | `CDE.all_units.motion["near_gt - recent_gt"].amp` |
| I12 | direction − magnitude | +15.64 pp [+8.99, +24.25] p 2.0e-31; ΔFDE +14.52 m; ΔD10 +18.44 m [+15.90, +20.09] p 1.8e-268 | 2,398 | — | McNemar / Wilcoxon | `CDE.all_units.motion["dir_wrong_mag_ok - dir_ok_mag_wrong"]` |
| I13 | matched-e first sub (exploratory) | +0.294 m [+0.253, +0.326], p 2.0e-99 | 2,398 | next-wp error at w3 | Wilcoxon | `exploratory_matched_first_substitution.json` `dir_wrong_minus_dir_ok` |
| S1 | SpatialVLA natural | 68/80 = 85.0% [77.5, 92.5]; coke 37/40, move near 31/40 | 80 | success = done at last step | episode boot | `outputs/cross_domain_temporal_replication/analysis.json` `closed_loop.success.natural`; per task recomputed |
| S2 | Rerun natural_gen | 75/80 = 93.75%; Δ +8.75 pp [+1.25, +16.25]; McNemar 9 vs 2 p 0.0654; disagreement 13.75%; divergence 0.1396 m [0.1002, 0.1855] | 80 | — | McNemar / episode boot | `closed_loop.vs_natural.natural_gen`, `discordant_vs_natural.natural_gen`, `final_traj_divergence_vs_natural_m.natural_gen` |
| S3 | Closed loop paired table | see `spatialvla_closed_loop_paired.csv` | 80 | — | same | recomputed: `_parts/scripts/svla_closed_loop_paired.py` (identical to `closed_loop.*`) |
| S4 | Token step-2/3/4 change | 10,542 / 6,824 / 5,286 of 18,975 = 55.56 / 35.96 / 27.86% | 18,975 | normal-row translation token ≠ R | descriptive (no CI in original) | recomputed: `_parts/scripts/svla_token_step_change.py` |
| S5 | Token normal amp / absorption | 22.06% [19.47, 24.70] / 36.67% [33.07, 40.18] | 18,975 units / 1,600 frames / 80 eps | A ≥ 1 / steps 2–4 tokens = R | episode boot | `token_level.all.means.normal.amp`, `.rec` |
| S6 | Correction | recent_ref − normal amp −8.31 pp [−9.31, −7.38] p 6.7e-261; full_ref −8.55 pp [−9.62, −7.54] p 2.0e-273; dev4 −0.0462 [−0.0514, −0.0413] (−82%) | 18,975 | — | McNemar / Wilcoxon | `token_level.all.vs_normal.{recent_ref,full_ref}` |
| S7 | Reverse | reverse − full_ref amp +5.10 pp [+4.39, +5.84] (1,180 vs 213) p 1.9e-162; dev4 +0.0468 [+0.0418, +0.0519] p≈0; reverse dev4 0.0566 vs normal 0.0561 | 18,975 | — | McNemar / Wilcoxon | `token_level.all.reverse_vs_full_ref` |
| S8 | Window | recent_ref 86.3%, win1 62.6% of full_ref dev4 reduction; win1 − full_ref amp +1.88 pp [+1.49, +2.32] p 6.3e-68 | 18,975 | (normal − row)/(normal − full_ref) on dev4 | derived; CI not reported | `token_level.all.means.*.dev4`, `win1_vs_full_ref` |
| S9 | Coexistence | d 0.15: 25.1% [21.8, 28.7]; d 0.30: 15.3% [12.8, 17.9] | 1,600 frames | same frame/d: ≥1 recovered & ≥1 amplified | episode boot | `token_level.B.d0.15/d0.3.coexist_recovered_and_amplified` |
| S10 | Direction vs magnitude | +1.61 pp [+0.40, +2.84] p 7.3e-5; dev4 +0.0126 [+0.0088, +0.0171] p 5.5e-17; near_ref − ref_trans +4.31 pp [+3.17, +5.63] p 4.2e-31 | 10,542 motion units | — | McNemar / Wilcoxon | `token_level.motion` |
| S11 | Ensemble weights | 0.5741 / 0.2579 / 0.1159 / 0.0521 (newest-first); steps 2–4 42.6% | — | exp(0.8 i)/Σ | analytic | `scripts/cross_domain_temporal/action_ensemble.py`; recomputed `svla_token_step_change.py`; also exp 35 `PROTOCOL.md` §0-1 |
| A1 | AutoVLA correction | 47.4% [38, 57] → 7.1% [4, 10]; Δ −40.3 pp [−50.3, −30.0] p 8e-42 | 365 A− units / 52 scenes | amp = A− ∧ FDE > 3 m | McNemar / log boot | `outputs/action_history_causal/ACTION_HISTORY_CAUSAL.md` (A− table) |
| A2 | AutoVLA reverse@emb | 4.4% → 33.4% [26.3, 43.2]; +29.0 pp [+21.1, +38.0] p 6.7e-31 | 365 | — | McNemar | `outputs/prev_action_state_patching/PREV_ACTION_STATE_PATCHING.md` l.10, l.333 |
| A3 | AutoVLA window | 46.6 → 26.6/14.2/8.5/5.8, full 4.1 → 47/76/90/96% | 365 | — | log boot | `outputs/temporal_feedback_window/TEMPORAL_FEEDBACK_WINDOW.md` §1 |
| A4 | AutoVLA direction > magnitude | +9.0 pp [+3.1, +18.7] p 6e-4; FDE +1.21 m | A− 365 | — | McNemar | `outputs/motion_semantics_ablation/RESULTS.md` l.26 |
| A5 | AutoVLA matched-distance | A− scenes amp 45.0% [37, 55] (313) vs A+ 4.8% [3, 7] (887) | — | — | log boot | `outputs/equal_distance_perturbation/EQUAL_DISTANCE.md` l.12, l.26 |
| A6 | AutoVLA natural | A− 52/2,747 = 1.9%; FDE5 0.66 m [0.48, 0.87] (500-scene subset) | — | — | — | `outputs/natural_reference_stabilization/NATURAL_REFERENCE.md` l.14; `outputs/fast_vs_cot_unbiased/FAST_VS_COT.md` l.9 |
