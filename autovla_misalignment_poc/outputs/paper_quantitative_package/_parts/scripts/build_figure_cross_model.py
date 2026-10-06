#!/usr/bin/env python
"""Build figure_cross_model.csv: summary points + CI + N + source. Values are READ from analysis.json files
(exp 28/33/34) or transcribed from AutoVLA report tables (path given). No new statistics except the SpatialVLA
closed-loop contrasts recomputed by svla_closed_loop_paired.py (read from its CSV)."""
import csv, json
B = "/root/VLA/autovla_misalignment_poc/outputs/"
O = json.load(open(B + "cross_vla_replication/analysis.json"))
I = json.load(open(B + "cross_vla_temporal_replication/analysis.json"))
S = json.load(open(B + "cross_domain_temporal_replication/analysis.json"))
rows = []
def add(model, exp, level, metric, cond, val, lo, hi, n, unit, cluster, test, p, src, key, note=""):
    rows.append(dict(model=model, experiment=exp, level=level, metric=metric, condition=cond, value=val, ci95_lo=lo, ci95_hi=hi,
                     N=n, unit=unit, cluster_unit=cluster, test=test, p=p, source=src, json_key=key, note=note))
def pc(x): return round(100 * x, 2)

# ---------------- AutoVLA (transcribed) ----------------
ah = "outputs/action_history_causal/ACTION_HISTORY_CAUSAL.md (all perturbations, A- table)"
add("AutoVLA", "7", "token/trajectory (open-loop)", "amplification rate (A- & FDE5>3m)", "normal", 47.4, 38, 57, "365 units / 52 scenes", "%", "log", "-", "", ah, "-")
add("AutoVLA", "7", "token/trajectory (open-loop)", "amplification rate", "recent_gt", 7.1, 4, 10, "365 units / 52 scenes", "%", "log", "-", "", ah, "-")
add("AutoVLA", "7", "token/trajectory (open-loop)", "amplification rate", "gt_history", 4.1, 1, 8, "365 units / 52 scenes", "%", "log", "-", "", ah, "-")
add("AutoVLA", "7", "token/trajectory (open-loop)", "delta amplification recent_gt - normal", "contrast", -40.3, -50.3, -30.0, "365 units", "pp", "log", "McNemar", "8e-42", ah, "-")
sp = "outputs/prev_action_state_patching/PREV_ACTION_STATE_PATCHING.md (line 10, table line 333)"
add("AutoVLA", "8", "token/trajectory (open-loop)", "amplification rate", "reverse@emb (GT-history + own prev-token embedding)", 33.4, 26.3, 43.2, "365 units", "%", "log", "-", "", sp, "-", "baseline gt_history 4.4% [1.6, 8.0] in this harness")
add("AutoVLA", "8", "token/trajectory (open-loop)", "delta amplification reverse - gt_history", "contrast", 29.0, 21.1, 38.0, "365 units", "pp", "log", "McNemar", "6.7e-31", sp, "-")
tw = "outputs/temporal_feedback_window/TEMPORAL_FEEDBACK_WINDOW.md §1 (Normal 46.6, win1-4 26.6/14.2/8.5/5.8, Full 4.1)"
for w, f in zip((1, 2, 3, 4), (47, 76, 90, 96)):
    add("AutoVLA", "9", "token/trajectory (open-loop)", "window fraction of full correction effect (amplification)", f"win{w}", f, "", "", "365 units", "%", "log", "-", "", tw, "-", "derived (N-w)/(N-F); 3-step 90% [79, 99] reported")
ms = "outputs/motion_semantics_ablation/RESULTS.md line 26"
add("AutoVLA", "32", "token/trajectory (open-loop)", "dir_wrong_mag_ok - dir_ok_mag_wrong amplification", "contrast", 9.0, 3.1, 18.7, "A- 365 units", "pp", "log", "McNemar", "6e-4", ms, "-")
ed = "outputs/equal_distance_perturbation/EQUAL_DISTANCE.md lines 12, 26"
add("AutoVLA", "6", "token/trajectory (open-loop)", "amplification under distance-matched alt token", "A- scenes", 45.0, 37, 55, "313 units", "%", "log", "-", "", ed, "-")
add("AutoVLA", "6", "token/trajectory (open-loop)", "amplification under distance-matched alt token", "A+ scenes", 4.8, 3, 7, "887 units", "%", "log", "-", "", ed, "-")

# ---------------- Impromptu (exp 33) ----------------
src = "outputs/cross_vla_temporal_replication/analysis.json"
AB, C = I["AB"], I["CDE"]["all_units"]
for k in ("a_minus", "amp", "fde"):
    v = AB["natural"][k]; s = 1 if k == "fde" else 100
    add("Impromptu VLA 3B", "33", "open-loop trajectory", f"natural {k}", "natural", round(s * v["mean"], 3), round(s * v["ci95"][0], 3), round(s * v["ci95"][1], 3), v["n"], "m" if k == "fde" else "%", "log (28)", "-", "", src, f"AB.natural.{k}")
v = AB["rerun_variability"]["natural_vs_rep_amin_discordant"]
add("Impromptu VLA 3B", "33", "open-loop trajectory", "rerun A- disagreement", "natural vs natural_rep", pc(v["mean"]), pc(v["ci95"][0]), pc(v["ci95"][1]), v["n"], "%", "log", "-", "", src, "AB.rerun_variability.natural_vs_rep_amin_discordant")
v = AB["A"]["baseline_flip_natural_ok_to_rep_a_minus"]
add("Impromptu VLA 3B", "33", "open-loop trajectory", "A+ -> A- by rerun only", "natural_rep", pc(v["mean"]), pc(v["ci95"][0]), pc(v["ci95"][1]), v["n"], "%", "log", "-", "", src, "AB.A.baseline_flip_natural_ok_to_rep_a_minus")
for m in ("m0.2", "m0.5"):
    for k, lab in (("flip_to_a_minus_given_m0_ok", "A+ -> A- after perturbation"), ("r_ge_3", "r>=3 (trajectory amplification)"), ("r_le_1", "r<=1 (absorption)")):
        v = AB["A"][m][k]
        add("Impromptu VLA 3B", "33", "open-loop trajectory", lab, f"|delta|={m[1:]} m", pc(v["mean"]), pc(v["ci95"][0]), pc(v["ci95"][1]), v["n"], "%", "log", "-", "", src, f"AB.A.{m}.{k}")
    v = AB["A"][m]["vs_m0_a_minus"]
    add("Impromptu VLA 3B", "33", "open-loop trajectory", "perturbed A- minus pert_m0 A- (paired)", f"|delta|={m[1:]} m", pc(v["mean"]), pc(v["ci95"][0]), pc(v["ci95"][1]), v["n"], "pp", "log", "McNemar", v["mcnemar"]["p"], src, f"AB.A.{m}.vs_m0_a_minus")
    v = AB["B"][m]["coexist_share"]
    add("Impromptu VLA 3B", "33", "open-loop trajectory", "prereg stable/unstable coexistence (same scene)", f"|delta|={m[1:]} m", pc(v["mean"]), pc(v["ci95"][0]), pc(v["ci95"][1]), v["n"], "%", "log", "-", "", src, f"AB.B.{m}.coexist_share", "criterion 3: must exceed rerun 5.9% -> FAIL")
for row in ("normal", "win1", "win2", "win3", "win4", "gt_history", "recent_gt", "reverse", "near_gt", "dir_ok_mag_wrong", "dir_wrong_mag_ok", "random_mag_matched"):
    v = C["means"][row]["amp"]
    add("Impromptu VLA 3B", "33", "open-loop trajectory (context-only)", "amplification rate", row, pc(v["mean"]), pc(v["ci95"][0]), pc(v["ci95"][1]), v["n"], "%", "log", "-", "", src, f"CDE.all_units.means.{row}.amp")
for row in ("win1", "win2", "win3", "win4", "gt_history", "recent_gt", "reverse"):
    v = C["vs_normal"][row]["amp"]
    add("Impromptu VLA 3B", "33", "open-loop trajectory (context-only)", f"delta amplification {row} - normal", "contrast", pc(v["mean"]), pc(v["ci95"][0]), pc(v["ci95"][1]), v["n"], "pp", "log", "McNemar", v["mcnemar"]["p"], src, f"CDE.all_units.vs_normal.{row}.amp")
for k in ("amp", "d10"):
    v = C["reverse_vs_gt_history"][k]; s = 100 if k == "amp" else 1
    add("Impromptu VLA 3B", "33", "open-loop trajectory (context-only)", f"reverse - gt_history {k}", "contrast", round(s * v["mean"], 3), round(s * v["ci95"][0], 3), round(s * v["ci95"][1], 3), v["n"], "pp" if k == "amp" else "m", "log",
        "McNemar" if k == "amp" else "Wilcoxon", v["mcnemar"]["p"] if k == "amp" else v["wilcoxon_p"], src, f"CDE.all_units.reverse_vs_gt_history.{k}")
for row in ("win1", "win2", "win3", "win4", "recent_gt"):
    v = C["window_frac_of_full"][row]["amp"]
    add("Impromptu VLA 3B", "33", "open-loop trajectory (context-only)", "fraction of full correction effect (amplification)", row, pc(v["frac"]), pc(v["ci95"][0]), pc(v["ci95"][1]), 2398, "%", "log", "-", "", src, f"CDE.all_units.window_frac_of_full.{row}.amp")
for c in ("dir_wrong_mag_ok - dir_ok_mag_wrong", "near_gt - recent_gt"):
    for k in ("amp", "d10"):
        v = C["motion"][c][k]; s = 100 if k == "amp" else 1
        add("Impromptu VLA 3B", "33", "open-loop trajectory (context-only)", f"{c} {k}", "contrast", round(s * v["mean"], 3), round(s * v["ci95"][0], 3), round(s * v["ci95"][1], 3), v["n"], "pp" if k == "amp" else "m", "log",
            "McNemar" if k == "amp" else "Wilcoxon", v["mcnemar"]["p"] if k == "amp" else v["wilcoxon_p"], src, f"CDE.all_units.motion.{c}.{k}")
X = json.load(open(B + "cross_vla_temporal_replication/exploratory_matched_first_substitution.json"))["dir_wrong_minus_dir_ok"]
add("Impromptu VLA 3B", "33", "open-loop trajectory (exploratory)", "matched-e first substitution: next-waypoint error dir_wrong - dir_ok", "contrast", round(X["mean"], 3), round(X["ci95"][0], 3), round(X["ci95"][1], 3), X["n"], "m", "log", "Wilcoxon", X["wilcoxon_p"],
    "outputs/cross_vla_temporal_replication/exploratory_matched_first_substitution.json", "dir_wrong_minus_dir_ok", "EXPLORATORY (not preregistered)")

# ---------------- OpenVLA (exp 28) ----------------
src = "outputs/cross_vla_replication/analysis.json"
add("OpenVLA-7B", "28", "closed-loop task", "natural success", "natural", 87.0, "", "", 100, "%", "task (10)", "-", "", src, "natural_success.rate", "87/100; per-condition CI not computed by script")
v = O["natural_rep_vs_natural"]
add("OpenVLA-7B", "28", "closed-loop task", "rerun success - natural", "natural_rep", pc(v["diff"]), pc(v["ci95_task_boot"][0]), pc(v["ci95_task_boot"][1]), 100, "pp", "task (10)", "McNemar exact / task sign-flip perm", f"{v['p_mcnemar']:.3g} / {v['p_perm_tasks']:.3g}", src, "natural_rep_vs_natural", "outcome disagreement 20/100")
for c in sorted(O["vs_natural"]):
    v = O["vs_natural"][c]
    add("OpenVLA-7B", "28", "closed-loop task", f"success - natural", c, pc(v["diff"]), pc(v["ci95_task_boot"][0]), pc(v["ci95_task_boot"][1]), v["n"], "pp", "task (10)", "McNemar exact / task sign-flip perm", f"{v['p_mcnemar']:.3g} / {v['p_perm_tasks']:.3g}", src, f"vs_natural.{c}")
for k in ("feedback_vs_corrected_d8", "feedback_vs_corrected_d24", "reverse_vs_natural_d8", "reverse_vs_natural_d24"):
    v = O[k]
    add("OpenVLA-7B", "28", "closed-loop task", k, "contrast", pc(v["diff"]), pc(v["ci95_task_boot"][0]), pc(v["ci95_task_boot"][1]), v["n"], "pp", "task (10)", "McNemar exact / task sign-flip perm", f"{v['p_mcnemar']:.3g} / {v['p_perm_tasks']:.3g}", src, k)
for key in ("d0_|8|_feedback", "d0_|24|_feedback", "d1_|8|_feedback", "d1_|24|_feedback", "d0_|8|_corrected", "d0_|8|_reverse",
            "d0_|8|_window_1", "d0_|8|_window_2", "d0_|8|_window_3", "d0_|8|_window_4"):
    v = O["phaseC"][key]
    add("OpenVLA-7B", "28", "within-step token (offline)", "downstream amplification (max|dbin|>=|delta|)", key, pc(v["amplified"]), pc(v["amplified_ci95"][0]), pc(v["amplified_ci95"][1]), v["n"], "%", "task (10), 1000 reps", "-", "", src, f"phaseC.{key}", f"D={v['D']:.2f} bins, recovered={pc(v['recovered'])}%")

# ---------------- SpatialVLA (exp 34) ----------------
src = "outputs/cross_domain_temporal_replication/analysis.json"
CL, TL = S["closed_loop"], S["token_level"]
for c, v in CL["success"].items():
    add("SpatialVLA-4B", "34", "closed-loop task", "success rate", c, pc(v["mean"]), pc(v["ci95"][0]), pc(v["ci95"][1]), v["n"], "%", "episode", "-", "", src, f"closed_loop.success.{c}")
for c, v in CL["vs_natural"].items():
    add("SpatialVLA-4B", "34", "closed-loop task", "success - natural", c, pc(v["mean"]), pc(v["ci95"][0]), pc(v["ci95"][1]), v["n"], "pp", "episode", "McNemar exact", v["mcnemar"]["p"], src, f"closed_loop.vs_natural.{c}")
for c, v in CL["final_traj_divergence_vs_natural_m"].items():
    add("SpatialVLA-4B", "34", "closed-loop trajectory", "endpoint divergence vs natural", c, round(v["mean"], 4), round(v["ci95"][0], 4), round(v["ci95"][1], 4), v["n"], "m", "episode", "-", "", src, f"closed_loop.final_traj_divergence_vs_natural_m.{c}")
for r in csv.DictReader(open(B + "paper_quantitative_package/spatialvla_closed_loop_paired.csv")):
    if r["task"] == "all" and r["block"] in ("context_feedback_effect", "executed_action_effect"):
        add("SpatialVLA-4B", "34", "closed-loop task", r["block"] + ": " + r["label"], "contrast", r["diff_pp_a_minus_b"], r["ci95_lo_pp"], r["ci95_hi_pp"], r["N_episodes"], "pp", "episode", "McNemar exact", r["mcnemar_exact_p"],
            "outputs/paper_quantitative_package/spatialvla_closed_loop_paired.csv", r["label"], "recomputed: _parts/scripts/svla_closed_loop_paired.py")
A = TL["all"]
for row in ("normal", "recent_ref", "full_ref", "win1", "reverse"):
    for k in ("amp", "dev4"):
        v = A["means"][row][k]; s = 100 if k == "amp" else 1
        add("SpatialVLA-4B", "34", "token (offline)", "amplification A>=1" if k == "amp" else "step-4 deviation (normalized)", row, round(s * v["mean"], 4), round(s * v["ci95"][0], 4), round(s * v["ci95"][1], 4), v["n"], "%" if k == "amp" else "norm. units", "episode (task,seed)", "-", "", src, f"token_level.all.means.{row}.{k}")
for row in ("recent_ref", "full_ref", "win1"):
    for k in ("amp", "dev4"):
        v = A["vs_normal"][row][k]; s = 100 if k == "amp" else 1
        add("SpatialVLA-4B", "34", "token (offline)", f"{row} - normal {k}", "contrast", round(s * v["mean"], 4), round(s * v["ci95"][0], 4), round(s * v["ci95"][1], 4), v["n"], "pp" if k == "amp" else "norm. units", "episode", "McNemar" if k == "amp" else "Wilcoxon", v["mcnemar"]["p"] if k == "amp" else v["wilcoxon_p"], src, f"token_level.all.vs_normal.{row}.{k}")
for k in ("amp", "dev4"):
    v = A["reverse_vs_full_ref"][k]; s = 100 if k == "amp" else 1
    add("SpatialVLA-4B", "34", "token (offline)", f"reverse - full_ref {k}", "contrast", round(s * v["mean"], 4), round(s * v["ci95"][0], 4), round(s * v["ci95"][1], 4), v["n"], "pp" if k == "amp" else "norm. units", "episode", "McNemar" if k == "amp" else "Wilcoxon", v["mcnemar"]["p"] if k == "amp" else v["wilcoxon_p"], src, f"token_level.all.reverse_vs_full_ref.{k}")
v = TL["motion"]["dir_wrong_mag_ok - dir_ok_mag_wrong"]["amp"]
add("SpatialVLA-4B", "34", "token (offline)", "dir_wrong_mag_ok - dir_ok_mag_wrong amp", "contrast", pc(v["mean"]), pc(v["ci95"][0]), pc(v["ci95"][1]), v["n"], "pp", "episode", "McNemar", v["mcnemar"]["p"], src, "token_level.motion.dir_wrong_mag_ok - dir_ok_mag_wrong.amp")
v = TL["motion"]["near_ref - ref_trans"]["amp"]
add("SpatialVLA-4B", "34", "token (offline)", "near_ref - ref_trans amp", "contrast", pc(v["mean"]), pc(v["ci95"][0]), pc(v["ci95"][1]), v["n"], "pp", "episode", "McNemar", v["mcnemar"]["p"], src, "token_level.motion.near_ref - ref_trans.amp")
for dd in ("d0.15", "d0.3"):
    v = TL["B"][dd]["coexist_recovered_and_amplified"]
    add("SpatialVLA-4B", "34", "token (offline)", "same-frame coexistence recovered & amplified", dd, pc(v["mean"]), pc(v["ci95"][0]), pc(v["ci95"][1]), v["n"], "%", "episode", "-", "", src, f"token_level.B.{dd}.coexist_recovered_and_amplified")
nr, fr = A["means"]["normal"]["dev4"]["mean"], A["means"]["full_ref"]["dev4"]["mean"]
for row in ("recent_ref", "win1"):
    f = (nr - A["means"][row]["dev4"]["mean"]) / (nr - fr)
    add("SpatialVLA-4B", "34", "token (offline)", "fraction of full_ref step-4 deviation reduction", row, round(100 * f, 1), "", "", TL["n_units"], "%", "episode", "-", "", src, f"derived from token_level.all.means.{{normal,full_ref,{row}}}.dev4", "derived ratio; CI not reported")
fields = list(rows[0].keys())
with open(B + "paper_quantitative_package/figure_cross_model.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
print(len(rows), "rows")
