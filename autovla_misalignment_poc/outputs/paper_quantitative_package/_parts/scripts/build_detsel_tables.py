#!/usr/bin/env python
"""
Build paper_detection_table.{csv,tex}, figure_detection_pre_post.csv and figure_candidate_count.csv
from EXISTING summary JSONs only (no recomputation of any statistic; only formatting and simple ratios
that are labelled 'derived'). CPU only, writes only into outputs/paper_quantitative_package/.

  nice -n 19 python build_detsel_tables.py
"""
import csv, json, os

O = "/root/VLA/autovla_misalignment_poc/outputs"
PKG = os.path.join(O, "paper_quantitative_package")
DET = os.path.join(O, "instability_detection_baselines/detection.json")
REL = lambda p: os.path.relpath(p, "/root/VLA")  # noqa: E731

d = json.load(open(DET))
dev, ho = d["dev"], d["heldout"]
ROWS = [("entropy", "Entropy"), ("loglik", "Log-likelihood"), ("margin", "Probability margin"),
        ("cand_variance", "Candidate variance (scene-level)"), ("disagreement", "Pairwise disagreement"),
        ("medoid_dist", "Medoid distance"), ("probe_logreg", "Hidden logistic probe"),
        ("probe_ridge", "Hidden ridge probe"), ("probe_mlp", "Hidden MLP probe")]

f4 = lambda x: f"{x:.4f}"  # noqa: E731
hdr = ["detector_key", "detector", "pre_auroc", "pre_auroc_ci95_lo", "pre_auroc_ci95_hi", "pre_within_scene_auroc",
       "post_auroc", "post_auroc_ci95_lo", "post_auroc_ci95_hi", "post_within_scene_auroc", "post_auprc",
       "post_auprc_ci95_lo", "post_auprc_ci95_hi", "pre_auprc", "prevalence_heldout", "n_amplifying_heldout",
       "dev_N", "heldout_N", "heldout_n_mixed_scenes", "dev_pre_auroc", "dev_post_auroc", "dev_value_type",
       "probe_layer", "probe_hparam", "note", "source_json_keys"]
rows = []
for k, name in ROWS:
    pre, post = ho[f"{k}_pre"], ho[f"{k}_post"]
    if k.startswith("probe"):
        dpre, dpost, dtype = f4(pre["dev_cv_auroc"]), f4(post["dev_cv_auroc"]), "dev log-grouped 3-fold CV AUROC (probe selection)"
        note = "trained on dev only; layer+hparam chosen by dev 3-fold CV; applied once to held-out"
        lay, hp = f"pre L{pre['layer']} / post L{post['layer']}", f"pre {pre['hparam']} / post {post['hparam']}"
    else:
        dpre, dpost, dtype = f4(dev[f"{k}_pre"]["auroc"]), f4(dev[f"{k}_post"]["auroc"]), "dev pooled AUROC (fixed sign, no training)"
        note, lay, hp = "fixed sign, no tuning", "", ""
    if k == "cand_variance":
        note += "; scene-level score (nearly constant within scene) so within-scene AUROC is not meaningful (report shows it in parentheses)"
    rows.append([k, name, f4(pre["auroc"]), f4(pre["auroc_ci95"][0]), f4(pre["auroc_ci95"][1]), f4(pre["within_scene_auroc"]),
                 f4(post["auroc"]), f4(post["auroc_ci95"][0]), f4(post["auroc_ci95"][1]), f4(post["within_scene_auroc"]),
                 f4(post["auprc"]), f4(post["auprc_ci95"][0]), f4(post["auprc_ci95"][1]), f4(pre["auprc"]),
                 f4(post["prevalence"]), round(post["prevalence"] * post["n"]), dev[f"entropy_pre"]["n"], post["n"],
                 post["n_mixed_scenes"], dpre, dpost, dtype, lay, hp, note,
                 f"{REL(DET)}: heldout.{k}_pre / heldout.{k}_post; dev.{k}_pre/post" + (" (dev_cv_auroc)" if k.startswith("probe") else "")])
with open(os.path.join(PKG, "paper_detection_table.csv"), "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(hdr); w.writerows(rows)

# figure (long format, dev + held-out)
fig = [["split", "detector_key", "window", "auroc", "auroc_ci95_lo", "auroc_ci95_hi", "auprc", "auprc_ci95_lo",
        "auprc_ci95_hi", "within_scene_auroc", "n_mixed_scenes", "prevalence", "n_candidates", "source"]]
for split, D in (("heldout", ho), ("dev", dev)):
    for k, _ in ROWS:
        for w_ in ("pre", "post"):
            key = f"{k}_{w_}"
            if key not in D:
                fig.append([split, k, w_] + ["not reported"] * 10 + [f"{REL(DET)}: {split}.{key} absent (probes have only dev_cv_auroc)"])
                continue
            m = D[key]
            fig.append([split, k, w_, f4(m["auroc"]), f4(m["auroc_ci95"][0]), f4(m["auroc_ci95"][1]), f4(m["auprc"]),
                        f4(m["auprc_ci95"][0]), f4(m["auprc_ci95"][1]), f4(m["within_scene_auroc"]), m["n_mixed_scenes"],
                        f4(m["prevalence"]), m["n"], f"{REL(DET)}: {split}.{key}"])
with open(os.path.join(PKG, "figure_detection_pre_post.csv"), "w", newline="") as fh:
    csv.writer(fh).writerows(fig)

# LaTeX
t3 = lambda x: f"{x:.3f}"  # noqa: E731
L = [r"\begin{table}[t]", r"\centering", r"\small", r"\setlength{\tabcolsep}{4pt}",
     r"\caption{Held-out detection of amplifying rollouts before (pre) and after (post) the first deviation from the GT token "
     r"(experiment 31/P4; AutoVLA, NAVSIM navtest held-out split, 52 logs, 4{,}814 scenes, RTX 5090 decodes). "
     r"Unit: candidate rollout (natural plan + 16 samples at $T{=}1.0$ per scene) that first deviates at $t^*<9$; "
     r"label: amplification (open-loop A$-$ and FDE$_{5s}>3$\,m). "
     rf"Held-out $N={ho['entropy_pre']['n']:,}$ candidates ({round(ho['entropy_pre']['prevalence']*ho['entropy_pre']['n']):,} amplifying, prevalence {ho['entropy_pre']['prevalence']:.3f} = AUPRC of a random detector); "
     rf"dev $N={dev['entropy_pre']['n']:,}$ (56 logs, prevalence {dev['entropy_pre']['prevalence']:.3f}). "
     r"Pre uses information up to and including the step that emits $t^*$; post uses steps $t^*{+}1..9$. "
     r"Heuristic detectors have signs fixed in advance; hidden-state probes (layer 18 selected for all) were trained on dev only and applied once. "
     r"95\% CIs: log-cluster bootstrap (1{,}000 resamples, percentile). Within-scene AUROC: mean over scenes containing both classes "
     rf"({ho['entropy_pre']['n_mixed_scenes']:,} held-out scenes); no CI reported. All detectors are scored on the same candidates (paired design), "
     r"but no paired test of AUROC differences between detectors was reported. "
     r"$^\dagger$Scene-level score, nearly constant within a scene, so its within-scene AUROC is not meaningful.}",
     r"\label{tab:detection}", r"\begin{tabular}{lccccccc}", r"\toprule",
     r" & \multicolumn{3}{c}{Pre-deviation} & \multicolumn{4}{c}{Post-deviation} \\",
     r"\cmidrule(lr){2-4}\cmidrule(lr){5-8}",
     r"Detector & AUROC & 95\% CI & Within & AUROC & 95\% CI & Within & AUPRC \\", r"\midrule"]
for i, (k, name) in enumerate(ROWS):
    if i == 6:
        L.append(r"\midrule")
    pre, post = ho[f"{k}_pre"], ho[f"{k}_post"]
    wpre, wpost = t3(pre["within_scene_auroc"]), t3(post["within_scene_auroc"])
    nm = name
    if k == "cand_variance":
        nm, wpre, wpost = "Candidate variance$^\\dagger$", f"({wpre})", f"({wpost})"
    L.append(f"{nm} & {t3(pre['auroc'])} & [{t3(pre['auroc_ci95'][0])}, {t3(pre['auroc_ci95'][1])}] & {wpre} & "
             f"{t3(post['auroc'])} & [{t3(post['auroc_ci95'][0])}, {t3(post['auroc_ci95'][1])}] & {wpost} & {t3(post['auprc'])} \\\\")
L += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
open(os.path.join(PKG, "paper_detection_table.tex"), "w").write("\n".join(L))

# candidate count curve
cc_rows = [["population", "decode_gpu", "N_candidates_incl_natural", "subset_scheme_openloop", "n_scenes", "n_logs",
            "a_minus", "d_a_minus", "d_a_minus_ci95_lo", "d_a_minus_ci95_hi", "ade5_m", "d_ade5_m", "fde5_m", "d_fde5_m",
            "d_fde5_ci95_lo", "d_fde5_ci95_hi", "pdms_prefix", "d_pdms", "d_pdms_ci95_lo", "d_pdms_ci95_hi", "d_pdms_wilcoxon_p",
            "saturation_frac_pdms(derived=dPDMS_N/dPDMS_17)", "saturation_frac_a_minus(derived=dA-_N/dA-_17)",
            "rel_inference_cost_3080Ti", "rel_inference_cost_5090", "source"]]
COST3080 = {1: "1.00", 4: "1.03", 8: "1.16", 16: "1.31"}           # SELECTION_VALIDATION.md table (3.2 min mean of two N=1 runs)
COST5090 = {1: "1.00", 4: "1.28 (elapsed 2.3/1.8 min; 1.24 by s/scene)", 8: "1.28 (elapsed 2.3/1.8 min; 1.24 by s/scene)",
            16: "1.28 (elapsed 2.3/1.8 min; 1.24 by s/scene)"}      # gpu5090_reanalysis/timing5090_N*.log; report states 1.2x
for pop, cc_dir, gpu, pdm_dir in (("dev 56 logs (exp 23)", "dev", "mixed (5090 39 logs + 3080 Ti 17 logs)", "dev_ncurve"),
                                  ("held-out 52 logs (exp 23)", "heldout", "mixed (5090 30 logs + 3080 Ti 22 logs)", "heldout_ncurve"),
                                  ("dev 56 logs (exp 23, 5090 rerun)", "dev_5090", "5090 only", "dev_5090_ncurve"),
                                  ("held-out 52 logs (exp 23, 5090 rerun)", "heldout_5090", "5090 only", "heldout_5090_ncurve")):
    C = json.load(open(os.path.join(O, "candidate_count_curve", cc_dir, "summary.json")))
    P = json.load(open(os.path.join(O, "pdm_score_best_of_n", pdm_dir, "summary.json")))["subsets"]["all"]
    gP, gA = P["ranksum_N17"]["d_pdms"], C["curve"]["17"]["d_a_minus"]
    for n, e in C["curve"].items():
        p = P[f"ranksum_N{n}"]
        cc_rows.append([pop, gpu, n, "natural + (N-1) random samples, 5 subsets/scene averaged (N=17: all candidates)",
                        C["n_scenes"], C["n_logs"], f4(e["a_minus"]), f4(e["d_a_minus"]), f4(e["d_a_minus_ci95"][0]),
                        f4(e["d_a_minus_ci95"][1]), f4(e["ade"]), f4(e["d_ade"]), f4(e["fde"]), f4(e["d_fde"]),
                        f4(e["d_fde_ci95"][0]), f4(e["d_fde_ci95"][1]), f4(p["pdms"]), f4(p["d_pdms"]), f4(p["d_pdms_ci95"][0]),
                        f4(p["d_pdms_ci95"][1]), f"{p['wilcoxon_p']:.3g}", f"{p['d_pdms']/gP:.3f}" if int(n) > 1 else "0",
                        f"{e['d_a_minus']/gA:.3f}" if int(n) > 1 else "0",
                        COST3080.get(int(n), "not measured") if "5090" not in cc_dir else "",
                        COST5090.get(int(n), "not measured") if "5090" in cc_dir else "",
                        f"{REL(os.path.join(O,'candidate_count_curve',cc_dir,'summary.json'))}: curve.{n}; "
                        f"{REL(os.path.join(O,'pdm_score_best_of_n',pdm_dir,'summary.json'))}: subsets.all.ranksum_N{n} (PDMS uses prefix subset natural+samples 1..N-1)"])
# exp 18/19: PoC 28 logs, 52 natural-failure + 297 normal scenes, population weighted 52/2695; single run per config, seed 0
EXP19 = [("best_of_n_selection", 9, "0.7", "3080 Ti"), ("best_of_n_selection_n4_T1.0", 5, "1.0", "3080 Ti"),
         ("best_of_n_selection_n8_T1.0", 9, "1.0", "3080 Ti"), ("best_of_n_selection_n16_T1", 17, "1.0", "3080 Ti"),
         ("best_of_n_selection_n32_T1.0", 33, "1.0", "5090"), ("best_of_n_selection_n4_T1.0_seed0_5090", 5, "1.0", "5090"),
         ("best_of_n_selection_n8_T1.0_seed0_5090", 9, "1.0", "5090"), ("best_of_n_selection_n16_T1.0_seed0_5090", 17, "1.0", "5090")]
for dname, n, T, gpu in EXP19:
    S = json.load(open(os.path.join(O, dname, "summary.json")))
    pr = S["population_a_minus_rate"]
    nr, rr = pr["normal"], pr["ranksum"]
    nr = nr["mean"] if isinstance(nr, dict) else nr; rr = rr["mean"] if isinstance(rr, dict) else rr
    cc_rows.append([f"PoC 28 logs, exp 18/19 (T={T}; population-weighted 52 fail/2695 normal; seed 0)", gpu, n,
                    "all candidates (natural + N-1 samples)", 349, 28, f4(rr), f4(rr - nr), "not reported", "not reported",
                    "", "", "", "", "", "", "not run", "not run", "", "", "", "", "", "", "",
                    f"{REL(os.path.join(O,dname,'summary.json'))}: population_a_minus_rate.{{normal,ranksum}}; see analysis_console.txt"])
with open(os.path.join(PKG, "figure_candidate_count.csv"), "w", newline="") as fh:
    csv.writer(fh).writerows(cc_rows)
print("ok")
