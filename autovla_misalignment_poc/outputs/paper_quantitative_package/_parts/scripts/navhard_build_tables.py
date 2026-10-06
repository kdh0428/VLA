#!/usr/bin/env python
"""
Build the navhard paper table (full navhard, 76 logs / 225 groups) from the EXISTING analysis JSONs only.
No statistic is recomputed: every mean / Δ / CI / p / group count is copied from the JSON key named in `source`.
Outputs (in outputs/paper_quantitative_package/):
  paper_navhard_table.csv, paper_navhard_table.tex, figure_navhard_methods.csv
"""
import csv
import json
import os

POC = "/root/VLA/autovla_misalignment_poc/outputs"
PKG = f"{POC}/paper_quantitative_package"
J26 = "navhard_full_validation/navhard_full_comparison.json"
J27 = "safety_filter_ablation/ablation.json"
J29 = "safety_filter_components/components.json"
J30 = "candidate_oracle/oracle_comparison.json"
D = {k: json.load(open(f"{POC}/{k}")) for k in (J26, J27, J29, J30)}


def from26(rule, scope="pooled"):
    r = D[J26][scope]["rules"][rule]
    e = r["epdms"]
    return dict(mean=e["mean"], diff=e.get("diff", 0.0), ci=e.get("ci95", [0.0, 0.0]), p_perm=e.get("p_perm_log"),
                p_wil=e.get("p_wilcoxon_group"), gbwt=r["groups_better_worse_tied"],
                n_groups=D[J26][scope]["n_groups"], n_logs=D[J26][scope]["n_logs"],
                source=f"outputs/{J26} :: {scope}.rules.{rule}.epdms (+groups_better_worse_tied)")


def fromC(j, mean_key, contrast, scope="full"):
    s = D[j][scope]
    e = s["contrasts"][contrast]["epdms"]
    return dict(mean=s["means"][mean_key]["epdms"], diff=e["diff"], ci=e["ci95"], p_perm=e["p_perm_log"], p_wil=e["p_wilcoxon_group"],
                gbwt=e["groups_better_worse_tied"], n_groups=s["n_groups"], n_logs=s["n_logs"],
                source=f"outputs/{j} :: {scope}.means.{mean_key}.epdms ; {scope}.contrasts['{contrast}'].epdms")


ROWS = [
    ("No selection (natural plan, cand. 0)", "normal", "exp26", from26("normal")),
    ("Rank-sum (entropy rank + log-lik rank)", "ranksum", "exp26", from26("ranksum")),
    ("Max log-likelihood", "max_loglik", "exp26", from26("max_loglik")),
    ("Collision constraint only", "collision_only (comp_collision)", "exp29", fromC(J29, "collision_only", "collision_only - no_filter")),
    ("Drivable-area constraint only", "dac_only (comp_dac)", "exp29", fromC(J29, "dac_only", "dac_only - no_filter")),
    ("Safety filter only (collision + DA, keep natural if safe)", "filter_only (= collision_dac)", "exp27", fromC(J27, "filter_only", "filter_only - normal")),
    ("Safety filter + random (3 seeds avg.)", "filter_random", "exp27", fromC(J27, "filter_random", "filter_random - normal")),
    ("Safety filter + rank-sum (F1, prereg. primary)", "F1 (= filter_ranksum)", "exp26", from26("F1")),
    ("Safety filter + max log-lik", "F1_maxll (= filter_maxll)", "exp26", from26("F1_maxll")),
    ("Oracle among 16 samples (analysis upper bound)", "oracle16", "exp30", fromC(J30, "oracle16", "oracle16 - no_selection")),
    ("Oracle among 17 candidates (analysis upper bound)", "oracle17", "exp30", fromC(J30, "oracle17", "oracle17 - no_selection")),
]


def fp(p):
    if p is None:
        return ""
    if p <= 5.0e-5 + 1e-9:
        return "5.0e-05 (floor)"
    return f"{p:.2g}" if p < 0.01 else f"{p:.3f}"


with open(f"{PKG}/paper_navhard_table.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["method", "condition_key", "experiment", "N_logs", "N_groups", "EPDMS", "delta_vs_no_selection", "ci95_low", "ci95_high",
                "p_perm_log_paired", "p_wilcoxon_group", "groups_improved", "groups_worsened", "groups_tied", "ci_method", "source"])
    for name, key, ex, r in ROWS:
        base = key == "normal"
        w.writerow([name, key, ex, r["n_logs"], r["n_groups"], f"{r['mean']:.4f}", "" if base else f"{r['diff']:+.4f}",
                    "" if base else f"{r['ci'][0]:+.4f}", "" if base else f"{r['ci'][1]:+.4f}",
                    "" if base else fp(r["p_perm"]), "" if base else f"{r['p_wil']:.2g}",
                    "" if base else r["gbwt"][0], "" if base else r["gbwt"][1], "" if base else r["gbwt"][2],
                    "log-cluster bootstrap 2,000 resamples seed 0 (percentile); perm = log-level paired sign-flip 20,000 seed 0 two-sided; Wilcoxon = group-level signed-rank",
                    r["source"]])

with open(f"{PKG}/figure_navhard_methods.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["method", "condition_key", "experiment", "EPDMS", "delta", "ci95_low", "ci95_high", "N_logs", "N_groups", "deployable", "source"])
    for name, key, ex, r in ROWS:
        w.writerow([name, key, ex, f"{r['mean']:.4f}", f"{r['diff']:+.4f}", f"{r['ci'][0]:+.4f}", f"{r['ci'][1]:+.4f}",
                    r["n_logs"], r["n_groups"], "no (uses scored future)" if key.startswith("oracle") else "yes", r["source"]])


def tex_w(p):
    if p < 0.01:
        m, e = f"{p:.1e}".split("e")
        return rf"${m}\times10^{{{int(e)}}}$"
    return f"{p:.3f}"


def tex_p(p):
    if p is None:
        return "--"
    if p <= 5.0e-5 + 1e-9:
        return r"$<10^{-4}$"
    if p < 0.01:
        m, e = f"{p:.1e}".split("e")
        return rf"${m}\times10^{{{int(e)}}}$"
    return f"{p:.3f}"


TEXNAME = {"normal": "No selection (natural plan)", "ranksum": "Rank-sum", "max_loglik": "Max log-lik.",
           "collision_only (comp_collision)": "Collision constraint only", "dac_only (comp_dac)": "Drivable-area constraint only",
           "filter_only (= collision_dac)": "Safety filter only", "filter_random": "Safety + random",
           "F1 (= filter_ranksum)": "Safety + rank-sum (F1)", "F1_maxll (= filter_maxll)": "Safety + max log-lik.",
           "oracle16": "Oracle (16 samples)$^\\dagger$", "oracle17": "Oracle (17 candidates)$^\\dagger$"}
L = [r"\begin{table}[t]", r"\centering", r"\small",
     r"\caption{NAVSIM v2 navhard two-stage pseudo closed-loop, full benchmark (76 logs, 225 groups; stage 1: 450 original scenes, "
     r"stage 2: 5{,}462 synthetic scenes). AutoVLA, 17 candidates per scene (T$=0.01$ natural plan + 16 samples at T$=1.0$), RTX 5090. "
     r"$\Delta$ = paired difference in EPDMS vs.\ no selection over the same 225 groups; 95\% CI = log-cluster bootstrap "
     r"(2{,}000 resamples, seed 0); $p_\text{perm}$ = log-level paired sign-flip permutation test (two-sided, 20{,}000 draws, seed 0; "
     r"floor $5\times10^{-5}$); $p_\text{W}$ = group-level Wilcoxon signed-rank. +/$-$/= = groups improved / worsened / tied. "
     r"No multiple-comparison correction. $^\dagger$Analysis-only upper bound that uses the scored future; not deployable. "
     r"The safety filter and the score share the PDM collision / drivable-area rules.}",
     r"\label{tab:navhard}", r"\begin{tabular}{lcccccc}", r"\toprule",
     r"Method & EPDMS & $\Delta$ & 95\% CI & $p_\text{perm}$ & $p_\text{W}$ & +/$-$/= \\", r"\midrule"]
for name, key, ex, r in ROWS:
    if key == "normal":
        L.append(rf"{TEXNAME[key]} & {r['mean']:.4f} & -- & -- & -- & -- & -- \\")
        continue
    if key.startswith("collision_only") or key.startswith("oracle16"):
        L.append(r"\midrule")
    L.append(rf"{TEXNAME[key]} & {r['mean']:.4f} & ${r['diff']:+.4f}$ & $[{r['ci'][0]:+.4f}, {r['ci'][1]:+.4f}]$ & {tex_p(r['p_perm'])} & "
             rf"{tex_w(r['p_wil'])} & {r['gbwt'][0]}/{r['gbwt'][1]}/{r['gbwt'][2]} \\")
L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
open(f"{PKG}/paper_navhard_table.tex", "w").write("\n".join(L) + "\n")
print(open(f"{PKG}/paper_navhard_table.csv").read())
print("\n".join(L))
