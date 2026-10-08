#!/usr/bin/env python
"""Recompute SpatialVLA (exp 34) closed-loop paired table from closed_loop/episodes.jsonl using EXACTLY
analyze_svla.py's boot() (episode-cluster bootstrap, 2000 reps, random.Random(0) per call, unstratified) and
paired() (McNemar exact binomial for binary, Wilcoxon for continuous). Imports those functions directly (no torch).
Trajectory = cumsum of executed action[:3] (world_vector), endpoint L2 distance (as analyze_svla.closed_loop).
Output: CSV rows for all conditions (overall + per task) vs natural, and the requested contrasts."""
import csv, json, os, sys
import numpy as np
sys.path.insert(0, "/root/VLA/autovla_misalignment_poc/scripts/cross_domain_temporal")
from analyze_svla import boot, paired  # noqa: E402  (module-level imports: numpy/scipy only)

SRC = "/root/VLA/autovla_misalignment_poc/outputs/cross_domain_temporal_replication/closed_loop/episodes.jsonl"
OUT = "/root/VLA/paper_quantitative_package/spatialvla_closed_loop_paired.csv"
E = {}
for l in open(SRC):
    r = json.loads(l)
    E[(r["task"], r["seed"], r["cond"])] = {"s": float(r["success"]),
                                           "end": np.cumsum(np.array([x["action"][:3] for x in r["rec"]]), 0)[-1]}
conds = sorted({k[2] for k in E}); keys = sorted({(k[0], k[1]) for k in E})
assert all(k + (c,) in E for k in keys for c in conds)
TASKS = {"all": keys}
for t in sorted({k[0] for k in keys}):
    TASKS[t] = [k for k in keys if k[0] == t]

def fmt(x): return f"{x:.4f}"
rows = []
def contrast(kind, label, a, b, ks, task):
    items = [{"cl": k, a: {"s": E[k + (a,)]["s"]}, b: {"s": E[k + (b,)]["s"]}} for k in ks]
    d = paired(items, a, b, "s", True)
    na = int(sum(E[k + (a,)]["s"] for k in ks)); nb = int(sum(E[k + (b,)]["s"] for k in ks)); N = len(ks)
    disc = float(np.mean([E[k + (a,)]["s"] != E[k + (b,)]["s"] for k in ks]))
    # endpoint distance between a and b trajectories (same metric as divergence vs natural)
    dv = boot([(k, float(np.linalg.norm(E[k + (a,)]["end"] - E[k + (b,)]["end"]))) for k in ks])
    row = {"block": kind, "task": task, "label": label, "cond_a": a, "cond_b": b, "N_episodes": N,
           "succ_a": na, "succ_a_pct": fmt(100 * na / N), "succ_b": nb, "succ_b_pct": fmt(100 * nb / N),
           "diff_pp_a_minus_b": fmt(100 * d["mean"]), "ci95_lo_pp": fmt(100 * d["ci95"][0]), "ci95_hi_pp": fmt(100 * d["ci95"][1]),
           "mcnemar_a_only": d["mcnemar"]["a_only"], "mcnemar_b_only": d["mcnemar"]["b_only"], "mcnemar_exact_p": f"{d['mcnemar']['p']:.4g}",
           "success_disagreement_pct": fmt(100 * disc),
           "traj_endpoint_dist_a_b_m": fmt(dv["mean"]), "traj_ci95_lo_m": fmt(dv["ci95"][0]), "traj_ci95_hi_m": fmt(dv["ci95"][1])}
    # per-condition success CI (boot over episodes)
    sa = boot([(k, E[k + (a,)]["s"]) for k in ks]); row["succ_a_ci95_pct"] = f"[{100*sa['ci95'][0]:.1f}, {100*sa['ci95'][1]:.1f}]"
    if b != "natural":
        # paired difference of divergence-vs-natural (Wilcoxon), only meaningful when neither is natural
        it2 = [{"cl": k, a: {"v": float(np.linalg.norm(E[k + (a,)]["end"] - E[k + ("natural",)]["end"]))},
                b: {"v": float(np.linalg.norm(E[k + (b,)]["end"] - E[k + ("natural",)]["end"]))}} for k in ks]
        d2 = paired(it2, a, b, "v", False)
        row["div_vs_nat_diff_m_a_minus_b"] = fmt(d2["mean"]); row["div_diff_ci95"] = f"[{d2['ci95'][0]:.4f}, {d2['ci95'][1]:.4f}]"
        row["div_diff_wilcoxon_p"] = f"{d2['wilcoxon_p']:.4g}"
    rows.append(row)

for task, ks in TASKS.items():
    for c in conds:
        if c != "natural":
            contrast("vs_natural", f"{c} - natural", c, "natural", ks, task)
CON = [("context_feedback_effect", "feedback - corrected (opposite)", "feedback_opposite", "corrected_opposite"),
       ("context_feedback_effect", "reverse - natural (opposite)", "reverse_opposite", "natural"),
       ("context_feedback_effect", "feedback - corrected (perp_left)", "feedback_perp_left", "corrected_perp_left"),
       ("context_feedback_effect", "reverse - natural (perp_left)", "reverse_perp_left", "natural"),
       ("executed_action_effect", "corrected - natural (opposite)", "corrected_opposite", "natural"),
       ("executed_action_effect", "feedback - reverse (opposite)", "feedback_opposite", "reverse_opposite"),
       ("executed_action_effect", "corrected - natural (perp_left)", "corrected_perp_left", "natural"),
       ("executed_action_effect", "feedback - reverse (perp_left)", "feedback_perp_left", "reverse_perp_left"),
       ("prereg_contrast", "corrected - feedback (opposite)", "corrected_opposite", "feedback_opposite"),
       ("prereg_contrast", "reverse - corrected (opposite)", "reverse_opposite", "corrected_opposite"),
       ("prereg_contrast", "corrected - feedback (perp_left)", "corrected_perp_left", "feedback_perp_left"),
       ("prereg_contrast", "reverse - corrected (perp_left)", "reverse_perp_left", "corrected_perp_left")]
for task, ks in TASKS.items():
    for kind, lab, a, b in CON:
        contrast(kind, lab, a, b, ks, task)
fields = list(dict.fromkeys(k for r in rows for k in r))
with open(OUT, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields + ["source"]); w.writeheader()
    for r in rows:
        r["source"] = "outputs/cross_domain_temporal_replication/closed_loop/episodes.jsonl (recomputed: _parts/scripts/svla_closed_loop_paired.py; methods imported from scripts/cross_domain_temporal/analyze_svla.py boot/paired)"
        w.writerow(r)
print("natural success by task:", {t: sum(E[k + ("natural",)]["s"] for k in ks) for t, ks in TASKS.items()})
for r in rows:
    print(r["task"][:16], r["label"], r["succ_a"], r["succ_b"], r["diff_pp_a_minus_b"], r["ci95_lo_pp"], r["ci95_hi_pp"], r["mcnemar_exact_p"], r["success_disagreement_pct"], r["traj_endpoint_dist_a_b_m"], r.get("div_vs_nat_diff_m_a_minus_b",""), r.get("div_diff_wilcoxon_p",""))

# Additional contrast (NOT in analyze_svla.py; same metric = endpoint divergence vs natural, same paired() Wilcoxon +
# episode bootstrap): context-only perturbation divergence vs numeric-path rerun divergence (natural_gen).
extra = []
for task, ks in TASKS.items():
    for c in ("reverse_opposite", "reverse_perp_left", "feedback_opposite", "corrected_opposite", "feedback_perp_left", "corrected_perp_left"):
        it = [{"cl": k, c: {"v": float(np.linalg.norm(E[k + (c,)]["end"] - E[k + ("natural",)]["end"]))},
               "natural_gen": {"v": float(np.linalg.norm(E[k + ("natural_gen",)]["end"] - E[k + ("natural",)]["end"]))}} for k in ks]
        d = paired(it, c, "natural_gen", "v", False)
        extra.append({"block": "additional_div_vs_rerun", "task": task, "label": f"div({c}) - div(natural_gen)", "cond_a": c, "cond_b": "natural_gen",
                      "N_episodes": len(ks), "div_vs_nat_diff_m_a_minus_b": fmt(d["mean"]), "div_diff_ci95": f"[{d['ci95'][0]:.4f}, {d['ci95'][1]:.4f}]",
                      "div_diff_wilcoxon_p": f"{d['wilcoxon_p']:.4g}",
                      "source": "outputs/cross_domain_temporal_replication/closed_loop/episodes.jsonl (recomputed, ADDITIONAL contrast not in original analysis: _parts/scripts/svla_closed_loop_paired.py)"})
        print(task[:16], extra[-1]["label"], extra[-1]["div_vs_nat_diff_m_a_minus_b"], extra[-1]["div_diff_ci95"], extra[-1]["div_diff_wilcoxon_p"])
with open(OUT, "a", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields + ["source"])
    for r in extra:
        w.writerow({k: r.get(k, "") for k in fields + ["source"]})
