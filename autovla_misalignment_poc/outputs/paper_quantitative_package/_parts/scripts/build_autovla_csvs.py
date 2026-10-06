#!/usr/bin/env python
"""Build the AutoVLA figure CSVs of the paper quantitative package.

This script only COPIES numbers out of existing summary JSON files (no new statistics). The only
derived quantities are (a) simple counts of distinct logs/scenes in existing records files, and
(b) descriptive spread (min / max / SD) across already-reported per-layer point estimates. Every
derived row is marked "(recomputed: build_autovla_csvs.py)" in its source column.
CPU only, no torch. Run with: nice -n 19 python build_autovla_csvs.py
"""
import csv
import json
import os
import statistics

OUT = "/root/VLA/autovla_misalignment_poc/outputs"
PKG = os.path.join(OUT, "paper_quantitative_package")
REL = "autovla_misalignment_poc/outputs/"


def J(p):
    return json.load(open(os.path.join(OUT, p)))


def pct(x):
    return "" if x is None else round(100.0 * x, 4)


def num(x, nd=4):
    return "" if x is None else round(float(x), nd)


def ci(d, scale=100.0):
    if not d or "ci95" not in d or d["ci95"] is None:
        return "", ""
    return round(scale * d["ci95"][0], 4), round(scale * d["ci95"][1], 4)


def write(name, header, rows):
    with open(os.path.join(PKG, name), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow([r.get(h, "") for h in header])
    print("wrote", name, len(rows))


# ---------------------------------------------------------------- equal distance (exp 6)
ED = "equal_distance_perturbation/summary.json"
ed = J(ED)
EDS = "paper_quantitative_package/_parts/recomputed/equal_distance_summary_strict_rerun.json"
eds = J(EDS)
rows = []
hdr = ["panel", "subset", "group", "condition", "metric", "value", "CI_low", "CI_high", "unit", "n", "n_scenes",
       "CI_method", "source"]
for tag, S, src in [("full", ed, REL + ED), ("strict", eds, REL + EDS + " (recomputed: rerun_equal_distance_analysis.sh, original script STRICT=1)")]:
    for g in ["A-", "A+"]:
        for cond, d in S["table"][g].items():
            if d["n"] == 0:
                continue
            for m, unit, sc in [("recovery", "%", 100), ("amplification", "%", 100), ("fde5", "m", 1), ("ade5", "m", 1),
                                ("downstream_err", "%", 100), ("entropy", "nat", 1)]:
                x = d.get(m)
                if not x:
                    continue
                lo, hi = ci(x, sc)
                rows.append({"panel": "summary_table", "subset": tag, "group": g, "condition": cond, "metric": m + "_mean",
                             "value": round(sc * x["mean"], 4), "CI_low": lo, "CI_high": hi, "unit": unit, "n": d["n"],
                             "n_scenes": S["n_scenes"][g], "CI_method": "log-cluster bootstrap 2000 reps seed 0 (percentile)",
                             "source": f"{src} :: table/{g}/{cond}/{m}"})
            if d.get("dist") and cond != "gt":
                for stat in ["median", "mean"]:
                    rows.append({"panel": "summary_table", "subset": tag, "group": g, "condition": cond,
                                 "metric": f"perturbation_distance_{stat}", "value": round(d["dist"][stat], 4),
                                 "CI_low": "" if stat == "median" else round(d["dist"]["ci95"][0], 4),
                                 "CI_high": "" if stat == "median" else round(d["dist"]["ci95"][1], 4),
                                 "unit": "m", "n": d["n"], "n_scenes": S["n_scenes"][g],
                                 "CI_method": "" if stat == "median" else "log-cluster bootstrap of the mean",
                                 "source": f"{src} :: table/{g}/{cond}/dist/{stat}"})
    for g in ["A-", "A+", "ALL"]:
        for k, v in S["Q1"][g].items():
            rows.append({"panel": "Q1_distance_vs_FDE", "subset": tag, "group": g, "condition": "original+alternatives",
                         "metric": k, "value": round(v["value"], 4), "CI_low": round(v["ci95"][0], 4), "CI_high": round(v["ci95"][1], 4),
                         "unit": "rho", "n": v["n"], "CI_method": "log-cluster bootstrap 2000 reps seed 0; p not computed by script",
                         "source": f"{src} :: Q1/{g}/{k}"})
        q = S["Q2"][g]
        for k in ["scenes_with_mixed_alt_outcomes", "scenes_all_alts_recover", "scenes_all_alts_amplify",
                  "noise_floor_outcome_flip_rate"]:
            v = q[k]
            rows.append({"panel": "Q2_mixing_and_noise", "subset": tag, "group": g, "condition": "alternatives" if "noise" not in k else "original vs original_reseed",
                         "metric": k, "value": round(100 * v["value"], 4), "CI_low": round(100 * v["ci95"][0], 4),
                         "CI_high": round(100 * v["ci95"][1], 4), "unit": "% of scenes", "n": v["n"], "n_scenes": v["n"],
                         "CI_method": "log-cluster bootstrap 2000 reps seed 0", "source": f"{src} :: Q2/{g}/{k}"})
        for k in ["within_scene_alt_fde_sd", "within_scene_alt_fde_range", "noise_floor_original_vs_reseed_fde_absdiff"]:
            v = q[k]
            rows.append({"panel": "Q2_mixing_and_noise", "subset": tag, "group": g, "condition": "alternatives" if "noise" not in k else "original vs original_reseed",
                         "metric": k + ("(mean over scenes)" if k.endswith("sd") or k.endswith("absdiff") else "(median over scenes)"),
                         "value": round(v["value"], 4), "CI_low": round(v["ci95"][0], 4), "CI_high": round(v["ci95"][1], 4),
                         "unit": "m", "n": v["n"], "n_scenes": v["n"], "CI_method": "log-cluster bootstrap 2000 reps seed 0",
                         "source": f"{src} :: Q2/{g}/{k}"})
        if "buckets" in q:
            for k in ["share_between_distance_buckets", "share_between_scenes", "share_within_scene"]:
                rows.append({"panel": "Q2_variance_decomposition", "subset": tag, "group": g, "condition": "original+alternatives",
                             "metric": k, "value": round(100 * q[k], 4), "unit": "% of FDE variance",
                             "CI_method": "no CI reported", "source": f"{src} :: Q2/{g}/{k}"})
            for b, d in q["buckets"].items():
                for k, unit, sc in [("fde_mean", "m", 1), ("fde_sd", "m", 1), ("fde_p10", "m", 1), ("fde_p90", "m", 1),
                                    ("amplification_rate", "%", 100), ("recovery_rate", "%", 100)]:
                    rows.append({"panel": "Q2_distance_quintile", "subset": tag, "group": g, "condition": f"quintile{b} [{d['dist_range_m'][0]:.3f}-{d['dist_range_m'][1]:.3f} m]",
                                 "metric": k, "value": round(sc * d[k], 4), "unit": unit, "n": d["n"],
                                 "CI_method": "no CI reported", "source": f"{src} :: Q2/{g}/buckets/{b}/{k}"})
        q5 = S["Q5"][g]
        for k, v in q5.items():
            if k == "wilcoxon_p_fde":
                rows.append({"panel": "Q5_original_vs_alternatives", "subset": tag, "group": g, "condition": "original - alternatives",
                             "metric": k, "value": v, "unit": "p (Wilcoxon signed-rank, scene units)", "source": f"{src} :: Q5/{g}/{k}"})
                continue
            sc = 100 if ("recovery" in k or "amplification" in k or "percentile" in k) else 1
            rows.append({"panel": "Q5_original_vs_alternatives", "subset": tag, "group": g, "condition": "original - alternatives",
                         "metric": k, "value": round(sc * v["value"], 4), "CI_low": round(sc * v["ci95"][0], 4),
                         "CI_high": round(sc * v["ci95"][1], 4), "unit": "%" if sc == 100 else "m or nat", "n": v["n"],
                         "CI_method": "log-cluster bootstrap 2000 reps seed 0", "source": f"{src} :: Q5/{g}/{k}"})
# cluster counts (recomputed counts only)
recs = [json.loads(l) for l in open(os.path.join(OUT, "equal_distance_perturbation/records.jsonl"))]
for g in ["A-", "A+", "ALL"]:
    sel = [r for r in recs if g == "ALL" or r["group"] == g]
    rows.append({"panel": "design_counts", "subset": "full", "group": g, "condition": "", "metric": "n_distinct_logs (bootstrap clusters)",
                 "value": len({r["log"] for r in sel}), "unit": "logs", "n_scenes": len(sel),
                 "source": f"{REL}equal_distance_perturbation/records.jsonl (recomputed: build_autovla_csvs.py, count of distinct 'log')"})
write("figure_equal_distance.csv", hdr, rows)

# ---------------------------------------------------------------- causal interventions (exp 5, 7, 8)
rows = []
hdr = ["experiment", "subset", "group", "condition", "metric", "value", "CI_low", "CI_high", "unit", "n_units", "n_scenes",
       "p", "test", "used_in_verdict", "source"]
FM = "first_mismatch_causal/summary.json"
fm = J(FM)
for g in ["A-", "A+ (step-matched)", "A+ (all)"]:
    for cond in ["original", "gt", "nn"]:
        for m, unit, sc in [("A5", "%", 100), ("A3", "%", 100), ("downstream_err", "%", 100), ("realign", "%", 100),
                            ("ade5", "m", 1), ("fde5", "m", 1), ("err_slope", "m/step", 1), ("entropy_after", "nat", 1)]:
            x = fm[g][cond][m]
            lo, hi = ci(x, sc)
            rows.append({"experiment": "5 first_mismatch_causal", "subset": "all first-mismatch scenes", "group": g,
                         "condition": {"original": "original", "gt": "GT 1-token at t*", "nn": "nearest-neighbour token at t*"}[cond],
                         "metric": m if m != "A5" else "A+ recovery (5 s)", "value": round(sc * x["mean"], 4), "CI_low": lo, "CI_high": hi,
                         "unit": unit, "n_units": x["n"], "n_scenes": x["n"], "used_in_verdict": "yes",
                         "source": f"{REL}{FM} :: {g}/{cond}/{m}" + (" (A+ step-matched = weighted mean; n is effective weighted n)" if "step" in g else "")})
for g in ["A-", "A+ (step-matched)", "A+ (all)"]:
    for k, v in fm[g]["contrasts"].items():
        for m, x in v.items():
            if not isinstance(x, dict) or "mean" not in x:
                continue
            sc = 100 if m in ("A5", "A3", "downstream_err", "realign") else 1
            lo, hi = ci(x, sc)
            p = x.get("mcnemar", {}).get("p", x.get("wilcoxon_p", ""))
            rows.append({"experiment": "5 first_mismatch_causal", "subset": "paired contrast", "group": g, "condition": k,
                         "metric": "delta " + m, "value": round(sc * x["mean"], 4), "CI_low": lo, "CI_high": hi,
                         "unit": "pp" if sc == 100 else "m", "n_units": x.get("n", ""), "p": p,
                         "test": "McNemar exact (binomial)" if "mcnemar" in x else "Wilcoxon signed-rank",
                         "used_in_verdict": "yes", "source": f"{REL}{FM} :: {g}/{k}/{m}"})

AH = "action_history_causal/summary.json"
ah = J(AH)
names = {"normal": "Normal AR", "hist_attn_mask": "Action-history attention mask", "recent_attn_mask": "Recent-action attention mask",
         "hist_emb_neutral": "[OOD] Action-history embedding neutralised", "recent_emb_neutral": "[OOD] Recent-action embedding neutralised",
         "gt_history": "GT-history", "recent_gt": "Recent-GT (previous 1 token = GT)"}
for sub in ah["subsets"]:
    for g in ["A-", "A+"]:
        G = ah["subsets"][sub][g]
        for cond, d in G["conditions"].items():
            for m, unit, sc in [("amplification", "%", 100), ("recovery", "%", 100), ("ade5", "m", 1), ("fde5", "m", 1),
                                ("downstream_err", "%", 100), ("realign", "%", 100), ("entropy", "nat", 1), ("err_slope", "m/step", 1)]:
                x = d.get(m)
                if not x:
                    continue
                lo, hi = ci(x, sc)
                rows.append({"experiment": "7 action_history_causal", "subset": sub, "group": g, "condition": names[cond],
                             "metric": m, "value": round(sc * x["mean"], 4), "CI_low": lo, "CI_high": hi, "unit": unit,
                             "n_units": x["n"], "n_scenes": G["n_scenes"], "used_in_verdict": "no (OOD)" if "emb_neutral" in cond else "yes",
                             "source": f"{REL}{AH} :: subsets/{sub}/{g}/conditions/{cond}/{m}"})
        for cond, d in G["vs_normal"].items():
            for m in ["amplification", "recovery", "fde5", "ade5", "downstream_err"]:
                x = d.get(m)
                if not x:
                    continue
                sc = 100 if m in ("amplification", "recovery", "downstream_err") else 1
                lo, hi = ci(x, sc)
                p = x["mcnemar"]["p"] if "mcnemar" in x else x.get("wilcoxon_p", "")
                rows.append({"experiment": "7 action_history_causal", "subset": sub, "group": g,
                             "condition": names[cond] + " - Normal AR", "metric": "delta " + m, "value": round(sc * x["mean"], 4),
                             "CI_low": lo, "CI_high": hi, "unit": "pp" if sc == 100 else "m", "n_units": x["n"],
                             "n_scenes": G["n_scenes"], "p": p, "test": "McNemar exact" if "mcnemar" in x else "Wilcoxon signed-rank",
                             "used_in_verdict": "no (OOD)" if "emb_neutral" in cond else "yes",
                             "source": f"{REL}{AH} :: subsets/{sub}/{g}/vs_normal/{cond}/{m}"})
# relative effect (derived from reported point estimates only)
a = ah["subsets"]["all perturbations"]["A-"]["conditions"]
n_, r_, g_ = a["normal"]["amplification"]["mean"], a["recent_gt"]["amplification"]["mean"], a["gt_history"]["amplification"]["mean"]
rows.append({"experiment": "7 action_history_causal", "subset": "all perturbations", "group": "A-",
             "condition": "Recent-GT share of GT-history effect", "metric": "(Normal-RecentGT)/(Normal-GThistory)",
             "value": round(100 * (n_ - r_) / (n_ - g_), 4), "unit": "%", "n_units": 365, "n_scenes": 52,
             "test": "CI not reported for exp 7", "used_in_verdict": "yes",
             "source": f"{REL}{AH} :: amplification means (recomputed: build_autovla_csvs.py, ratio of reported means)"})
rows.append({"experiment": "7 action_history_causal", "subset": "all perturbations", "group": "A-",
             "condition": "Recent-GT relative reduction of Normal amplification", "metric": "(Normal-RecentGT)/Normal",
             "value": round(100 * (n_ - r_) / n_, 4), "unit": "%", "n_units": 365, "n_scenes": 52, "test": "CI not reported",
             "source": f"{REL}{AH} :: amplification means (recomputed: build_autovla_csvs.py, ratio of reported means)"})

SP = "prev_action_state_patching/summary.json"
sp = J(SP)
for g in ["A-", "A+"]:
    G = sp["groups"][g]["all"]
    for cond in ["normal", "recent_gt", "gt_history"] + [f"reverse@{l}" for l in ["emb", "L0", "L4", "L8", "L12", "L16", "L20", "L24", "L28", "L35"]]:
        d = G["rows"][cond]
        for m, unit, sc in [("amplification", "%", 100), ("recovery", "%", 100), ("fde5", "m", 1)]:
            x = d["level"][m]
            lo, hi = ci(x, sc)
            rows.append({"experiment": "8 prev_action_state_patching", "subset": "all perturbations", "group": g, "condition": cond,
                         "metric": m, "value": round(sc * x["mean"], 4), "CI_low": lo, "CI_high": hi, "unit": unit, "n_units": x["n"],
                         "n_scenes": G["n_scenes"], "used_in_verdict": "yes", "source": f"{REL}{SP} :: groups/{g}/all/rows/{cond}/level/{m}"})
        vb = d.get("vs_base") or {}
        for m in ["amplification", "recovery", "fde5"]:
            x = vb.get(m)
            if not x:
                continue
            sc = 100 if m != "fde5" else 1
            lo, hi = ci(x, sc)
            p = x["mcnemar"]["p"] if "mcnemar" in x else x.get("wilcoxon_p", "")
            rows.append({"experiment": "8 prev_action_state_patching", "subset": "all perturbations", "group": g,
                         "condition": f"{cond} - {d.get('baseline', 'baseline')}", "metric": "delta " + m, "value": round(sc * x["mean"], 4),
                         "CI_low": lo, "CI_high": hi, "unit": "pp" if sc == 100 else "m", "n_units": x["n"], "n_scenes": G["n_scenes"],
                         "p": p, "test": "McNemar exact" if "mcnemar" in x else "Wilcoxon signed-rank", "used_in_verdict": "yes",
                         "source": f"{REL}{SP} :: groups/{g}/all/rows/{cond}/vs_base/{m}"})
    fr = G.get("fraction_of_gt_history_effect", {})
    for cond in ["recent_gt"]:
        x = fr.get(cond)
        if x:
            rows.append({"experiment": "8 prev_action_state_patching", "subset": "all perturbations", "group": g,
                         "condition": "Recent-GT share of GT-history effect", "metric": "(Normal-row)/(Normal-GThistory) amplification",
                         "value": round(100 * x.get("mean", x.get("frac", 0)), 4), "CI_low": ci(x)[0], "CI_high": ci(x)[1], "unit": "%",
                         "n_units": G["n"], "n_scenes": G["n_scenes"], "test": "log-cluster bootstrap", "used_in_verdict": "yes",
                         "source": f"{REL}{SP} :: groups/{g}/all/fraction_of_gt_history_effect/{cond}/amplification"})
write("figure_causal_intervention.csv", hdr, rows)

# ---------------------------------------------------------------- layer patching (exp 8)
rows = []
hdr = ["layer", "amplification", "CI_low", "CI_high", "condition", "group", "recovery", "recovery_CI_low", "recovery_CI_high",
       "fde5_m", "fde5_CI_low", "fde5_CI_high", "frac_of_gt_history_effect", "frac_CI_low", "frac_CI_high", "n_units", "n_scenes", "source"]
for g in ["A-", "A+"]:
    G = sp["groups"][g]["all"]
    fr = G.get("fraction_of_gt_history_effect", {})
    for name, d in G["rows"].items():
        if name in ("normal", "recent_gt", "gt_history"):
            layer, cond = "baseline", name
        elif "@" in name:
            kind, layer = name.split("@")
            cond = {"patch_full": "full_replacement"}.get(kind, kind.replace("patch_delta", "delta_alpha"))
        else:
            continue
        L = d["level"]
        f = fr.get(name)
        rows.append({"layer": layer, "amplification": pct(L["amplification"]["mean"]), "CI_low": ci(L["amplification"])[0],
                     "CI_high": ci(L["amplification"])[1], "condition": cond, "group": g,
                     "recovery": pct(L["recovery"]["mean"]), "recovery_CI_low": ci(L["recovery"])[0], "recovery_CI_high": ci(L["recovery"])[1],
                     "fde5_m": num(L["fde5"]["mean"]), "fde5_CI_low": ci(L["fde5"], 1)[0], "fde5_CI_high": ci(L["fde5"], 1)[1],
                     "frac_of_gt_history_effect": "" if not f else pct(f.get("mean", f.get("frac"))),
                     "frac_CI_low": "" if not f else ci(f)[0], "frac_CI_high": "" if not f else ci(f)[1],
                     "n_units": L["amplification"]["n"], "n_scenes": G["n_scenes"],
                     "source": f"{REL}{SP} :: groups/{g}/all/rows/{name}/level"})
    # descriptive spread across the 37 full-replacement layers (from reported point estimates only)
    fulls = [G["rows"][f"patch_full@{l}"]["level"]["amplification"] for l in (["emb"] + [f"L{i}" for i in range(36)])]
    vals = [100 * x["mean"] for x in fulls]
    widths = [100 * (x["ci95"][1] - x["ci95"][0]) for x in fulls]
    for lab, v in [("min_over_37_layers", min(vals)), ("max_over_37_layers", max(vals)), ("range_max_minus_min", max(vals) - min(vals)),
                   ("SD_over_37_layers", statistics.pstdev(vals)), ("median_CI_width_over_37_layers", statistics.median(widths))]:
        rows.append({"layer": "summary", "amplification": round(v, 4), "condition": "full_replacement:" + lab, "group": g,
                     "n_units": fulls[0]["n"], "n_scenes": G["n_scenes"],
                     "source": f"{REL}{SP} :: groups/{g}/all/rows/patch_full@*/level/amplification (recomputed: build_autovla_csvs.py, descriptive over reported point estimates; population SD)"})
write("figure_layer_patching.csv", hdr, rows)

# ---------------------------------------------------------------- temporal window (exp 9, 10)
rows = []
hdr = ["experiment", "subset", "group", "condition", "free_horizon_N", "metric", "value", "CI_low", "CI_high", "unit",
       "fraction_of_full_effect", "fraction_CI_low", "fraction_CI_high", "fraction_formula", "n_units", "n_scenes", "source"]
TW = "temporal_feedback_window/summary.json"
tw = J(TW)
for sub, S in tw["subsets"].items():
    for g in ["A-", "A+"]:
        G = S.get(g)
        if not G or not G.get("rows"):
            continue
        for cond, d in G["rows"].items():
            for m, unit, sc in [("amplification", "%", 100), ("recovery", "%", 100), ("ade5", "m", 1), ("fde5", "m", 1),
                                ("downstream_err", "%", 100)]:
                x = d["level"][m]
                lo, hi = ci(x, sc)
                fr = G.get("fraction", {}).get(m, {}).get(cond)
                vn = d.get("vs_normal", {}).get(m)
                rows.append({"experiment": "9 temporal_feedback_window", "subset": sub, "group": g, "condition": cond, "metric": m,
                             "value": round(sc * x["mean"], 4), "CI_low": lo, "CI_high": hi, "unit": unit,
                             "fraction_of_full_effect": "" if not fr else pct(fr["mean"]),
                             "fraction_CI_low": "" if not fr else ci(fr)[0], "fraction_CI_high": "" if not fr else ci(fr)[1],
                             "fraction_formula": "(Normal - row)/(Normal - GT-history)" if fr else "",
                             "n_units": x["n"], "n_scenes": G["n_scenes"],
                             "source": f"{REL}{TW} :: subsets/{sub}/{g}/rows/{cond}/level/{m}; fraction: subsets/{sub}/{g}/fraction/{m}/{cond}"})
                if vn and cond != "normal":
                    lo2, hi2 = ci(vn, sc)
                    rows.append({"experiment": "9 temporal_feedback_window", "subset": sub, "group": g, "condition": cond + " - normal",
                                 "metric": "delta " + m + " (absolute reduction if negative)", "value": round(sc * vn["mean"], 4),
                                 "CI_low": lo2, "CI_high": hi2, "unit": "pp" if sc == 100 else "m", "n_units": vn["n"], "n_scenes": G["n_scenes"],
                                 "source": f"{REL}{TW} :: subsets/{sub}/{g}/rows/{cond}/vs_normal/{m} (p in json: mcnemar/wilcoxon)"})
HC = "horizon_controlled_window/summary.json"
hc = J(HC)
for tsel in ["t*=0", "t*<=1"]:
    for g in ["A-", "A+"]:
        X = hc["analyses"]["in10"][tsel][g]
        for N, ND in X["N"].items():
            for w, d in ND["windows"].items():
                for m, unit, sc in [("fde", "m", 1), ("ade", "m", 1), ("amp", "%", 100), ("amp2", "%", 100)]:
                    x = d["level"][m]
                    fr = d.get("fraction", {}).get(m)
                    lo, hi = ci(x, sc)
                    rows.append({"experiment": "10 horizon_controlled_window", "subset": f"in-distribution 10-token, {tsel}", "group": g,
                                 "condition": "normal" if w == "0" else f"win{w}", "free_horizon_N": N,
                                 "metric": {"amp": "amplification (FDE@release+N > 3 m)", "amp2": "FDE@release+N > 2 m"}.get(m, m + "@release+N"),
                                 "value": round(sc * x["mean"], 4), "CI_low": lo, "CI_high": hi, "unit": unit,
                                 "fraction_of_full_effect": "" if (not fr or w == "0") else pct(fr["mean"]),
                                 "fraction_CI_low": "" if (not fr or w == "0") else ci(fr)[0],
                                 "fraction_CI_high": "" if (not fr or w == "0") else ci(fr)[1],
                                 "fraction_formula": "(Normal_same_poses - win)/(Normal_same_poses - Full_same_poses)" if (fr and w != "0") else "",
                                 "n_units": x["n"], "n_scenes": d["n_scenes"],
                                 "source": f"{REL}{HC} :: analyses/in10/{tsel}/{g}/N/{N}/windows/{w}/level/{m}"})
                if w != "0" and d.get("rediverge_given_stab"):
                    for k in ["rediverge_given_stab", "rediverge_given_stab_full"]:
                        x = d[k]
                        lo, hi = ci(x)
                        rows.append({"experiment": "10 horizon_controlled_window", "subset": f"in-distribution 10-token, {tsel}", "group": g,
                                     "condition": f"win{w}" if k.endswith("stab") else f"Full (same poses as win{w})", "free_horizon_N": N,
                                     "metric": "re-divergence: FDE>2 m after N given error<=1 m at release", "value": round(100 * x["mean"], 4),
                                     "CI_low": lo, "CI_high": hi, "unit": "%", "n_units": x["n"],
                                     "source": f"{REL}{HC} :: analyses/in10/{tsel}/{g}/N/{N}/windows/{w}/{k}"})
write("figure_temporal_window.csv", hdr, rows)

# ---------------------------------------------------------------- motion semantics (exp 11, 32)
rows = []
hdr = ["experiment", "group", "condition", "label", "metric", "value", "CI_low", "CI_high", "unit", "frac_of_recent_gt_effect",
       "frac_CI_low", "frac_CI_high", "p", "test", "n_units", "n_scenes", "source"]
PI = "prev_action_identity_decomposition/summary.json"
MS = "motion_semantics_ablation/summary.json"
lab11 = {"normal": "self token (Normal AR)", "gt_history": "GT-history", "recent_gt": "GT token (Recent-GT)",
         "geo_nn_gt": "GT-motion nearest other token", "emb_nn_gt": "GT-embedding nearest other token",
         "geo_nn_self": "self-motion nearest other token", "emb_nn_self": "self-embedding nearest other token",
         "random_tok": "random trained token", "mean_emb": "[OOD] mean action embedding"}
lab32 = {"normal": "self token (Normal AR)", "recent_gt": "GT token (Recent-GT)", "dir_ok_mag_wrong": "direction-correct / magnitude-wrong",
         "dir_wrong_mag_ok": "magnitude-correct / direction-wrong", "mirror_same_dist": "opposite side of GT, matched distance",
         "random_mag_matched": "magnitude-matched random"}
for exp, P, labs in [("11 prev_action_identity_decomposition", PI, lab11), ("32 motion_semantics_ablation", MS, lab32)]:
    S = J(P)
    for g in ["A-", "A+"]:
        G = S["groups"][g]
        for cond, d in G["conditions"].items():
            fr = G.get("frac_of_recent_gt_effect", {}).get(cond, {})
            vn = G.get("vs_normal", {}).get(cond, {})
            for m, unit, sc in [("amplification", "%", 100), ("fde5", "m", 1), ("recovery", "%", 100), ("ade5", "m", 1)]:
                x = d.get(m)
                if not x:
                    continue
                lo, hi = ci(x, sc)
                f = fr.get(m if m != "recovery" and m != "ade5" else "__none__")
                v = vn.get(m, {})
                p = v.get("mcnemar", {}).get("p", v.get("wilcoxon_p", ""))
                rows.append({"experiment": exp, "group": g, "condition": cond, "label": labs.get(cond, cond), "metric": m,
                             "value": round(sc * x["mean"], 4), "CI_low": lo, "CI_high": hi, "unit": unit,
                             "frac_of_recent_gt_effect": "" if not f else pct(f["frac"]),
                             "frac_CI_low": "" if not f else ci(f)[0], "frac_CI_high": "" if not f else ci(f)[1],
                             "p": p if cond != "normal" else "", "test": ("vs Normal: " + ("McNemar exact" if "mcnemar" in v else "Wilcoxon")) if v else "",
                             "n_units": x["n"], "n_scenes": S["n_scenes"][g] if "n_scenes" in S else (52 if g == "A-" else 156),
                             "source": f"{REL}{P} :: groups/{g}/conditions/{cond}/{m}"})
        for cond, d in G.get("vs_recent_gt", {}).items():
            for m in ["amplification", "fde5"]:
                x = d.get(m)
                if not x:
                    continue
                sc = 100 if m == "amplification" else 1
                lo, hi = ci(x, sc)
                p = x.get("mcnemar", {}).get("p", x.get("wilcoxon_p", ""))
                rows.append({"experiment": exp, "group": g, "condition": f"{cond} - recent_gt", "label": labs.get(cond, cond) + " minus GT token",
                             "metric": "delta " + m, "value": round(sc * x["mean"], 4), "CI_low": lo, "CI_high": hi,
                             "unit": "pp" if sc == 100 else "m", "p": p, "test": "McNemar exact" if "mcnemar" in x else "Wilcoxon signed-rank",
                             "n_units": x["n"], "source": f"{REL}{P} :: groups/{g}/vs_recent_gt/{cond}/{m}"})
        for cond, d in G.get("vs_normal", {}).items():
            for m in ["amplification", "fde5"]:
                x = d.get(m)
                if not x:
                    continue
                sc = 100 if m == "amplification" else 1
                lo, hi = ci(x, sc)
                p = x.get("mcnemar", {}).get("p", x.get("wilcoxon_p", ""))
                rows.append({"experiment": exp, "group": g, "condition": f"{cond} - normal", "label": labs.get(cond, cond) + " minus self token",
                             "metric": "delta " + m, "value": round(sc * x["mean"], 4), "CI_low": lo, "CI_high": hi,
                             "unit": "pp" if sc == 100 else "m", "p": p, "test": "McNemar exact" if "mcnemar" in x else "Wilcoxon signed-rank",
                             "n_units": x["n"], "source": f"{REL}{P} :: groups/{g}/vs_normal/{cond}/{m}"})
RC = "motion_semantics_ablation/row_contrasts.json"
rc = J(RC)
for k, x in rc.items():
    g, contrast, m = k.split(" ")
    sc = 100 if m == "amplification" else 1
    lo, hi = ci(x, sc)
    p = x.get("mcnemar", {}).get("p", x.get("wilcoxon_p", ""))
    rows.append({"experiment": "32 motion_semantics_ablation", "group": g, "condition": contrast, "label": "row contrast (paired)",
                 "metric": "delta " + m, "value": round(sc * x["mean"], 4), "CI_low": lo, "CI_high": hi, "unit": "pp" if sc == 100 else "m",
                 "p": p, "test": "McNemar exact" if "mcnemar" in x else "Wilcoxon signed-rank", "n_units": x["n"],
                 "source": f"{REL}{RC} :: {k}"})
pi = J(PI)
for g in ["A-", "A+", "both"]:
    X = pi["geometry_vs_embedding"][g]
    for m in ["fde5", "amplification"]:
        for v in ["dgeo", "dcos"]:
            sc = 100 if m == "amplification" else 1
            rows.append({"experiment": "11 prev_action_identity_decomposition", "group": g, "condition": f"joint regression coef per 1 SD of {v}",
                         "label": "geometry (motion) vs embedding", "metric": "delta " + m, "value": round(sc * X[m][v], 4),
                         "CI_low": round(sc * X[m][v + "_ci95"][0], 4), "CI_high": round(sc * X[m][v + "_ci95"][1], 4),
                         "unit": "pp per SD" if sc == 100 else "m per SD", "test": "log-cluster bootstrap 1000 reps", "n_units": X["n"],
                         "source": f"{REL}{PI} :: geometry_vs_embedding/{g}/{m}/{v}"})
write("figure_motion_semantics.csv", hdr, rows)
