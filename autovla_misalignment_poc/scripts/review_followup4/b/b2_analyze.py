#!/usr/bin/env python
"""
B2 matched-prefix analysis (CPU; written with the protocol, before any main B2 data; MATCHED_PREFIX_PROTOCOL.md §6-§8).

  python b2_analyze.py --run <dir with bank.jsonl, cells.jsonl> --cohort {p1c,dev} [--cohort-manifest p1c_cohort_t4.json]
                       --out <dir> [--mask]
Unit outcome per cell (analyze_action_history.unit_metrics, unchanged): severe = (not A+) and FDE5 > 3 m; A+; FDE5.
Unit difference d_u = Y(cell_a) - Y(cell_b); scene d_s = mean over the scene's units; estimators, drive-cluster exact
sign-flip, drive bootstrap (B = 10,000, seed 20261017) and cluster-t are imported unchanged from analyze_p1c.py.
Estimands: F (natural A-, pi = 1), S (A+, recomputed domain weights), P = F u S (Hajek) = the t* <= 4 domain E4 of the
P1-C eligible set. Dev cohort: unweighted, descriptive (development set), drive clusters.
Families (Holm): B2-F primary {C_GT, C_NAT, INT, C_PROV} on theta_F, severe, primary anchor, alpha 0.05;
B2-P {same} on theta_P, alpha 0.025. Everything else is secondary (CIs, no family claim).
--mask replaces effect-bearing values by MASKED (pipeline test on gate/pilot output).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

sys.path.insert(0, "/root/VLA/autovla_misalignment_poc/scripts/review_followup/p1c")
import analyze_p1c as AP  # noqa: E402

SEED = 20261017
POC = "/root/VLA/autovla_misalignment_poc"
P1C_RUN = f"{POC}/outputs/review_followup/P1C_new_log_replication/run_20261008"
ALPHA = {"F": 0.05, "P": 0.025}
# (id, cell_a, cell_b) ; INT handled separately
CONTRASTS = [("C_GT", "GT_NSRC", "GT_REF"),          # latest-block source effect inside a GT older prefix (matched one-shot Reverse - Full)
             ("C_NAT", "NAT_NSRC", "NAT_REF"),       # same inside a natural older prefix (Normal - one-shot Recent-GT)
             ("C_PROV", "GT_CSRC", "GT_NSRC")]       # corrected-branch source vs Normal source inside GT older prefix (value + provenance)
SECONDARY = [("O_main_REF", "NAT_REF", "GT_REF"), ("O_main_NSRC", "NAT_NSRC", "GT_NSRC"), ("O_main_CSRC", "NAT_CSRC", "GT_CSRC"),
             ("C_PROV_NAT", "NAT_CSRC", "NAT_NSRC"), ("CSRC_vs_REF_GT", "GT_CSRC", "GT_REF")]
METRICS = ["severe", "A_plus", "fde5", "tok_a_changed", "suffix_changed_frac", "local_disp_change_m", "local_dyaw_change_rad",
           "endpoint_dist_m"]
MASK = "MASKED"


def drive_of(log):
    return log.split("_veh-")[0] + "_veh-" + log.split("_veh-")[1].split("_")[0]


def dayveh_of(log):
    d = log.split(".")
    return f"{d[0]}.{d[1]}.{d[2]}_veh-{log.split('_veh-')[1].split('_')[0]}"


def load(run, cohort, cman):
    cells = defaultdict(dict)
    for l in open(os.path.join(run, "cells.jsonl")):
        x = json.loads(l)
        cells[(x["unit_id"], x["n_old"])][x["cell"]] = x
    bank = {json.loads(l)["unit_id"]: json.loads(l) for l in open(os.path.join(run, "bank.jsonl"))}
    if cohort == "p1c":
        W = json.load(open(cman))["scene_weights"]
    else:
        W = None
    ed = {}
    src = f"{P1C_RUN}/units/records.jsonl" if cohort == "p1c" else f"{POC}/outputs/equal_distance_perturbation/records.jsonl"
    for l in open(src):
        r = json.loads(l); ed[r["token"]] = r
    return cells, bank, W, ed


def unit_rows(cells, bank, W, ed, cohort):
    """long table: one row per (unit, n_old, contrast) with d for every metric."""
    out = []
    for (uid, n_old), cs in cells.items():
        scene = uid.split("|")[0]
        r = ed[scene]
        if W is not None:
            w = W[scene]; stratum, weight, drive, dv, log = w["stratum"], w["weight"], w["drive"], w["day_vehicle"], w["log"]
        else:
            stratum, weight, log = ("F" if r["group"] == "A-" else "S"), 1.0, r["log"]
            drive, dv = drive_of(log), dayveh_of(log)
        bk = bank[uid]
        t = bk["t_star"]
        a = t + 1 + n_old + 1
        older_equal = all(bk["normal_own"][j - t - 1] == r["gt"][j] for j in range(t + 1, a - 1))
        srcN, srcF, gtl = bk["normal_own"][a - 1 - t - 1], bk["full_own"][a - 1 - t - 1], r["gt"][a - 1]
        base = {"unit_id": uid, "scene": scene, "n_old": n_old, "anchor": a, "t_star": t, "stratum": stratum, "weight": weight,
                "log": log, "drive": drive, "day_vehicle": dv, "older_values_equal": older_equal,
                "srcF_eq_srcN": srcF == srcN, "srcN_eq_gt": srcN == gtl, "srcF_eq_gt": srcF == gtl}
        pairs = CONTRASTS + SECONDARY + [("INT_a", "GT_NSRC", "GT_REF"), ("INT_b", "NAT_NSRC", "NAT_REF")]
        for cid, ca, cb in pairs:
            if ca not in cs or cb not in cs:
                continue
            A, B = cs[ca], cs[cb]
            valid = len(A["acts"]) == 10 and len(B["acts"]) == 10 and np.isfinite(A["fde5"]) and np.isfinite(B["fde5"])
            da = np.asarray(A["tok_a_motion"]); db = np.asarray(B["tok_a_motion"])
            row = base | {"contrast": cid, "cell_a": ca, "cell_b": cb, "valid": bool(valid),
                          "severe": A["severe"] - B["severe"], "A_plus": A["A_plus"] - B["A_plus"],
                          "fde5": (A["fde5"] - B["fde5"]) if valid else np.nan,
                          "tok_a_changed": int(A["tok_a"] != B["tok_a"]),
                          "suffix_changed_frac": float(np.mean([x != y for x, y in zip(A["own"], B["own"])])),
                          "local_disp_change_m": float(np.linalg.norm(da[:2] - db[:2])),
                          "local_dyaw_change_rad": float(abs(da[2] - db[2])),
                          "endpoint_dist_m": float(np.linalg.norm(np.asarray(A["endpoint"]) - np.asarray(B["endpoint"])))}
            out.append(row)
    return pd.DataFrame(out)


def scene_tab(df, cid, metric, n_old, extra=None):
    u = df[(df.contrast == cid) & (df.n_old == n_old) & df.valid]
    if extra is not None:
        u = u[extra(u)]
    u = u.dropna(subset=[metric])
    if cid == "INT":
        raise ValueError
    g = u.groupby("scene")
    sc = pd.DataFrame({"d": g[metric].mean(), "dsum": g[metric].sum(), "n_units": g[metric].size(), "log": g.log.first(),
                       "drive": g.drive.first(), "day_vehicle": g.day_vehicle.first(), "stratum": g.stratum.first(),
                       "w": g.weight.first().astype(float)}).reset_index()
    sc["cluster"] = sc.drive
    return sc


def int_tab(df, metric, n_old, extra=None):
    """INT = (GT_NSRC - GT_REF) - (NAT_NSRC - NAT_REF) per unit."""
    a = df[(df.contrast == "INT_a") & (df.n_old == n_old) & df.valid].set_index("unit_id")
    b = df[(df.contrast == "INT_b") & (df.n_old == n_old) & df.valid].set_index("unit_id")
    j = a.join(b[[metric]], rsuffix="_b", how="inner")
    j[metric] = j[metric] - j[f"{metric}_b"]
    j = j.reset_index()
    j["contrast"] = "INT"
    if extra is not None:
        j = j[extra(j)]
    return scene_tab(j.assign(contrast="INTX"), "INTX", metric, n_old)


def est(sc, which, cohort, clus="drive"):
    s = AP.estimand_table(sc, which).copy()
    s["cluster"] = s[clus]
    return AP.estimate(s, "scene", SEED, with_ci_inversion=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--cohort", required=True, choices=["p1c", "dev"])
    ap.add_argument("--cohort-manifest", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mask", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    cells, bank, W, ed = load(args.run, args.cohort, args.cohort_manifest)
    df = unit_rows(cells, bank, W, ed, args.cohort)
    res = []
    whiches = ["F", "S", "P"]
    for n_old in sorted(df.n_old.unique()):
        for metric in METRICS:
            for cid in [c[0] for c in CONTRASTS] + ["INT"] + [c[0] for c in SECONDARY]:
                for subset, extra in (("all", None), ("older_values_differ", lambda u: ~u.older_values_equal),
                                      ("src_values_differ", lambda u: ~u.srcF_eq_srcN)):
                    if subset == "src_values_differ" and cid != "C_PROV":
                        continue
                    if subset == "older_values_differ" and cid != "INT":
                        continue
                    sc = int_tab(df, metric, n_old, extra) if cid == "INT" else scene_tab(df, cid, metric, n_old, extra)
                    for which in whiches:
                        for clus in (["drive", "day_vehicle"] if (metric == "severe" and subset == "all") else ["drive"]):
                            e = est(sc, which, args.cohort, clus)
                            res.append({"n_old": int(n_old), "anchor_rule": f"a=t*+{n_old + 2}", "metric": metric, "contrast": cid,
                                        "subset": subset, "estimand": which, "cluster": clus, **e})
    R = pd.DataFrame(res)
    # Holm families: primary anchor (n_old = 1), severe, subset all, drive clusters
    R["family"] = None; R["holm_p"] = np.nan
    for which in ("F", "P"):
        m = (R.n_old == 1) & (R.metric == "severe") & (R.subset == "all") & (R.cluster == "drive") & (R.estimand == which) & \
            R.contrast.isin(["C_GT", "C_NAT", "INT", "C_PROV"])
        if m.sum():
            R.loc[m, "holm_p"] = AP.holm(R.loc[m, "p_signflip"].values)
            R.loc[m, "family"] = f"B2-{which}"
            R.loc[m, "family_alpha"] = ALPHA[which]
            R.loc[m, "reject"] = R.loc[m, "holm_p"] < ALPHA[which]
    # descriptives: source values, older-prefix emptiness, denominators
    desc = []
    for n_old in sorted(df.n_old.unique()):
        u = df[(df.n_old == n_old) & (df.contrast == "C_GT")]
        for which in ("F", "S", "all"):
            v = u if which == "all" else u[u.stratum == which]
            desc.append({"n_old": int(n_old), "stratum": which, "units": len(v), "scenes": v.scene.nunique(), "drives": v.drive.nunique(),
                         "day_vehicle": v.day_vehicle.nunique(), "share_older_values_equal": float(v.older_values_equal.mean()) if len(v) else None,
                         "share_srcF_eq_srcN": float(v.srcF_eq_srcN.mean()) if len(v) else None,
                         "share_srcN_eq_gt": float(v.srcN_eq_gt.mean()) if len(v) else None,
                         "share_srcF_eq_gt": float(v.srcF_eq_gt.mean()) if len(v) else None, "invalid_units": int((~v.valid).sum())})
    D = pd.DataFrame(desc)
    # P1-C input verification (images re-fetched): bank Normal vs stored P1-C H7 normal tokens
    chk = None
    if args.cohort == "p1c":
        st = {}
        for l in open(f"{P1C_RUN}/mech/action_history_causal/records.jsonl"):
            x = json.loads(l)
            for pn, pr in x["perturbations"].items():
                st[f"{x['token']}|{pn}"] = pr["conditions"]["normal"].get("action_idx")
        eq = [bank[u]["normal_acts"] == st.get(u) for u in bank if u in st]
        chk = {"units_compared": len(eq), "normal_tokens_equal_stored_P1C_H7": int(sum(eq)),
               "share": float(np.mean(eq)) if eq else None}
    if args.mask:
        for c in R.columns:
            if c not in ("n_old", "anchor_rule", "metric", "contrast", "subset", "estimand", "cluster", "n_scenes", "n_units",
                         "n_clusters", "weighting", "family", "family_alpha"):
                R[c] = MASK
        D = D[["n_old", "stratum", "units", "scenes", "drives", "day_vehicle", "invalid_units"]]
        df = df[["unit_id", "scene", "n_old", "stratum", "drive", "contrast", "valid"]]
    R.to_csv(os.path.join(args.out, "MATCHED_PREFIX_RESULTS.csv"), index=False)
    D.to_csv(os.path.join(args.out, "matched_prefix_descriptives.csv"), index=False)
    df.to_parquet(os.path.join(args.out, "unit_contrasts.parquet"))
    json.dump({"run": os.path.abspath(args.run), "cohort": args.cohort, "masked": args.mask, "input_check_p1c": chk,
               "seed": SEED, "n_rows": len(R)}, open(os.path.join(args.out, "analysis_meta.json"), "w"), indent=1, default=str)
    print(D.to_string()); print("input check", chk)
    print(R[(R.family.notna())][["estimand", "contrast", "n_scenes", "n_clusters", "theta", "p_signflip", "holm_p"]].to_string())


if __name__ == "__main__":
    main()
