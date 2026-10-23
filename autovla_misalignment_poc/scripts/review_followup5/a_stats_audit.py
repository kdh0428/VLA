#!/usr/bin/env python
"""review_followup5 A_STATS step 2 (CPU): old vs new intervals / margin verdicts for every AutoVLA row produced by the
affected sign-flip machinery (protocol §1). Runs after a_stats_coverage.py wrote A_METHOD_SELECTION.json.

For binary-outcome rows (amplification / severe / recovery): the sign-flip-inversion CI and the shifted sign-flip TOST are
invalid (ties); the replacement is the interval selected for the row's cluster structure by the coverage study
(cluster-robust t, G - 1 df) or NR. Cluster-t CIs are taken from the stored clt columns, or derived exactly from the stored
theta, two-sided cluster-t p and G (se = |theta| / t_{G-1}^{-1}(1 - p/2)); rows where that is impossible are NR.
Continuous-metric rows (FDE, endpoint m) are not affected by ties and are carried unchanged. The delta = 0 sign-flip p,
Holm families, margins and estimands are unchanged.

  nice -n 19 /root/miniforge3/envs/autovla/bin/python a_stats_audit.py
"""
from __future__ import annotations

import json
import math
import os

import numpy as np
import pandas as pd
from scipy.stats import t as tdist

POC = "/root/VLA/autovla_misalignment_poc"
O = f"{POC}/outputs"
OUT = f"{O}/review_followup5/A_STATS"
SEL = json.load(open(f"{OUT}/A_METHOD_SELECTION.json"))["selection"]
BIN = {"amplification", "severe", "recovery", "primary_outcome", "A-", "Aminus"}
LAB = "POST-HOC CORRECTION (review_followup5 A_STATS); original values preserved"


def cohort_of(dataset, stratum, cluster, G):
    """map a row's structure to a simulated cohort (protocol §1 rule 3); None = structure not simulated."""
    if cluster != "drive":
        # only the P1-C day-vehicle structure was simulated; dev day / vehicle / component axes and crossed axes are not
        return "P1C_P_DV" if cluster in ("day_vehicle", "P1C:day_vehicle") and dataset == "P1C" else None
    if G is not None and G <= 7:
        return "SUB7"
    if dataset == "DEV" and stratum in ("A-", "Am", "P1B"):
        return "DEV_Am"
    if dataset == "P1C" and stratum == "F":
        return "P1C_F"
    if dataset == "P1C" and stratum in ("P", "S"):
        return "P1C_P"
    return None


def selected(coh, lev):
    if coh is None:
        return "NR (structure not covered by the coverage study)"
    s = SEL[coh][lev]["selected"]
    return "cluster_t" if s == "t" else ("bootstrap" if s == "boot" else f"NR (no method reached the required coverage for {coh})")


def t_ci(theta, p, G, level, clt=None):
    if clt is not None and np.all(np.isfinite(clt)) and level == 0.95:
        return [float(clt[0]), float(clt[1])], (clt[1] - clt[0]) / (2 * tdist.ppf(0.975, G - 1))
    if not (np.isfinite(theta) and np.isfinite(p) and G and G > 1) or p >= 1 or theta == 0:
        return [np.nan, np.nan], np.nan
    tt = tdist.isf(p / 2, G - 1)
    se = abs(theta) / tt
    q = tdist.ppf(1 - (1 - level) / 2, G - 1)
    return [theta - q * se, theta + q * se], se


def tost_t(theta, se, G, m):
    if not (np.isfinite(se) and se > 0):
        return np.nan
    return float(max(tdist.sf((theta + m) / se, G - 1), tdist.sf((m - theta) / se, G - 1)))


def gint(x):
    try:
        return int(float(x))
    except (TypeError, ValueError):
        return None


def degenerate(lo, hi):
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return ""
    if hi - lo < 1e-9:
        return "zero width"
    if abs(lo) < 1e-9 or abs(hi) < 1e-9:
        return "end pinned at 0"
    return ""


def rows():
    out = []
    # ---- review_followup2 A3 cluster re-analysis (dev A-/A+, P1-C F/P)
    cr = pd.read_csv(f"{O}/review_followup2/A_autovla_cluster/run_20261013/CLUSTER_REANALYSIS.csv")
    for _, r in cr.iterrows():
        if str(r.experiment).startswith("exp33"):
            continue                                   # Impromptu (descriptive, out of AutoVLA scope)
        ds = "P1C" if str(r.experiment).startswith("P1C") else "DEV"
        st = r.group_or_estimand if ds == "P1C" else ("A-" if r.group_or_estimand == "A-" else r.group_or_estimand)
        out.append(dict(source="review_followup2/A_autovla_cluster/run_20261013/CLUSTER_REANALYSIS.csv", row=r.contrast_id,
                        dataset=ds, stratum=st, metric=r.metric, cluster="drive", G=int(r.n_drives), theta=r.drive_estimate_scene,
                        old_ci95=[r.drive_sf_ci95_lo, r.drive_sf_ci95_hi], old_ci90=[r.drive_sf_ci90_lo, r.drive_sf_ci90_hi],
                        boot95=[r.drive_boot_ci95_lo, r.drive_boot_ci95_hi], clt95=[r.drive_clt_ci95_lo, r.drive_clt_ci95_hi],
                        p0=r.drive_p_signflip, p_t=r.drive_p_cluster_t, margin=r.equiv_margin,
                        old_equiv=(f"sign-flip TOST p {r.drive_tost_p_equiv:.3g}; equivalent by sf 90% CI: {r.drive_equiv_by_sf_ci90}"
                                   if np.isfinite(r.equiv_margin) else ""), holm_old=r.drive_holm_p))
    # ---- P1-C frozen analysis: primary decisions + windows equivalence
    pe = pd.read_csv(f"{O}/review_followup/P1C_new_log_replication/run_20261008/analysis/paired_effects.csv")
    pe = pe[pe.sf_ci95_lo.notna() | pe.equiv_margin.notna()]
    for _, r in pe.iterrows():
        cl = "day_vehicle" if "day" in str(r.variant).lower() and "vehicle" in str(r.variant).lower() else "drive"
        out.append(dict(source="review_followup/P1C_new_log_replication/run_20261008/analysis/paired_effects.csv",
                        row=f"{r.kind}|{r.contrast}|{r.estimand}|{r.variant}|{r.weighting}", dataset="P1C", stratum=r.estimand,
                        metric=r.metric, cluster=cl, G=int(r.n_clusters) if np.isfinite(r.n_clusters) else None, theta=r.theta,
                        old_ci95=[r.sf_ci95_lo, r.sf_ci95_hi], old_ci90=[r.sf_ci90_lo, r.sf_ci90_hi], boot95=[r.boot_ci95_lo, r.boot_ci95_hi],
                        clt95=[r.clt_ci95_lo, r.clt_ci95_hi], p0=r.p_signflip, p_t=r.p_cluster_t, margin=r.equiv_margin,
                        mme=r.mme, old_decision=r.decision,
                        old_equiv=(f"equivalent by sf 90% CI: {r.equivalent_sf90}; by bootstrap 90% CI: {r.equivalent_boot90}"
                                   if np.isfinite(r.equiv_margin) else ""), holm_old=r.holm_p))
    # ---- P1-B early - late (c2) and C1 predeclared (amplification rows only carry ties)
    c2 = pd.read_csv(f"{O}/review_followup2/C_baseline_window/run_20261013/c2_position_length.csv")
    for _, r in c2.iterrows():
        out.append(dict(source="review_followup2/C_baseline_window/run_20261013/c2_position_length.csv",
                        row=f"{r.analysis}|{r.window_kind}|{r.contrast}|w{r.w}|{r.metric}|{r.subset}", dataset="DEV", stratum="P1B",
                        metric=r.metric, cluster="drive", G=int(r.n_drives), theta=r.est_scene, old_ci95=[r.sf_ci95_lo, r.sf_ci95_hi],
                        old_ci90=[np.nan, np.nan], boot95=[r.ci95_lo, r.ci95_hi], clt95=[np.nan, np.nan], p0=r.p_signflip,
                        p_t=r.p_cluster_t, margin=r.tost_margin,
                        old_equiv=(f"sign-flip TOST p {r.tost_p_equiv:.3g} (Holm m3 {r.tost_p_equiv_holm_m3:.3g}); {r.margin_status}"
                                   if np.isfinite(r.tost_margin) else ""), holm_old=r.tost_p_equiv_holm_m3, family="P1-B TOST Holm m=3 per (analysis, metric)"))
    c1 = pd.read_csv(f"{O}/review_followup2/C_baseline_window/run_20261013/c1_predeclared_and_gt_drive.csv")
    for _, r in c1.iterrows():
        out.append(dict(source="review_followup2/C_baseline_window/run_20261013/c1_predeclared_and_gt_drive.csv",
                        row=f"{r.source}|{r.family}|{r.group}|{r.contrast}|{r.metric}", dataset="DEV",
                        stratum=("A-" if r.group == "A-" else r.group), metric=r.metric, cluster="drive", G=int(r.n_drives),
                        theta=r.est_scene, old_ci95=[r.sf_ci95_lo, r.sf_ci95_hi], old_ci90=[np.nan, np.nan],
                        boot95=[r.ci95_lo, r.ci95_hi], clt95=[np.nan, np.nan], p0=r.p_signflip, p_t=r.p_cluster_t,
                        margin=r.tost_margin, old_equiv=(f"sign-flip TOST p {r.tost_p_equiv:.3g}" if np.isfinite(r.tost_margin) else ""),
                        holm_old=r.p_holm_m3_drive))
    # ---- review_followup3 P0-A population effects and P0-D dependence
    pop = pd.read_csv(f"{O}/review_followup3/P0/POPULATION_EFFECTS.csv")
    for _, r in pop.iterrows():
        ds = "P1C" if str(r.cohort).startswith("P1") else "DEV"
        st = r.estimand if ds == "P1C" else ("A-" if str(r.estimand).startswith("A-") else r.estimand)
        out.append(dict(source="review_followup3/P0/POPULATION_EFFECTS.csv", row=r.row_id, dataset=ds, stratum=st, metric=r.metric,
                        cluster=str(r.cluster), G=gint(r.n_clusters), theta=r.theta, old_ci95=[r.sf_ci95_lo, r.sf_ci95_hi],
                        old_ci90=[r.sf_ci90_lo, r.sf_ci90_hi], boot95=[r.boot_ci95_lo, r.boot_ci95_hi], clt95=[r.clt_ci95_lo, r.clt_ci95_hi],
                        p0=r.p_signflip_raw, p_t=np.nan, margin=np.nan, holm_old=r.p_holm))
    dep = pd.read_csv(f"{O}/review_followup3/P0/DEPENDENCE_SENSITIVITY.csv")
    for _, r in dep.iterrows():
        ds = "P1C" if str(r.contrast_id).startswith("P1C") else "DEV"
        st = str(r.contrast_id).split()[-1] if ds == "P1C" else "A-"
        out.append(dict(source="review_followup3/P0/DEPENDENCE_SENSITIVITY.csv", row=r.row_id, dataset=ds, stratum=st,
                        metric="amplification", cluster=("drive" if r.axis == "drive" else f"{ds}:{r.axis}"), G=gint(r.G), theta=r.theta,
                        old_ci95=[r.sf_ci95_lo, r.sf_ci95_hi], old_ci90=[r.sf_ci90_lo, r.sf_ci90_hi], boot95=[r.boot_ci95_lo, r.boot_ci95_hi],
                        clt95=[r.clt_ci95_lo, r.clt_ci95_hi], p0=r.p_signflip, p_t=r.p_cluster_t, margin=np.nan, holm_old=r.holm_p))
    ds_ = pd.read_csv(f"{O}/review_followup3/P0/DEPENDENCE_disjoint_subsets.csv")
    for _, r in ds_.iterrows():
        out.append(dict(source="review_followup3/P0/DEPENDENCE_disjoint_subsets.csv", row=f"{r.reference_set}|{r.subset}|{r.contrast_id}",
                        dataset="SUB", stratum="subset", metric="amplification", cluster="drive", G=gint(r.drive_G), theta=r.theta,
                        old_ci95=[r.sf_ci95_lo, r.sf_ci95_hi], old_ci90=[np.nan, np.nan], boot95=[r.boot_ci95_lo, r.boot_ci95_hi],
                        clt95=[np.nan, np.nan], p0=r.drive_p, p_t=np.nan, margin=np.nan, holm_old=np.nan))
    sb = pd.read_csv(f"{O}/review_followup2/A_autovla_cluster/run_20261013/p1c_subsets_drive_level.csv")
    for _, r in sb.iterrows():
        out.append(dict(source="review_followup2/A_autovla_cluster/run_20261013/p1c_subsets_drive_level.csv",
                        row=f"{r.contrast_id}|{r.subset}", dataset="SUB", stratum="subset", metric="amplification", cluster="drive",
                        G=gint(r.n_drives), theta=r.theta, old_ci95=[r.sf_ci95_lo, r.sf_ci95_hi], old_ci90=[np.nan, np.nan],
                        boot95=[r.boot_ci95_lo, r.boot_ci95_hi], clt95=[np.nan, np.nan], p0=r.p_signflip, p_t=np.nan, margin=np.nan,
                        holm_old=np.nan))
    return out


def main():
    res = []
    for r in rows():
        binary = r["metric"] in BIN or str(r["metric"]).startswith("exceed")
        coh = cohort_of(r["dataset"], r["stratum"], r["cluster"], r["G"])
        o95, o90 = [float(x) for x in r["old_ci95"]], [float(x) for x in r["old_ci90"]]
        rec = {"source": r["source"], "row": r["row"], "dataset": r["dataset"], "stratum": r["stratum"], "metric": r["metric"],
               "outcome_type": "binary (ties)" if binary else "continuous", "cluster": r["cluster"], "G": r["G"],
               "estimate": r["theta"], "p_signflip_delta0 (unchanged, valid)": r["p0"], "holm_original": r.get("holm_old"),
               "old_ci95_signflip_inv": o95, "old_ci90_signflip_inv": o90, "old_degeneracy": degenerate(*o95) or degenerate(*o90),
               "bootstrap_ci95 (reported primary)": [float(x) for x in r["boot95"]], "margin": r.get("margin"), "mme": r.get("mme"),
               "old_margin_verdict": r.get("old_equiv", ""), "old_decision": r.get("old_decision", "")}
        if not binary:
            rec.update({"cohort_sim": "", "new_method": "unchanged (continuous metric; no ties)", "new_ci95": o95, "new_ci90": o90,
                        "new_margin_verdict": rec["old_margin_verdict"], "change": "none"})
            res.append(rec); continue
        m95, m90 = selected(coh, "95"), selected(coh, "90")
        clt = np.array(r["clt95"], float)
        G = r["G"] or 0
        if m95 == "cluster_t":
            ci95, se = t_ci(r["theta"], r["p_t"], G, 0.95, clt)
            ci90, _ = t_ci(r["theta"], r["p_t"], G, 0.90, None) if not np.isfinite(se) else ([r["theta"] - tdist.ppf(0.95, G - 1) * se, r["theta"] + tdist.ppf(0.95, G - 1) * se], se)
            if not np.all(np.isfinite(ci95)):
                m95 = m90 = "NR (cluster-t not derivable from the stored row)"
        else:
            ci95 = ci90 = [np.nan, np.nan]; se = np.nan
        verdict = ""
        if np.isfinite(r.get("margin") or np.nan):
            mg = float(r["margin"])
            if m90 == "cluster_t" and np.isfinite(se):
                pe = tost_t(r["theta"], se, G, mg)
                inside = ci90[0] > -mg and ci90[1] < mg
                verdict = f"cluster-t TOST p {pe:.3g}; 90% CI inside +-{mg:g}: {bool(inside)}"
                rec["new_p_equiv_t"] = pe
            else:
                verdict = "inconclusive (no valid interval for the margin test)"
        mmev = ""
        if np.isfinite(r.get("mme") or np.nan) and r.get("old_decision"):
            if r["old_decision"] == "replicated":
                mmev = "replicated (unchanged: rests on the delta = 0 sign-flip Holm p and the sign)"
            elif m90 == "cluster_t":
                mmev = ("not replicated, below MME" if ci90[0] > -abs(r["mme"]) and ci90[1] < abs(r["mme"]) else "inconclusive")
            else:
                mmev = "inconclusive (no valid 90% interval)"
        ch = []
        if rec["old_degeneracy"]:
            ch.append(f"old interval {rec['old_degeneracy']}")
        if r.get("old_equiv") and verdict:
            ch.append("margin verdict re-evaluated")
        if mmev and r.get("old_decision") and not mmev.startswith(r["old_decision"]):
            ch.append(f"decision {r['old_decision']} -> {mmev}")
        rec.update({"cohort_sim": coh or "", "new_method": m95, "new_ci95": [float(x) for x in ci95], "new_ci90": [float(x) for x in ci90],
                    "new_margin_verdict": verdict, "new_decision": mmev, "change": "; ".join(ch) or "interval replaced (label only)"})
        res.append(rec)
    df = pd.DataFrame(res)
    # Holm m = 3 over w1..w3 for the P1-B TOST families (same family definition as the original c2 table)
    df["new_p_equiv_t_holm_m3"] = np.nan
    c2 = df.source.str.contains("c2_position_length") & df.get("new_p_equiv_t", pd.Series(np.nan, index=df.index)).notna()
    if c2.any():
        key = df.loc[c2, "row"].str.replace(r"\|w\d\|", "|w*|", regex=True)
        for k, idx in df.loc[c2].groupby(key).groups.items():
            ps = df.loc[idx, "new_p_equiv_t"].values
            order = np.argsort(ps); m = len(ps); adj = np.empty(m); run = 0.0
            for i, j in enumerate(order):
                run = max(run, min(1.0, (m - i) * ps[j])); adj[j] = run
            df.loc[idx, "new_p_equiv_t_holm_m3"] = adj
    df["status"] = LAB
    df.to_csv(f"{OUT}/AUTOVLA_CI_AUDIT.csv", index=False)
    b = df[df.outcome_type.str.startswith("binary")]
    summ = {"rows_total": len(df), "rows_binary": len(b), "rows_continuous_unchanged": int((df.outcome_type == "continuous").sum()),
            "binary_new_method_counts": b.new_method.value_counts().to_dict(),
            "old_degenerate_rows": int((df.old_degeneracy != "").sum()),
            "margin_rows": int((df.old_margin_verdict.astype(str) != "").sum()),
            "decisions_changed": int(b.change.str.contains("decision").sum())}
    json.dump(summ, open(f"{OUT}/a_stats_audit_summary.json", "w"), indent=1, default=str)
    print(json.dumps(summ, indent=1, default=str))
    m = df[df.old_margin_verdict.astype(str) != ""][["source", "row", "estimate", "margin", "old_margin_verdict", "new_method", "new_ci90", "new_margin_verdict"]]
    pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 80)
    print(m.to_string())


if __name__ == "__main__":
    main()
