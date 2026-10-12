#!/usr/bin/env python
"""
A3 (review_followup2, 2026-10-13): drive-level cluster re-analysis of the AutoVLA / Impromptu primary contrasts.

*** POST-HOC ANALYSIS OF EXISTING DATA. *** No model inference, no GPU. Reads existing units tables only, writes only to
outputs/review_followup2/A_autovla_cluster/run_20261013/. Nothing here is preregistered (the P1-C rows re-derive a
preregistered analysis and are labelled as such).

Preserved from the original analyses:
  * estimand weighting: scene-weighted (P0-B / P1-C primary) kept as primary; unit-weighted and cluster-equal
    (log-equal or drive-equal) reported as separate estimands in separate columns;
  * hierarchy drive -> log -> scene -> perturbation unit, pairing of conditions inside each unit (each contrast uses its
    own experiment/block, i.e. its own Normal);
  * contrast definitions, groups (A-), metrics (amplification = not A+ and FDE5 > 3 m) and exclusion rules (none for
    AutoVLA primary conditions; Impromptu parse failures excluded as in the original) -- inherited from the units tables.
Changed: the independent unit for resampling/testing is the DRIVE (`<date.time>_veh-NN`, the log-name prefix, same
definition as P1-C split.csv); logs of one drive move together. Log-level results are recomputed for comparison.

Usage: nice -n 19 python a3_cluster_reanalysis.py
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import stats_core as SC  # noqa: E402

POC = "/root/VLA/autovla_misalignment_poc"
RF = os.path.join(POC, "outputs/review_followup")
OUT = os.path.join(POC, "outputs/review_followup2/A_autovla_cluster/run_20261013")
os.makedirs(OUT, exist_ok=True)

P0B_SEED = 20261008      # P0-B: log bootstrap seed = 20261008 + 100 i (AutoVLA) / + 700 + 10 i (Impromptu)
P1C_SEED = 20261008      # P1-C frozen script: drive bootstrap seed 20261008
SEED = 20261013          # this run (drive level)
B = 10_000
B_LOO = 2_000
MC = 200_000
T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:7.1f}s]", *a, flush=True)


def drive_of(log_id: str) -> str:
    p = log_id.split("_")
    return f"{p[0]}_{p[1]}"


def dayveh_of(log_id: str) -> str:
    p = log_id.split("_")
    return f"{p[0][:10]}_{p[1]}"


# =================================================================================================== load units
def load():
    u = pd.read_parquet(os.path.join(RF, "P0B_cluster_stats/run_20261008/units.parquet"))
    u = u[u["model"].isin(["AutoVLA", "Impromptu-VLA-3B"]) & u["condition"].notna()].copy()
    u = u[u["exclusion_reason"].isna()]
    r8 = pd.read_parquet(os.path.join(RF, "R8R9_raw_recovery/run_20261008/units.parquet"),
                         columns=["experiment", "unit_id", "log_id", "scene_id", "group", "condition", "amplification",
                                  "fde5", "exclusion_reason"])
    r8 = r8[(r8["experiment"] == "R8") & r8["exclusion_reason"].isna()].copy()
    r8["experiment"] = "R8_prev_action_state_patching"
    c = pd.read_parquet(os.path.join(RF, "P1C_new_log_replication/run_20261008/analysis/units.parquet"))
    for df in (u, r8, c):
        df["drive_id"] = df["log_id"].map(drive_of)
        df["dayveh_id"] = df["log_id"].map(dayveh_of)
    assert (c["drive_id"] == c["drive_id"]).all()
    return u, r8, c


# =================================================================================================== contrast specs
# id, family, model, source, experiment, group, cond_a, cond_b, metric, predicted sign, margin (equivalence question)
AV = [
    ("AV-P1 Recent-Normal", "AutoVLA primary (P0-B, Holm m=3)", "AutoVLA", "P0B", "exp07_action_history_causal", "A-",
     "recent_gt", "normal", "amplification", -1, None, 0),
    ("AV-P2 Reverse-Full", "AutoVLA primary (P0-B, Holm m=3)", "AutoVLA", "R8", "R8_prev_action_state_patching", "A-",
     "reverse@emb", "gt_history", "amplification", +1, None, None),
    ("AV-P3 Direction-Magnitude", "AutoVLA primary (P0-B, Holm m=3)", "AutoVLA", "P0B", "exp32_motion_semantics", "A-",
     "dir_wrong_mag_ok", "dir_ok_mag_wrong", "amplification", +1, None, 1),
    ("AV-S GeoNN(GT)-RecentGT [equivalence]", "AutoVLA secondary (equivalence question, margin +-5 pp)", "AutoVLA",
     "P0B", "exp11_prev_action_identity", "A-", "geo_nn_gt", "recent_gt", "amplification", None, 0.05, 6),
    ("AV-S Recent-Normal FDE", "AutoVLA secondary", "AutoVLA", "P0B", "exp07_action_history_causal", "A-",
     "recent_gt", "normal", "fde5", -1, None, 2),
    ("AV-S Reverse-Full FDE", "AutoVLA secondary", "AutoVLA", "R8", "R8_prev_action_state_patching", "A-",
     "reverse@emb", "gt_history", "fde5", +1, None, None),
    ("AV-S Direction-Magnitude FDE", "AutoVLA secondary", "AutoVLA", "P0B", "exp32_motion_semantics", "A-",
     "dir_wrong_mag_ok", "dir_ok_mag_wrong", "fde5", +1, None, 3),
    ("AV-S Recent-Normal A+", "AutoVLA secondary", "AutoVLA", "P0B", "exp07_action_history_causal", "A+",
     "recent_gt", "normal", "amplification", -1, None, 8),
    ("AV-S Reverse-Full A+", "AutoVLA secondary", "AutoVLA", "R8", "R8_prev_action_state_patching", "A+",
     "reverse@emb", "gt_history", "amplification", +1, None, None),
    ("AV-S Direction-Magnitude A+", "AutoVLA secondary", "AutoVLA", "P0B", "exp32_motion_semantics", "A+",
     "dir_wrong_mag_ok", "dir_ok_mag_wrong", "amplification", +1, None, 9),
]
IMP = [
    ("IMP Recent-Normal", "recent_gt", "normal", -1, 0),
    ("IMP GT-history-Normal", "gt_history", "normal", -1, 1),
    ("IMP Reverse-Full", "reverse", "gt_history", +1, 2),
    ("IMP Direction-Magnitude", "dir_wrong_mag_ok", "dir_ok_mag_wrong", +1, 3),
]
P1C = [("RN", "H7", "recent_gt", "normal", -1, {"F": -0.10, "P": -0.02}),
       ("RV", "H8", "reverse@emb", "gt_history", +1, {"F": +0.10, "P": +0.02}),
       ("DM", "H32", "dir_wrong_mag_ok", "dir_ok_mag_wrong", +1, {"F": +0.05, "P": +0.02})]
P1C_ALPHA = {"F": 0.05, "P": 0.025}

INTERVENTION_NOTE = {
    "recent_gt": "context-only; sliding replacement of position k-1 by the GT token at EVERY step k>=t*+2 (8-t* times; "
                 "repeated, not a single correction); uses future GT",
    "gt_history": "context-only; all previous action tokens replaced by GT (uses future GT)",
    "reverse@emb": "AutoVLA exp 8 / P1-C H8: GT-history context, embedding at k-1 copied from the SEPARATE Normal row of "
                   "the same batch (Normal-branch token injection; needs the normal counterfactual rollout)",
    "reverse": "Impromptu exp 33: corrected context with the branch's OWN k-1 waypoint re-inserted (self-error "
               "re-insertion) -- a different intervention from AutoVLA reverse@emb",
    "dir_wrong_mag_ok": "sliding k-1 substitute with GT magnitude, wrong direction (built from GT and own token; uses GT)",
    "dir_ok_mag_wrong": "sliding k-1 substitute with GT direction, wrong magnitude (uses GT)",
    "geo_nn_gt": "sliding k-1 substitute = token whose motion is nearest to GT, excluding GT itself (uses GT)",
}


def orig_tables():
    cmp_ = pd.read_csv(os.path.join(RF, "P0B_cluster_stats/run_20261008/comparison_with_original.csv")).set_index("contrast_id")
    pe = pd.read_csv(os.path.join(RF, "P0B_cluster_stats/run_20261008/paired_effects.csv"))
    pe = pe[pe["weighting"] == "scene-weighted (primary)"].set_index("contrast_id")
    r8 = pd.read_csv(os.path.join(RF, "R8R9_raw_recovery/run_20261008/paired_effects.csv"))
    dec = json.load(open(os.path.join(RF, "P1C_new_log_replication/run_20261008/analysis/decisions.json")))["decisions"]
    return cmp_, pe, r8, dec


def fmt(x, d=4):
    return None if x is None or (isinstance(x, float) and not np.isfinite(x)) else round(float(x), d)


# =================================================================================================== core runner
def run_levels(paired, levels, *, w=None, seeds, margin=None, extra=("log_id",)):
    """paired: rows with columns d, scene_id, + cluster columns; returns {level: {weighting: result}} and LOO."""
    res = {}
    for lev, col in levels.items():
        sc = SC.scene_table(paired, col, "scene_id", d="d", w=w, extra=extra)
        res[lev] = {}
        for wt in SC.WEIGHTINGS:
            r = SC.analyze(sc, wt, B=B, seed=seeds[lev], reps=MC, margin=margin)
            r.pop("_arrays")
            res[lev][wt] = r
        res[lev]["_sc"] = sc
    return res


def lodo(sc, wt="scene", seed=SEED):
    return SC.leave_one_cluster_out(sc, wt, B=B_LOO, seed=seed, reps=20_000)


def main():
    u, r8, c = load()
    cmp_, pe, r8pe, dec = orig_tables()
    log("loaded", len(u), len(r8), len(c))
    ROWS, LONG, LOO = [], [], []

    # ------------------------------------------------------------------------------------------- AutoVLA (PoC set)
    for cid, fam, model, src, exp, grp, a, b, met, sign, margin, p0b_i in AV:
        base = r8 if src == "R8" else u[u["experiment"] == exp]
        base = base[base["group"] == grp]
        pr, rep = SC.pair_rows(base, ["unit_id"], "condition", met, a, b,
                               keep_cols=["scene_id", "log_id", "drive_id", "dayveh_id"])
        seeds = {"log": P0B_SEED + 100 * p0b_i if p0b_i is not None else SEED + 1, "drive": SEED}
        res = run_levels(pr, {"log": "log_id", "drive": "drive_id"}, seeds=seeds, margin=margin)
        lo = lodo(res["drive"]["_sc"])
        lo.insert(0, "contrast_id", cid); LOO.append(lo)
        # original values
        if cid == "AV-P2 Reverse-Full":
            o = cmp_.loc["AV-P2 Reverse-Full"]
            orig = dict(est=o.orig_estimate, lo=o.orig_ci95_lo, hi=o.orig_ci95_hi, p=o.orig_p, meth=o.orig_method)
            rr = r8pe[(r8pe.family == "R8_primary") & (r8pe.metric == met)].iloc[0]
            rep_log = dict(est=rr.scene_est, lo=rr.scene_ci95_lo, hi=rr.scene_ci95_hi, p=rr.scene_p_signflip)
        elif src == "R8":
            fam_ = "R8_secondary_A-" if grp == "A-" else "R8_secondary_A+"
            rr = r8pe[(r8pe.family == fam_) & (r8pe.contrast == "reverse@emb - gt_history") & (r8pe.metric == met)].iloc[0]
            orig = dict(est=rr.unit_est, lo=np.nan, hi=np.nan, p=np.nan, meth="(no original per-unit p; R8R9 recovery)")
            rep_log = dict(est=rr.scene_est, lo=rr.scene_ci95_lo, hi=rr.scene_ci95_hi, p=rr.scene_p_signflip)
        else:
            o = cmp_.loc[cid] if cid in cmp_.index else None
            orig = (dict(est=o.orig_estimate, lo=o.orig_ci95_lo, hi=o.orig_ci95_hi, p=o.orig_p, meth=o.orig_method)
                    if o is not None else dict(est=np.nan, lo=np.nan, hi=np.nan, p=np.nan, meth="(not in original summary)"))
            q = pe.loc[cid]
            rep_log = dict(est=q.estimate, lo=q.ci95_lo, hi=q.ci95_hi, p=q.p_signflip)
        ROWS.append(dict(contrast_id=cid, family=fam, model=model, experiment=exp, group_or_estimand=grp, cond_a=a,
                         cond_b=b, metric=met, predicted_sign=sign, margin=margin, orig=orig, rep_log=rep_log, res=res,
                         lodo=lo, pair_report=rep, notes=INTERVENTION_NOTE.get(a, "")))
        log(cid, "drive", res["drive"]["scene"]["theta"], res["drive"]["scene"]["p_signflip"])

    # ------------------------------------------------------------------------------------------- Impromptu exp 33
    imp = u[u["experiment"] == "exp33_impromptu_cde"]
    for cid, a, b, sign, i in IMP:
        pr, rep = SC.pair_rows(imp, ["unit_id"], "condition", "amplification", a, b,
                               keep_cols=["scene_id", "log_id", "drive_id", "dayveh_id"])
        res = run_levels(pr, {"log": "log_id", "drive": "drive_id"}, seeds={"log": P0B_SEED + 700 + 10 * i, "drive": SEED})
        lo = lodo(res["drive"]["_sc"]); lo.insert(0, "contrast_id", cid); LOO.append(lo)
        o = cmp_.loc[cid]; q = pe.loc[cid]
        ROWS.append(dict(contrast_id=cid, family="Impromptu exp 33 (P0-B secondary family, own Holm m=4)",
                         model="Impromptu-VLA-3B", experiment="exp33_impromptu_cde", group_or_estimand="all units",
                         cond_a=a, cond_b=b, metric="amplification", predicted_sign=sign, margin=None,
                         orig=dict(est=o.orig_estimate, lo=o.orig_ci95_lo, hi=o.orig_ci95_hi, p=o.orig_p, meth=o.orig_method),
                         rep_log=dict(est=q.estimate, lo=q.ci95_lo, hi=q.ci95_hi, p=q.p_signflip), res=res, lodo=lo,
                         pair_report=rep, notes=INTERVENTION_NOTE.get(a, "")))
        log(cid, res["drive"]["scene"]["theta"], res["drive"]["scene"]["p_signflip"])

    # ------------------------------------------------------------------------------------------- P1-C (drive-level by design)
    split = pd.read_csv(os.path.join(RF, "P1C_new_log_replication/design_20261008/split.csv"), dtype=str).set_index("log")
    nonnav = {l for l in split.index if split.loc[l, "role"] == "eval" and split.loc[l, "navhard_source"] == "False"}
    dvdis = {l for l in split.index if split.loc[l, "role"] == "eval" and split.loc[l, "dayveh_overlaps_poc"] == "False"}
    P1C_SUB = []
    for cid, blk, a, b, sign, mme in P1C:
        cb = c[c["block"] == blk]
        pr, rep = SC.pair_rows(cb, ["unit_id"], "condition", "amplification", a, b,
                               keep_cols=["scene_id", "log_id", "drive_id", "dayveh_id", "stratum", "weight"])
        for est in ("F", "P"):
            sub = pr[pr["stratum"] == "F"].copy() if est == "F" else pr.copy()
            if est == "F":
                sub["weight"] = 1.0
            res = run_levels(sub, {"log": "log_id", "drive": "drive_id", "dayveh": "dayveh_id"}, w="weight",
                             seeds={"log": SEED + 2, "drive": P1C_SEED, "dayveh": SEED + 3})
            lo = lodo(res["drive"]["_sc"]); lo.insert(0, "contrast_id", f"P1C {cid} {est}"); LOO.append(lo)
            d0 = next(x for x in dec if x["contrast"] == cid and x["estimand"] == est)
            ROWS.append(dict(contrast_id=f"P1C {cid} {est}", family=f"P1-C family {est} (preregistered b212637+amend., "
                             f"Holm m=3, alpha={P1C_ALPHA[est]})", model="AutoVLA", experiment=f"P1C_{blk}",
                             group_or_estimand=est, cond_a=a, cond_b=b, metric="amplification", predicted_sign=sign,
                             margin=None, mme=mme[est], orig=dict(est=d0["theta"], lo=d0["boot_ci95_lo"], hi=d0["boot_ci95_hi"],
                             p=d0["p_signflip"], meth="P1-C preregistered: drive sign-flip exact, drive bootstrap",
                             holm=d0["holm_p"], decision=d0["decision"], sf90=(d0["sf_ci90_lo"], d0["sf_ci90_hi"])),
                             rep_log=None, res=res, lodo=lo, pair_report=rep, notes=INTERVENTION_NOTE.get(a, "")))
            # sensitivity subsets at drive level (post-hoc recomputation of prespecified S1/S2)
            for sname, logs in (("S1_non_navhard_source", nonnav), ("S2_dayveh_disjoint", dvdis)):
                ss = sub[sub["log_id"].isin(logs)]
                if ss["drive_id"].nunique() < 2:
                    continue
                sc = SC.scene_table(ss, "drive_id", "scene_id", w="weight")
                r = SC.analyze(sc, "scene", B=B, seed=SEED, reps=MC)
                P1C_SUB.append(dict(contrast_id=f"P1C {cid} {est}", subset=sname, n_drives=r["n_clusters"],
                                    n_scenes=r["n_scenes"], theta=r["theta"], boot_ci95_lo=r["boot_ci95"][0],
                                    boot_ci95_hi=r["boot_ci95"][1], sf_ci95_lo=r["sf_ci95"][0], sf_ci95_hi=r["sf_ci95"][1],
                                    p_signflip=r["p_signflip"], sf_method=r["sf_method"],
                                    p_min_attainable=r["p_min_attainable"]))
            log("P1C", cid, est, res["drive"]["scene"]["theta"], res["drive"]["scene"]["p_signflip"])

    # =============================================================================================== Holm + judgments
    def fam_holm(prefix_ids, level):
        ids = [r for r in ROWS if r["contrast_id"] in prefix_ids]
        adj = SC.holm([r["res"][level]["scene"]["p_signflip"] for r in ids])
        for r, h in zip(ids, adj):
            r.setdefault("holm", {})[level] = float(h)

    av_fam = ["AV-P1 Recent-Normal", "AV-P2 Reverse-Full", "AV-P3 Direction-Magnitude"]
    imp_fam = [x[0] for x in IMP]
    for lev in ("log", "drive"):
        fam_holm(av_fam, lev); fam_holm(imp_fam, lev)
    for est in ("F", "P"):
        for lev in ("log", "drive", "dayveh"):
            fam_holm([f"P1C {x[0]} {est}" for x in P1C], lev)

    # =============================================================================================== output tables
    out_rows = []
    for r in ROWS:
        dr, lg = r["res"]["drive"], r["res"]["log"]
        ds, ls = dr["scene"], lg["scene"]
        alpha = P1C_ALPHA.get(r["group_or_estimand"], 0.05) if r["contrast_id"].startswith("P1C") else 0.05
        hol = r.get("holm", {})
        h_d = hol.get("drive", ds["p_signflip"]); h_l = hol.get("log", ls["p_signflip"])
        in_family = "holm" in r
        o = r["orig"]
        # judgments
        if r["margin"] is not None:
            t = ds["tost"]; tl = ls["tost"]
            j_d = SC.judge(ds["theta"], ds["boot_ci95"], ds["p_signflip"], 0.05, t["p_equiv"], r["margin"])
            j_l = SC.judge(ls["theta"], ls["boot_ci95"], ls["p_signflip"], 0.05, tl["p_equiv"], r["margin"])
            j_orig = "described as 'as effective as GT' (equivalence claim, no margin/test)"
            jd, jl = j_d["label"], j_l["label"]
        elif r["contrast_id"].startswith("P1C"):
            def p1c_dec(res_, holm_p, sf90):
                th = res_["theta"]
                if np.isfinite(holm_p) and holm_p < alpha and np.sign(th) == r["predicted_sign"]:
                    return "replicated"
                if np.all(np.isfinite(sf90)) and -abs(r["mme"]) < sf90[0] and sf90[1] < abs(r["mme"]):
                    return "not_replicated_below_MME"
                return "inconclusive"
            jd = p1c_dec(ds, h_d, ds["sf_ci90"])
            jl = p1c_dec(ls, h_l, ls["sf_ci90"] if np.all(np.isfinite(ls["sf_ci90"])) else ls["boot_ci90"])
            j_orig = o["decision"]
        else:
            def sig(res_, p):
                ok = np.isfinite(p) and p < alpha
                if r["predicted_sign"] is not None:
                    return "significant, sign as predicted" if ok and np.sign(res_["theta"]) == r["predicted_sign"] else (
                        "significant, opposite sign" if ok else "not significant")
                return "significant" if ok else "not significant"
            jd, jl = sig(ds, h_d), sig(ls, h_l)
            j_orig = ("significant (unit-level p)" if np.isfinite(o["p"]) and o["p"] < 0.05 else
                      ("no original per-unit test" if not np.isfinite(o["p"]) else "not significant"))
        lo = r["lodo"]
        G = ds["n_clusters"]
        caveat = []
        if G <= 15:
            caveat.append(f"only {G} independent drives; percentile bootstrap may under-cover (sim coverage 0.86-0.93 at G=13)")
        caveat.append(f"min attainable exact two-sided p = {ds['p_min_attainable']:.2g} ({ds['n_informative']} informative drives)")
        if np.isfinite(ds["p_signflip"]) and ds["p_signflip"] <= ds["p_min_attainable"] * 1.0001:
            caveat.append("observed p equals the attainable floor (all informative drives same sign)")
        n_logs = r["res"]["log"]["scene"]["n_clusters"]
        row = {
            "contrast_id": r["contrast_id"], "family": r["family"], "model": r["model"], "experiment": r["experiment"],
            "group_or_estimand": r["group_or_estimand"], "cond_a": r["cond_a"], "cond_b": r["cond_b"], "metric": r["metric"],
            "analysis_status": ("re-derivation of preregistered drive-level analysis" if r["contrast_id"].startswith("P1C")
                                else "POST-HOC re-analysis of existing data"),
            "n_units": ds["n_units"], "n_scenes": ds["n_scenes"], "n_logs": n_logs, "n_drives": G,
            "n_informative_drives": ds["n_informative"],
            # original
            "orig_estimate": o["est"], "orig_ci95_lo": o["lo"], "orig_ci95_hi": o["hi"], "orig_p": o["p"],
            "orig_method": o["meth"], "orig_holm_p": o.get("holm"),
            # log level, previously reported (P0-B / R8R9)
            "logrep_estimate_scene": (r["rep_log"] or {}).get("est"), "logrep_ci95_lo": (r["rep_log"] or {}).get("lo"),
            "logrep_ci95_hi": (r["rep_log"] or {}).get("hi"), "logrep_p_signflip": (r["rep_log"] or {}).get("p"),
            # log level, recomputed here
            "log_estimate_scene": ls["theta"], "log_ci95_lo": ls["boot_ci95"][0], "log_ci95_hi": ls["boot_ci95"][1],
            "log_p_signflip": ls["p_signflip"], "log_sf_method": ls["sf_method"], "log_holm_p": hol.get("log") if in_family else None,
            # drive level (primary new result, scene weighting = original primary estimand)
            "drive_estimate_scene": ds["theta"], "drive_boot_ci95_lo": ds["boot_ci95"][0], "drive_boot_ci95_hi": ds["boot_ci95"][1],
            "drive_boot_ci90_lo": ds["boot_ci90"][0], "drive_boot_ci90_hi": ds["boot_ci90"][1],
            "drive_sf_ci95_lo": ds["sf_ci95"][0], "drive_sf_ci95_hi": ds["sf_ci95"][1],
            "drive_sf_ci90_lo": ds["sf_ci90"][0], "drive_sf_ci90_hi": ds["sf_ci90"][1],
            "drive_p_signflip": ds["p_signflip"], "drive_sf_method": ds["sf_method"],
            "drive_p_min_attainable": ds["p_min_attainable"], "drive_p_cluster_t": ds["p_cluster_t"],
            "drive_clt_ci95_lo": ds["clt_ci95"][0], "drive_clt_ci95_hi": ds["clt_ci95"][1],
            "drive_holm_p": hol.get("drive") if in_family else None,
            # other estimands, drive clusters
            "drive_estimate_unit": dr["unit"]["theta"], "drive_unit_ci95_lo": dr["unit"]["boot_ci95"][0],
            "drive_unit_ci95_hi": dr["unit"]["boot_ci95"][1], "drive_unit_p_signflip": dr["unit"]["p_signflip"],
            "drive_estimate_drive_equal": dr["cluster_equal"]["theta"],
            "drive_equal_ci95_lo": dr["cluster_equal"]["boot_ci95"][0], "drive_equal_ci95_hi": dr["cluster_equal"]["boot_ci95"][1],
            "drive_equal_p_signflip": dr["cluster_equal"]["p_signflip"],
            "log_estimate_log_equal": lg["cluster_equal"]["theta"],
            # leave one drive out
            "lodo_min": lo["theta"].min(), "lodo_max": lo["theta"].max(), "lodo_max_p": lo["p_signflip"].max(),
            "lodo_n_ci_includes_0": int(lo["ci_includes_0"].sum()), "lodo_n_sign_change": int(lo["sign_change"].sum()),
            "lodo_most_influential": lo.loc[lo["influence"].abs().idxmax(), "left_out"],
            # judgments
            "judgment_original": j_orig, "judgment_log": jl, "judgment_drive": jd,
            "judgment_change_log_to_drive": "no change" if jl == jd else f"{jl} -> {jd}",
            "small_cluster_caveat": "; ".join(caveat),
            "intervention_note": r["notes"],
        }
        if r["margin"] is not None:
            t = ds["tost"]
            row.update({"equiv_margin": r["margin"], "drive_tost_p_lower": t["p_lower"], "drive_tost_p_upper": t["p_upper"],
                        "drive_tost_p_equiv": t["p_equiv"], "drive_tost_method": t["p_method"],
                        "drive_tost_boot_p_equiv": SC.tost(*SC.cluster_arrays(r["res"]["drive"]["_sc"], "scene")[1:],
                                                           r["margin"], "bootstrap", B, SEED)["p_equiv"],
                        "drive_equiv_by_boot_ci90": t["equiv_boot_ci90"], "drive_equiv_by_sf_ci90": t["equiv_sf_ci90"]})
            for wt in ("unit", "cluster_equal"):
                tt = r["res"]["drive"][wt]["tost"]
                row[f"drive_{wt}_tost_p_equiv"] = tt["p_equiv"]
                row[f"drive_{wt}_ci90"] = f"[{tt['boot_ci90'][0]:.4f}, {tt['boot_ci90'][1]:.4f}]"
        out_rows.append(row)
        # long format: every level x weighting
        for lev, levres in r["res"].items():
            for wt in SC.WEIGHTINGS:
                x = levres[wt]
                LONG.append({"contrast_id": r["contrast_id"], "cluster_level": lev, "weighting": wt,
                             "estimand_name": {"unit": "unit-weighted", "scene": "scene-weighted (orig. primary)",
                                               "cluster_equal": f"{lev}-equal"}[wt],
                             "n_clusters": x["n_clusters"], "n_informative": x["n_informative"], "n_scenes": x["n_scenes"],
                             "n_units": x["n_units"], "estimate": x["theta"], "boot_ci95_lo": x["boot_ci95"][0],
                             "boot_ci95_hi": x["boot_ci95"][1], "boot_ci90_lo": x["boot_ci90"][0],
                             "boot_ci90_hi": x["boot_ci90"][1], "sf_ci95_lo": x["sf_ci95"][0], "sf_ci95_hi": x["sf_ci95"][1],
                             "sf_ci90_lo": x["sf_ci90"][0], "sf_ci90_hi": x["sf_ci90"][1], "p_signflip": x["p_signflip"],
                             "sf_method": x["sf_method"], "p_min_attainable": x["p_min_attainable"],
                             "p_cluster_t": x["p_cluster_t"], "boot_B": x["B"], "boot_seed": x["seed"],
                             "tost_p_equiv": x.get("tost", {}).get("p_equiv")})

    main_df = pd.DataFrame(out_rows)
    main_df.to_csv(os.path.join(OUT, "CLUSTER_REANALYSIS.csv"), index=False)
    pd.DataFrame(LONG).to_csv(os.path.join(OUT, "cluster_reanalysis_long.csv"), index=False)
    pd.concat(LOO, ignore_index=True).to_csv(os.path.join(OUT, "leave_one_drive_out.csv"), index=False)
    pd.DataFrame(P1C_SUB).to_csv(os.path.join(OUT, "p1c_subsets_drive_level.csv"), index=False)
    # per-drive effects (scene-weighted) for the figure / forest
    pdrv = []
    for r in ROWS:
        sc = r["res"]["drive"]["_sc"]
        for g, s in sc.groupby("cluster"):
            pdrv.append({"contrast_id": r["contrast_id"], "drive_id": g, "n_scenes": len(s), "n_logs": s["log_id"].nunique(),
                         "n_units": int(s["n_units"].sum()), "effect_scene_weighted": float((s.w * s.dmean).sum() / s.w.sum())})
    pd.DataFrame(pdrv).to_csv(os.path.join(OUT, "per_drive_effects.csv"), index=False)
    # verification against previously reported numbers
    ver = []
    for r in ROWS:
        ls, ds = r["res"]["log"]["scene"], r["res"]["drive"]["scene"]
        if r["rep_log"] is not None:
            ver.append({"contrast_id": r["contrast_id"], "check": "log-level scene estimate & exact/MC p vs P0-B/R8R9",
                        "reported_est": r["rep_log"]["est"], "recomputed_est": ls["theta"],
                        "reported_p": r["rep_log"]["p"], "recomputed_p": ls["p_signflip"],
                        "reported_ci": f"[{r['rep_log']['lo']:.5f}, {r['rep_log']['hi']:.5f}]",
                        "recomputed_ci": f"[{ls['boot_ci95'][0]:.5f}, {ls['boot_ci95'][1]:.5f}]",
                        "est_match": bool(np.isclose(r["rep_log"]["est"], ls["theta"], atol=1e-12)),
                        "p_match": bool(np.isclose(r["rep_log"]["p"], ls["p_signflip"], rtol=1e-6) or
                                        (("Monte Carlo" in ls["sf_method"] or ls["n_clusters"] > 20) and abs(r["rep_log"]["p"] - ls["p_signflip"]) < 5e-3))})
        if r["contrast_id"].startswith("P1C"):
            o = r["orig"]
            ver.append({"contrast_id": r["contrast_id"], "check": "drive-level theta, exact p, Holm, bootstrap CI, sf 90% CI, decision vs decisions.json",
                        "reported_est": o["est"], "recomputed_est": ds["theta"], "reported_p": o["p"], "recomputed_p": ds["p_signflip"],
                        "reported_ci": f"[{o['lo']:.5f}, {o['hi']:.5f}] holm={o['holm']:.3g} sf90=[{o['sf90'][0]:.4f}, {o['sf90'][1]:.4f}] {o['decision']}",
                        "recomputed_ci": f"[{ds['boot_ci95'][0]:.5f}, {ds['boot_ci95'][1]:.5f}] holm={r['holm']['drive']:.3g} sf90=[{ds['sf_ci90'][0]:.4f}, {ds['sf_ci90'][1]:.4f}] "
                                         f"{main_df.loc[main_df.contrast_id == r['contrast_id'], 'judgment_drive'].iloc[0]}",
                        "est_match": bool(np.isclose(o["est"], ds["theta"], atol=1e-12)),
                        "p_match": bool(np.isclose(o["p"], ds["p_signflip"], rtol=1e-9))})
    pd.DataFrame(ver).to_csv(os.path.join(OUT, "verification_vs_reported.csv"), index=False)
    meta = {"created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "status": "POST-HOC analysis of existing data",
            "B": B, "B_LOO": B_LOO, "MC_reps": MC, "seed_drive": SEED, "seed_log_P0B_scheme": "20261008 + 100 i / 700 + 10 i",
            "seed_drive_P1C": P1C_SEED, "exact_max_clusters": SC.EXACT_MAX,
            "pair_reports": {r["contrast_id"]: r["pair_report"] for r in ROWS},
            "inputs": ["P0B_cluster_stats/run_20261008/units.parquet", "R8R9_raw_recovery/run_20261008/units.parquet",
                       "P1C_new_log_replication/run_20261008/analysis/units.parquet",
                       "P0B .../comparison_with_original.csv, paired_effects.csv", "R8R9 .../paired_effects.csv",
                       "P1C .../decisions.json", "P1C design split.csv"]}
    json.dump(meta, open(os.path.join(OUT, "a3_run_meta.json"), "w"), indent=1, default=str)
    log("done")


if __name__ == "__main__":
    main()
