#!/usr/bin/env python
"""
P0-D (review_followup3, 2026-10-14): higher-level dependence (day / vehicle / day-vehicle) sensitivity.

*** POST-HOC RE-ANALYSIS OF EXISTING DATA. *** CPU only, no model inference. Reads existing units tables; writes only
to outputs/review_followup3/P0/. Nothing here is preregistered. The drive-level rows reproduce the original analyses
(AutoVLA primary family: review_followup2 A3; P1-C: preregistered decisions.json) before any sensitivity is added.

Hierarchy (open-loop AutoVLA): day x vehicle (crossed) -> day-vehicle -> drive (<date.time>_veh-NN) -> log -> scene ->
perturbation unit -> condition. drive is nested in day-vehicle, day-vehicle in day and in vehicle; day and vehicle are
CROSSED (a vehicle can drive on several days, a day can have several vehicles).

Estimand: unchanged (scene-weighted; P1-C P = Hajek with design weights w = 1/pi; P1-C F weight 1). Changing the
cluster axis changes only inference (resampling / sign-flip unit), not theta. The 'cluster_equal' weighting (each group
of the axis equally weighted) is a DIFFERENT estimand and reported in its own columns.

Axes:
  drive         original
  day_vehicle   <date>_veh-NN
  vehicle       veh-NN (ignores day correlation)
  day           <date> (ignores vehicle correlation)
  dv_component  connected components of the bipartite day-vehicle graph within the analysed set = the finest
                partition in which BOTH day and vehicle are nested (a valid one-way clustering for two-way dependence)
  twoway_CGM    Cameron-Gelbach-Miller two-way cluster-robust variance V_day + V_vehicle - V_dayvehicle for the ratio
                estimator (linearised), t with min(G_day, G_vehicle) - 1 df. Auxiliary (asymptotic in BOTH G).

Usage: nice -n 19 /root/miniforge3/envs/autovla/bin/python p0_d_dependence.py
"""
from __future__ import annotations

import json
import math
import os
import sys
import time

import numpy as np
import pandas as pd
import yaml
from scipy.stats import t as tdist

POC = "/root/VLA/autovla_misalignment_poc"
sys.path.insert(0, os.path.join(POC, "scripts/review_followup2"))
import stats_core as SC  # noqa: E402
import a3_cluster_reanalysis as A3  # noqa: E402  (load() and contrast specs; importing does not run main)

RF = os.path.join(POC, "outputs/review_followup")
RF2 = os.path.join(POC, "outputs/review_followup2/A_autovla_cluster/run_20261013")
OUT = os.path.join(POC, "outputs/review_followup3/P0")
os.makedirs(OUT, exist_ok=True)
NAVTRAIN = "/root/VLA/autovla/navsim/navsim/planning/script/config/common/train_test_split/scene_filter/navtrain.yaml"
NAVTEST = "/root/VLA/autovla/navsim/navsim/planning/script/config/common/train_test_split/scene_filter/navtest.yaml"
# full AutoVLA training split (OpenScene v1.1 trainval, all scenes; see p0_d_disjoint_inventory.py docstring)
TRAINVAL = os.path.join(POC, "outputs/review_followup3/P0/_external/openscene_v1.1_trainval_log_metadata.csv")

B, B_LOO, MC = 10_000, 2_000, 200_000
SEED_AV_DRIVE, SEED_P1C_DRIVE = 20261013, 20261008   # original seeds (A3 / P1-C frozen script)
SEED = 20261014
AXES = ["drive", "day_vehicle", "vehicle", "day", "dv_component"]
T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:7.1f}s]", *a, flush=True)


def parts(drive):
    d, v = drive.split("_")
    return d[:10], v, f"{d[:10]}_{v}"


def components(df):
    """connected components of the bipartite day-vehicle graph of the rows in df -> dict day_vehicle -> comp id"""
    parent = {}

    def f(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for d, v in df[["day", "vehicle"]].drop_duplicates().itertuples(index=False):
        a, b = f("D:" + d), f("V:" + v)
        if a != b:
            parent[a] = b
    m = {}
    for d, v in df[["day", "vehicle"]].drop_duplicates().itertuples(index=False):
        m[f"{d}_{v}"] = f("D:" + d)
    mem = {}
    for k, r in m.items():
        mem.setdefault(r, []).append(k)
    labels = {r: "{" + "+".join(sorted(v)) + "}" for r, v in mem.items()}
    return {k: labels[r] for k, r in m.items()}


def add_axes(pr):
    p = pr.copy()
    dd = p["drive_id"].map(parts)
    p["day"] = dd.map(lambda x: x[0]); p["vehicle"] = dd.map(lambda x: x[1]); p["day_vehicle"] = dd.map(lambda x: x[2])
    p["drive"] = p["drive_id"]
    cm = components(p)
    p["dv_component"] = p["day_vehicle"].map(cm)
    return p


def structure(p):
    dv = p[["drive", "day", "vehicle", "day_vehicle"]].drop_duplicates()
    vd = dv.groupby("vehicle")["day"].nunique()
    dvv = dv.groupby("day")["vehicle"].nunique()
    return {"n_drives": dv.drive.nunique(), "n_day_vehicle": dv.day_vehicle.nunique(), "n_vehicle": dv.vehicle.nunique(),
            "n_day": dv.day.nunique(), "n_dv_component": p["dv_component"].nunique(),
            "vehicles_on_multiple_days": int((vd > 1).sum()), "days_with_multiple_vehicles": int((dvv > 1).sum()),
            "max_days_per_vehicle": int(vd.max()), "max_vehicles_per_day": int(dvv.max()),
            "crossed": bool((vd > 1).any() and (dvv > 1).any()),
            "multi_vehicle_days": ";".join(f"{d}:{'+'.join(sorted(dv[dv.day == d].vehicle.unique()))}"
                                           for d in dvv[dvv > 1].index),
            "multi_day_vehicles": ";".join(f"{v}:{'+'.join(sorted(dv[dv.vehicle == v].day.unique()))}"
                                           for v in vd[vd > 1].index)}


def scene_tab(p, axis, w):
    return SC.scene_table(p, axis, "scene_id", d="d", w=w, extra=["day", "vehicle", "day_vehicle", "drive"])


def cgm_twoway(p, w):
    """two-way (day x vehicle) cluster-robust SE of the scene-weighted ratio; intersection = day_vehicle"""
    sc = scene_tab(p, "drive", w)
    N, D = (sc.w * sc.dmean).values, sc.w.values
    th = N.sum() / D.sum()
    e = (N - th * D) / D.sum()

    def V(col):
        g = pd.Series(e).groupby(sc[col].values).sum().values
        G = len(g)
        return G / (G - 1) * float((g ** 2).sum()) if G > 1 else float("nan"), G

    vd, Gd = V("day"); vv, Gv = V("vehicle"); vi, Gi = V("day_vehicle")
    v2 = vd + vv - vi
    note = ""
    if not v2 > 0:
        v2 = max(vd, vv); note = "V negative -> max(V_day, V_vehicle)"
    se = math.sqrt(v2)
    df = min(Gd, Gv) - 1
    q = tdist.ppf(0.975, df)
    p_ = float(2 * tdist.sf(abs(th / se), df)) if se > 0 else float("nan")
    return {"theta": th, "se": se, "df": df, "ci95": [th - q * se, th + q * se], "p": p_, "G_day": Gd, "G_vehicle": Gv,
            "G_dayvehicle": Gi, "note": note, "se_day": math.sqrt(vd), "se_vehicle": math.sqrt(vv),
            "se_dayvehicle": math.sqrt(vi)}


def clean(o):
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if hasattr(o, "item"):
        o = o.item()
    if isinstance(o, float) and not np.isfinite(o):
        return None
    return o


# ============================================================================================ data sets
def contrasts():
    u, r8, c = A3.load()
    out = []
    for cid, fam, model, src, exp, grp, a, b, met, sign, margin, _ in A3.AV[:3]:
        base = r8 if src == "R8" else u[u["experiment"] == exp]
        base = base[base["group"] == grp]
        pr, rep = SC.pair_rows(base, ["unit_id"], "condition", met, a, b, keep_cols=["scene_id", "log_id", "drive_id"])
        out.append(dict(cid=cid, family="AV_primary_m3", alpha=0.05, sign=sign, p=add_axes(pr), w=None,
                        seed_drive=SEED_AV_DRIVE, pair=rep, a=a, b=b))
    for cid, blk, a, b, sign, _ in A3.P1C:
        cb = c[c["block"] == blk]
        pr, rep = SC.pair_rows(cb, ["unit_id"], "condition", "amplification", a, b,
                               keep_cols=["scene_id", "log_id", "drive_id", "stratum", "weight"])
        for est in ("F", "P"):
            sub = pr[pr["stratum"] == "F"].copy() if est == "F" else pr.copy()
            if est == "F":
                sub["weight"] = 1.0
            out.append(dict(cid=f"P1C {cid} {est}", family=f"P1C_{est}_m3", alpha=0.05 if est == "F" else 0.025,
                            sign=sign, p=add_axes(sub), w="weight", seed_drive=SEED_P1C_DRIVE, pair=rep, a=a, b=b))
    return out, u, r8, c


# ============================================================================================ main
def main():
    CT, u, r8, c = contrasts()
    log("contrasts", len(CT))
    ROWS, LOGO, STRUCT = [], [], []
    for k, ct in enumerate(CT):
        p = ct["p"]
        st = structure(p); st["contrast_id"] = ct["cid"]; STRUCT.append(st)
        for ai, ax in enumerate(AXES):
            sc = scene_tab(p, ax, ct["w"])
            seed = ct["seed_drive"] if ax == "drive" else SEED + 10 * ai
            res = {}
            for wt in ("scene", "cluster_equal"):
                r = SC.analyze(sc, wt, B=B, seed=seed, reps=MC); r.pop("_arrays")
                res[wt] = r
            r = res["scene"]
            lo = SC.leave_one_cluster_out(sc, "scene", B=B_LOO, seed=seed, reps=20_000)
            lo.insert(0, "axis", ax); lo.insert(0, "contrast_id", ct["cid"]); LOGO.append(lo)
            row = dict(contrast_id=ct["cid"], family=ct["family"], alpha=ct["alpha"], axis=ax, cond_a=ct["a"],
                       cond_b=ct["b"], estimand="scene-weighted" + (" Hajek (w=1/pi)" if ct["cid"].endswith(" P") else ""),
                       n_units=r["n_units"], n_scenes=r["n_scenes"], n_drives=p.drive.nunique(), G=r["n_clusters"],
                       G_informative=r["n_informative"], p_min_attainable=r["p_min_attainable"], theta=r["theta"],
                       boot_ci95_lo=r["boot_ci95"][0], boot_ci95_hi=r["boot_ci95"][1],
                       sf_ci95_lo=r["sf_ci95"][0], sf_ci95_hi=r["sf_ci95"][1], sf_ci90_lo=r["sf_ci90"][0],
                       sf_ci90_hi=r["sf_ci90"][1], p_signflip=r["p_signflip"], sf_method=r["sf_method"],
                       p_at_floor=bool(np.isclose(r["p_signflip"], r["p_min_attainable"])),
                       p_cluster_t=r["p_cluster_t"], clt_ci95_lo=r["clt_ci95"][0], clt_ci95_hi=r["clt_ci95"][1],
                       boot_seed=seed, B=B,
                       theta_group_equal=res["cluster_equal"]["theta"],
                       group_equal_boot_ci95_lo=res["cluster_equal"]["boot_ci95"][0],
                       group_equal_boot_ci95_hi=res["cluster_equal"]["boot_ci95"][1],
                       group_equal_p_signflip=res["cluster_equal"]["p_signflip"],
                       logo_theta_min=lo.theta.min() if len(lo) else np.nan,
                       logo_theta_max=lo.theta.max() if len(lo) else np.nan,
                       logo_max_p=lo.p_signflip.max() if len(lo) else np.nan,
                       logo_n_sign_change=int(lo.sign_change.sum()) if len(lo) else 0,
                       logo_n_ci_includes_0=int(lo.ci_includes_0.sum()) if len(lo) else 0,
                       logo_n=len(lo),
                       logo_most_influential=(lo.loc[lo.influence.abs().idxmax(), "left_out"] if len(lo) else ""),
                       logo_max_p_left_out=(lo.loc[lo.p_signflip.idxmax(), "left_out"] if len(lo) else ""))
            # group weight concentration (scene-weight share of largest group)
            sh = sc.groupby("cluster")["w"].sum(); sh = sh / sh.sum()
            row["max_group_weight_share"] = float(sh.max()); row["max_group"] = sh.idxmax()
            row["inference_possible_alpha05"] = bool(r["p_min_attainable"] < 0.05)
            ROWS.append(row)
        tw = cgm_twoway(p, ct["w"])
        ROWS.append(dict(contrast_id=ct["cid"], family=ct["family"], alpha=ct["alpha"], axis="twoway_CGM_day_x_vehicle",
                         cond_a=ct["a"], cond_b=ct["b"], estimand="scene-weighted" + (" Hajek" if ct["cid"].endswith(" P") else ""),
                         n_units=len(p), n_scenes=p.scene_id.nunique(), n_drives=p.drive.nunique(),
                         G=f"day {tw['G_day']} x vehicle {tw['G_vehicle']} (int. {tw['G_dayvehicle']})",
                         theta=tw["theta"], clt_ci95_lo=tw["ci95"][0], clt_ci95_hi=tw["ci95"][1], p_cluster_t=tw["p"],
                         sf_method=f"CGM two-way, t df={tw['df']}; se={tw['se']:.4f} (day {tw['se_day']:.4f}, vehicle "
                                   f"{tw['se_vehicle']:.4f}, dv {tw['se_dayvehicle']:.4f}) {tw['note']}",
                         inference_possible_alpha05=None))
        log(ct["cid"], "done")

    df = pd.DataFrame(ROWS)
    # Holm within original family, per axis (scene-weighted, sign-flip p)
    df["holm_p"] = np.nan
    for (fam, ax), g in df[df.axis.isin(AXES)].groupby(["family", "axis"]):
        df.loc[g.index, "holm_p"] = SC.holm(g.p_signflip.values)
    df["significant_holm"] = df["holm_p"] < df["alpha"]
    # change vs drive
    for cid, g in df.groupby("contrast_id"):
        base = g[g.axis == "drive"].iloc[0]
        for i in g.index:
            df.loc[i, "ci_width_ratio_vs_drive_boot"] = ((df.loc[i, "boot_ci95_hi"] - df.loc[i, "boot_ci95_lo"]) /
                                                         (base.boot_ci95_hi - base.boot_ci95_lo)) if pd.notna(df.loc[i, "boot_ci95_hi"]) else np.nan
            df.loc[i, "ci_width_ratio_vs_drive_clt"] = ((df.loc[i, "clt_ci95_hi"] - df.loc[i, "clt_ci95_lo"]) /
                                                        (base.clt_ci95_hi - base.clt_ci95_lo))
    df.insert(0, "row_id", [f"P0D-{i:03d}" for i in range(len(df))])
    df["analysis_status"] = np.where(df.axis == "drive", "reproduction of original drive-level analysis",
                                     "POST-HOC sensitivity (not preregistered)")

    # ------------------------------------------------------------------ reproduction check against originals
    ref = pd.read_csv(os.path.join(RF2, "CLUSTER_REANALYSIS.csv")).set_index("contrast_id")
    dec = json.load(open(os.path.join(RF, "P1C_new_log_replication/run_20261008/analysis/decisions.json")))["decisions"]
    VER = []
    for _, r in df[df.axis == "drive"].iterrows():
        cid = r.contrast_id
        q = ref.loc[cid]
        VER.append(dict(contrast_id=cid, source="review_followup2 CLUSTER_REANALYSIS.csv", orig_theta=q.drive_estimate_scene,
                        new_theta=r.theta, orig_lo=q.drive_boot_ci95_lo, new_lo=r.boot_ci95_lo, orig_hi=q.drive_boot_ci95_hi,
                        new_hi=r.boot_ci95_hi, orig_p=q.drive_p_signflip, new_p=r.p_signflip, orig_holm=q.drive_holm_p,
                        new_holm=r.holm_p,
                        match=bool(np.isclose(q.drive_estimate_scene, r.theta, atol=1e-12) and
                                   np.isclose(q.drive_boot_ci95_lo, r.boot_ci95_lo, atol=1e-9) and
                                   np.isclose(q.drive_boot_ci95_hi, r.boot_ci95_hi, atol=1e-9) and
                                   np.isclose(q.drive_p_signflip, r.p_signflip, rtol=1e-9) and
                                   np.isclose(q.drive_holm_p, r.holm_p, rtol=1e-9))))
        if cid.startswith("P1C"):
            _, cc, est = cid.split()
            d0 = next(x for x in dec if x["contrast"] == cc and x["estimand"] == est)
            VER.append(dict(contrast_id=cid, source="P1-C decisions.json (preregistered)", orig_theta=d0["theta"],
                            new_theta=r.theta, orig_lo=d0["boot_ci95_lo"], new_lo=r.boot_ci95_lo, orig_hi=d0["boot_ci95_hi"],
                            new_hi=r.boot_ci95_hi, orig_p=d0["p_signflip"], new_p=r.p_signflip, orig_holm=d0["holm_p"],
                            new_holm=r.holm_p,
                            match=bool(np.isclose(d0["theta"], r.theta, atol=1e-12) and
                                       np.isclose(d0["boot_ci95_lo"], r.boot_ci95_lo, atol=1e-9) and
                                       np.isclose(d0["p_signflip"], r.p_signflip, rtol=1e-9) and
                                       np.isclose(d0["holm_p"], r.holm_p, rtol=1e-9))))
    ver = pd.DataFrame(VER)
    ver.to_csv(os.path.join(OUT, "DEPENDENCE_reproduction_check.csv"), index=False)
    assert ver.match.all(), ver[~ver.match]
    log("reproduction OK", len(ver))

    # ------------------------------------------------------------------ arm-sign swap test (a<->b gives -theta, same p)
    ct0 = CT[0]
    pr_sw = ct0["p"].copy(); pr_sw["d"] = -pr_sw["d"]
    r_sw = SC.analyze(scene_tab(pr_sw, "day_vehicle", None), "scene", B=200, seed=1, reps=MC)
    r_or = SC.analyze(scene_tab(ct0["p"], "day_vehicle", None), "scene", B=200, seed=1, reps=MC)
    swap_ok = bool(np.isclose(r_sw["theta"], -r_or["theta"]) and np.isclose(r_sw["p_signflip"], r_or["p_signflip"]))
    assert swap_ok

    # ------------------------------------------------------------------ drive map, set memberships, navtrain overlap
    mem = pd.read_csv(os.path.join(RF2, "drive_membership.csv"))
    nt = yaml.safe_load(open(NAVTRAIN))["log_names"]
    ntest = yaml.safe_load(open(NAVTEST))["log_names"]
    ntr = pd.DataFrame({"log": nt}); ntr["drive"] = ntr.log.map(A3.drive_of)
    ntr[["day", "vehicle", "day_vehicle"]] = pd.DataFrame(ntr.drive.map(parts).tolist(), index=ntr.index)
    sets_tr = {k: set(ntr[k]) for k in ("log", "drive", "day", "vehicle", "day_vehicle")}   # navtrain (subset proxy)
    tvl = pd.read_csv(TRAINVAL).log_name.tolist()
    ftv = pd.DataFrame({"log": tvl}); ftv["drive"] = ftv.log.map(A3.drive_of)
    ftv[["day", "vehicle", "day_vehicle"]] = pd.DataFrame(ftv.drive.map(parts).tolist(), index=ftv.index)
    sets_tv = {k: set(ftv[k]) for k in ("log", "drive", "day", "vehicle", "day_vehicle")}
    dm = mem.rename(columns={"drive_id": "drive"}).copy()
    dm[["day", "vehicle", "day_vehicle"]] = pd.DataFrame(dm.drive.map(parts).tolist(), index=dm.index)
    flags = [c_ for c_ in mem.columns if c_ != "drive_id"]
    for c_ in flags:
        dm[c_] = dm[c_] > 0
    p1c_drv = set(c.drive_id); p1c_F = set(c[c.stratum == "F"].drive_id)
    dm["p1c_eval_units_F"] = dm.drive.isin(p1c_F) | dm["p1c_eval_units_F"]
    ntest_drv = {A3.drive_of(l) for l in ntest}
    dm["in_navtest_loglist"] = dm.drive.isin(ntest_drv)
    for k in ("drive", "day_vehicle", "day", "vehicle"):
        dm[f"navtrain_shares_{k}"] = dm[k].isin(sets_tr[k])
    dm["navtrain_n_logs_same_day_vehicle"] = dm.day_vehicle.map(ntr.groupby("day_vehicle").size()).fillna(0).astype(int)
    dm["navtrain_n_logs_same_day"] = dm.day.map(ntr.groupby("day").size()).fillna(0).astype(int)
    dm["navtrain_n_logs_same_vehicle"] = dm.vehicle.map(ntr.groupby("vehicle").size()).fillna(0).astype(int)
    for k in ("drive", "day_vehicle", "day", "vehicle"):
        dm[f"trainval_shares_{k}"] = dm[k].isin(sets_tv[k])
    dm["trainval_n_logs_same_day"] = dm.day.map(ftv.groupby("day").size()).fillna(0).astype(int)
    dm["trainval_n_logs_same_vehicle"] = dm.vehicle.map(ftv.groupby("vehicle").size()).fillna(0).astype(int)
    # logs of each drive (from the units tables + split.csv) in navtrain?
    split = pd.read_csv(os.path.join(RF, "P1C_new_log_replication/design_20261008/split.csv"), dtype=str)
    all_logs = set(split.log) | set(u.log_id.dropna()) | set(r8.log_id) | set(c.log_id)
    dm["any_known_log_in_navtrain"] = dm.drive.map(lambda d_: any(l.startswith(d_) and l in sets_tr["log"] for l in all_logs))
    # dev-set membership summaries
    dev_ref = {"A-": set(dm[dm.mech_poc_Aminus].drive), "mech_poc": set(dm[dm.mech_poc_units].drive),
               "dev_discovery": set(dm[dm.poc_dev_discovery].drive)}
    dm.to_csv(os.path.join(OUT, "DEPENDENCE_drive_map.csv"), index=False)

    # ------------------------------------------------------------------ overlap facts to verify
    def dvset(drv):
        return {parts(x)[2] for x in drv}

    def dset(drv, i):
        return {parts(x)[i] for x in drv}

    facts = {}
    for nm, ref_ in dev_ref.items():
        facts[nm] = {"n_drives": len(ref_), "n_day_vehicle": len(dvset(ref_)), "n_day": len(dset(ref_, 0)),
                     "n_vehicle": len(dset(ref_, 1)),
                     "p1c_eval_shared_drives": len(ref_ & p1c_drv),
                     "p1c_eval_shared_day_vehicle": len(dvset(ref_) & dvset(p1c_drv)),
                     "p1c_eval_shared_days": len(dset(ref_, 0) & dset(p1c_drv, 0)),
                     "p1c_eval_shared_vehicles": len(dset(ref_, 1) & dset(p1c_drv, 1)),
                     "shared_day_vehicle_list": sorted(dvset(ref_) & dvset(p1c_drv))}
    for nm, drv in (("P1C_eval_24", p1c_drv), ("P1C_F_18", p1c_F), ("A-_13", dev_ref["A-"])):
        facts[f"navtrain_{nm}"] = {k: int(sum(parts(x)[i] in sets_tr[k] for x in drv)) for i, k in
                                   ((0, "day"), (1, "vehicle"), (2, "day_vehicle"))}
        facts[f"navtrain_{nm}"]["drive"] = len(drv & sets_tr["drive"])
        facts[f"navtrain_{nm}"]["n"] = len(drv)
        facts[f"trainval_{nm}"] = {k: int(sum(parts(x)[i] in sets_tv[k] for x in drv)) for i, k in
                                   ((0, "day"), (1, "vehicle"), (2, "day_vehicle"))}
        facts[f"trainval_{nm}"]["drive"] = len(drv & sets_tv["drive"])
        facts[f"trainval_{nm}"]["n"] = len(drv)
    facts["trainval_n_logs"] = len(tvl)
    facts["navtrain_navtest_shared_logs"] = len(set(nt) & set(ntest))
    facts["navtrain_n_logs"] = len(nt)

    # ------------------------------------------------------------------ day- / vehicle-disjoint P1-C subsets
    SUB = []
    for refnm, ref_ in dev_ref.items():
        rd, rv, rdv = dset(ref_, 0), dset(ref_, 1), dvset(ref_)
        for sname, keep in (("dayveh_disjoint", lambda x: parts(x)[2] not in rdv),
                            ("day_disjoint", lambda x: parts(x)[0] not in rd),
                            ("vehicle_disjoint", lambda x: parts(x)[1] not in rv),
                            ("day_and_vehicle_disjoint", lambda x: parts(x)[0] not in rd and parts(x)[1] not in rv)):
            for ct in CT:
                if not ct["cid"].startswith("P1C"):
                    continue
                p = ct["p"]; ss = p[p.drive.map(keep)]
                row = dict(reference_set=refnm, subset=sname, contrast_id=ct["cid"], n_drives=ss.drive.nunique(),
                           n_day_vehicle=ss.day_vehicle.nunique(), n_days=ss.day.nunique(), n_vehicles=ss.vehicle.nunique(),
                           n_scenes=ss.scene_id.nunique(), n_units=len(ss),
                           drives=";".join(sorted(ss.drive.unique())))
                if ss.drive.nunique() >= 2:
                    for ax in ("drive", "day_vehicle", "dv_component"):
                        ss2 = ss.copy()
                        if ax == "dv_component":
                            ss2["dv_component"] = ss2.day_vehicle.map(components(ss2))
                        sc = scene_tab(ss2, ax, ct["w"])
                        r = SC.analyze(sc, "scene", B=B, seed=SEED + 500, reps=MC)
                        row.update({f"{ax}_G": r["n_clusters"], f"{ax}_p_min": r["p_min_attainable"],
                                    f"{ax}_p": r["p_signflip"]})
                        if ax == "drive":
                            row.update(theta=r["theta"], boot_ci95_lo=r["boot_ci95"][0], boot_ci95_hi=r["boot_ci95"][1],
                                       sf_ci95_lo=r["sf_ci95"][0], sf_ci95_hi=r["sf_ci95"][1])
                SUB.append(row)
    sub = pd.DataFrame(SUB)
    sub.to_csv(os.path.join(OUT, "DEPENDENCE_disjoint_subsets.csv"), index=False)

    df["status"] = "PROVISIONAL (AutoVLA no-op parity, P1 Stage 0, not judged); " + df["analysis_status"]
    df.to_csv(os.path.join(OUT, "DEPENDENCE_SENSITIVITY.csv"), index=False)
    pd.concat(LOGO, ignore_index=True).to_csv(os.path.join(OUT, "DEPENDENCE_leave_one_group_out.csv"), index=False)
    st = pd.DataFrame(STRUCT)
    st.to_csv(os.path.join(OUT, "DEPENDENCE_structure.csv"), index=False)

    # ------------------------------------------------------------------ manifest fragment (one row per drive)
    MAN = []
    for _, r in dm.iterrows():
        sets_ = [f for f in flags + ["in_navtest_loglist"] if r[f]]
        MAN.append(dict(dataset="drive_map", experiment="all AutoVLA/Impromptu open-loop sets", model="AutoVLA;Impromptu-VLA-3B",
                        task="navtest open-loop planning", unit_id=r.drive, log="", drive=r.drive, day=r.day,
                        vehicle=r.vehicle, day_vehicle=r.day_vehicle, scene="", perturbation="", initial_physical_state="",
                        instruction="", rng_seed="", condition_set="", n_rows="", included=True, missing_reason="",
                        denominator_note=("member of: " + ";".join(sets_) +
                                          f" | trainval(AutoVLA training) shares day_vehicle={r.trainval_shares_day_vehicle}, "
                                          f"day={r.trainval_shares_day}, vehicle={r.trainval_shares_vehicle}")))
    pd.DataFrame(MAN).to_csv(os.path.join(OUT, "_frag_manifest_D.csv"), index=False)

    # ------------------------------------------------------------------ numbers fragment
    src = "outputs/review_followup3/P0/DEPENDENCE_SENSITIVITY.csv"
    nums = {"_meta": {"created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                      "status": "POST-HOC sensitivity; drive rows reproduce original",
                      "script": "scripts/review_followup3/p0_d_dependence.py", "B": B, "B_logo": B_LOO, "MC": MC,
                      "seeds": {"drive_AV": SEED_AV_DRIVE, "drive_P1C": SEED_P1C_DRIVE, "other_axes": f"{SEED}+10*axis_index"},
                      "swap_test_ok": swap_ok, "reproduction_rows_matched": int(ver.match.sum()),
                      "inputs": ["P0B_cluster_stats/run_20261008/units.parquet", "R8R9_raw_recovery/run_20261008/units.parquet",
                                 "P1C_new_log_replication/run_20261008/analysis/units.parquet",
                                 "review_followup2/A_autovla_cluster/run_20261013/drive_membership.csv", NAVTRAIN, NAVTEST, TRAINVAL]},
            "overlap_facts": facts, "rows": {}}
    for _, r in df.iterrows():
        nums["rows"][r.row_id] = {"source": src, "contrast_id": r.contrast_id, "axis": r.axis, "G": r.G,
                                  "theta": r.theta, "boot_ci95": [r.boot_ci95_lo, r.boot_ci95_hi],
                                  "sf_ci95": [r.sf_ci95_lo, r.sf_ci95_hi], "p_signflip": r.p_signflip,
                                  "p_min_attainable": r.p_min_attainable, "holm_p": r.holm_p,
                                  "clt_ci95": [r.clt_ci95_lo, r.clt_ci95_hi], "p_cluster_t": r.p_cluster_t}
    json.dump(clean(nums), open(os.path.join(OUT, "_frag_numbers_D.json"), "w"), indent=1)
    json.dump(facts, open(os.path.join(OUT, "DEPENDENCE_overlap_facts.json"), "w"), indent=1, default=str)
    log("done")


if __name__ == "__main__":
    main()
