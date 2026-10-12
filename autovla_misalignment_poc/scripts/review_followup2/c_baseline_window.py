#!/usr/bin/env python
"""
C (review_followup2, 2026-10-13): extrapolation-baseline and correction-window RECHECK.  POST-HOC analysis of existing
data only (CPU, no model inference).  Regenerates every table / figure of
outputs/review_followup2/C_baseline_window/run_20261013/ from the source files:

  P1-A  outputs/review_followup/P1A_gt_free_reference/run_20261008/{raw,raw_fixedpos}/records.jsonl,
        units.parquet, fixedpos/units.parquet
  P1-B  outputs/review_followup/P1B_start_position/run_20261008/units.parquet
  R8R9  outputs/review_followup/R8R9_raw_recovery/run_20261008/units.parquet  (+ equal-distance records for GT tokens)

C1  per unit, perturbed - unperturbed counterfactual change for the MODEL (OC vs U, same batch / seed) and for each
    no-inference baseline built from the SAME reference tokens r:
        Geo_p = r[:t] + [p] + r[t+1:]     Geo_0 = r                       (pure coordinate accumulation)
        CM_p  = r[:t] + [p]*(10-t)        CM_0  = r[:t] + [r_t]*(10-t)    (constant-motion extrapolation)
    endpoint change  = || X_p(pose 9) - X_0(pose 9) ||  (m, x/y, decoder frame = ego frame at t=0)
    suffix change    = mean_{k>t} || disp[X_p,k] - disp[X_0,k] ||  (m per 0.5 s segment, token-local frame)
    model endpoint decomposition (exact identities of AutoVLA: OC suffix == C1 suffix, O1 suffix == U suffix):
        coordinate accumulation = || OC9 - C1_9 ||  (p vs r_t composed under the same, context-changed suffix)
        local-motion (context)  = || C1_9 - U9 ||   (same executed token at t, suffix changed only via context)
    excess = model change - baseline change (per unit), drive-cluster inference (stats_core).
    CM_0 is decoded with p1_common.rollout_np (numpy copy of ActionTokenizer.rollout; checked against the stored
    decoder output of CM_p for every unit).
C2  P1-B early - late per w (total horizon AND common post-release horizon from the per-pose GT error curves),
    recovery fractions (Normal - window) / (Normal - Full) within the same run with drive-cluster CIs (P1-B, R8R9).

drive_id = '<date.time>_veh-NN' prefix of log_id.  Primary resampling unit = drive (13 drives for A-).
Run:  nice -n 19 /root/miniforge3/envs/autovla/bin/python c_baseline_window.py
"""
from __future__ import annotations

import json
import math
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "review_followup", "p1ab"))
import stats_core as SC  # noqa: E402
import p1_common as C  # noqa: E402  (module import only: numpy/pickle; no torch, no GPU)

POC = "/root/VLA/autovla_misalignment_poc"
RF = f"{POC}/outputs/review_followup"
P1A = f"{RF}/P1A_gt_free_reference/run_20261008"
P1B = f"{RF}/P1B_start_position/run_20261008"
R89 = f"{RF}/R8R9_raw_recovery/run_20261008"
OUT = f"{POC}/outputs/review_followup2/C_baseline_window/run_20261013"
SEED = 20261013
B = 10_000
T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:7.1f}s]", *a, flush=True)


def drive_of(log_id: str) -> str:
    p = log_id.split("_")
    return f"{p[0]}_{p[1]}"


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


# ============================================================================================ generic drive analysis
def drive_row(paired: pd.DataFrame, d: str, margin=None, seed=SEED, tost_method="signflip"):
    """paired: one row per unit with drive_id, scene_id and column d.  Returns a flat dict (scene-weighted primary)."""
    x = paired[["drive_id", "scene_id", "log_id", d]].dropna()
    sc = SC.scene_table(x, "drive_id", "scene_id", d=d)
    r = SC.analyze(sc, "scene", B=B, seed=seed, margin=margin, tost_method=tost_method)
    out = {"n_units": r["n_units"], "n_scenes": r["n_scenes"], "n_logs": int(x.log_id.nunique()), "n_drives": r["n_clusters"],
           "n_informative_drives": r["n_informative"], "est_scene": r["theta"],
           "ci95_lo": r["boot_ci95"][0], "ci95_hi": r["boot_ci95"][1], "ci90_lo": r["boot_ci90"][0], "ci90_hi": r["boot_ci90"][1],
           "sf_ci95_lo": r["sf_ci95"][0], "sf_ci95_hi": r["sf_ci95"][1],
           "p_signflip": r["p_signflip"], "sf_method": r["sf_method"], "p_min_attainable": r["p_min_attainable"],
           "p_cluster_t": r["p_cluster_t"]}
    for wt in ("unit", "cluster_equal"):
        lab, N, D = SC.cluster_arrays(sc, wt)
        out[f"est_{'unit' if wt == 'unit' else 'drive_equal'}"] = SC.ratio(N, D)
    # log-level p for comparison (same estimand, old cluster unit)
    scl = SC.scene_table(x, "log_id", "scene_id", d=d)
    out["p_signflip_loglevel"] = SC.analyze(scl, "scene", B=200, seed=seed, ci_inversion=False)["p_signflip"]
    if margin is not None:
        t = r["tost"]
        out.update({"tost_margin": margin, "tost_p_lower": t["p_lower"], "tost_p_upper": t["p_upper"], "tost_p_equiv": t["p_equiv"],
                    "tost_method": t["p_method"], "equiv_90ci_inside": t["equiv_boot_ci90"]})
    else:
        out.update({"tost_margin": np.nan, "tost_p_equiv": np.nan, "equiv_90ci_inside": None})
    return out


# ============================================================================================ C1
def c1_units():
    cb, disp, dyaw = C.codebook_geometry()
    rows, checks = [], []
    for src, path in (("primary", f"{P1A}/raw/records.jsonl"), ("fixedpos", f"{P1A}/raw_fixedpos/records.jsonl")):
        for line in open(path):
            r = json.loads(line)
            t = r["t_p"]; ref = r["ref_tokens"]; p = r["perturbation_info"]["token_p"]
            if p == ref[t]:
                continue                                   # E:original (no perturbation) excluded, as in P1-A
            reft = np.asarray(r["ref_traj_xyh"], float)
            R = {k: np.asarray(v["traj_xyh"], float) for k, v in r["rows"].items()}
            A = {k: v["action_idx"] for k, v in r["rows"].items()}
            geo = np.asarray(r["baselines"]["Geo"]["traj_xyh"], float); cm = np.asarray(r["baselines"]["CM"]["traj_xyh"], float)
            geo_a = r["baselines"]["Geo"]["action_idx"]; cm_a = r["baselines"]["CM"]["action_idx"]
            cm0_a = list(ref[:t]) + [ref[t]] * (C.N_ACT - t)
            cm0 = C.rollout_np(cb, cm0_a)
            cm_np_err = float(np.abs(C.rollout_np(cb, cm_a) - cm).max())
            ref_np_err = float(np.abs(C.rollout_np(cb, ref) - reft).max())
            # matching checks: prefix tokens, first realized perturbation, pose at t, horizon
            chk = {"unit_id": r["unit_id"], "source": src, "t_p": t,
                   "prefix_equal": A["OC"][:t] == geo_a[:t] == cm_a[:t] == list(ref[:t]),
                   "first_token_p_all": A["OC"][t] == geo_a[t] == cm_a[t] == p,
                   "pose_t_maxdiff_OC_Geo_CM": float(max(np.abs(R["OC"][t] - geo[t]).max(), np.abs(R["OC"][t] - cm[t]).max())),
                   "pose_t_maxdiff_U_ref_CM0": float(max(np.abs(R["U"][t] - reft[t]).max(), np.abs(cm0[t] - reft[t]).max())),
                   "n_poses_equal_10": len(R["OC"]) == len(geo) == len(cm) == len(cm0) == 10,
                   "cm_rollout_np_vs_decoder": cm_np_err, "ref_rollout_np_vs_file": ref_np_err,
                   "OC_suffix_eq_C1": A["OC"][t + 1:] == A["C1"][t + 1:], "U_eq_ref": A["U"] == list(ref)}
            checks.append(chk)
            sfx = range(t + 1, C.N_ACT)
            d0 = float(np.linalg.norm(disp[p] - disp[ref[t]]))
            e = lambda a, b: float(np.linalg.norm(a[9, :2] - b[9, :2]))
            dev = lambda a: float(np.linalg.norm(a[9, :2] - reft[9, :2]))
            m_end = e(R["OC"], R["U"])
            g_end = e(geo, reft)
            c_end = e(cm, cm0)
            m_sfx = float(np.mean([np.linalg.norm(disp[A["OC"][k]] - disp[A["U"][k]]) for k in sfx]))
            c_sfx = float(np.mean([np.linalg.norm(disp[cm_a[k]] - disp[cm0_a[k]]) for k in sfx]))   # == d0
            coord = e(R["OC"], R["C1"]); local = e(R["C1"], R["U"])
            rows.append({
                "source": src, "family": r["family"], "group": r["group"], "unit_id": r["unit_id"], "scene_id": r["token"],
                "log_id": r["log"], "drive_id": drive_of(r["log"]), "t_p": t, "t_star": r["t_star"], "d0": d0,
                "model_end": m_end, "geo_end": g_end, "cm_end": c_end,
                "model_sfx": m_sfx, "geo_sfx": 0.0, "cm_sfx": c_sfx,
                "model_coord": coord, "model_local": local,
                "model_dref": dev(R["OC"]) - dev(R["U"]), "geo_dref": dev(geo) - 0.0 * dev(reft), "cm_dref": dev(cm) - dev(cm0),
                "oc_dev9": dev(R["OC"]), "cm_dev9": dev(cm), "geo_dev9": dev(geo), "cm0_dev9": dev(cm0), "u_dev9": dev(R["U"]),
                "model_head9": abs(wrap(R["OC"][9, 2] - R["U"][9, 2])), "cm_head9": abs(wrap(cm[9, 2] - cm0[9, 2])),
                "geo_head9": abs(wrap(geo[9, 2] - reft[9, 2])),
            })
    U = pd.DataFrame(rows)
    U["ex_geo_end"] = U.model_end - U.geo_end
    U["ex_cm_end"] = U.model_end - U.cm_end
    U["ex_geo_sfx"] = U.model_sfx - U.geo_sfx
    U["ex_cm_sfx"] = U.model_sfx - U.cm_sfx
    U["ex_geo_coord"] = U.model_coord - U.geo_end
    U["ex_geo_local"] = U.model_local
    U["ex_geo_dref"] = U.model_dref - U.geo_dref
    U["ex_cm_dref"] = U.model_dref - U.cm_dref
    U["orig_oc_minus_cm_dev9"] = U.oc_dev9 - U.cm_dev9
    U["cm0_dev9_level"] = U.cm0_dev9
    return U, pd.DataFrame(checks)


C1_METRICS = [  # (metric id, column, description, key for Holm)
    ("model_end", "model_end", "model counterfactual endpoint change ||OC9-U9|| (m)", False),
    ("geo_end", "geo_end", "Geo counterfactual endpoint change ||Geo9-r9|| = pure coordinate accumulation (m)", False),
    ("cm_end", "cm_end", "CM counterfactual endpoint change ||CM_p9-CM_0_9|| (m)", False),
    ("ex_geo_end", "ex_geo_end", "EXCESS endpoint change model - Geo (m)", True),
    ("ex_cm_end", "ex_cm_end", "EXCESS endpoint change model - CM (m)", True),
    ("model_sfx", "model_sfx", "model suffix local-motion change mean||disp OC_k - disp U_k|| (m/seg); Geo = 0", False),
    ("cm_sfx", "cm_sfx", "CM suffix local-motion change (= d0 every step) (m/seg)", False),
    ("ex_geo_sfx", "ex_geo_sfx", "EXCESS suffix local-motion change model - Geo (= model, Geo suffix unchanged) (m/seg)", True),
    ("ex_cm_sfx", "ex_cm_sfx", "EXCESS suffix local-motion change model - CM (m/seg)", True),
    ("model_coord", "model_coord", "model endpoint component: coordinate accumulation ||OC9-C1_9|| (m)", False),
    ("model_local", "model_local", "model endpoint component: local-motion/context ||C1_9-U9|| (m)", False),
    ("ex_geo_coord", "ex_geo_coord", "model coordinate-accumulation component - Geo change (m)", False),
    ("ex_geo_dref", "ex_geo_dref", "same-reference variant: [dev9_ref(OC)-dev9_ref(U)] - [dev9_ref(Geo)-0] (m)", False),
    ("ex_cm_dref", "ex_cm_dref", "same-reference variant: [dev9_ref(OC)-dev9_ref(U)] - [dev9_ref(CM_p)-dev9_ref(CM_0)] (m)", False),
    ("orig_oc_minus_cm_dev9", "orig_oc_minus_cm_dev9", "ORIGINAL P1-A secondary OC-CM: dev9_ref(OC)-dev9_ref(CM_p) (CM_0 mismatch NOT removed) (m)", False),
    ("cm0_dev9_level", "cm0_dev9_level", "level: unperturbed CM_0 deviation from r at pose 9 (m) - CM's own extrapolation error", False),
]


def c1_tables(U):
    out = []
    seed = SEED
    for (src, fam), gf in U.groupby(["source", "family"]):
        for grp in ("A-", "A+", "all"):
            g = gf if grp == "all" else gf[gf.group == grp]
            if g.empty:
                continue
            role = "primary (family G)" if fam == "G" else ("secondary (GT-anchored E)" if fam == "E" else "secondary (fixed position, GT-free)")
            rows = []
            for mid, col, desc, key in C1_METRICS:
                r = drive_row(g, col, margin=None, seed=seed)
                seed += 11
                rows.append({"family": fam, "role": role, "group": grp, "metric": mid, "description": desc, "holm_key": key, **r})
            keys = [i for i, x in enumerate(rows) if x["holm_key"]]
            adj = SC.holm([rows[i]["p_signflip"] for i in keys])
            for i, a in zip(keys, adj):
                rows[i]["p_holm_posthoc_m4"] = a
            for x in rows:
                x["equivalence"] = "not tested (no predeclared margin for this post-hoc contrast)"
                j = SC.judge(x["est_scene"], [x["ci95_lo"], x["ci95_hi"]], x.get("p_holm_posthoc_m4", x["p_signflip"]))
                x["judgment_difference"] = ("different_from_0" if j["diff_from_0"] else "not detected") if x["holm_key"] else (
                    "descriptive level / component" if x["metric"] in ("model_end", "geo_end", "cm_end", "model_sfx", "cm_sfx",
                                                                       "model_coord", "model_local", "cm0_dev9_level") else
                    ("different_from_0 (unadjusted)" if j["diff_from_0"] else "not detected (unadjusted)"))
            out += rows
    return pd.DataFrame(out)


def c1_predeclared(seed=SEED + 5000):
    """drive-level re-analysis of the predeclared P1-A primary contrasts (margins predeclared: 0.02 m/seg, 0.25 m) and of
    GT metrics, kept separate from the reference-consistency metric."""
    rows = []
    for src, path in (("primary", f"{P1A}/units.parquet"), ("fixedpos", f"{P1A}/fixedpos/units.parquet")):
        u = pd.read_parquet(path, columns=["unit_id", "log_id", "scene_id", "group", "family", "condition", "exclusion_reason",
                                           "dev9_ref", "suffix_loc", "amplification", "fde5", "intervention_count",
                                           "intervention_count_effective"])
        u = u[u.exclusion_reason.isna()]
        u["drive_id"] = u.log_id.map(drive_of)
        specs = [("P1A-1 C1-U suffix_loc", "C1", "U", "suffix_loc", 0.02, "reference-consistency (predeclared primary)"),
                 ("P1A-2 OC-Geo dev9_ref", "OC", "Geo", "dev9_ref", 0.25, "reference-consistency (predeclared primary)"),
                 ("P1A-3 OC_refR-OC dev9_ref", "OC_refR", "OC", "dev9_ref", 0.25, "reference-consistency recovery (predeclared primary) - NOT GT task improvement"),
                 ("OC-CM dev9_ref (original secondary)", "OC", "CM", "dev9_ref", None, "reference-relative, CM_0 mismatch not removed"),
                 ("OC-O1 dev9_ref (noise-free geometry contrast)", "OC", "O1", "dev9_ref", None, "reference-consistency (secondary)"),
                 ("GT amplification OC_refR-OC", "OC_refR", "OC", "amplification", None, "GT metric (separate)"),
                 ("GT FDE5 OC_refR-OC", "OC_refR", "OC", "fde5", None, "GT metric (separate)"),
                 ("GT amplification OC-U", "OC", "U", "amplification", None, "GT metric (separate)")]
        for fam in sorted(u.family.unique()):
            for grp in ("A-", "A+", "all"):
                g = u[(u.family == fam) & ((u.group == grp) if grp != "all" else True)]
                block = []
                for cid, a, b, met, margin, kind in specs:
                    pr, rep = SC.pair_rows(g, ["unit_id"], "condition", met, a, b, keep_cols=["drive_id", "scene_id", "log_id"])
                    r = drive_row(pr, "d", margin=margin, seed=seed); seed += 11
                    reps = g[g.condition == "OC_refR"]
                    block.append({"source": src, "family": fam, "group": grp, "contrast": cid, "metric": met, "kind": kind,
                                  "missing_pairs": rep["n_unpaired_or_missing"], **r,
                                  "OC_refR_mean_applications_per_unit": float(reps.intervention_count.mean()) if "refR" in cid else np.nan,
                                  "OC_refR_mean_effective_applications": float(reps.intervention_count_effective.mean()) if "refR" in cid else np.nan})
                if fam == "G" and grp == "A-" and src == "primary":
                    adj = SC.holm([x["p_signflip"] for x in block[:3]])
                    adje = SC.holm([x["tost_p_equiv"] for x in block[:3]])
                    for x, a, ae in zip(block[:3], adj, adje):
                        x["p_holm_m3_drive"] = a; x["tost_p_equiv_holm_m3"] = ae
                rows += block
    return pd.DataFrame(rows)


# ============================================================================================ C2
def ratio_boot(dfA: pd.DataFrame, cols_num, cols_den, weighting, seed):
    """fraction f = sum_g N_g / sum_g D_g  with N, D per drive (scene-weighted: sum of scene means; unit: sums)."""
    x = dfA.copy()
    x["num"] = x[cols_num[0]] - x[cols_num[1]]
    x["den"] = x[cols_den[0]] - x[cols_den[1]]
    if weighting == "scene":
        s = x.groupby(["drive_id", "scene_id"])[["num", "den"]].mean().groupby("drive_id").sum()
    else:
        s = x.groupby("drive_id")[["num", "den"]].sum()
    N, D = s.num.values, s.den.values
    f = N.sum() / D.sum() if D.sum() != 0 else np.nan
    Cn = SC.boot_counts(len(N), B, seed)
    with np.errstate(divide="ignore", invalid="ignore"):
        fb = (Cn @ N) / (Cn @ D)
    fb = fb[np.isfinite(fb)]
    return f, SC.pct_ci(fb, 0.95), float(np.mean(fb >= 0.9)), len(N)


def p1b_load():
    u = pd.read_parquet(f"{P1B}/units.parquet")
    u["drive_id"] = u.log_id.map(drive_of)
    u["e_curve"] = u.err_gt_curve.map(lambda s: np.asarray(json.loads(s), float))
    return u


def p1b_wide(u, metric):
    return u.pivot_table(index=["unit_id", "scene_id", "log_id", "drive_id", "group", "t_star"], columns="condition",
                         values=metric, aggfunc="first").reset_index()


def c2_position(u):
    rows = []
    seed = SEED + 20000
    A = u[(u.group == "A-")]
    # post-release horizon facts (from data)
    rel = u[(u.group == "A-") & (u.t_star <= 2) & u.release_pose.notna()].groupby("condition")["release_pose"].mean()
    # ---------------- total horizon (pose 9) early - late, s1 vs s2/3/4
    for kind in ("P", "T"):
        for s_late in (3, 2, 4):
            for met, margin in (("amplification", 0.10), ("fde5", 0.5)):
                block = []
                for w in (1, 2, 3):
                    ce, cl = f"{kind}_gt_s1_w{w}", f"{kind}_gt_s{s_late}_w{w}"
                    sub = A[A.t_star <= 2]
                    pr, rep = SC.pair_rows(sub, ["unit_id"], "condition", met, ce, cl, keep_cols=["drive_id", "scene_id", "log_id"])
                    r = drive_row(pr, "d", margin=margin, seed=seed); seed += 11
                    pre = (kind == "P" and s_late == 3 and met == "amplification")
                    block.append({"analysis": "total horizon (pose 9 = 5 s)", "window_kind": "persistent" if kind == "P" else "transient",
                                  "contrast": f"s1 - s{s_late}", "w": w, "metric": met, "subset": "A-, t*<=2 (common)",
                                  "status": "PREDECLARED primary (re-analysed at drive level)" if pre else "predeclared secondary (drive level)",
                                  "margin_status": "predeclared" if met in ("amplification", "fde5") else "post-hoc",
                                  "eval_pose_early": 9, "eval_pose_late": 9,
                                  "post_release_poses_early": 9 - rel.get(ce, np.nan), "post_release_poses_late": 9 - rel.get(cl, np.nan),
                                  "equal_post_release_horizon": False, "missing_pairs": rep["n_unpaired_or_missing"], **r})
                adj = SC.holm([b["p_signflip"] for b in block]); adje = SC.holm([b["tost_p_equiv"] for b in block])
                for b, a, ae in zip(block, adj, adje):
                    b["p_holm_m3"] = a; b["tost_p_equiv_holm_m3"] = ae
                rows += block
    # ---------------- common post-release horizon N (same units for all w and both starts)
    for kind in ("P", "T"):
        for N in (1, 2, 3):
            tmax = 3 - N                     # rel_late + N <= 9 with rel = t* + s + w, s <= 3, w <= 3
            sub = A[A.t_star <= tmax]
            for met in ("err_diffindiff", "err_raw", "exceed3_diffindiff", "exceed2_diffindiff"):
                block = []
                for w in (1, 2, 3):
                    recs = []
                    for uid, g in sub.groupby("unit_id"):
                        gi = g.set_index("condition")
                        ce, cl = f"{kind}_gt_s1_w{w}", f"{kind}_gt_s3_w{w}"
                        re_, rl = int(gi.loc[ce, "release_pose"]), int(gi.loc[cl, "release_pose"])
                        pe, pl = re_ + N, rl + N
                        en, ee, el = gi.loc["normal", "e_curve"], gi.loc[ce, "e_curve"], gi.loc[cl, "e_curve"]
                        if met == "err_raw":
                            d = ee[pe] - el[pl]
                        elif met == "err_diffindiff":
                            d = (ee[pe] - en[pe]) - (el[pl] - en[pl])
                        else:
                            tau = 3.0 if met.startswith("exceed3") else 2.0
                            d = (float(ee[pe] > tau) - float(en[pe] > tau)) - (float(el[pl] > tau) - float(en[pl] > tau))
                        recs.append({"unit_id": uid, "drive_id": g.drive_id.iat[0], "scene_id": g.scene_id.iat[0],
                                     "log_id": g.log_id.iat[0], "d": d, "pe": pe, "pl": pl})
                    pr = pd.DataFrame(recs)
                    margin = 0.10 if met.startswith("exceed") else 0.5
                    r = drive_row(pr, "d", margin=margin, seed=seed); seed += 11
                    block.append({"analysis": f"common post-release horizon N={N} (pose rel+N)", "window_kind": "persistent" if kind == "P" else "transient",
                                  "contrast": "s1 - s3", "w": w, "metric": met, "subset": f"A-, t*<={tmax}",
                                  "status": ("predeclared secondary (exp-10 extension), drive level" if (N == 2 and met == "err_diffindiff")
                                             or (N == 3 and met == "err_diffindiff") else "POST-HOC (this recheck)"),
                                  "margin_status": "post-hoc margin (not predeclared for this horizon)",
                                  "eval_pose_early": float(pr.pe.mean()), "eval_pose_late": float(pr.pl.mean()),
                                  "post_release_poses_early": N, "post_release_poses_late": N, "equal_post_release_horizon": True,
                                  "missing_pairs": 0, **r})
                adj = SC.holm([b["p_signflip"] for b in block]); adje = SC.holm([b["tost_p_equiv"] for b in block])
                for b, a, ae in zip(block, adj, adje):
                    b["p_holm_m3"] = a; b["tost_p_equiv_holm_m3"] = ae
                rows += block
    return pd.DataFrame(rows)


def c2_levels(u):
    """scene-weighted levels per condition (A-, t*<=2) with drive CIs, for the position/length figure."""
    A = u[(u.group == "A-") & (u.t_star <= 2)]
    rows = []
    for cond, g in A.groupby("condition"):
        for met in ("amplification", "fde5"):
            sc = SC.scene_table(g.assign(dd=g[met]), "drive_id", "scene_id", d="dd")
            r = SC.analyze(sc, "scene", B=2000, seed=SEED, ci_inversion=False)
            rows.append({"condition": cond, "metric": met, "level": r["theta"], "ci95_lo": r["boot_ci95"][0], "ci95_hi": r["boot_ci95"][1],
                         "n_drives": r["n_clusters"]})
    return pd.DataFrame(rows)


def c2_recovery(u):
    rows = []
    seed = SEED + 40000
    # ------------------------------------------------ P1-B (total horizon, pose 9), A-, t*<=2
    A = u[(u.group == "A-") & (u.t_star <= 2)]
    cnt = A.groupby("condition")[["intervention_count", "corrected_positions_n", "corrected_pairs", "corrected_pairs_effective"]].mean()
    rel = A.groupby("condition")["release_pose"].mean()
    for met in ("amplification", "fde5"):
        W = p1b_wide(A, met)
        conds = [c for c in W.columns if c.startswith(("P_gt", "T_gt"))] + ["recent_gt"]
        for c in conds:
            for wt in ("scene", "unit"):
                f, ci, p90, G = ratio_boot(W, ("normal", c), ("normal", "full_gt"), wt, seed); seed += 3
                kind = "persistent window" if c.startswith("P_") else ("transient window" if c.startswith("T_") else "Recent-GT (repeated)")
                s = int(c.split("_s")[1][0]) if "_s" in c else np.nan
                w = int(c.split("_w")[1]) if "_w" in c else np.nan
                rows.append({"run": "P1-B", "group": "A-", "subset": "t*<=2 common (46 scenes / 326 units)", "condition": c,
                             "intervention_type": kind, "s": s, "w": w, "metric": met, "weighting": wt, "horizon": "pose 9 (total)",
                             "normal": W["normal"].mean() if wt == "unit" else W.groupby("scene_id")["normal"].mean().mean(),
                             "full": W["full_gt"].mean() if wt == "unit" else W.groupby("scene_id")["full_gt"].mean().mean(),
                             "window": W[c].mean() if wt == "unit" else W.groupby("scene_id")[c].mean().mean(),
                             "fraction": f, "ci95_lo": ci[0], "ci95_hi": ci[1], "boot_share_f_ge_0.9": p90,
                             "denominator": "Normal - full_gt (same P1-B run, same units)", "n_units": len(W), "n_scenes": W.scene_id.nunique(),
                             "n_logs": W.log_id.nunique(), "n_drives": G,
                             "corrected_positions": cnt.loc[c, "corrected_positions_n"] if c in cnt.index else np.nan,
                             "mean_corrected_forwards_per_unit": cnt.loc[c, "intervention_count"],
                             "mean_corrected_pairs_per_unit": cnt.loc[c, "corrected_pairs"],
                             "mean_effective_pairs_per_unit": cnt.loc[c, "corrected_pairs_effective"],
                             "mean_post_release_free_poses": 9 - rel.get(c, np.nan) if c != "recent_gt" else 0.0})
    # ------------------------------------------------ P1-B, common post-release horizon N=1 (continuous GT error)
    for c in [f"{k}_gt_s{s}_w{w}" for k in ("P", "T") for s in (1, 2, 3, 4) for w in (1, 2, 3)]:
        recs = []
        for uid, g in A.groupby("unit_id"):
            gi = g.set_index("condition")
            pe = int(gi.loc[c, "release_pose"]) + 1
            if pe > 9:
                continue
            recs.append({"drive_id": g.drive_id.iat[0], "scene_id": g.scene_id.iat[0], "log_id": g.log_id.iat[0],
                         "normal": gi.loc["normal", "e_curve"][pe], "full_gt": gi.loc["full_gt", "e_curve"][pe], c: gi.loc[c, "e_curve"][pe]})
        W = pd.DataFrame(recs)
        f, ci, p90, G = ratio_boot(W, ("normal", c), ("normal", "full_gt"), "scene", seed); seed += 3
        rows.append({"run": "P1-B", "group": "A-", "subset": f"t*<=2 units with rel+1<=9 (n={len(W)})", "condition": c,
                     "intervention_type": "persistent window" if c.startswith("P_") else "transient window",
                     "s": int(c.split("_s")[1][0]), "w": int(c.split("_w")[1]), "metric": "gt_err_at_rel+1", "weighting": "scene",
                     "horizon": "post-release N=1 (pose rel+1)", "fraction": f, "ci95_lo": ci[0], "ci95_hi": ci[1], "boot_share_f_ge_0.9": p90,
                     "denominator": "Normal - full_gt at the same pose (same P1-B run)", "n_units": len(W), "n_scenes": W.scene_id.nunique(),
                     "n_logs": W.log_id.nunique(), "n_drives": G, "corrected_positions": int(c.split("_w")[1]),
                     "mean_post_release_free_poses": 1.0})
    # ------------------------------------------------ R8R9
    r = pd.read_parquet(f"{R89}/units.parquet")
    r["drive_id"] = r.log_id.map(drive_of)
    ed = {x["token"]: x for x in map(json.loads, open(C.ED_RECORDS))}
    for exp, conds, full in (("R9", ["win1", "win2", "win3", "win4", "win5", "delay2_len1", "delay3_len1", "delay4_len1"], "gt_history"),
                             ("R8", ["recent_gt"], "gt_history")):
        E = r[(r.experiment == exp) & (r.group == "A-")]
        for subset, smask in (("all A- (52 scenes / 365 units)", E.t_star >= 0), ("A-, t*<=2 (P1-B common basis)", E.t_star <= 2)):
            S = E[smask]
            tst = S.drop_duplicates("unit_id").set_index("unit_id")["t_star"]
            for met in ("amplification", "fde5"):
                W = S.pivot_table(index=["unit_id", "scene_id", "log_id", "drive_id"], columns="condition", values=met, aggfunc="first").reset_index()
                for c in conds:
                    tt = W.unit_id.map(tst)
                    if c.startswith("win"):
                        w = int(c[3:]); s = 1
                        # persistent window positions t*+1 .. min(t*+w, 8); forwards that see it: k = t*+2..9
                        npos = np.minimum(w, 8 - tt)
                        fwd = (9 - (tt + 1)).astype(float)
                        pairs = np.array([sum(min(w, k - 1 - t) for k in range(t + 2, 10)) for t in tt])
                        eff = np.full(len(W), np.nan)
                        kind, post = "persistent window", (9 - (tt + 1 + w)).clip(lower=0)
                    elif c.startswith("delay"):
                        s = int(c[5]); w = int(c[-1])
                        npos = np.minimum(w, (8 - tt - s + 1).clip(lower=0))
                        fwd = (9 - (tt + s)).clip(lower=0).astype(float)
                        pairs = np.array([sum(max(0, min(w, k - 1 - t - s + 1)) for k in range(t + 2, 10)) for t in tt])
                        eff = np.full(len(W), np.nan)
                        kind, post = "persistent window (delayed start)", (9 - (tt + s + w)).clip(lower=0)
                    else:
                        s = np.nan; w = np.nan
                        npos = (8 - tt).astype(float)          # positions t*+1..8, one at a time
                        fwd = (8 - tt).astype(float)           # forwards k = t*+2..9, each replaces position k-1
                        pairs = fwd.values
                        acts = S[S.condition == "recent_gt"].set_index("unit_id")["action_idx"].map(
                            lambda a: json.loads(a) if isinstance(a, str) else list(a))      # stored as JSON string
                        eff = np.array([sum(int(acts[u][j] != ed[u.split("|")[0]]["gt"][j]) for j in range(t + 1, 9))
                                        for u, t in zip(W.unit_id, tt)], float)
                        kind, post = "Recent-GT (repeated last-position replacement)", tt * 0.0
                    for wt in ("scene", "unit"):
                        f, ci, p90, G = ratio_boot(W, ("normal", c), ("normal", full), wt, seed); seed += 3
                        lv = (lambda col: W[col].mean()) if wt == "unit" else (lambda col: W.groupby("scene_id")[col].mean().mean())
                        rows.append({"run": exp + " (recovery re-run)", "group": "A-", "subset": subset, "condition": c, "intervention_type": kind,
                                     "s": s, "w": w, "metric": met, "weighting": wt, "horizon": "pose 9 (total)",
                                     "normal": lv("normal"), "full": lv(full), "window": lv(c),
                                     "fraction": f, "ci95_lo": ci[0], "ci95_hi": ci[1], "boot_share_f_ge_0.9": p90,
                                     "denominator": f"Normal - {full} (same {exp} run, same units)", "n_units": len(W),
                                     "n_scenes": W.scene_id.nunique(), "n_logs": W.log_id.nunique(), "n_drives": G,
                                     "corrected_positions": float(np.mean(npos)), "mean_corrected_forwards_per_unit": float(np.mean(fwd)),
                                     "mean_corrected_pairs_per_unit": float(np.mean(pairs)),
                                     "mean_effective_pairs_per_unit": float(np.nanmean(eff)) if np.isfinite(eff).any() else np.nan,
                                     "mean_post_release_free_poses": float(np.mean(post))})
    # P1-A Recent-reference / Recent-GT arms (reference-relative; for the Recent-GT definition column only)
    return pd.DataFrame(rows)


# ============================================================================================ figures
COL = {"blue": "#2a78d6", "orange": "#eb6834", "aqua": "#1baf7a", "yellow": "#eda100", "ink": "#0b0b0b", "ink2": "#52514e", "grid": "#e4e3df"}


def _style(ax):
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.spines["left"].set_color(COL["ink2"]); ax.spines["bottom"].set_color(COL["ink2"])
    ax.tick_params(colors=COL["ink2"], labelsize=8)
    ax.grid(axis="y", color=COL["grid"], lw=0.6); ax.set_axisbelow(True)


def fig_p1a(T):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    G = T[(T.family == "G")]
    fig, axs = plt.subplots(1, 3, figsize=(13.5, 4.4))
    groups = ["A-", "A+", "all"]
    x = np.arange(3)
    # panel 1: counterfactual endpoint change, model vs Geo vs CM
    ax = axs[0]
    for i, (m, lab, c) in enumerate((("model_end", "model (OC vs U)", COL["blue"]), ("geo_end", "Geo (coordinate accumulation)", COL["orange"]),
                                      ("cm_end", "CM (constant motion)", COL["aqua"]))):
        v = G[G.metric == m].set_index("group").loc[groups]
        ax.bar(x + (i - 1) * 0.26, v.est_scene, 0.24, color=c, label=lab)
        ax.errorbar(x + (i - 1) * 0.26, v.est_scene, yerr=[v.est_scene - v.ci95_lo, v.ci95_hi - v.est_scene], fmt="none", ecolor=COL["ink"], lw=1, capsize=2)
    ax.set_xticks(x); ax.set_xticklabels(groups); ax.set_ylabel("perturbed - unperturbed endpoint change (m)", fontsize=8)
    ax.set_title("A. Counterfactual endpoint change (family G, pose 9)", fontsize=9, loc="left"); ax.legend(fontsize=7, frameon=False); _style(ax)
    # panel 2: excess endpoint and decomposition
    ax = axs[1]
    for i, (m, lab, c) in enumerate((("ex_geo_end", "model - Geo", COL["orange"]), ("ex_cm_end", "model - CM", COL["aqua"]),
                                      ("model_local", "model local-motion component", COL["blue"]), ("ex_geo_coord", "model coord. comp. - Geo", COL["yellow"]))):
        v = G[G.metric == m].set_index("group").loc[groups]
        xx = x + (i - 1.5) * 0.2
        ax.errorbar(xx, v.est_scene, yerr=[v.est_scene - v.ci95_lo, v.ci95_hi - v.est_scene], fmt="o", ms=5, color=c, ecolor=c, lw=1.5, capsize=2, label=lab)
    ax.axhline(0, color=COL["ink2"], lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(groups); ax.set_ylabel("m (drive-cluster 95% CI)", fontsize=8)
    ax.set_title("B. Excess endpoint change and decomposition", fontsize=9, loc="left"); ax.legend(fontsize=7, frameon=False); _style(ax)
    # panel 3: suffix local motion excess
    ax = axs[2]
    for i, (m, lab, c) in enumerate((("ex_geo_sfx", "model - Geo (Geo suffix = 0)", COL["orange"]), ("ex_cm_sfx", "model - CM (CM = d0)", COL["aqua"]))):
        v = G[G.metric == m].set_index("group").loc[groups]
        xx = x + (i - 0.5) * 0.25
        ax.errorbar(xx, v.est_scene, yerr=[v.est_scene - v.ci95_lo, v.ci95_hi - v.est_scene], fmt="o", ms=5, color=c, ecolor=c, lw=1.5, capsize=2, label=lab)
    ax.axhline(0, color=COL["ink2"], lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(groups); ax.set_ylabel("m per 0.5 s segment", fontsize=8)
    ax.set_title("C. Excess suffix local-motion change", fontsize=9, loc="left"); ax.legend(fontsize=7, frameon=False); _style(ax)
    fig.suptitle("P1-A (post-hoc): model vs no-inference baselines, same reference tokens, drive-cluster bootstrap 95% CI "
                 "(A- 13 drives, A+ 19, all 19)", fontsize=9)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig_p1a_excess.png", dpi=150); plt.close(fig)


def fig_pos_len(L, P, R):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(2, 2, figsize=(12, 8.4))
    wcol = {1: COL["blue"], 2: COL["orange"], 3: COL["aqua"]}
    # A: position effect, total horizon, levels vs s per w (persistent GT)
    ax = axs[0, 0]
    Lv = L[L.metric == "amplification"].set_index("condition")
    for w in (1, 2, 3):
        v = Lv.loc[[f"P_gt_s{s}_w{w}" for s in (1, 2, 3, 4)]]
        ax.errorbar(np.arange(1, 5) + (w - 2) * 0.06, v.level * 100, yerr=[(v.level - v.ci95_lo) * 100, (v.ci95_hi - v.level) * 100],
                    marker="o", ms=5, lw=2, color=wcol[w], capsize=2, label=f"w={w}")
    for c, ls in (("normal", "--"), ("full_gt", ":")):
        ax.axhline(Lv.loc[c, "level"] * 100, color=COL["ink2"], ls=ls, lw=1)
        ax.text(4.25, Lv.loc[c, "level"] * 100, "Normal" if c == "normal" else "Full GT", fontsize=7, color=COL["ink2"], va="center")
    ax.set_xticks([1, 2, 3, 4]); ax.set_xlabel("start offset s (fixed length w)", fontsize=8); ax.set_ylabel("GT amplification at pose 9 (%)", fontsize=8)
    ax.set_title("A. POSITION effect, total horizon (P1-B persistent GT, A-, t*<=2)\nlate starts have 2 fewer post-release poses", fontsize=9, loc="left")
    ax.legend(fontsize=7, frameon=False); _style(ax)
    # B: position effect early-late per w: total horizon vs common post-release horizon
    ax = axs[0, 1]
    specs = [("total horizon (pose 9) amplification, pp", (P.analysis.str.startswith("total")) & (P.window_kind == "persistent") & (P.contrast == "s1 - s3") & (P.metric == "amplification"), 100, COL["blue"]),
             ("common horizon N=1: exceed-3m diff-in-diff, pp", (P.analysis.str.contains("N=1")) & (P.window_kind == "persistent") & (P.metric == "exceed3_diffindiff"), 100, COL["orange"]),
             ("common horizon N=2: exceed-3m diff-in-diff, pp", (P.analysis.str.contains("N=2")) & (P.window_kind == "persistent") & (P.metric == "exceed3_diffindiff"), 100, COL["aqua"])]
    for i, (lab, m, k, c) in enumerate(specs):
        v = P[m].set_index("w").loc[[1, 2, 3]]
        xx = np.array([1, 2, 3]) + (i - 1) * 0.15
        ax.errorbar(xx, v.est_scene * k, yerr=[(v.est_scene - v.ci95_lo) * k, (v.ci95_hi - v.est_scene) * k], fmt="o", ms=5, color=c, ecolor=c, lw=1.5, capsize=2, label=lab)
    ax.axhspan(-10, 10, color=COL["grid"], alpha=0.6, lw=0); ax.text(3.3, 9, "±10 pp margin\n(predeclared for\ntotal-horizon primary)", fontsize=6.5, color=COL["ink2"], va="top")
    ax.axhline(0, color=COL["ink2"], lw=0.8)
    ax.set_xticks([1, 2, 3]); ax.set_xlim(0.6, 3.9); ax.set_xlabel("correction length w", fontsize=8); ax.set_ylabel("early (s=1) - late (s=3), pp (drive 95% CI)", fontsize=8)
    ax.set_title("B. POSITION effect per w: unequal vs equal post-release horizon", fontsize=9, loc="left"); ax.legend(fontsize=7, frameon=False, loc="lower left"); _style(ax)
    # C: length effect, recovery fraction vs w at s=1 (binary amplification, scene weighted)
    ax = axs[1, 0]
    Rs = R[(R.metric == "amplification") & (R.weighting == "scene") & (R.horizon == "pose 9 (total)")]
    series = [("P1-B persistent GT, s=1", Rs[(Rs.run == "P1-B") & Rs.condition.str.match(r"P_gt_s1_w\d")], COL["blue"]),
              ("P1-B transient GT, s=1", Rs[(Rs.run == "P1-B") & Rs.condition.str.match(r"T_gt_s1_w\d")], COL["orange"]),
              ("R9 win_w (all A-)", Rs[(Rs.run.str.startswith("R9")) & Rs.condition.str.match(r"win[1-4]$") & Rs.subset.str.startswith("all")], COL["aqua"])]
    for i, (lab, v, c) in enumerate(series):
        v = v.sort_values("w")
        xx = v.w.values + (i - 1) * 0.07
        ax.errorbar(xx, v.fraction * 100, yerr=[(v.fraction - v.ci95_lo) * 100, (v.ci95_hi - v.fraction) * 100], marker="o", ms=5, lw=2, color=c, capsize=2, label=lab)
    ax.axhline(90, color=COL["ink2"], ls="--", lw=0.8); ax.text(0.85, 91.5, "90%", fontsize=7, color=COL["ink2"])
    ax.set_xticks([1, 2, 3, 4]); ax.set_xlabel("correction length w (start fixed at s=1)", fontsize=8)
    ax.set_ylabel("recovery fraction (Normal - window)/(Normal - Full), %", fontsize=8)
    ax.set_title("C. LENGTH effect, total horizon (GT amplification, scene-weighted, drive CI)", fontsize=9, loc="left"); ax.legend(fontsize=7, frameon=False); _style(ax)
    # D: start x length recovery map at common post-release horizon N=1 (continuous)
    ax = axs[1, 1]
    Rn = R[(R.horizon.str.startswith("post-release")) & R.condition.str.startswith("P_gt")]
    for w in (1, 2, 3):
        v = Rn[Rn.w == w].sort_values("s")
        ax.errorbar(v.s + (w - 2) * 0.06, v.fraction * 100, yerr=[(v.fraction - v.ci95_lo) * 100, (v.ci95_hi - v.fraction) * 100], marker="o", ms=5, lw=2, color=wcol[w], capsize=2, label=f"w={w}")
    ax.set_xticks([1, 2, 3, 4]); ax.set_xlabel("start offset s", fontsize=8); ax.set_ylabel("recovery of GT error at pose rel+1, %", fontsize=8)
    ax.set_title("D. POSITION x LENGTH at equal post-release horizon N=1\n(P1-B persistent GT, continuous GT error, same-pose Normal/Full)", fontsize=9, loc="left")
    ax.legend(fontsize=7, frameon=False); _style(ax)
    fig.suptitle("Correction position vs length (post-hoc recheck; drive-cluster bootstrap 95% CI, 12 drives for A- t*<=2, 13 for R9 all A-)", fontsize=9)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig_position_vs_length.png", dpi=150); plt.close(fig)


# ============================================================================================ main
def main():
    os.makedirs(OUT, exist_ok=True)
    log("C1: building per-unit counterfactual table from P1-A raw records")
    U, CK = c1_units()
    U.to_parquet(f"{OUT}/c1_units_counterfactual.parquet", index=False)
    ck = {"n_units": len(CK), "prefix_equal": float(CK.prefix_equal.mean()), "first_token_p_all": float(CK.first_token_p_all.mean()),
          "pose_t_maxdiff_OC_Geo_CM_max": float(CK.pose_t_maxdiff_OC_Geo_CM.max()),
          "pose_t_maxdiff_U_ref_CM0_max_over_U_eq_ref_units": float(CK[CK.U_eq_ref].pose_t_maxdiff_U_ref_CM0.max()),
          "n_poses_equal_10": float(CK.n_poses_equal_10.mean()), "cm_rollout_np_vs_decoder_max_m": float(CK.cm_rollout_np_vs_decoder.max()),
          "ref_rollout_np_vs_file_max_m": float(CK.ref_rollout_np_vs_file.max()), "OC_suffix_eq_C1": float(CK.OC_suffix_eq_C1.mean()),
          "U_eq_ref": float(CK.U_eq_ref.mean()),
          "cm_sfx_equals_d0_maxabs": float((U.cm_sfx - U.d0).abs().max())}
    json.dump(ck, open(f"{OUT}/c1_matching_checks.json", "w"), indent=1)
    log("C1 checks", ck)
    T = c1_tables(U)
    PD = c1_predeclared()
    T.to_csv(f"{OUT}/c1_excess_effects.csv", index=False)
    PD.to_csv(f"{OUT}/c1_predeclared_and_gt_drive.csv", index=False)
    log("C1 tables written", len(T), len(PD))
    fig_p1a(T)

    log("C2: P1-B")
    u = p1b_load()
    P = c2_position(u)
    P.to_csv(f"{OUT}/c2_position_length.csv", index=False)
    L = c2_levels(u)
    L.to_csv(f"{OUT}/c2_levels_p1b.csv", index=False)
    log("C2 recovery")
    R = c2_recovery(u)
    R.to_csv(f"{OUT}/c2_recovery_fractions.csv", index=False)
    fig_pos_len(L, P, R)
    meta = {"script": os.path.abspath(__file__), "seed": SEED, "bootstrap_B": B, "stats_core": os.path.join(HERE, "stats_core.py"),
            "inputs": [f"{P1A}/raw/records.jsonl", f"{P1A}/raw_fixedpos/records.jsonl", f"{P1A}/units.parquet", f"{P1A}/fixedpos/units.parquet",
                       f"{P1B}/units.parquet", f"{R89}/units.parquet", C.ED_RECORDS, C.CODEBOOK],
            "inputs_sha256": {p: C.sha256_file(p) for p in (f"{P1A}/raw/records.jsonl", f"{P1A}/raw_fixedpos/records.jsonl", f"{P1B}/units.parquet", f"{R89}/units.parquet")},
            "elapsed_s": time.time() - T0, "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "nature": "POST-HOC analysis of existing data; CPU only; no model inference"}
    json.dump(meta, open(f"{OUT}/c_run_meta.json", "w"), indent=1)
    log("done")


if __name__ == "__main__":
    main()
