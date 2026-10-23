#!/usr/bin/env python
"""review_followup5 B_NUMERIC analysis (CPU; protocol §3). Reads B_NUMERIC/run/records.jsonl (S0 / SE / SF, same units).

Per unit u and setting s:  D_model(u,s) = ||X_OC(u,s,9) - X_U(u,s,9)||   (endpoint response, not FDE)
Frozen baselines D_b(u) (review_followup3 P2): {Geo, CM, CV0, CTRV-K3, CTRA-2}_end from p2_units.parquet; learned
prefix extrapolator E_prefix_end (seed mean) and E_prefix_ego_end (secondary) from extrap_units.parquet.
E_b(u,s) = D_model(u,s) - D_b(u);  Delta(u,s) = E_b(u,s) - E_b(u,S0) = D_model(u,s) - D_model(u,S0).
Inference: c_baseline_window.drive_row (unit -> scene mean, scene-weighted, drive bootstrap B = 10,000, exact drive
sign-flip at 0), seed 20261018 (+11 per row). Primary family: A- x {SE, SF} x {Geo, CM, CV0, CTRV-K3, CTRA-2, E_prefix},
Holm alpha 0.05 (missing -> p = 1). Materiality: |Delta| > 0.5 m, or |E(S0)| >= 1 m and |Delta|/|E(S0)| > 0.25, or sign flip.

  nice -n 19 /root/miniforge3/envs/autovla/bin/python b_numeric_analyze.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

POC = "/root/VLA/autovla_misalignment_poc"
sys.path.insert(0, f"{POC}/scripts/review_followup2")
sys.path.insert(0, f"{POC}/scripts/review_followup3")
sys.path.insert(0, f"{POC}/scripts")
import stats_core as SC  # noqa: E402
import c_baseline_window as CB  # noqa: E402
import p2_kinematics as K  # noqa: E402
from analyze_action_history import unit_metrics  # noqa: E402

B = f"{POC}/outputs/review_followup5/B_NUMERIC"
RUN = f"{B}/run"
MAN = f"{B}/manifest_G_full.json"
RAW = f"{POC}/outputs/review_followup/P1A_gt_free_reference/run_20261008/raw/records.jsonl"
P2 = f"{POC}/outputs/review_followup3/P2"
SEED = 20261018
BASE = ["Geo", "CM", "CV0", "CTRV-K3", "CTRA-2", "E_prefix"]
SECOND_BASE = ["E_prefix_ego"]


def main():
    man = json.load(open(MAN))
    stored = {u["unit_id"]: u["stored_rows_acts"] for u in man["units"]}
    gtinfo = {}
    for line in open(RAW):
        if '"unit_id"' not in line[:200]:
            continue
        x = json.loads(line)
        if x["family"] == "G":
            gtinfo[x["unit_id"]] = (x["gt"], x["trajectory_gt"], np.asarray(x["ref_traj_xyh"])[9, :2])
    rec = {}
    only = os.environ.get("B5_SETTINGS")          # code test on S0 only while the GPU run is still writing
    for line in open(f"{RUN}/records.jsonl"):
        try:
            x = json.loads(line)
        except json.JSONDecodeError:
            continue
        if only and x["setting"] not in only.split(","):
            continue
        rec[(x["setting"], x["unit_id"], x["row"])] = x
    OUTB = os.environ.get("B5_OUT", B)
    settings = [s for s in ("S0", "SE", "SF") if any(k[0] == s for k in rec)]
    units = man["units"]
    rows, agree = [], []
    for u in units:
        uid = u["unit_id"]
        gt, gtraj, r9 = gtinfo[uid]
        t = u["t_p"]
        base = {"unit_id": uid, "scene_id": u["scene"], "log_id": u["log"], "drive_id": CB.drive_of(u["log"]), "group": u["group"], "t_p": t}
        for s in settings:
            if (s, uid, "OC") not in rec or (s, uid, "U") not in rec:
                continue
            oc, uu = rec[(s, uid, "OC")], rec[(s, uid, "U")]
            Xo, Xu = np.asarray(oc["poses"]), np.asarray(uu["poses"])
            lo, lu = K.body_increments(Xo)[:, :2], K.body_increments(Xu)[:, :2]
            mo = unit_metrics(oc["acts"], Xo[:, :2], [], gt, gtraj, t)
            mu = unit_metrics(uu["acts"], Xu[:, :2], [], gt, gtraj, t)
            d9 = {n: float(np.linalg.norm(np.asarray(rec[(s, uid, n)]["poses"])[9, :2] - r9)) for n in ("OC", "OC_refR", "OC_refH") if (s, uid, n) in rec}
            r = dict(base, setting=s, D_model=float(np.linalg.norm(Xo[9, :2] - Xu[9, :2])),
                     sfx=float(np.mean(np.linalg.norm(lo[t + 1:] - lu[t + 1:], axis=1))),
                     severe_OC=int(mo["amplification"]), severe_U=int(mu["amplification"]), Aminus_OC=int(not mo["recovery"]),
                     Aminus_U=int(not mu["recovery"]), fde5_OC=mo["fde5"], fde5_U=mu["fde5"],
                     selfR=d9.get("OC_refR", np.nan) - d9["OC"], selfH=d9.get("OC_refH", np.nan) - d9["OC"],
                     prefix_lock=bool(oc["prefix_lock"] and uu["prefix_lock"]))
            if s != "S0":
                o0, u0 = rec[("S0", uid, "OC")], rec[("S0", uid, "U")]
                r["tok_changed_OC"] = float(np.mean([a != b for a, b in zip(o0["own"], oc["own"])]))
                r["tok_changed_U"] = float(np.mean([a != b for a, b in zip(u0["own"], uu["own"])]))
            rows.append(r)
            if s == "S0":
                agree.append({"unit_id": uid, **{f"eq_{n}": list(rec[("S0", uid, n)]["acts"]) == list(stored[uid][n])
                                                 for n in stored[uid] if ("S0", uid, n) in rec}})
    U = pd.DataFrame(rows)
    p2 = pd.read_parquet(f"{P2}/p2_units.parquet")
    p2 = p2[(p2.family == "G") & (p2.source == "primary")].set_index("unit_id")
    ex = pd.read_parquet(f"{P2}/extrap_units.parquet")
    ex = ex[(ex.family == "G") & (ex.source == "primary")].set_index("unit_id")
    for b in ["Geo", "CM", "CV0", "CTRV-K3", "CTRA-2"]:
        U[f"Db_{b}"] = U.unit_id.map(p2[f"{b}_end"])
    U["Db_E_prefix"] = U.unit_id.map(ex["E_prefix_end"])
    U["Db_E_prefix_ego"] = U.unit_id.map(ex["E_prefix_ego_end"])
    U["model_end_P2_S0_stored"] = U.unit_id.map(p2["model_end"])
    for b in BASE + SECOND_BASE:
        U[f"E_{b}"] = U.D_model - U[f"Db_{b}"]
    s0 = U[U.setting == "S0"].set_index("unit_id")
    for s in settings:
        m = U.setting == s
        U.loc[m, "Delta"] = U.loc[m, "D_model"].values - U.loc[m, "unit_id"].map(s0.D_model).values
        for col in ("severe_OC", "fde5_OC", "selfR", "selfH", "sfx"):
            U.loc[m, f"d_{col}_vs_S0"] = U.loc[m, col].values - U.loc[m, "unit_id"].map(s0[col]).values
    U.to_parquet(f"{OUTB}/b_numeric_units.parquet", index=False)
    # ---- inference
    out = []
    seed = SEED
    for s in settings:
        for grp in ("A-", "A+", "all"):
            g = U[(U.setting == s) & ((U.group == grp) if grp != "all" else True)]
            for b in BASE + SECOND_BASE:
                r = CB.drive_row(g, f"E_{b}", seed=seed); seed += 11
                out.append({"setting": s, "group": grp, "quantity": f"E_{b}", "baseline": b, "role": "excess", **r})
            for q in ("D_model", "sfx", "selfR", "selfH"):
                r = CB.drive_row(g, q, seed=seed); seed += 11
                out.append({"setting": s, "group": grp, "quantity": q, "baseline": "", "role": "level", **r})
            if s != "S0":
                for q in ("Delta", "d_sfx_vs_S0", "d_severe_OC_vs_S0", "d_fde5_OC_vs_S0", "d_selfR_vs_S0", "d_selfH_vs_S0"):
                    r = CB.drive_row(g, q, seed=seed); seed += 11
                    out.append({"setting": s, "group": grp, "quantity": q, "baseline": "", "role": "setting change vs S0", **r})
    R = pd.DataFrame(out)
    R["family"] = ""; R["p_holm"] = np.nan
    fam = (R.group == "A-") & R.setting.isin(["SE", "SF"]) & R.quantity.isin([f"E_{b}" for b in BASE])
    planned = [(s, f"E_{b}") for s in ("SE", "SF") for b in BASE]
    ps = []
    for s, q in planned:
        x = R[fam & (R.setting == s) & (R.quantity == q)]
        ps.append(float(x.p_signflip.iloc[0]) if len(x) and np.isfinite(x.p_signflip.iloc[0]) else 1.0)
    adj = SC.holm(ps)
    for (s, q), a in zip(planned, adj):
        m = fam & (R.setting == s) & (R.quantity == q)
        R.loc[m, "p_holm"] = a
        R.loc[m, "family"] = "primary: G/A-/endpoint excess, {SE,SF} x 6 baselines, Holm m = 12, alpha 0.05"
    # materiality (descriptive)
    mat = []
    for grp in ("A-", "A+", "all"):
        for b in BASE + SECOND_BASE:
            e0 = R[(R.setting == "S0") & (R.group == grp) & (R.quantity == f"E_{b}")].est_scene.iloc[0]
            for s in [x for x in settings if x != "S0"]:
                es = R[(R.setting == s) & (R.group == grp) & (R.quantity == f"E_{b}")].est_scene.iloc[0]
                dl = R[(R.setting == s) & (R.group == grp) & (R.quantity == "Delta")].iloc[0]
                a = abs(es - e0) > 0.5
                bb = abs(e0) >= 1 and abs(es - e0) / abs(e0) > 0.25
                c = np.sign(es) != np.sign(e0)
                mat.append({"group": grp, "baseline": b, "setting": s, "E_S0": e0, "E_s": es, "change": es - e0,
                            "Delta_paired_est": dl.est_scene, "Delta_ci95": json.dumps([float(dl.ci95_lo), float(dl.ci95_hi)]),
                            "abs_change_gt_0.5m": bool(a), "rel_change_gt_25pct": bool(bb), "sign_flip": bool(c),
                            "material": bool(a or bb or c)})
    M = pd.DataFrame(mat)
    R.to_csv(f"{OUTB}/B_NUMERIC_RESULTS.csv", index=False)
    M.to_csv(f"{OUTB}/B_NUMERIC_MATERIALITY.csv", index=False)
    A = pd.DataFrame(agree)
    eqc = [c for c in A.columns if c.startswith("eq_")]
    s0u = U[U.setting == "S0"]
    meta = {"settings_present": settings, "n_units_per_setting": U.groupby("setting").size().to_dict(),
            "prefix_lock_all": bool(U.prefix_lock.all()),
            "S0_vs_stored_raw": {"units": len(A), "rows_equal": int(A[eqc].values.sum()), "rows": int(A[eqc].size),
                                 "units_all_rows_equal": int(A[eqc].all(axis=1).sum())},
            "S0_D_model_vs_P2_model_end_max_abs": float(np.nanmax(np.abs(s0u.D_model - s0u.model_end_P2_S0_stored))),
            "missing_baselines": {b: int(U[f"Db_{b}"].isna().sum()) for b in BASE + SECOND_BASE},
            "token_change_vs_S0": (U[U.setting != "S0"].groupby("setting")[["tok_changed_OC", "tok_changed_U"]].mean().to_dict()
                                   if "tok_changed_OC" in U else {})}
    json.dump(meta, open(f"{OUTB}/b_numeric_meta.json", "w"), indent=1, default=str)
    pd.set_option("display.width", 250)
    print(json.dumps(meta, indent=1, default=str))
    print(R[R.family != ""][["setting", "quantity", "est_scene", "ci95_lo", "ci95_hi", "p_signflip", "p_holm"]].to_string())
    print(M.round(3).to_string())


if __name__ == "__main__":
    main()
