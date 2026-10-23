#!/usr/bin/env python
"""Author query Q1 (2026-10-19): contrast-specific coverage check of the cluster-robust t interval for four representative
AutoVLA contrasts. Plan fixed beforehand: outputs/manuscript_author_queries_20261019/COVERAGE_PLAN.md (sha256 in
COVERAGE_PLAN_LOCK.json). Post-hoc sensitivity on already analysed data. CPU only, vectorised.

  nice -n 19 /root/miniforge3/envs/autovla/bin/python q1_coverage.py
"""
from __future__ import annotations

import json
import math
import os
import time

import numpy as np
import pandas as pd
from scipy.stats import t as tdist

POC = "/root/VLA/autovla_misalignment_poc"
O = f"{POC}/outputs"
OUT = f"{O}/manuscript_author_queries_20261019"
R, SEED = 5000, 20261019


def drive_of(log):
    p = str(log).split("_")
    return f"{p[0]}_{p[1]}"


def paired(df, a, b, key="unit_id"):
    A = df[df.condition == a].set_index(key)
    B = df[df.condition == b].set_index(key)
    j = A.join(B[["y"]], rsuffix="_ref", how="inner")
    j["d"] = j.y - j.y_ref
    return j.reset_index()


def structures():
    out = {}
    d = pd.read_parquet(f"{O}/review_followup/P0B_cluster_stats/run_20261008/units.parquet")
    d = d[(d.experiment == "exp07_action_history_causal") & (d.group == "A-") & d.exclusion_reason.isna()].copy()
    d["y"] = d.amplification.astype(float)
    j = paired(d, "recent_gt", "normal")
    j["drive"] = j.log_id.map(drive_of)
    out["DEV_RN"] = (j[["scene_id", "drive", "d", "y_ref"]], "normal", "dev A-")
    u = pd.read_parquet(f"{O}/review_followup/P1C_new_log_replication/run_20261008/analysis/units.parquet")
    u = u[(u.stratum == "F") & u.exclusion_reason.isna()].copy()
    u["y"] = u.amplification.astype(float)
    for cid, blk, a, b in (("P1C_F_RN", "H7", "recent_gt", "normal"), ("P1C_F_RV", "H8", "reverse@emb", "gt_history")):
        j = paired(u[u.block == blk], a, b)
        out[cid] = (j.rename(columns={"drive_id": "drive"})[["scene_id", "drive", "d", "y_ref"]], b, "P1-C F")
    W = json.load(open(f"{O}/review_followup4/B/B2/p1c_cohort_t4.json"))["scene_weights"]
    cells = {}
    for line in open(f"{O}/review_followup4/B/B2/p1c_main/cells.jsonl"):
        x = json.loads(line)
        if x["n_old"] == 1 and x["cell"] in ("NAT_NSRC", "NAT_REF"):
            cells.setdefault(x["unit_id"], {})[x["cell"]] = x["severe"]
    rows = []
    for uid, c in cells.items():
        sc = uid.split("|")[0]
        if len(c) == 2 and W[sc]["stratum"] == "F":
            rows.append({"scene_id": sc, "drive": W[sc]["drive"], "d": c["NAT_NSRC"] - c["NAT_REF"], "y_ref": c["NAT_REF"]})
    out["B2_F_CNAT"] = (pd.DataFrame(rows), "NAT_REF", "P1-C B2 F (primary anchor)")
    return out


def scene_table(df):
    g = df.groupby("scene_id")
    sc = pd.DataFrame({"n": g.d.size(), "qp": g.d.apply(lambda s: float((s > 0).mean())), "qm": g.d.apply(lambda s: float((s < 0).mean())),
                       "pc": g.y_ref.mean(), "drive": g.drive.first(), "dmean": g.d.mean()}).reset_index()
    return sc


def cluster_t(Nmat, D, level):
    """Nmat: (R, G) cluster numerators; D: (G,) denominators (weights)."""
    G = len(D)
    th = Nmat.sum(1) / D.sum()
    r = Nmat - th[:, None] * D[None, :]
    se = np.sqrt(G / (G - 1) * (r ** 2).sum(1)) / D.sum()
    q = tdist.ppf(1 - (1 - level) / 2, G - 1)
    return th - q * se, th + q * se, se


def simulate(sc, model, delta, tau, rng):
    labs, inv = np.unique(sc.drive.values, return_inverse=True)
    G = len(labs)
    n = sc.n.values.astype(int)
    S = len(sc)
    eta = rng.normal(0, tau, (R, G))[:, inv] if tau > 0 else np.zeros((R, S))
    if model == "M1":
        Dd = (sc.qp + sc.qm).values[None, :]
        ds = np.clip(delta + eta, -Dd, Dd)
        qp, qm = (Dd + ds) / 2, (Dd - ds) / 2
    else:
        pc = sc.pc.values[None, :]
        pt = np.clip(pc + delta + eta, 0, 1)
        qp, qm = pt * (1 - pc), pc * (1 - pt)
    theta = (qp - qm).mean(1)                       # scene weights = 1 in these strata (scene-weighted ratio)
    cp = rng.binomial(np.broadcast_to(n, (R, S)), qp)
    rest = n[None, :] - cp
    with np.errstate(invalid="ignore", divide="ignore"):
        cond = np.where(qp < 1, qm / (1 - qp), 0.0)
    cm = rng.binomial(rest, np.clip(cond, 0, 1))
    dmean = (cp - cm) / n[None, :]
    Nmat = np.zeros((R, G))
    for gi in range(G):
        Nmat[:, gi] = dmean[:, inv == gi].sum(1)
    Dg = np.bincount(inv, minlength=G).astype(float)
    return theta, Nmat, Dg


def wilson(c, n, z=1.96):
    p = c / n
    den = 1 + z * z / n
    mid = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return mid - half, mid + half


def main():
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.default_rng(SEED)
    t0 = time.time()
    plan = {"DEV_RN": [0.0, -0.10, 0.10, -0.40], "P1C_F_RN": [0.0, -0.10, 0.10, -0.34],
            "P1C_F_RV": [0.0, 0.10, -0.10, 0.27], "B2_F_CNAT": [0.0, 0.10, -0.10, 0.255]}
    scen_rows, res_rows, struct_rows = [], [], []
    for cid, (df, ref, coh) in structures().items():
        sc = scene_table(df)
        G = sc.drive.nunique()
        struct_rows.append({"contrast_id": cid, "cohort": coh, "reference_arm": ref, "n_units": int(sc.n.sum()), "n_scenes": len(sc),
                            "n_drives": G, "observed_theta_scene_weighted": float(sc.dmean.mean()),
                            "reference_arm_rate_scene_mean": float(sc.pc.mean()),
                            "observed_discordance_scene_mean": float((sc.qp + sc.qm).mean()),
                            "drive_sizes_scenes": ";".join(str(x) for x in sc.groupby("drive").size().sort_values().values)})
        for model in ("M1", "M2"):
            for delta in plan[cid]:
                for tau in (0.0, 0.05):
                    theta, Nmat, Dg = simulate(sc, model, delta, tau, rng)
                    sid = f"{cid}|{model}|d={delta:+.3f}|tau={tau}"
                    scen_rows.append({"scenario_id": sid, "contrast_id": cid, "model": model, "target_delta": delta, "tau": tau,
                                      "realised_theta_mean": float(theta.mean()), "realised_theta_min": float(theta.min()),
                                      "realised_theta_max": float(theta.max()),
                                      "target_shortfall": float(abs(delta) - abs(theta.mean())) if delta else 0.0,
                                      "note": ("target not reachable under the observed discordance (clipped)"
                                               if delta and abs(theta.mean()) < 0.95 * abs(delta) else "")})
                    for lev in (0.95, 0.90):
                        lo, hi, se = cluster_t(Nmat, Dg, lev)
                        cov = (lo <= theta) & (theta <= hi)
                        c = int(cov.sum())
                        wl, wh = wilson(c, R)
                        res_rows.append({"scenario_id": sid, "contrast_id": cid, "model": model, "target_delta": delta, "tau": tau,
                                         "method": f"cluster_t_{int(lev * 100)}", "reps": R, "coverage": c / R,
                                         "mc_se": math.sqrt((c / R) * (1 - c / R) / R), "mc_wilson95_lo": wl, "mc_wilson95_hi": wh,
                                         "P_zero_width": float(np.mean(se == 0)), "mean_width": float(np.mean(hi - lo))})
                    print(f"[{time.time()-t0:5.0f}s] {sid}", flush=True)
    pd.DataFrame(struct_rows).to_csv(f"{OUT}/COVERAGE_STRUCTURES.csv", index=False)
    pd.DataFrame(scen_rows).to_csv(f"{OUT}/COVERAGE_SCENARIOS.csv", index=False)
    Rr = pd.DataFrame(res_rows)
    Rr.to_csv(f"{OUT}/COVERAGE_RESULTS.csv", index=False)
    summ = Rr.groupby(["contrast_id", "method"]).agg(min_cov=("coverage", "min"), min_mc_hi=("mc_wilson95_hi", "min"),
                                                     max_cov=("coverage", "max"), max_zero=("P_zero_width", "max")).reset_index()
    summ.to_csv(f"{OUT}/COVERAGE_SUMMARY.csv", index=False)
    print(pd.DataFrame(struct_rows).to_string()); print(summ.to_string())


if __name__ == "__main__":
    main()
