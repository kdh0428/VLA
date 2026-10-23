#!/usr/bin/env python
"""CVPR next validation (2026-10-19), A2/A3: coverage of cluster-t and drive-bootstrap intervals for weighted E / E4
contrasts and for the B2 interaction INT, under a finite-population design (D3) and a superpopulation design (D2).
Plan fixed beforehand: outputs/cvpr_next_validation_20261019/STATISTICAL_CHECK_PLAN.md (sha256 in the lock file).
CPU only; vectorised over replicates.

  nice -n 19 /root/miniforge3/envs/autovla/bin/python sim_weighted.py [--quick]
"""
from __future__ import annotations

import glob
import json
import math
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy.stats import t as tdist

POC = "/root/VLA/autovla_misalignment_poc"
O = f"{POC}/outputs"
OUT = f"{O}/cvpr_next_validation_20261019"
P1C = f"{O}/review_followup/P1C_new_log_replication/run_20261008"
B2 = f"{O}/review_followup4/B/B2"
SEED = 20261019
R_T, R_B, BB = 5000, 1000, 999
VALS = (-2, -1, 1, 2)


def drive_of(log):
    p = str(log).split("_")
    return f"{p[0]}_{p[1]}"


# ------------------------------------------------------------------------------------------------ data
def p1c_units():
    u = pd.read_parquet(f"{P1C}/analysis/units.parquet")
    return u[u.exclusion_reason.isna()].copy()


def p1c_frame():
    """per log: N_S eligible, F count (frame), from the stage-1 selection shards."""
    fr = {}
    for f in sorted(glob.glob(f"{P1C}/selection/shard_*.json")):
        if not os.path.basename(f).replace("shard_", "").replace(".json", "").isdigit():
            continue
        d = json.load(open(f))
        for lg, den in d["denominators"].items():
            fr[lg] = {"N_S": int(den["S_eligible"]), "N_F": int(den["F"])}
    return fr


def contrast_units(cid):
    """unit table: scene, log, drive, stratum, d (difference, may be -2..2 for INT)."""
    if cid in ("E_RN", "F_RN", "E_RV"):
        u = p1c_units()
        blk, a, b = ("H7", "recent_gt", "normal") if cid in ("E_RN", "F_RN") else ("H8", "reverse@emb", "gt_history")
        u = u[u.block == blk]
        A = u[u.condition == a].set_index("unit_id")
        Bb = u[u.condition == b].set_index("unit_id")
        j = A[["scene_id", "log_id", "drive_id", "stratum"]].join(Bb[["amplification"]].rename(columns={"amplification": "yb"}), how="inner")
        j["d"] = A.loc[j.index, "amplification"].astype(int) - j.yb.astype(int)
        j = j.reset_index().rename(columns={"scene_id": "scene", "log_id": "log", "drive_id": "drive"})
        frame = p1c_frame()
    else:
        W = json.load(open(f"{B2}/p1c_cohort_t4.json"))["scene_weights"]
        cells = {}
        for line in open(f"{B2}/p1c_main/cells.jsonl"):
            x = json.loads(line)
            if x["n_old"] == 1 and x["cell"] in ("GT_NSRC", "GT_REF", "NAT_NSRC", "NAT_REF"):
                cells.setdefault(x["unit_id"], {})[x["cell"]] = int(x["severe"])
        rows = []
        for uid, c in cells.items():
            if len(c) < 4:
                continue
            sc = uid.split("|")[0]
            w = W[sc]
            d = {"E4_CNAT": c["NAT_NSRC"] - c["NAT_REF"], "F_CNAT": c["NAT_NSRC"] - c["NAT_REF"], "E4_CGT": c["GT_NSRC"] - c["GT_REF"],
                 "E4_INT": (c["GT_NSRC"] - c["GT_REF"]) - (c["NAT_NSRC"] - c["NAT_REF"]),
                 "F_INT": (c["GT_NSRC"] - c["GT_REF"]) - (c["NAT_NSRC"] - c["NAT_REF"])}[cid]
            rows.append({"unit_id": uid, "scene": sc, "log": w["log"], "drive": w["drive"], "stratum": w["stratum"], "d": d,
                         "N_S_dom": w.get("N_S_dom"), "m_S_dom": w.get("m_S_dom")})
        j = pd.DataFrame(rows)
        frame = {}
        for lg, g in j.groupby("log"):
            s = g[g.stratum == "S"]
            frame[lg] = {"N_S": int(s.N_S_dom.iloc[0]) if len(s) else 0, "N_F": int(g[g.stratum == "F"].scene.nunique())}
    if cid.startswith("F_"):
        j = j[j.stratum == "F"]
    return j, frame


def build(cid):
    """per drive: list of logs; per log: F scene class-count arrays and S pseudo-population arrays."""
    j, frame = contrast_units(cid)
    sc = []
    for s, g in j.groupby("scene"):
        cnt = [int((g.d == v).sum()) for v in VALS]
        sc.append({"scene": s, "log": g.log.iloc[0], "drive": g.drive.iloc[0], "stratum": g.stratum.iloc[0], "n": len(g), "cnt": cnt,
                   "dmean": float(g.d.mean())})
    sc = pd.DataFrame(sc)
    logs, dropped = {}, []
    for lg, g in sc.groupby("log"):
        F = g[g.stratum == "F"]
        S = g[g.stratum == "S"]
        N_S = frame.get(lg, {}).get("N_S", 0) if not cid.startswith("F_") else 0
        if N_S > 0 and len(S) == 0:
            dropped.append(lg); N_S = 0
        idx = [i % len(S) for i in range(N_S)] if N_S else []
        logs[lg] = {"drive": g.drive.iloc[0],
                    "F_cnt": np.array(F.cnt.tolist(), float).reshape(-1, 4), "F_n": F.n.values.astype(float),
                    "S_cnt": np.array(S.cnt.tolist(), float).reshape(-1, 4)[idx] if N_S else np.zeros((0, 4)),
                    "S_n": S.n.values.astype(float)[idx] if N_S else np.zeros(0),
                    "N_S": N_S, "m": min(len(S), N_S),
                    "F_dmean": F.dmean.values, "S_dmean": S.dmean.values[idx] if N_S else np.zeros(0)}
    drives = sorted({v["drive"] for v in logs.values()})
    num = sum(v["F_dmean"].sum() + v["S_dmean"].sum() for v in logs.values())
    den = sum(len(v["F_dmean"]) + len(v["S_dmean"]) for v in logs.values())
    obs_w = (sc[sc.stratum == "F"].dmean.sum() + sum((logs[lg]["N_S"] / logs[lg]["m"]) * sc[(sc.log == lg) & (sc.stratum == "S")].dmean.sum()
                                                     for lg in logs if logs[lg]["m"])) / (len(sc[sc.stratum == "F"]) + sum(logs[lg]["N_S"] for lg in logs))
    info = {"contrast": cid, "n_units": int(len(j)), "n_sampled_scenes": int(len(sc)), "n_drives": len(drives),
            "frame_scenes": int(den), "theta_pop": float(num / den), "observed_hajek_with_sim_weights": float(obs_w),
            "logs_dropped_no_sampled_S": dropped}
    return logs, drives, info


# ------------------------------------------------------------------------------------------------ simulation
def scene_sum(cnt, q, coupling, rng):
    """cnt: (R, k, 4) value-class counts; returns (R, k) scene sums after label swaps."""
    v = np.array(VALS, float)
    if coupling == "U":
        keep = rng.binomial(cnt.astype(int), q)
        return ((2 * keep - cnt) * v).sum(-1)
    s = np.where(rng.random(cnt.shape[:2]) < q, 1.0, -1.0)
    return s * (cnt * v).sum(-1)


def drive_draw(logs, drive, R, q, coupling, rng):
    """(R,) N and D of one drive under D3 (S re-sampled per log, F census, swaps)."""
    N = np.zeros(R); D = np.zeros(R)
    for lg, L in logs.items():
        if L["drive"] != drive:
            continue
        if len(L["F_n"]):
            cF = np.broadcast_to(L["F_cnt"], (R,) + L["F_cnt"].shape)
            N += (scene_sum(cF, q, coupling, rng) / L["F_n"]).sum(1)
            D += len(L["F_n"])
        if L["N_S"] and L["m"]:
            pick = np.argsort(rng.random((R, L["N_S"])), 1)[:, :L["m"]]
            cS = L["S_cnt"][pick]                      # (R, m, 4)
            nS = L["S_n"][pick]
            w = L["N_S"] / L["m"]
            N += w * (scene_sum(cS, q, coupling, rng) / nS).sum(1)
            D += w * L["m"]
    return N, D


def ci_t(N, D, level=0.95):
    G = N.shape[1]
    th = N.sum(1) / D.sum(1)
    r = N - th[:, None] * D
    se = np.sqrt(G / (G - 1) * (r ** 2).sum(1)) / D.sum(1)
    q = tdist.ppf(1 - (1 - level) / 2, G - 1)
    return th, th - q * se, th + q * se, se


def ci_boot(N, D, rng, level=0.95):
    Rr, G = N.shape
    lo = np.empty(Rr); hi = np.empty(Rr)
    cnt = rng.multinomial(G, np.full(G, 1 / G), size=BB).astype(float)       # (B, G), reused across replicates
    for i in range(Rr):
        bs = (cnt @ N[i]) / (cnt @ D[i])
        lo[i], hi[i] = np.percentile(bs, [(1 - level) / 2 * 100, (1 + level) / 2 * 100])
    return lo, hi


def wilson(c, n, z=1.96):
    p = c / n
    den = 1 + z * z / n
    mid = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return mid - half, mid + half


def run(cid, design, q, coupling, logs, drives, theta_pop, rng):
    R = R_T
    G = len(drives)
    per = {d: drive_draw(logs, d, R, q, coupling, rng) for d in drives}
    if design == "D3":
        N = np.stack([per[d][0] for d in drives], 1); D = np.stack([per[d][1] for d in drives], 1)
    else:
        sel = rng.integers(0, G, (R, G))
        N = np.empty((R, G)); D = np.empty((R, G))
        for k in range(G):
            shift = (np.arange(R) + 7919 * k) % R            # independent copy for each draw position
            for gi, d in enumerate(drives):
                m = sel[:, k] == gi
                N[m, k] = per[d][0][shift[m]]; D[m, k] = per[d][1][shift[m]]
    theta = (2 * q - 1) * theta_pop
    est, lo, hi, se = ci_t(N, D)
    ok = np.isfinite(lo) & np.isfinite(hi)
    out = []
    c = int(((lo <= theta) & (theta <= hi) & ok).sum())
    wl, wh = wilson(c, R)
    out.append({"contrast": cid, "design": design, "q": q, "coupling": coupling, "method": "cluster_t_95", "reps": R, "true_theta": theta,
                "coverage": c / R, "mc_se": math.sqrt((c / R) * (1 - c / R) / R), "mc_wilson95_lo": wl, "mc_wilson95_hi": wh,
                "bias": float(np.nanmean(est) - theta), "mean_width": float(np.nanmean(hi - lo)), "P_zero_width": float(np.mean(se == 0)),
                "failure_rate": float(np.mean(~ok))})
    Rb = R_B
    blo, bhi = ci_boot(N[:Rb], D[:Rb], rng)
    okb = np.isfinite(blo) & np.isfinite(bhi)
    c = int(((blo <= theta) & (theta <= bhi) & okb).sum())
    wl, wh = wilson(c, Rb)
    out.append({"contrast": cid, "design": design, "q": q, "coupling": coupling, "method": "drive_bootstrap_95", "reps": Rb, "true_theta": theta,
                "coverage": c / Rb, "mc_se": math.sqrt((c / Rb) * (1 - c / Rb) / Rb), "mc_wilson95_lo": wl, "mc_wilson95_hi": wh,
                "bias": float(np.nanmean(est[:Rb]) - theta), "mean_width": float(np.nanmean(bhi - blo)), "P_zero_width": float(np.mean((bhi - blo) < 1e-12)),
                "failure_rate": float(np.mean(~okb))})
    return out


def main():
    global R_T, R_B
    if "--quick" in sys.argv:
        R_T, R_B = 300, 100
    rng = np.random.default_rng(SEED)
    t0 = time.time()
    plan = {"E_RN": ["D3", "D2"], "E_RV": ["D3", "D2"], "E4_CNAT": ["D3", "D2"], "E4_CGT": ["D3", "D2"], "E4_INT": ["D3", "D2"],
            "F_RN": ["D2"], "F_CNAT": ["D2"], "F_INT": ["D2"]}
    res, infos = [], []
    for cid, designs in plan.items():
        logs, drives, info = build(cid)
        infos.append(info)
        for design in designs:
            for q in (0.5, 0.75, 0.25, 1.0):
                for coupling in ("U", "S"):
                    res += run(cid, design, q, coupling, logs, drives, info["theta_pop"], rng)
                    print(f"[{time.time()-t0:6.0f}s] {cid} {design} q={q} {coupling}", flush=True)
    suffix = "_quick" if "--quick" in sys.argv else ""
    pd.DataFrame(infos).to_csv(f"{OUT}/SIM_POPULATIONS{suffix}.csv", index=False)
    df = pd.DataFrame(res)
    df.to_csv(f"{OUT}/SIM_RESULTS{suffix}.csv", index=False)
    s = df.groupby(["contrast", "method"]).agg(min_cov=("coverage", "min"), min_wilson_hi=("mc_wilson95_hi", "min"), max_cov=("coverage", "max"),
                                               max_abs_bias=("bias", lambda x: float(np.max(np.abs(x)))), max_zero=("P_zero_width", "max"),
                                               max_fail=("failure_rate", "max")).reset_index()
    s.to_csv(f"{OUT}/SIM_SUMMARY{suffix}.csv", index=False)
    print(pd.DataFrame(infos).to_string()); print(s.round(4).to_string())


if __name__ == "__main__":
    main()
