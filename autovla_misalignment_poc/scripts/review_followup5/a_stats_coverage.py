#!/usr/bin/env python
"""review_followup5 A_STATS step 1: coverage study on the REAL AutoVLA cluster structures (CPU; written and run before
any affected AutoVLA row is recomputed; protocol §1 rule 3). Uses only the natural / Normal-arm event rates per scene.

Cohorts (structure = scenes, units per scene, scene weights, cluster labels):
  P1C_F      P1-C failure stratum, H7 Normal rows, drive clusters (18)
  P1C_P      P1-C all sampled scenes with design weights, drive clusters (24)
  P1C_P_DV   same, day-vehicle clusters (16)
  DEV_Am     development A- stratum (exp 7 Normal), drive clusters (13)
  SUB7       P1-C F restricted to the first 7 drives (sorted id): the size of the dependence / disjoint subsets (G = 6-7)
Paired binary unit outcomes: a ~ Bern(p0_s) (Normal-arm scene rate), b = latent-coupled Bern(p1_s), p1_s = clip(p0_s + delta
+ eta_g), coupling rho = 0.7 (same uniform draw with prob rho), eta_g ~ N(0, tau). True theta = scene-weighted mean of
(p1_s - p0_s). Methods: drive percentile bootstrap (B = 999 in the simulation; 10,000 in the real rows), cluster-robust t
(G - 1 df), and the old sign-flip inversion (reference; fewer replicates). R = 400 replicates per scenario, seed 20261018.

Selection (fixed in the protocol): exclude P(zero width) > 0 under true discordance > 0; highest minimum coverage, required
>= 0.925 (95%) / >= 0.875 (90%); else NR for that cohort.

  nice -n 19 /root/miniforge3/envs/autovla/bin/python a_stats_coverage.py
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, "/root/VLA/autovla_misalignment_poc/scripts/review_followup2")
import stats_core as SC  # noqa: E402

POC = "/root/VLA/autovla_misalignment_poc"
OUT = f"{POC}/outputs/review_followup5/A_STATS"
P1C_UNITS = f"{POC}/outputs/review_followup/P1C_new_log_replication/run_20261008/analysis/units.parquet"
DEV_UNITS = f"{POC}/outputs/review_followup/P0B_cluster_stats/run_20261008/units.parquet"
R, R_SF, BB, SEED = 400, 60, 999, 20261018


def structures():
    u = pd.read_parquet(P1C_UNITS)
    n = u[(u.block == "H7") & (u.condition == "normal") & u.exclusion_reason.isna()]
    g = n.groupby("scene_id").agg(p0=("amplification", "mean"), n=("amplification", "size"), w=("weight", "first"),
                                  drive=("drive_id", "first"), dv=("day_vehicle", "first"), stratum=("stratum", "first")).reset_index()
    out = {}
    f = g[g.stratum == "F"]
    out["P1C_F"] = f.assign(cl=f.drive, w=1.0)
    out["P1C_P"] = g.assign(cl=g.drive)
    out["P1C_P_DV"] = g.assign(cl=g.dv)
    d7 = sorted(f.drive.unique())[:7]
    out["SUB7"] = f[f.drive.isin(d7)].assign(cl=lambda x: x.drive, w=1.0)
    d = pd.read_parquet(DEV_UNITS)
    d = d[(d.experiment.astype(str).str.contains("action_history")) & (d.condition == "normal") & (d.group == "A-")]
    if len(d) == 0:
        d = pd.read_parquet(DEV_UNITS)
        d = d[(d.condition == "normal") & (d.group == "A-")]
    d["drive"] = d.log_id.map(lambda s: "_".join(s.split("_")[:2]))
    gd = d.groupby("scene_id").agg(p0=("amplification", "mean"), n=("amplification", "size"), drive=("drive", "first")).reset_index()
    out["DEV_Am"] = gd.assign(cl=gd.drive, w=1.0)
    return out


def simulate(st, delta, tau, rng):
    p0 = st.p0.values.astype(float)
    cls = st.cl.values
    lab, inv = np.unique(cls, return_inverse=True)
    eta = rng.normal(0, tau, len(lab))[inv] if tau > 0 else 0.0
    p1 = np.clip(p0 + delta + eta, 0, 1)
    w = st.w.values.astype(float)
    theta = float(np.sum(w * (p1 - p0)) / np.sum(w))
    N = np.zeros(len(lab)); D = np.zeros(len(lab))
    for s in range(len(st)):
        k = int(st.n.values[s])
        u1 = rng.random(k)
        u2 = np.where(rng.random(k) < 0.7, u1, rng.random(k))
        dm = float(np.mean((u2 < p1[s]).astype(float) - (u1 < p0[s]).astype(float)))
        N[inv[s]] += w[s] * dm; D[inv[s]] += w[s]
    return theta, N, D


def main():
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    sts = structures()
    scen = {"P1C_F": [0.0, 0.1, 0.3], "SUB7": [0.0, 0.1, 0.3], "DEV_Am": [0.0, 0.1, 0.4],
            "P1C_P": [0.0, 0.02, 0.04], "P1C_P_DV": [0.0, 0.02, 0.04]}
    rows = []
    for coh, st in sts.items():
        G = st.cl.nunique()
        for delta in scen[coh]:
            for tau in (0.0, 0.05 if delta >= 0.1 else 0.01):
                cov = {m: [] for m in ("boot95", "boot90", "t95", "t90", "sf95", "sf90")}
                zw = {m: 0 for m in cov}
                disc = 0
                for r in range(R):
                    th, N, D = simulate(st, delta, tau, rng)
                    disc += int(np.any(N != 0))
                    bt = SC.cluster_bootstrap(N, D, B=BB, seed=int(rng.integers(1 << 30)))
                    for lev, key in ((0.95, "95"), (0.90, "90")):
                        lo, hi = SC.pct_ci(bt, lev)
                        cov["boot" + key].append(lo <= th <= hi); zw["boot" + key] += int(hi - lo < 1e-9)
                        _, (lo, hi) = SC.cluster_t(N, D, lev)
                        cov["t" + key].append(lo <= th <= hi); zw["t" + key] += int(hi - lo < 1e-9)
                        if r < R_SF:
                            lo, hi = SC.signflip_ci(N, D, SC.ratio(N, D), level=lev)
                            cov["sf" + key].append(lo <= th <= hi); zw["sf" + key] += int(hi - lo < 1e-9)
                for m, v in cov.items():
                    rows.append({"cohort": coh, "G": G, "n_scenes": len(st), "delta": delta, "tau": tau, "method": m,
                                 "reps": len(v), "coverage": float(np.mean(v)) if v else np.nan, "P_zero_width": zw[m] / max(len(v), 1),
                                 "share_reps_with_discordance": disc / R})
                print(f"[{time.time()-t0:6.0f}s] {coh} delta={delta} tau={tau}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(f"{OUT}/A_COVERAGE_SIM.csv", index=False)
    sel = {}
    for coh in sts:
        sel[coh] = {}
        for lev, need in (("95", 0.925), ("90", 0.875)):
            cand = {}
            for m in ("boot", "t"):
                x = df[(df.cohort == coh) & (df.method == m + lev)]
                cand[m] = {"min_coverage": float(x.coverage.min()), "mean_abs_dev": float((x.coverage - float(lev) / 100).abs().mean()),
                           "max_P_zero_width": float(x.P_zero_width.max())}
            ok = {m: c for m, c in cand.items() if c["max_P_zero_width"] == 0 and c["min_coverage"] >= need}
            pick = (max(ok, key=lambda m: (round(ok[m]["min_coverage"], 3), -ok[m]["mean_abs_dev"])) if ok else "NR")
            sf = df[(df.cohort == coh) & (df.method == "sf" + lev)]
            sel[coh][lev] = {"candidates": cand, "selected": pick, "required": need,
                             "old_signflip_inversion_reference": {"min_coverage": float(sf.coverage.min()),
                                                                  "max_P_zero_width": float(sf.P_zero_width.max())}}
    json.dump({"rule": __doc__, "selection": sel, "runtime_s": time.time() - t0}, open(f"{OUT}/A_METHOD_SELECTION.json", "w"), indent=1)
    print(json.dumps({c: {l: v["selected"] for l, v in s.items()} for c, s in sel.items()}, indent=1))


if __name__ == "__main__":
    main()
