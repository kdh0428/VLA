#!/usr/bin/env python
"""Author query Q1, ADDED AFTER THE PLAN (amendment QA-1, written after q1_coverage.py showed that the observed-scale targets
cannot be reached by a constant shift under the observed discordance): 'empirical alternative' scenario in which every
scene keeps its observed q+ / q- (true theta = the observed estimate exactly), plus the same with drive heterogeneity
(a per-drive shift eta ~ N(0, 0.05^2) moved within the scene's discordance). Same cluster-t evaluation, R = 5,000.

  nice -n 19 /root/miniforge3/envs/autovla/bin/python q1_empirical_alt.py
"""
import math
import numpy as np
import pandas as pd
import q1_coverage as Q

def main():
    rng = np.random.default_rng(Q.SEED + 1)
    rows = []
    for cid, (df, ref, coh) in Q.structures().items():
        sc = Q.scene_table(df)
        labs, inv = np.unique(sc.drive.values, return_inverse=True)
        G, n, S = len(labs), sc.n.values.astype(int), len(sc)
        for tau in (0.0, 0.05):
            Dd = (sc.qp + sc.qm).values[None, :]
            base = (sc.qp - sc.qm).values[None, :]
            eta = rng.normal(0, tau, (Q.R, G))[:, inv] if tau else 0.0
            ds = np.clip(base + eta, -Dd, Dd)
            qp, qm = (Dd + ds) / 2, (Dd - ds) / 2
            theta = (qp - qm).mean(1)
            cp = rng.binomial(np.broadcast_to(n, (Q.R, S)), qp)
            with np.errstate(invalid="ignore", divide="ignore"):
                cond = np.where(qp < 1, qm / (1 - qp), 0.0)
            cm = rng.binomial(n[None, :] - cp, np.clip(cond, 0, 1))
            dmean = (cp - cm) / n[None, :]
            Nmat = np.stack([dmean[:, inv == g].sum(1) for g in range(G)], 1)
            Dg = np.bincount(inv, minlength=G).astype(float)
            for lev in (0.95, 0.90):
                lo, hi, se = Q.cluster_t(Nmat, Dg, lev)
                c = int(((lo <= theta) & (theta <= hi)).sum())
                wl, wh = Q.wilson(c, Q.R)
                rows.append({"scenario_id": f"{cid}|EMP|tau={tau}", "contrast_id": cid, "model": "EMP (observed cells; amendment QA-1)",
                             "target_delta": float(sc.dmean.mean()), "realised_theta_mean": float(theta.mean()), "tau": tau,
                             "method": f"cluster_t_{int(lev*100)}", "reps": Q.R, "coverage": c / Q.R,
                             "mc_se": math.sqrt((c / Q.R) * (1 - c / Q.R) / Q.R), "mc_wilson95_lo": wl, "mc_wilson95_hi": wh,
                             "P_zero_width": float(np.mean(se == 0)), "mean_width": float(np.mean(hi - lo))})
    d = pd.DataFrame(rows)
    d.to_csv(f"{Q.OUT}/COVERAGE_RESULTS_EMPIRICAL_ALT.csv", index=False)
    print(d[["scenario_id", "method", "realised_theta_mean", "coverage", "mc_wilson95_lo", "mc_wilson95_hi"]].round(4).to_string())

if __name__ == "__main__":
    main()
