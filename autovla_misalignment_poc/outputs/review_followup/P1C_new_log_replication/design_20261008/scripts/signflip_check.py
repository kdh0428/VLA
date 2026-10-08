#!/usr/bin/env python
"""Calibration of the prespecified primary test (drive-level sign-flip on scene-weighted totals) at the real
eval pool (24 drives). For each replicate: drive totals T_g = sum of (HT-weighted) scene effects, statistic
S = sum_g T_g / sum_g n_g; null distribution by flipping signs of T_g (2000 random flips; exact enumeration is
used in the real analysis). Reports type I and power at alpha .05 and at the Holm first step (.05/3)."""
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import power_sim as P

R = int(os.environ.get("REPS", 1000)); F = 2000
rng = np.random.default_rng(7)
out = []
for cl in ("drive", "segment"):
    for m in (4, 8):
        reps = [P.replicate(24, m, 0, cl) for _ in range(R)]
        for estd in ("F", "P"):
            for c in ("rn", "rv", "dm"):
                pv = {h: [] for h in ("H0", "MME", "dev")}
                for X in reps:
                    nF, fr, fd, NS, sr, sd, nu, ns = X.T
                    if c == "rv":
                        cF, cS = -P.RV_SCALE["A-"] * fr, -P.RV_SCALE["A+"] * sr
                    else:
                        cF, cS = (fr, sr) if c == "rn" else (fd, sd)
                    for h in pv:
                        if estd == "F":
                            mu = {"H0": 0.0, "MME": P.MME[("F", c)], "dev": P.DEVF[c]}[h]
                            T = cF + nF * mu; n = nF
                        else:
                            pd_ = (nF.sum() * P.DEVF[c] + NS.sum() * P.DEVS[c]) / (nF.sum() + NS.sum())
                            sc = {"H0": 0.0, "MME": P.MME[("P", c)] / pd_, "dev": 1.0}[h]
                            T = cF + cS + sc * (nF * P.DEVF[c] + NS * P.DEVS[c]); n = nF + NS
                        keep = n > 0; T = T[keep]
                        s0 = abs(T.sum())
                        flips = rng.choice([-1.0, 1.0], size=(F, len(T)))
                        pv[h].append((1 + np.sum(np.abs(flips @ T) >= s0 - 1e-12)) / (F + 1))
                row = dict(cluster_model=cl, aplus_per_log=m, estimand=estd, contrast=c, weighting="scene", reps=R)
                for h, v in pv.items():
                    v = np.array(v)
                    row[f"rej05_{h}"] = float(np.mean(v < 0.05)); row[f"rejHolm_{h}"] = float(np.mean(v < 0.05 / 3))
                out.append(row); print(row, flush=True)
import pandas as pd
pd.DataFrame(out).to_csv(os.path.join(P.D, "work", "signflip_calibration.csv"), index=False, float_format="%.4f")
