#!/usr/bin/env python
"""Author query Q2 (2026-10-19): power sensitivity to within-component correlation of drive effects (planning only, CPU).
Component mean effect ~ N(target, sd_drive^2 * (1 + (m-1) rho) / m): equal drive variance, equal weights, exchangeable
correlation rho within a component; components independent and identically distributed. sd_drive = SD of the observed
P1-C failure-stratum drive effects (B2_CLUSTER_CONTRIBUTIONS, drive axis; includes sampling noise of small drives).
Test: exact two-sided sign-flip at 0 over all 2^G patterns (G = 10 -> 1,024 patterns, enumerated once and reused).
Power criteria: raw p < 0.05 and raw p < 0.0125 (= alpha / m for the 4-contrast F family; a conservative stand-in, NOT a joint
simulation of the Holm procedure). R = 4,000 replicates, seed 20261019.

  nice -n 19 /root/miniforge3/envs/autovla/bin/python q2_power.py
"""
import itertools, math
import numpy as np, pandas as pd

POC = "/root/VLA/autovla_misalignment_poc"
OUT = f"{POC}/outputs/manuscript_author_queries_20261019"
CON = f"{POC}/outputs/review_followup5/A_B2_CLUSTER/B2_CLUSTER_CONTRIBUTIONS.csv"
R, G, M = 4000, 10, 3

def main():
    rng = np.random.default_rng(20261019)
    signs = np.array(list(itertools.product([-1, 1], repeat=G)), float)       # (1024, G)
    c = pd.read_csv(CON)
    rows = []
    for contrast in ("C_NAT", "C_GT"):
        d = c[(c.contrast == contrast) & (c.estimand == "F") & (c.cluster_axis == "drive")]
        eff = (d.N_g / d.D_g).values
        theta = float(d.N_g.sum() / d.D_g.sum())
        sd = float(np.std(eff, ddof=1))
        for frac in (0.5, 1.0):
            for rho in (0.0, 0.25, 0.5, 1.0):
                sdc = sd * math.sqrt((1 + (M - 1) * rho) / M)
                T = theta * frac + rng.normal(0, sdc, (R, G))
                obs = np.abs(T.sum(1))
                p = (np.abs(T @ signs.T) >= obs[:, None] - 1e-12).mean(1)
                for crit in (0.05, 0.0125):
                    pw = float(np.mean(p < crit))
                    rows.append({"contrast": contrast, "estimand": "F", "observed_theta": theta, "effect_fraction": frac,
                                 "target_theta": theta * frac, "sd_drive_effect": sd, "drives_per_component_m": M, "rho_component": rho,
                                 "sd_component_mean": sdc, "G_components": G, "exact_p_floor": 2 / 2 ** G, "criterion": f"raw p < {crit}",
                                 "criterion_note": "alpha 0.05 single test" if crit == 0.05 else "alpha/m = 0.05/4 (conservative stand-in for Holm; not a joint Holm simulation)",
                                 "power": pw, "mc_se": math.sqrt(pw * (1 - pw) / R), "reps": R})
    d = pd.DataFrame(rows)
    d.to_csv(f"{OUT}/POWER_SENSITIVITY.csv", index=False)
    print(d[d.criterion == "raw p < 0.0125"].pivot_table(index=["contrast", "effect_fraction"], columns="rho_component", values="power").round(3))

if __name__ == "__main__":
    main()
