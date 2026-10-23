#!/usr/bin/env python
"""review_followup5 C_DISJOINT power / precision plan (CPU). For a future replication on data whose day AND vehicle are
disjoint from dev + P1-C, the independent unit is the connected day-vehicle component. Inputs: the stored B2 P1-C
component-level contributions (no new outcome). For G new components, cluster effects are resampled from the observed
component-level effects of P1-C (shifted to a target effect), and the exact two-sided sign-flip test at 0 is applied;
power = P(Holm-adjusted p < alpha) in a family of 4 (worst case: the tested contrast has the largest p of the family is
not assumed; we report raw-p power and the Holm attainability floor).

  nice -n 19 /root/miniforge3/envs/autovla/bin/python c_power.py
"""
import itertools
import json
import os

import numpy as np
import pandas as pd

POC = "/root/VLA/autovla_misalignment_poc"
OUT = f"{POC}/outputs/review_followup5/C_DISJOINT"
CON = f"{POC}/outputs/review_followup5/A_B2_CLUSTER/B2_CLUSTER_CONTRIBUTIONS.csv"
UC = f"{POC}/outputs/review_followup4/B/B2/p1c_analysis/unit_contrasts.parquet"


def exact_p(T):
    G = len(T)
    obs = abs(T.sum())
    signs = np.array(list(itertools.product([-1, 1], repeat=G)))
    return float(np.mean(np.abs(signs @ T) >= obs - 1e-12))


def main():
    rng = np.random.default_rng(20261018)
    c = pd.read_csv(CON)
    rows = []
    for est, contrast in (("F", "C_NAT"), ("F", "C_GT"), ("P", "C_NAT")):
        # drive-level effects (more clusters than components -> better spread estimate); scaled to a component of 3 drives
        d = c[(c.contrast == contrast) & (c.estimand == est) & (c.cluster_axis == "drive")]
        eff = (d.N_g / d.D_g).values
        theta = d.N_g.sum() / d.D_g.sum()
        sd_drive = float(np.std(eff, ddof=1))
        for G in (6, 8, 10, 12, 16, 20):
            for target in (theta, theta / 2):
                for drives_per_comp in (1, 3):
                    sd = sd_drive / np.sqrt(drives_per_comp)
                    rej, ps = 0, []
                    R = 400 if G <= 16 else 200
                    for _ in range(R):
                        T = target + rng.normal(0, sd, G)
                        p = exact_p(T) if G <= 16 else float(np.mean([abs((rng.choice([-1, 1], G) * T).sum()) >= abs(T.sum()) for _ in range(4000)]))
                        ps.append(p)
                    ps = np.array(ps)
                    rows.append({"estimand": est, "contrast": contrast, "observed_theta": theta, "target_theta": target,
                                 "sd_drive_effect": sd_drive, "drives_per_component": drives_per_comp, "G_components": G,
                                 "p_floor": 2 / 2 ** G, "holm_floor_m4": min(1, 4 * 2 / 2 ** G),
                                 "power_raw_p_lt_0.05": float(np.mean(ps < 0.05)),
                                 "power_p_lt_0.0125 (Holm m4 worst case)": float(np.mean(ps < 0.0125)),
                                 "power_p_lt_0.00625 (P family alpha 0.025, Holm m4 worst case)": float(np.mean(ps < 0.00625))})
    df = pd.DataFrame(rows)
    df.to_csv(f"{OUT}/POWER_PRECISION_PLAN.csv", index=False)
    u = pd.read_parquet(UC)
    u = u[(u.contrast == "C_NAT") & (u.n_old == 1)]
    per_dv = u.groupby("day_vehicle").agg(F_scenes=("stratum", lambda s: int((s == "F").sum())), scenes=("scene", "nunique"))
    json.dump({"F_unit_rows_per_day_vehicle_median": float(per_dv.F_scenes.median()),
               "scenes_per_day_vehicle_median": float(per_dv.scenes.median())}, open(f"{OUT}/power_inputs.json", "w"), indent=1)
    print(df[(df.drives_per_component == 3)].round(3).to_string())


if __name__ == "__main__":
    main()
