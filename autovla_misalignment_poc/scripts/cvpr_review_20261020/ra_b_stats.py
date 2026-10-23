#!/usr/bin/env python
"""B1-B3 (post_hoc/exploratory): component map check, per-component and leave-one-component-out effects,
exact sign-flip resolution with K = 6 and Holm, E/E4 weights and Kish ESS. CPU only.
"""
import itertools
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ra_common as R  # noqa: E402
sys.path.insert(0, f"{R.SCR}/review_followup/p1c")
import analyze_p1c as AP  # noqa: E402


def main():
    uc = pd.read_parquet(f"{R.B2}/p1c_analysis/unit_contrasts.parquet")
    uc = uc[(uc.n_old == 1) & uc.valid]
    cmap = pd.read_csv(f"{R.RF5}/A_B2_CLUSTER/COMPONENT_MAP.csv")
    comp = dict(zip(cmap.scene, cmap.component))
    # ---- B1 mapping check
    miss = sorted(set(uc.scene) - set(comp))
    m = uc[["scene", "drive", "day_vehicle", "stratum", "weight"]].drop_duplicates("scene")
    m["component"] = m.scene.map(comp)
    m["dv_consistent"] = m.day_vehicle == m.scene.map(dict(zip(cmap.scene, cmap.day_vehicle)))
    summ = m.groupby("component").agg(n_scenes=("scene", "size"), n_F=("stratum", lambda s: int((s == "F").sum())),
                                      n_drives=("drive", "nunique"), n_day_vehicle=("day_vehicle", "nunique"),
                                      sum_weight_E4=("weight", "sum")).reset_index()
    nunits = uc[uc.contrast == "C_GT"].assign(component=lambda d: d.scene.map(comp)).groupby(["component", "stratum"]).size().unstack(fill_value=0)
    summ = summ.merge(nunits.rename(columns={"F": "n_units_F", "S": "n_units_S"}), left_on="component", right_index=True)
    summ.to_csv(f"{R.OUT}/tables/B1_component_summary.csv", index=False)
    # ---- per component and LOCO for C_GT / C_NAT severe (scene-weighted ratio estimator as in B2)
    rows, loco = [], []
    for con in ("C_GT", "C_NAT"):
        d = uc[uc.contrast == con]
        sc = d.groupby(["scene", "stratum"]).agg(w=("weight", "first"), d=("severe", "mean"), n_units=("unit_id", "size")).reset_index()
        sc["component"] = sc.scene.map(comp)
        for which in ("F", "P"):
            s = sc if which == "P" else sc[sc.stratum == "F"]
            g = s.assign(N=s.w * s.d).groupby("component").agg(N=("N", "sum"), D=("w", "sum"), n_scenes=("scene", "size"))
            theta = g.N.sum() / g.D.sum()
            for k, r in g.iterrows():
                rows.append({"contrast": con, "estimand": "F" if which == "F" else "E4 (P)", "component": k,
                             "n_scenes": int(r.n_scenes), "N_k": r.N, "D_k": r.D, "theta_k_pp": 100 * r.N / r.D,
                             "share_of_numerator": r.N / g.N.sum()})
                rest = g.drop(k)
                loco.append({"contrast": con, "estimand": "F" if which == "F" else "E4 (P)", "left_out": k,
                             "theta_all_pp": 100 * theta, "theta_loco_pp": 100 * rest.N.sum() / rest.D.sum(),
                             "sign_kept": np.sign(rest.N.sum()) == np.sign(theta), "n_components_left": len(rest)})
    pd.DataFrame(rows).to_csv(f"{R.OUT}/tables/B1_component_effects.csv", index=False)
    pd.DataFrame(loco).to_csv(f"{R.OUT}/tables/B1_leave_one_component_out.csv", index=False)
    # ---- B2 enumeration
    enum = []
    for con in ("C_GT", "C_NAT"):
        for which, est in (("F", "F"), ("P", "E4 (P)")):
            T = pd.DataFrame(rows)
            Nk = T[(T.contrast == con) & (T.estimand == est)].N_k.values
            obs = abs(Nk.sum())
            sums = np.array([abs((np.array(sv) * Nk).sum()) for sv in itertools.product([1, -1], repeat=len(Nk))])
            p = float(np.mean(sums >= obs - 1e-12 * max(1, obs)))
            enum.append({"contrast": con, "estimand": est, "K": len(Nk), "n_sign_vectors": len(sums),
                         "p_enumerated": p, "p_stored_signflip_exact": AP.signflip_exact(Nk),
                         "min_attainable_p_two_sided": 2 / len(sums), "n_vectors_at_least_obs": int((sums >= obs - 1e-12 * max(1, obs)).sum())})
    E = pd.DataFrame(enum)
    st = pd.read_csv(f"{R.RF5}/A_B2_CLUSTER/B2_CLUSTER_RESULTS.csv")
    fam = st[(st.cluster_axis == "component") & st.family_member]
    holm = []
    for which in ("F", "P"):
        f = fam[fam.estimand == which]
        adj = AP.holm(f.p_signflip_exact.values)
        for (_, r), a in zip(f.iterrows(), adj):
            holm.append({"family": f"B2-{which} (m = 4) on component clusters", "contrast": r.contrast, "p_raw": r.p_signflip_exact,
                         "holm_recomputed": a, "holm_stored": r.holm_p, "alpha": 0.05 if which == "F" else 0.025,
                         "best_possible_holm_if_all_four_at_floor": min(1.0, 4 * 0.03125)})
    E.to_csv(f"{R.OUT}/tables/B2_signflip_enumeration.csv", index=False)
    pd.DataFrame(holm).to_csv(f"{R.OUT}/tables/B2_holm_component_family.csv", index=False)
    # attainable p grid for K = 6 (two-sided, generic distinct |N_k|): multiples of 1/32
    # ---- B3 weights
    cw = json.load(open(f"{R.B2}/p1c_cohort_t4.json"))["scene_weights"]
    W4 = pd.DataFrame([{"scene": k, **v} for k, v in cw.items()])
    U = pd.read_parquet(f"{R.P1C}/analysis/units.parquet")
    WE = U[["scene_id", "log_id", "stratum", "pi", "weight"]].drop_duplicates("scene_id")
    out = []
    for name, df, wcol, pcol in (("E (P1-C, 2,191 eligible)", WE, "weight", "pi"), ("E4 (B2, 1,775 eligible)", W4, "weight", "pi")):
        for st_ in ("F", "S", "all"):
            x = df if st_ == "all" else df[df.stratum == st_]
            w = x[wcol].astype(float)
            out.append({"frame": name, "stratum": st_, "n_sampled_scenes": len(x), "sum_weights": w.sum(),
                        "min_w": w.min(), "median_w": w.median(), "max_w": w.max(), "kish_ess": w.sum() ** 2 / (w ** 2).sum(),
                        "check_w_eq_1_over_pi": bool(np.allclose(w, 1 / x[pcol].astype(float))),
                        "note": "Kish ESS ignores drive/day-vehicle clustering; not an independent sample size"})
    pd.DataFrame(out).to_csv(f"{R.OUT}/tables/B3_weights_kish.csv", index=False)
    chk = {"scenes_missing_component": miss, "day_vehicle_consistent": bool(m.dv_consistent.all()),
           "n_components": int(m.component.nunique())}
    json.dump(chk, open(f"{R.OUT}/logs/B_checks.json", "w"), indent=1)
    print(chk); print(summ.to_string()); print(E.to_string()); print(pd.DataFrame(holm).to_string())
    print(pd.DataFrame(loco).to_string()); print(pd.DataFrame(out).to_string())


if __name__ == "__main__":
    main()
