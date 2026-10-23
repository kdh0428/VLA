#!/usr/bin/env python
"""review_followup5 A_B2_CLUSTER (CPU, no GPU; protocol §2): B2 matched-prefix contrasts re-tested with drive, day-vehicle
and connected day-vehicle component clusters. Uses the stored unit-level contrasts written by b2_analyze.py and its
unchanged estimation functions (scene_tab / int_tab / est -> analyze_p1c.estimate).

Component map: bipartite graph (day nodes, vehicle nodes) over all 312 P1-C scenes of the B2 cohort; each scene links its
day and its vehicle. One map for F, S, P. A second map over dev + P1-C logs is described only.

  nice -n 19 /root/miniforge3/envs/autovla/bin/python a_b2_cluster.py
"""
from __future__ import annotations

import hashlib
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "/root/VLA/autovla_misalignment_poc/scripts/review_followup4/b")
import b2_analyze as BA  # noqa: E402
AP = BA.AP

POC = "/root/VLA/autovla_misalignment_poc"
B2 = f"{POC}/outputs/review_followup4/B/B2"
OUT = f"{POC}/outputs/review_followup5/A_B2_CLUSTER"
UC = f"{B2}/p1c_analysis/unit_contrasts.parquet"
ALPHA = {"F": 0.05, "P": 0.025}
PRIMARY = ["C_GT", "C_NAT", "INT", "C_PROV"]
SECOND = ["O_main_REF", "O_main_NSRC", "O_main_CSRC"]


def components(pairs):
    """pairs: iterable of (day, vehicle) -> dict day_vehicle -> component id (union-find)."""
    par = {}

    def find(x):
        par.setdefault(x, x)
        while par[x] != x:
            par[x] = par[par[x]]
            x = par[x]
        return x
    for d, v in pairs:
        a, b = find(("d", d)), find(("v", v))
        if a != b:
            par[a] = b
    roots = {}
    out = {}
    for d, v in sorted(set(pairs)):
        r = find(("d", d))
        roots.setdefault(r, f"K{len(roots) + 1}")
        out[f"{d}_{v}"] = roots[r]
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    df = pd.read_parquet(UC)
    dv = df[["scene", "day_vehicle"]].drop_duplicates()
    pairs = [tuple(x.split("_", 1)) for x in dv.day_vehicle.unique()]
    cmap = components(pairs)
    df["component"] = df.day_vehicle.map(cmap)
    # map file
    mp = dv.assign(day=dv.day_vehicle.str.split("_").str[0], vehicle=dv.day_vehicle.str.split("_").str[1],
                   component=dv.day_vehicle.map(cmap))
    st = df[["scene", "stratum"]].drop_duplicates().set_index("scene").stratum
    mp["stratum"] = mp.scene.map(st)
    mp.sort_values(["component", "day_vehicle", "scene"]).to_csv(f"{OUT}/COMPONENT_MAP.csv", index=False)
    summ = mp.groupby("component").agg(day_vehicles=("day_vehicle", lambda s: ";".join(sorted(set(s)))), n_scenes=("scene", "size"),
                                       n_F=("stratum", lambda s: int((s == "F").sum())), n_S=("stratum", lambda s: int((s == "S").sum()))).reset_index()
    summ.to_csv(f"{OUT}/COMPONENT_SUMMARY.csv", index=False)
    rows, contrib = [], []
    for clus in ("drive", "day_vehicle", "component"):
        for cid in PRIMARY + SECOND:
            sc = BA.int_tab(df, "severe", 1) if cid == "INT" else BA.scene_tab(df, cid, "severe", 1)
            sc["component"] = sc.day_vehicle.map(cmap)
            for which in ("F", "P"):
                e = BA.est(sc, which, "p1c", clus)
                s = AP.estimand_table(sc, which).copy()
                s["cluster"] = s[clus]
                labs, N, D = AP.cluster_arrays(s, "scene")
                g_inf = int(np.sum(np.abs(N) > 1e-12))
                floor = 2.0 / 2 ** g_inf if g_inf else 1.0
                boot = e.get("boot_ci95") if clus != "component" else None
                rows.append({"contrast": cid, "family_member": cid in PRIMARY, "estimand": which, "cluster_axis": clus,
                             "theta": e["theta"], "n_scenes": e["n_scenes"], "n_units": e["n_units"], "G": e["n_clusters"],
                             "G_informative": g_inf, "p_signflip_exact": e["p_signflip"], "p_floor": floor,
                             "p_cluster_t": e["p_cluster_t"], "boot_ci95": None if boot is None else list(boot),
                             "boot_ci95_note": "NR (too few clusters for a bootstrap CI)" if clus == "component" else ""})
                for lab, n, d in zip(labs, N, D):
                    contrib.append({"contrast": cid, "estimand": which, "cluster_axis": clus, "cluster": lab,
                                    "N_g": n, "D_g": d, "share_of_theta_numerator": n / D.sum()})
    R = pd.DataFrame(rows)
    R["holm_p"] = np.nan; R["family"] = ""; R["alpha"] = np.nan; R["reject_H0"] = None
    for clus in ("drive", "day_vehicle", "component"):
        for which in ("F", "P"):
            m = (R.cluster_axis == clus) & (R.estimand == which) & R.family_member
            R.loc[m, "holm_p"] = AP.holm(R.loc[m, "p_signflip_exact"].values)
            R.loc[m, "family"] = f"B2-{which} (m = 4) on {clus} clusters"
            R.loc[m, "alpha"] = ALPHA[which]
            R.loc[m, "reject_H0"] = R.loc[m, "holm_p"] < ALPHA[which]
    R.to_csv(f"{OUT}/B2_CLUSTER_RESULTS.csv", index=False)
    pd.DataFrame(contrib).to_csv(f"{OUT}/B2_CLUSTER_CONTRIBUTIONS.csv", index=False)
    # check against the stored B2 drive / day-vehicle rows (must match exactly)
    old = pd.read_csv(f"{B2}/p1c_analysis/MATCHED_PREFIX_RESULTS.csv")
    old = old[(old.n_old == 1) & (old.metric == "severe") & (old.subset == "all")]
    chk = []
    for _, r in R[R.cluster_axis != "component"].iterrows():
        o = old[(old.contrast == r.contrast) & (old.estimand == r.estimand) & (old.cluster == r.cluster_axis)]
        if len(o):
            chk.append({"contrast": r.contrast, "estimand": r.estimand, "cluster": r.cluster_axis,
                        "theta_equal": bool(np.isclose(o.theta.iloc[0], r.theta, rtol=1e-12)),
                        "p_equal": bool(np.isclose(o.p_signflip.iloc[0], r.p_signflip_exact, rtol=1e-12))})
    # dev + P1-C combined graph (description only)
    dev = [json.loads(l) for l in open(f"{POC}/outputs/equal_distance_perturbation/records.jsonl")]
    dvd = {BA.dayveh_of(r["log"]) for r in dev}
    allp = [tuple(x.split("_", 1)) for x in set(dv.day_vehicle) | dvd]
    cm2 = components(allp)
    comb = {"n_day_vehicles_p1c": len(set(dv.day_vehicle)), "n_day_vehicles_dev": len(dvd),
            "shared_day_vehicles": len(set(dv.day_vehicle) & dvd), "n_components_p1c": len(set(cmap.values())),
            "n_components_dev_plus_p1c": len(set(cm2.values())),
            "components_dev_plus_p1c": {k: sorted(x for x, c in cm2.items() if c == k) for k in sorted(set(cm2.values()))}}
    meta = {"input": UC, "input_sha256": hashlib.sha256(open(UC, "rb").read()).hexdigest(), "n_components": len(set(cmap.values())),
            "check_vs_stored_B2": {"rows": len(chk), "all_theta_equal": all(c["theta_equal"] for c in chk),
                                   "all_p_equal": all(c["p_equal"] for c in chk)}, "combined_graph": comb}
    json.dump(meta, open(f"{OUT}/b2_cluster_meta.json", "w"), indent=1)
    show = R[R.family_member][["contrast", "estimand", "cluster_axis", "theta", "G", "G_informative", "p_signflip_exact", "p_floor", "holm_p", "reject_H0"]]
    print(show.to_string())
    print(json.dumps({k: v for k, v in meta.items() if k != "combined_graph"}, indent=1), comb["n_components_dev_plus_p1c"], comb["shared_day_vehicles"])


if __name__ == "__main__":
    main()
