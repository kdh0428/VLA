#!/usr/bin/env python
"""Addendum 4: paired PDMS (NAVSIM v1, non-reactive, official scorer, stored plans) for RN and B2 C_GT / C_NAT."""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ra_common as R  # noqa: E402

SUB = ["score", "no_at_fault_collisions", "drivable_area_compliance", "ego_progress", "time_to_collision_within_bound",
       "comfort", "driving_direction_compliance"]


def main():
    rows = []
    for line in open(f"{R.OUT}/data/pdms_raw.jsonl"):
        x = json.loads(line)
        for k, v in x.items():
            if k in ("token", "log"):
                continue
            src, rest = k.split("|", 1)
            unit, cond = rest.rsplit("|", 1)
            rows.append({"source": src, "scene_id": x["token"], "unit_id": unit, "cond": cond, **{s: v[s] for s in SUB}})
    P = pd.DataFrame(rows)
    U = pd.read_parquet(f"{R.P1C}/analysis/units.parquet")
    U = U[(U.block == "H7") & (U.condition == "normal")][["unit_id", "stratum", "weight", "drive_id"]]
    uc = pd.read_parquet(f"{R.B2}/p1c_analysis/unit_contrasts.parquet")
    uc = uc[(uc.n_old == 1) & (uc.contrast == "C_GT")][["unit_id", "stratum", "weight", "drive"]].rename(columns={"drive": "drive_id"})
    P1 = P[P.source == "P1C"].merge(U, on="unit_id", how="inner")
    P2 = P[P.source == "B2"].merge(uc, on="unit_id", how="inner")
    P1.to_csv(f"{R.OUT}/data/pdms_units_P1C.csv", index=False); P2.to_csv(f"{R.OUT}/data/pdms_units_B2.csv", index=False)
    out = []
    for con, a, b, df in (("RN", "recent_gt", "normal", P1), ("C_GT", "GT_NSRC", "GT_REF", P2), ("C_NAT", "NAT_NSRC", "NAT_REF", P2)):
        A = df[df.cond == a].set_index("unit_id"); B = df[df.cond == b].set_index("unit_id")
        idx = A.index.intersection(B.index)
        for frame, sel in (("F", A.loc[idx].stratum == "F"), ("E (P1-C)" if con == "RN" else "E4 (B2)", pd.Series(True, index=idx))):
            ii = idx[sel.values]
            D = (A.loc[ii, SUB] - B.loc[ii, SUB]); D["scene_id"] = A.loc[ii, "scene_id"]; D["w"] = A.loc[ii, "weight"] if frame != "F" else 1.0
            sc = D.groupby("scene_id").agg({**{s: "mean" for s in SUB}, "w": "first"})
            Am = A.loc[ii].groupby("scene_id")[SUB].mean(); Bm = B.loc[ii].groupby("scene_id")[SUB].mean()
            for s in SUB:
                out.append({"analysis_id": f"ADD4.PDMS.{con}.{frame}.{s}", "analysis_status": "post_hoc/exploratory",
                            "contrast": con, "intervention": a, "reference": b, "target_population": frame, "metric": f"PDMS:{s}",
                            "unit": "score in [0,1] (higher = better)", "horizon": "4 s non-reactive (NAVSIM v1 default)",
                            "n_complete": len(ii), "n_scenes": len(sc), "n_clusters": A.loc[ii, "drive_id"].nunique(),
                            "cluster_unit": "drive", "weighting_method": "equal scene" if frame == "F" else "Hajek (stored weights)",
                            "estimate_condition": float(np.average(Am[s], weights=sc.w)),
                            "estimate_reference": float(np.average(Bm[s], weights=sc.w)),
                            "delta_intervention_minus_reference": float(np.average(sc[s], weights=sc.w)),
                            "interval_status": "not computed (exploratory; existing interval policy covers severe only)",
                            "source_file": "data/pdms_units_*.csv"})
    T = pd.DataFrame(out)
    T.to_csv(f"{R.OUT}/tables/ADD4_pdms_paired.csv", index=False)
    pd.set_option("display.width", 250)
    print(T.pivot_table(index=["contrast", "target_population"], columns="metric", values="delta_intervention_minus_reference").round(4).to_string())
    print(T[T.metric == "PDMS:score"][["contrast", "target_population", "estimate_condition", "estimate_reference", "delta_intervention_minus_reference", "n_complete", "n_scenes"]].round(4).to_string())


if __name__ == "__main__":
    main()
