#!/usr/bin/env python
"""E2 (2026-10-23, CPU only): DESCRIPTIVE per-component effects and leave-one-component-out (LOCO) summary for the
B2 matched-prefix contrasts. NOT an external / independent validation: all six components come from the same P1-C
cohort (navtest logs), and they share days/vehicles with the development cohort (5 components over dev + P1-C).

Inputs (read-only):
  review_followup4/B/B2/p1c_analysis/unit_contrasts.parquet   (stored unit-level contrasts, sha256 checked)
  review_followup5/A_B2_CLUSTER/COMPONENT_MAP.csv             (fixed component map K1..K6; not redefined here)
  review_followup5/A_B2_CLUSTER/B2_CLUSTER_RESULTS.csv        (reference theta for the reproduction check)
Estimators: unchanged b2_analyze.scene_tab / int_tab and analyze_p1c.estimand_table / cluster_arrays
(scene-weighted ratio theta = sum_g N_g / sum_g D_g; primary anchor a = t*+3, n_old = 1, metric severe).
Estimands: F (failure stratum) and P (population, E4, per-scene weights w). Contrast families unchanged; no p-values are
computed (LOCO with G = 5 has a sign-flip floor of 2/2^5 = 0.0625, so it is reported as effect sizes only).

  PYTHONDONTWRITEBYTECODE=1 nice -n 19 /root/miniforge3/envs/autovla/bin/python e2_leave_component_out.py
"""
from __future__ import annotations

import hashlib
import json
import os
import sys

sys.dont_write_bytecode = True  # do not create __pycache__ next to the imported (read-only) analysis scripts
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, "/root/VLA/autovla_misalignment_poc/scripts/review_followup4/b")
import b2_analyze as BA  # noqa: E402
AP = BA.AP

POC = "/root/VLA/autovla_misalignment_poc"
UC = f"{POC}/outputs/review_followup4/B/B2/p1c_analysis/unit_contrasts.parquet"
UC_SHA = "c2dd2b65213b71a225d4d013042a757d680377e311560b45fa5cc1c927b0a35b"
CMAP = f"{POC}/outputs/review_followup5/A_B2_CLUSTER/COMPONENT_MAP.csv"
REF = f"{POC}/outputs/review_followup5/A_B2_CLUSTER/B2_CLUSTER_RESULTS.csv"
OUT = "/root/VLA/v05_work/autovla_E"
REQUESTED = ["C_NAT", "C_GT"]
SUPPLEMENTARY = ["INT", "C_PROV"]  # same B2 family members, shown for context only
LABEL = "DESCRIPTIVE ONLY - same-cohort re-analysis (P1-C navtest logs); NOT external or day/vehicle-independent validation"


def main():
    sha = hashlib.sha256(open(UC, "rb").read()).hexdigest()
    assert sha == UC_SHA, f"input hash changed: {sha}"
    df = pd.read_parquet(UC)
    cm = pd.read_csv(CMAP)
    dv2k = cm.drop_duplicates("day_vehicle").set_index("day_vehicle").component.to_dict()
    ref = pd.read_csv(REF)
    rows, summ, checks = [], [], []
    for cid in REQUESTED + SUPPLEMENTARY:
        sc = BA.int_tab(df, "severe", 1) if cid == "INT" else BA.scene_tab(df, cid, "severe", 1)
        sc["component"] = sc.day_vehicle.map(dv2k)
        assert sc.component.notna().all(), "scene without component"
        for which in ("F", "P"):
            s = AP.estimand_table(sc, which).copy()
            s["cluster"] = s["component"]
            labs, N, D = AP.cluster_arrays(s, "scene")
            theta = N.sum() / D.sum()
            r = ref[(ref.contrast == cid) & (ref.estimand == which) & (ref.cluster_axis == "component")]
            checks.append({"contrast": cid, "estimand": which, "theta": theta, "ref_theta": float(r.theta.iloc[0]),
                           "equal": bool(np.isclose(theta, r.theta.iloc[0], rtol=1e-12))})
            loo = []
            for i, k in enumerate(labs):
                sk = s[s.component == k]
                th_g = N[i] / D[i]
                th_loo = (N.sum() - N[i]) / (D.sum() - D[i])
                loo.append(th_loo)
                rows.append({"label": LABEL, "contrast": cid, "requested": cid in REQUESTED,
                             "estimand": which + (" (E4, per-scene weights)" if which == "P" else " (failure stratum)"),
                             "component": k, "day_vehicles": ";".join(sorted(sk.day_vehicle.unique())),
                             "n_scenes": int(len(sk)), "n_units": int(sk.n_units.sum()), "n_drives": int(sk.drive.nunique()),
                             "weight_D_g": D[i], "weight_share": D[i] / D.sum(),
                             "theta_component": th_g, "share_of_theta_numerator": N[i] / N.sum() if N.sum() else np.nan,
                             "theta_full": theta, "theta_leave_component_out": th_loo,
                             "loco_minus_full": th_loo - theta, "loco_over_full": th_loo / theta if theta else np.nan,
                             "loco_same_sign_as_full": bool(np.sign(th_loo) == np.sign(theta))})
            th_c = N / D
            summ.append({"contrast": cid, "requested": cid in REQUESTED, "estimand": which, "G_components": len(labs),
                         "theta_full": theta,
                         "theta_component_min": float(th_c.min()), "theta_component_max": float(th_c.max()),
                         "n_components_same_sign_as_full": int(np.sum(np.sign(th_c) == np.sign(theta))),
                         "loco_min": float(min(loo)), "loco_max": float(max(loo)),
                         "loco_min_dropped": labs[int(np.argmin(loo))], "loco_max_dropped": labs[int(np.argmax(loo))],
                         "loco_all_same_sign_as_full": bool(all(np.sign(x) == np.sign(theta) for x in loo)),
                         "max_abs_loco_change": float(max(abs(x - theta) for x in loo))})
    R = pd.DataFrame(rows)
    R.to_csv(f"{OUT}/leave_component_out_20261023.csv", index=False)
    S = pd.DataFrame(summ)
    S.to_csv(f"{OUT}/leave_component_out_summary_20261023.csv", index=False)
    meta = {"label": LABEL, "script": os.path.abspath(__file__), "input": UC, "input_sha256": sha, "component_map": CMAP,
            "component_map_sha256": hashlib.sha256(open(CMAP, "rb").read()).hexdigest(),
            "estimator": "scene-weighted ratio sum N_g / sum D_g (analyze_p1c.cluster_arrays, 'scene'); metric severe; n_old 1 (anchor t*+3)",
            "no_p_values": "LOCO leaves G = 5 components; exact sign-flip floor 0.0625 > 0.05; only effect sizes reported",
            "reproduction_check_vs_A_B2_CLUSTER": {"rows": len(checks), "all_theta_equal": all(c["equal"] for c in checks)},
            "summary": S.to_dict(orient="records")}
    json.dump(meta, open(f"{OUT}/leave_component_out_meta_20261023.json", "w"), indent=1, default=float)
    pd.set_option("display.width", 250)
    print(S.to_string())
    print(meta["reproduction_check_vs_A_B2_CLUSTER"])


if __name__ == "__main__":
    main()
