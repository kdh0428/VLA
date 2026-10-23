#!/usr/bin/env python
"""CPU-only re-check of vla0_work/pkg/paper/pdms_paired_cluster_t.csv from stored per-unit PDMS (no scoring rerun).
Cluster-robust t (G-1 df), drive clusters, ratio linearisation (supplement Sec. B formula)."""
import numpy as np, pandas as pd
from scipy import stats
D = "/root/VLA/autovla_misalignment_poc/outputs/cvpr_review_and_contribution_20261020/data"
def est(df, a, b, weighted):
    A = df[df.cond == a].set_index("unit_id"); B = df[df.cond == b].set_index("unit_id")
    u = A.index.intersection(B.index); d = (A.loc[u, "score"] - B.loc[u, "score"]).rename("d").to_frame()
    d["scene"] = A.loc[u, "scene_id"]; d["drive"] = A.loc[u, "drive_id"]; d["w"] = A.loc[u, "weight"] if weighted else 1.0
    s = d.groupby("scene").agg(d=("d", "mean"), w=("w", "first"), drive=("drive", "first"))
    th = (s.w * s.d).sum() / s.w.sum()
    g = s.assign(N=s.w * s.d).groupby("drive").agg(N=("N", "sum"), Dg=("w", "sum")); G = len(g)
    se = np.sqrt(G / (G - 1) * ((g.N - th * g.Dg) ** 2).sum()) / g.Dg.sum(); t = stats.t.ppf(0.975, G - 1)
    return th, th - t * se, th + t * se, len(s), G
P = pd.read_csv(f"{D}/pdms_units_P1C.csv"); B2 = pd.read_csv(f"{D}/pdms_units_B2.csv")
rows = [("RN", "F", est(P[P.stratum == "F"], "recent_gt", "normal", False)), ("RN", "E", est(P, "recent_gt", "normal", True)),
        ("C_GT", "F", est(B2[B2.stratum == "F"], "GT_NSRC", "GT_REF", False)), ("C_GT", "E4", est(B2, "GT_NSRC", "GT_REF", True)),
        ("C_NAT", "F", est(B2[B2.stratum == "F"], "NAT_NSRC", "NAT_REF", False)), ("C_NAT", "E4", est(B2, "NAT_NSRC", "NAT_REF", True))]
out = pd.DataFrame([{"contrast": c, "pop": p, "delta": r[0], "lo": r[1], "hi": r[2], "n_scenes": r[3], "G": r[4]} for c, p, r in rows])
out.to_csv("/root/VLA/v05_work/A_materials/A1_pdms_ci_recheck_20261023.csv", index=False); print(out.to_string())
