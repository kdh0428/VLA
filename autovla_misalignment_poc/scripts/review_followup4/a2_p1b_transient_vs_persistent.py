#!/usr/bin/env python
"""
A2 supplement (review_followup4, POST-HOC, CPU only, read-only inputs).

P1-B already contains a schedule in which GT is NOT retained: T_gt_s1_w3 (GT at context position t*+o only while it
is the most recent position, o = 1..3; afterwards that position reverts to the model's own token, so from step
t*+5 on no GT is in the context). P_gt_s1_w3 is the exp 9 persistent rule (positions t*+1..t*+3 hold GT at every
later step). Same run, same units, same batch (B = 53), same seeds.

This script reports, for A-, both subsets (common t* <= 2 = predeclared P1-B support; all units):
  scene-weighted levels, fraction of the Normal -> Full effect f = sum_s(N-X)/sum_s(N-F), and the paired
  difference T - P (amplification pp, FDE5 m) with a drive-cluster bootstrap (joint resample of num/den).
It is a descriptive post-hoc comparison of two existing rows; T and P also differ in GT exposure count
(T: each corrected position seen in 1 forward; P: in every later forward), so T - P is not a pure "release" effect.
"""
import json
import os

import numpy as np
import pandas as pd

POC = "/root/VLA/autovla_misalignment_poc"
SRC = os.path.join(POC, "outputs/review_followup/P1B_start_position/run_20261008/units.parquet")
OUT = os.path.join(POC, "outputs/review_followup4/A2_A3")
SEED, NB = 20261018, 10000
CONDS = ["normal", "full_gt", "P_gt_s1_w3", "T_gt_s1_w3", "recent_gt"]


def main():
    u = pd.read_parquet(SRC, columns=["unit_id", "scene_id", "log_id", "group", "t_star", "condition", "amplification", "fde5",
                                      "corrected_pairs", "corrected_pairs_effective", "corrected_positions_n"])
    u = u[(u.group == "A-") & u.condition.isin(CONDS)].copy()
    u["drive"] = u.log_id.map(lambda l: "_".join(l.split("_")[:2]))
    rng = np.random.default_rng(SEED)
    res = {}
    for sub, mask in (("common_tstar_le2", u.t_star <= 2), ("all_units", u.t_star >= 0)):
        d = u[mask]
        W = d.pivot_table(index=["unit_id", "scene_id", "drive"], columns="condition", values=["amplification", "fde5"])
        S = W.groupby(level=["scene_id", "drive"]).mean()          # scene means first, scenes equal
        drives = sorted(S.index.get_level_values("drive").unique())
        di = {x: i for i, x in enumerate(drives)}
        sd = np.array([di[x] for x in S.index.get_level_values("drive")])
        R = rng.integers(0, len(drives), size=(NB, len(drives)))
        cnt = np.stack([np.bincount(r, minlength=len(drives)) for r in R])      # drive multiplicity per replicate
        wts = cnt[:, sd]                                                           # scene weight per replicate
        out = {"n_units": int(W.shape[0]), "n_scenes": int(S.shape[0]), "n_drives": len(drives)}
        for m in ("amplification", "fde5"):
            X = {c: S[(m, c)].to_numpy() for c in CONDS}
            out[f"{m}_levels"] = {c: float(X[c].mean()) for c in CONDS}
            den = (X["normal"] - X["full_gt"])
            for c in ("P_gt_s1_w3", "T_gt_s1_w3", "recent_gt"):
                num = X["normal"] - X[c]
                bs = (wts @ num) / (wts @ den)
                out[f"{m}_fraction_{c}"] = {"est": float(num.sum() / den.sum()),
                                            "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]}
            diff = X["T_gt_s1_w3"] - X["P_gt_s1_w3"]
            bs = (wts @ diff) / wts.sum(1)
            out[f"{m}_T_minus_P"] = {"est": float(diff.mean()), "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                                     "unit": "pp/100" if m == "amplification" else "m"}
            if m == "amplification":
                wu = W[("amplification", "T_gt_s1_w3")].to_numpy(), W[("amplification", "P_gt_s1_w3")].to_numpy()
                out["amplification_unit_discordant"] = {"T_only": int(((wu[0] == 1) & (wu[1] == 0)).sum()),
                                                        "P_only": int(((wu[0] == 0) & (wu[1] == 1)).sum())}
        ex = d[d.condition.isin(["P_gt_s1_w3", "T_gt_s1_w3"])].groupby("condition")[["corrected_positions_n", "corrected_pairs", "corrected_pairs_effective"]].mean()
        out["exposure_mean_per_unit"] = ex.to_dict(orient="index")
        res[sub] = out
    res["note"] = ("POST-HOC descriptive; scene-weighted, drive-cluster bootstrap (10,000, joint num/den); "
                   "T and P differ in exposure (pairs) as well as retention.")
    json.dump(res, open(os.path.join(OUT, "a2_p1b_transient_vs_persistent.json"), "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
