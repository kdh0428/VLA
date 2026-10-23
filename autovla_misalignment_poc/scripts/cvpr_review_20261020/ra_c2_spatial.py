#!/usr/bin/env python
"""C2 (post_hoc): SpatialVLA P1-D per-task success, paired 2x2 counts, task-equal vs pooled, physical-state level."""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ra_common as R  # noqa: E402

VE = f"{R.POC}/outputs/cvpr_visual_evidence_20261019"


def main():
    E = pd.read_csv(f"{VE}/SPATIAL_EPISODES.csv")
    E = E[E.eligible & E.arm.isin(["N", "F", "C", "R"])]
    rows, phys = [], []
    for cfg in ("r1e1", "r2e1", "r4e1", "r1e4"):
        for con, (a, b) in (("C-F", ("C", "F")), ("R-N", ("R", "N"))):
            A = E[(E.config == cfg) & (E.arm == a)].set_index(["task", "seed"])
            B = E[(E.config == cfg) & (E.arm == b)].set_index(["task", "seed"])
            idx = A.index.intersection(B.index)
            per = {}
            for task in ("google_robot_pick_coke_can", "google_robot_move_near"):
                ii = [i for i in idx if i[0] == task]
                ya, yb = A.loc[ii, "success"].astype(int).values, B.loc[ii, "success"].astype(int).values
                n11 = int(((ya == 1) & (yb == 1)).sum()); n00 = int(((ya == 0) & (yb == 0)).sum())
                n10 = int(((ya == 1) & (yb == 0)).sum()); n01 = int(((ya == 0) & (yb == 1)).sum())
                per[task] = ya.mean() - yb.mean()
                rows.append({"config": cfg, "contrast": con, "intervention": a, "reference": b, "task": task, "n_pairs": len(ii),
                             "success_intervention": int(ya.sum()), "success_reference": int(yb.sum()),
                             "rate_intervention": ya.mean(), "rate_reference": yb.mean(),
                             "delta_intervention_minus_reference_pp": 100 * (ya.mean() - yb.mean()),
                             "n11_both_success": n11, "n10_only_intervention": n10, "n01_only_reference": n01, "n00_both_fail": n00,
                             "n_physical_states": int(A.loc[ii, "physical_state_id"].nunique())})
                st = pd.DataFrame({"state": A.loc[ii, "physical_state_id"].values, "d": ya - yb})
                for s, g in st.groupby("state"):
                    phys.append({"config": cfg, "contrast": con, "task": task, "physical_state_id": s, "n_units": len(g),
                                 "delta_pp": 100 * g.d.mean()})
            ya = A.loc[idx, "success"].astype(int).values; yb = B.loc[idx, "success"].astype(int).values
            rows.append({"config": cfg, "contrast": con, "intervention": a, "reference": b, "task": "episode-pooled (both tasks)",
                         "n_pairs": len(idx), "delta_intervention_minus_reference_pp": 100 * (ya.mean() - yb.mean())})
            rows.append({"config": cfg, "contrast": con, "intervention": a, "reference": b, "task": "task-equal mean",
                         "n_pairs": len(idx), "delta_intervention_minus_reference_pp": 100 * np.mean(list(per.values())),
                         "note": "equal to pooled here because each task has 120 pairs"})
    T = pd.DataFrame(rows)
    T.to_csv(f"{R.OUT}/tables/C2_spatialvla_task_success.csv", index=False)
    P = pd.DataFrame(phys)
    P.to_csv(f"{R.OUT}/tables/C2_spatialvla_physical_state.csv", index=False)
    f = pd.read_csv(f"{R.AQ}/FINAL_PAPER_NUMBERS.csv")
    s = f[f.claim_id.str.startswith("SVLA")]
    s.to_csv(f"{R.OUT}/tables/C2_existing_equivalence_rows.csv", index=False)
    pd.set_option("display.width", 250)
    print(T[T.config.isin(["r4e1", "r1e4"])][["config", "contrast", "task", "n_pairs", "success_intervention", "success_reference",
                                              "delta_intervention_minus_reference_pp", "n11_both_success", "n10_only_intervention",
                                              "n01_only_reference", "n00_both_fail", "n_physical_states"]].to_string())
    print(s[["claim_id", "estimate", "ci_low", "ci_high", "ci_method", "ci_level", "test", "p_raw", "n_units", "status"]].to_string())


if __name__ == "__main__":
    main()
