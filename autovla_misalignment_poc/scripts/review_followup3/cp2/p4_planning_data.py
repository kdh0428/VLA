#!/usr/bin/env python
"""P4 planning data (CP2, 2026-10-14). Existing SpatialVLA data used ONLY as planning input; no hypothesis is tested.

Regenerated from scratch for CP2 (the stopped v1 attempt in outputs/review_followup3/_stopped_v1_20261014/P4/ was
reviewed; same source table and contrast definitions, plus the per-layout within-state structure for move_near).

Source: outputs/review_followup2/B_spatialvla_direct/run_20261013/episodes.csv (exp 35 and P1-D, SimplerEnv Google
Robot, closed loop, success at step 80). Contrasts per unit: dR = S(R) - S(N), dC = S(C) - S(F), for r1e4 and r4e1.

Writes outputs/review_followup3/P4/planning/
  planning_params.csv     experiment x task x config x contrast: n units, primary/physical states, mean, P(+1), P(-1),
                          discordance p_d, SD of the paired difference
  planning_corr.json      experiment x task: correlation of the 4 paired differences across units (joint structure)
  planning_within_state.csv  move_near: between-layout SD and ICC(1) of the paired differences (units nested in the
                          10 physical layouts) = within-state correlation; coke has 1 unit per physical state (ICC n/a)
  planning_units_p1d.npz  P1-D unit x 8-outcome matrix per task (N,F,C,R x r1e4,r4e1) + layout ids

  source /root/miniforge3/etc/profile.d/conda.sh && conda activate autovla
  nice -n 19 python p4_planning_data.py
"""
import json
import os

import numpy as np
import pandas as pd

ROOT = "/root/VLA/autovla_misalignment_poc"
EPIS = f"{ROOT}/outputs/review_followup2/B_spatialvla_direct/run_20261013/episodes.csv"
OUT = f"{ROOT}/outputs/review_followup3/P4/planning"
CFGS = ("r1e4", "r4e1")
MODES = ("natural", "feedback", "corrected", "reverse")


def unit_table(e, exp, task):
    d = e[(e.experiment == exp) & (e.task == task) & (e.config.isin(CFGS)) & (e["mode"].isin(MODES))]
    w = d.pivot_table(index=["seed", "cluster_primary", "cluster_physical"], columns=["config", "mode"],
                      values="success", aggfunc="first")
    n0 = len(w)
    w = w.dropna()
    assert len(w) == n0, f"incomplete pairing {exp} {task}: {n0 - len(w)} units dropped"
    return w


def contrasts(w):
    out = {}
    for c in CFGS:
        out[f"dR_{c}"] = (w[(c, "reverse")] - w[(c, "natural")]).to_numpy()
        out[f"dC_{c}"] = (w[(c, "corrected")] - w[(c, "feedback")]).to_numpy()
    return out


def icc1(d, groups):
    g = pd.Series(d).groupby(groups)
    means, sizes = g.mean(), g.size()
    k, n = len(means), len(d)
    n0 = (n - (sizes ** 2).sum() / n) / (k - 1)
    msb = (sizes * (means - d.mean()) ** 2).sum() / (k - 1)
    msw = ((d - means.reindex(groups).to_numpy()) ** 2).sum() / (n - k)
    s2b = max((msb - msw) / n0, 0.0)
    return means, sizes, s2b, msw


def main():
    os.makedirs(OUT, exist_ok=True)
    e = pd.read_csv(EPIS)
    rows, corr, ws, npz = [], {}, [], {}
    for exp in ("P1D", "exp35"):
        for task in ("coke", "move_near"):
            w = unit_table(e, exp, task)
            cs = contrasts(w)
            keys = list(cs)
            M = np.column_stack([cs[k] for k in keys])
            corr[f"{exp}/{task}"] = {"keys": keys, "n": int(len(M)), "corr": np.round(np.corrcoef(M.T), 4).tolist()}
            lay = np.array(w.index.get_level_values("cluster_physical"))
            if exp == "P1D":
                cols = [(c, m) for c in CFGS for m in MODES]
                npz[f"{task}_Y"] = w[cols].to_numpy().astype(np.int8)
                npz[f"{task}_layout"] = lay
            for k in keys:
                d = cs[k]
                rows.append({"experiment": exp, "task": task, "config": k.split("_")[1], "contrast": k.split("_")[0],
                             "n_units": len(d),
                             "n_primary_states": w.index.get_level_values("cluster_primary").nunique(),
                             "n_physical_states": len(set(lay)),
                             "mean_pp": 100 * d.mean(), "p_plus": (d > 0).mean(), "p_minus": (d < 0).mean(),
                             "discordance": (d != 0).mean(), "sd_d": d.std(ddof=1)})
                if len(set(lay)) < len(d):
                    means, sizes, s2b, msw = icc1(d, lay)
                    ws.append({"experiment": exp, "task": task, "config": k.split("_")[1],
                               "contrast": k.split("_")[0], "n_layouts": len(means),
                               "units_per_layout_min": int(sizes.min()), "units_per_layout_max": int(sizes.max()),
                               "sd_layout_means_pp": 100 * means.std(ddof=1),
                               "sigma_between_pp": 100 * np.sqrt(s2b), "sd_within_pp": 100 * np.sqrt(msw),
                               "icc1": s2b / (s2b + msw) if (s2b + msw) > 0 else 0.0})
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "planning_params.csv"), index=False, float_format="%.4f")
    pd.DataFrame(ws).to_csv(os.path.join(OUT, "planning_within_state.csv"), index=False, float_format="%.4f")
    json.dump(corr, open(os.path.join(OUT, "planning_corr.json"), "w"), indent=1)
    np.savez(os.path.join(OUT, "planning_units_p1d.npz"), **npz)
    print(pd.DataFrame(rows).to_string(float_format=lambda x: f"{x:.3f}"))
    print(pd.DataFrame(ws).to_string(float_format=lambda x: f"{x:.3f}"))


if __name__ == "__main__":
    main()
