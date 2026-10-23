#!/usr/bin/env python
"""P4 planning data (post-hoc use of existing data ONLY as planning input; no hypothesis test is run here).

Reads the per-episode table built by review_followup2/B (experiments 35 and P1-D, SpatialVLA, SimplerEnv) and writes
  planning_params.csv     per experiment x task x config x contrast: n units, discordance, P(+1), P(-1), mean, SD
  planning_corr.json      per experiment x task: correlation matrix of the 4 paired differences
                          (dR r1e4, dR r4e1, dC r1e4, dC r4e1) across units (joint structure for the Holm simulation)
  planning_cluster.csv    move_near: layout (physical) level between-layout SD and ICC of the paired differences,
                          and repeat concordance of identical primary states (exp 35 duplicate seeds)
  planning_units_p1d.npz  P1-D unit x 8-outcome matrix per task (N,F,C,R x r1e4,r4e1) for the bootstrap simulation

Contrasts: dR = S(R) - S(N); dC = S(C) - S(F) (B's sign convention; C-F).
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
    w = w.dropna()
    return w


def contrasts(w):
    out = {}
    for c in CFGS:
        out[f"dR_{c}"] = (w[(c, "reverse")] - w[(c, "natural")]).to_numpy()
        out[f"dC_{c}"] = (w[(c, "corrected")] - w[(c, "feedback")]).to_numpy()
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    e = pd.read_csv(EPIS)
    rows, corr, clus, npz = [], {}, [], {}
    for exp in ("P1D", "exp35"):
        for task in ("coke", "move_near"):
            w = unit_table(e, exp, task)
            cs = contrasts(w)
            keys = list(cs)
            M = np.column_stack([cs[k] for k in keys])
            corr[f"{exp}/{task}"] = {"keys": keys, "n": int(len(M)),
                                     "corr": np.round(np.corrcoef(M.T), 4).tolist()}
            if exp == "P1D":
                cols = [(c, m) for c in CFGS for m in MODES]
                npz[f"{task}_Y"] = w[cols].to_numpy().astype(np.int8)
                npz[f"{task}_layout"] = np.array(w.index.get_level_values("cluster_physical"))
            for k in keys:
                d = cs[k]
                rows.append({"experiment": exp, "task": task, "config": k.split("_")[1], "contrast": k.split("_")[0],
                             "n_units": len(d),
                             "n_primary_states": w.index.get_level_values("cluster_primary").nunique(),
                             "n_physical_states": w.index.get_level_values("cluster_physical").nunique(),
                             "mean_pp": 100 * d.mean(), "p_plus": (d > 0).mean(), "p_minus": (d < 0).mean(),
                             "discordance": (d != 0).mean(), "sd_d": d.std(ddof=1)})
                if task == "move_near":
                    lay = np.array(w.index.get_level_values("cluster_physical"))
                    g = pd.Series(d).groupby(lay)
                    means, sizes = g.mean(), g.size()
                    # one-way ANOVA ICC(1) of the unit-level paired difference within layouts
                    k_, N = len(means), len(d)
                    n0 = (N - (sizes ** 2).sum() / N) / (k_ - 1)
                    msb = (sizes * (means - d.mean()) ** 2).sum() / (k_ - 1)
                    msw = ((d - means.reindex(lay).to_numpy()) ** 2).sum() / (N - k_)
                    s2b = max((msb - msw) / n0, 0.0)
                    clus.append({"experiment": exp, "config": k.split("_")[1], "contrast": k.split("_")[0],
                                 "n_layouts": k_, "units_per_layout_min": int(sizes.min()),
                                 "units_per_layout_max": int(sizes.max()),
                                 "sd_layout_means_pp": 100 * means.std(ddof=1),
                                 "sigma_between_pp_anova": 100 * np.sqrt(s2b), "sd_within_pp": 100 * np.sqrt(msw),
                                 "icc1": s2b / (s2b + msw) if (s2b + msw) > 0 else 0.0,
                                 "layout_means_pp": ";".join(f"{x:.1f}" for x in 100 * means.to_numpy())})
            # success ICC within physical layouts (natural arm) and repeat concordance of identical primary states
            if task == "move_near":
                for c in CFGS:
                    s = w[(c, "natural")]
                    prim = np.array(w.index.get_level_values("cluster_primary"))
                    ss = pd.Series(s.to_numpy()).groupby(prim)
                    agree = [float((v == v.iloc[0]).all()) for _, v in ss if len(v) > 1]
                    clus.append({"experiment": exp, "config": c, "contrast": "natural_success_repeat",
                                 "n_layouts": len(set(prim)),
                                 "repeat_groups": len(agree),
                                 "repeat_all_identical_frac": float(np.mean(agree)) if agree else np.nan})
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "planning_params.csv"), index=False, float_format="%.4f")
    pd.DataFrame(clus).to_csv(os.path.join(OUT, "planning_cluster.csv"), index=False, float_format="%.4f")
    json.dump(corr, open(os.path.join(OUT, "planning_corr.json"), "w"), indent=1)
    np.savez(os.path.join(OUT, "planning_units_p1d.npz"), **npz)
    print(pd.DataFrame(rows).to_string(float_format=lambda x: f"{x:.3f}"))
    print(pd.DataFrame(clus).to_string(float_format=lambda x: f"{x:.3f}"))
    print(json.dumps(corr))


if __name__ == "__main__":
    main()
