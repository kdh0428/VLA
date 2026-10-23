#!/usr/bin/env python
"""Follow-up to the package verification (2026-10-20, post_hoc): (1) save fold assignments + environment for F and G,
(2) re-derive out-of-fold scores with saved folds, (3) a post-hoc EQUAL-QUOTA heldout check (top-q by each score in the
heldout batch; NOT a deployable fixed-threshold evaluation), labelled separately from the frozen-threshold results."""
import json
import os
import platform
import sys

import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ra_common as R  # noqa: E402

G = f"{R.OUT}/selective_intervention"
D = f"{R.OUT}/decomposition"


def env():
    return {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__,
            "scikit-learn": sklearn.__version__, "platform": platform.platform(),
            "GroupKFold": "sklearn GroupKFold(n_splits=5), deterministic (no shuffle); groups = log name",
            "Ridge": "alpha=10, StandardScaler", "Logistic": "max_iter=3000 (G) / 2000 (F), C=1, lbfgs"}


def main():
    json.dump(env(), open(f"{R.OUT}/logs/ENVIRONMENT.json", "w"), indent=1)
    X = pd.read_parquet(f"{G}/G_runtime_features.parquet")
    Y = pd.read_parquet(f"{G}/G_evaluation_outcomes.parquet")
    df = X.merge(Y, on=["split", "group_id", "scene_id"])
    feat = [c for c in X.columns if c not in ("split", "group_id", "scene_id")]
    dev, ho = df[df.split == "development"].reset_index(drop=True), df[df.split == "heldout"].reset_index(drop=True)
    folds = np.zeros(len(dev), int)
    for f, (_, te) in enumerate(GroupKFold(5).split(dev[feat], groups=dev.group_id)):
        folds[te] = f
    pd.DataFrame({"scene_id": dev.scene_id, "group_id": dev.group_id, "fold": folds}).to_csv(f"{G}/G_dev_fold_assignment.csv", index=False)
    mg = make_pipeline(StandardScaler(), Ridge(alpha=10.0)).fit(dev[feat].values, dev.gain_rs.values)
    mr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000)).fit(dev[feat].values, dev.severe_keep.values)
    S = {"gain gate": mg.predict(ho[feat].values), "risk gate": mr.predict_proba(ho[feat].values)[:, 1],
         "entropy gate": ho.keep_ent_mean.values}
    n = len(ho)
    rows = []
    for q in (0.05, 0.10, 0.20, 0.30, 0.50):
        k = int(round(q * n))
        for name, s in S.items():
            trig = np.zeros(n, bool); trig[np.argsort(-s, kind="stable")[:k]] = True
            fde = np.where(trig, ho.fde_rs, ho.fde_keep)
            sev = np.where(trig, ho.severe_rs, ho.severe_keep)
            rows.append({"quota": q, "k_triggered": k, "gate": name, "actual_rate": k / n,
                         "plan_changes": int((trig & ~ho.rs_is_keep.values).sum()),
                         "delta_mean_fde_m": float((fde - ho.fde_keep).mean()), "delta_severe_pp": float(100 * (sev - ho.severe_keep).mean())})
        rows.append({"quota": q, "k_triggered": k, "gate": "random gate (expectation)", "actual_rate": k / n,
                     "plan_changes": float(k / n * (~ho.rs_is_keep.values).sum()),
                     "delta_mean_fde_m": float(k / n * (ho.fde_rs - ho.fde_keep).mean()),
                     "delta_severe_pp": float(100 * k / n * (ho.severe_rs - ho.severe_keep).mean())})
    Q = pd.DataFrame(rows)
    Q["status"] = "post-hoc equal-quota check within the heldout batch (ranking only; not a fixed-threshold deployment evaluation)"
    Q.to_csv(f"{G}/G_equal_quota_heldout_posthoc.csv", index=False)
    print(Q.pivot_table(index="quota", columns="gate", values="delta_mean_fde_m").round(5).to_string())
    # F folds
    U = pd.read_csv(f"{D}/F_unit_decomposition.csv")
    ff = np.zeros(len(U), int)
    for f, (_, te) in enumerate(GroupKFold(5).split(U, groups=U.log_id)):
        ff[te] = f
    pd.DataFrame({"unit_id": U.unit_id, "log_id": U.log_id, "fold": ff}).to_csv(f"{D}/F_fold_assignment.csv", index=False)
    print(env())


if __name__ == "__main__":
    main()
