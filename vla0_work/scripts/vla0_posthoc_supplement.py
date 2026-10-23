#!/usr/bin/env python
"""Post-hoc supplementary analyses (2026-10-21, after CVPR_VLA0_VERIFICATION_20261021): conditional older effect and a
repetition-penalty token-set sensitivity subset. Raw pilot records only; task-cluster bootstrap (10,000, seed 20261021)."""
import json, sys
import numpy as np, pandas as pd

ACT, H, ANCHOR, BINS = 7, 8, 3, 1000
rec_path, out = sys.argv[1], sys.argv[2]
R = [json.loads(l) for l in open(rec_path)]
rows = []
for r in R:
    if r["status"] != "OK":
        continue
    S = {k: np.array(v["audit"]["numbers"][:35], float).reshape(5, 7) for k, v in r["arms"].items()}
    A = np.array(r["reference"]["audit"]["numbers"], float).reshape(H, ACT)
    D = {k: np.abs(S[k][:, :6] - S["R"][:, :6]).mean() / BINS for k in S}
    sets = {k: set(r["arms"][k]["prefix_ids"]) for k in ("R", "O", "L")}
    dO = np.abs(np.array(r["donors"]["2"]["vector"][:6]) - A[1, :6]).mean() / BINS
    dL = np.abs(np.array(r["donors"]["3"]["vector"][:6]) - A[2, :6]).mean() / BINS
    rows.append({"obs_id": r["obs_id"], "task": f"{r['suite']}/{r['task_index']}", "suite": r["suite"],
                 "abs_OL_minus_L": np.abs(S["OL"][:, :6] - S["L"][:, :6]).mean() / BINS,
                 "D_OL_minus_D_L": D["OL"] - D["L"], "OL_equals_L_all_ints": bool(np.array_equal(S["OL"], S["L"])),
                 "L_minus_O": D["L"] - D["O"], "same_prefix_token_sets_R_O_L": sets["R"] == sets["O"] == sets["L"],
                 "matched_src": (dL / dO) if dO > 0 else np.nan})
U = pd.DataFrame(rows)
U["matched_src"] = U["matched_src"].between(0.8, 1.25)
U.to_csv(f"{out}/posthoc_supplement_per_observation.csv", index=False)
rng = np.random.default_rng(20261021)


def boot(d, col):
    g = d.groupby("task")[col]
    s, c = g.sum().values, g.size().values
    bs = [s[i].sum() / c[i].sum() for i in (rng.integers(0, len(s), len(s)) for _ in range(10000))]
    return float(d[col].mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)), len(d), len(s)


res = []
for lab, col, mask in (("mean |A_OL - A_L| / 1000 (older edit added to latest edit)", "abs_OL_minus_L", None),
                       ("D_OL - D_L (distance-from-R summary)", "D_OL_minus_D_L", None),
                       ("L - O, prefix token sets of R/O/L identical (rep.-penalty token set unchanged)", "L_minus_O", "same_prefix_token_sets_R_O_L"),
                       ("L - O, identical token sets AND matched source distance", "L_minus_O", "both")):
    d = U if mask is None else (U[U[mask]] if mask != "both" else U[U.same_prefix_token_sets_R_O_L & U.matched_src])
    e, lo, hi, n, t = boot(d, col)
    res.append({"quantity": lab, "estimate": e, "ci95_lo": lo, "ci95_hi": hi, "n_obs": n, "n_tasks": t,
                "status": "post hoc (2026-10-21), task-cluster bootstrap 10,000, seed 20261021"})
res.append({"quantity": "observations with OL == L on all 35 suffix integers", "estimate": int(U.OL_equals_L_all_ints.sum()),
            "n_obs": len(U), "status": "count"})
pd.DataFrame(res).to_csv(f"{out}/posthoc_supplement_effects.csv", index=False)
print(pd.DataFrame(res).round(6).to_string())
