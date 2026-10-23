#!/usr/bin/env python
"""CPU analysis of the VLA-0 within-call pilot (protocol: pkg/PROTOCOL_BEFORE_VLA0_GPU.md)."""
import json
import os
import sys

import numpy as np
import pandas as pd

ACT, H, ANCHOR, BINS = 7, 8, 3, 1000
SEED, B = 20261020, 10000


def suffix(arm):
    n = arm["audit"]["numbers"]
    # strict: exactly 35 integers, no bad fields, in range, EOS (2026-10-21)
    return np.array(n, float).reshape(H - ANCHOR, ACT) if len(n) == (H - ANCHOR) * ACT and \
        arm["audit"]["n_bad_fields"] == 0 and arm["audit"]["in_range"] and arm["audit"]["eos"] else None


def main(rec_path, out):
    os.makedirs(out, exist_ok=True)
    R = [json.loads(l) for l in open(rec_path)]
    cov, rows, curves = [], [], []
    for r in R:
        d = r.get("donors", {})
        c = {"obs_id": r["obs_id"], "suite": r["suite"], "task": f"{r['suite']}/{r['task_index']}", "status": r["status"],
             "donor_v1": d.get("1") is not None, "donor_v2": d.get("2") is not None, "donor_v3": d.get("3") is not None,
             "n_valid_sources": sum(1 for s in r.get("sources", []) if s["audit"]["complete"] and s["audit"]["in_range"]),
             "latency_s": r.get("latency_s"), "peak_vram_gb": r.get("peak_vram_gb"), "n_generate_calls": r.get("n_generate_calls")}
        cov.append(c)
        if r["status"] != "OK":
            continue
        A = np.array(r["reference"]["audit"]["numbers"], float).reshape(H, ACT)
        arms = {k: suffix(v) for k, v in r["arms"].items()}
        base = arms.get("R")
        row = {"obs_id": r["obs_id"], "suite": r["suite"], "task": c["task"], "selfpatch_identity": r["checks"]["selfpatch_token_identity"]}
        for k, s in arms.items():
            row[f"{k}_suffix_ok"] = s is not None
            if s is None or base is None:
                continue
            diff = np.abs(s[:, :6] - base[:, :6]) / BINS
            row[f"resp_{k}"] = float(diff.mean())
            row[f"grip_flip_{k}"] = float((np.sign(s[:, 6] - BINS / 2) != np.sign(base[:, 6] - BINS / 2)).mean())
            changed = np.where(np.any(s != base, axis=1))[0]
            row[f"first_change_t_{k}"] = int(changed[0] + ANCHOR + 1) if len(changed) else None
            row[f"token_changed_{k}"] = r["arms"][k]["ids"] != r["arms"]["R"]["ids"]
            for t in range(H - ANCHOR):
                curves.append({"obs_id": r["obs_id"], "task": c["task"], "suite": r["suite"], "arm": k, "timestep": t + ANCHOR + 1,
                               "resp": float(diff[t].mean())})
        for j, name in ((2, "O"), (3, "L"), (1, "v1")):
            dj = r["donors"].get(str(j))
            if dj is not None:
                row[f"src_dist_{name}"] = float(np.abs(np.array(dj["vector"][:6], float) - A[j - 1, :6]).mean() / BINS)
                row[f"src_grip_only_{name}"] = bool(np.all(np.array(dj["vector"][:6]) == A[j - 1, :6]))
        if all(arms.get(k) is not None for k in ("R", "O", "L", "OL")):
            I = arms["OL"] - arms["L"] - arms["O"] + arms["R"]
            row["interaction_abs"] = float(np.abs(I[:, :6]).mean() / BINS)
            row["interaction_signed"] = float(I[:, :6].mean() / BINS)
        rows.append(row)
    C = pd.DataFrame(cov); U = pd.DataFrame(rows); Cu = pd.DataFrame(curves)
    C.to_csv(f"{out}/coverage_per_observation.csv", index=False)
    U.to_csv(f"{out}/paired_per_observation.csv", index=False)
    Cu.to_csv(f"{out}/timestep_curves_long.csv", index=False)
    rng = np.random.default_rng(SEED)

    def boot(df, col):
        d = df.dropna(subset=[col])
        if d.empty:
            return np.nan, np.nan, np.nan, 0, 0
        tm = d.groupby("task")[col].mean()
        tasks = tm.index.values; vals = tm.values
        est = float(d[col].mean())
        bs = []
        g = d.groupby("task")[col]
        sums, cnts = g.sum().values, g.size().values
        for _ in range(B):
            idx = rng.integers(0, len(tasks), len(tasks))
            bs.append(sums[idx].sum() / cnts[idx].sum())
        return est, float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)), len(d), len(tasks)

    if not U.empty:
        if "resp_L" in U and "resp_O" in U:
            U["L_minus_O"] = U["resp_L"] - U["resp_O"]
        if "src_dist_L" in U and "src_dist_O" in U:
            ratio = U["src_dist_L"] / U["src_dist_O"].replace(0, np.nan)
            U["matched_src"] = ratio.between(0.8, 1.25)
        if "resp_H" in U and "resp_L" in U:
            U["H_minus_L"] = U["resp_H"] - U["resp_L"]
    eff = []
    for col, lab, sub in (("L_minus_O", "PRIMARY L - O response", None), ("L_minus_O", "L - O, source-distance-matched subset", "matched_src"),
                          ("resp_O", "response O", None), ("resp_L", "response L", None), ("resp_OL", "response OL", None),
                          ("resp_H", "response H (secondary)", None), ("resp_HL", "response HL (secondary)", None),
                          ("H_minus_L", "H - L (secondary)", None), ("interaction_abs", "|I| = |A_OL-A_L-A_O+A_R| (descriptive)", None),
                          ("interaction_signed", "signed I mean (descriptive)", None), ("src_dist_O", "source distance O", None),
                          ("src_dist_L", "source distance L", None)):
        if U.empty or col not in U:
            continue
        d = U if sub is None else U[U[sub] == True]  # noqa: E712
        for suite in [None] + sorted(U.suite.unique()):
            dd = d if suite is None else d[d.suite == suite]
            est, lo, hi, n, nt = boot(dd, col)
            eff.append({"quantity": lab, "suite": suite or "all", "estimate": est, "ci95_lo": lo if suite is None else np.nan,
                        "ci95_hi": hi if suite is None else np.nan, "n_obs": n, "n_tasks": nt,
                        "interval": "task-cluster bootstrap 10,000, seed 20261020, percentile" if suite is None else "descriptive (no interval)"})
    E = pd.DataFrame(eff)
    E.to_csv(f"{out}/effects.csv", index=False)
    print(C.groupby("suite")[["donor_v1", "donor_v2", "donor_v3"]].sum())
    print(E.round(5).to_string())
    return U, E, Cu


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
