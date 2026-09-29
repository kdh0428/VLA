#!/usr/bin/env python
"""
Per-candidate features and outcomes for learned best-of-N selection (CPU only).

For every scene of one or more expanded_best_of_n runs and every candidate (row 0 = natural plan):
  confidence   mean / max / late (steps 5-9) entropy, summed / min log-prob
  consensus    mean ADE to the other candidates, fraction of candidates within 1 m ADE
  anchor       ADE to the natural plan (row 0), is_natural flag
  kinematics   path length, max |second difference| of positions (jerk proxy), final |lateral|
Outcomes (for training / evaluation only): A- (P/R/A 5 s), FDE5, ADE5.

Output: an .npz with X (n_scene, n_cand, n_feat), A (n_scene, n_cand), FDE, ADE, logs, tokens, feature names.
  python scripts/selector_features.py <out.npz> <run_dir> [<run_dir> ...]
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_action_history import a_eval   # noqa: E402

FEATS = ["ent_mean", "ent_max", "ent_late", "lp_sum", "lp_min", "cons_ade", "cons_frac1m", "ade_to_natural",
         "is_natural", "path_len", "jerk", "final_lat"]


def main() -> None:
    out, runs = sys.argv[1], sys.argv[2:]
    X, A, F, D, logs, toks = [], [], [], [], [], []
    for run in runs:
        for l in open(os.path.join(run, "records.jsonl")):
            r = json.loads(l)
            if r.get("stub_cot"):
                continue
            c = r["candidates"]
            T = np.asarray([x["trajectory_pred"] for x in c], float)                  # (n, 10, 2)
            gt = np.asarray(r["trajectory_gt"], float)[:, :2]
            pair = np.linalg.norm(T[:, None] - T[None], axis=-1).mean(-1)          # (n, n) ADE
            n = len(c)
            rows, a_, f_, d_ = [], [], [], []
            for i, x in enumerate(c):
                e = np.asarray(x["entropy_steps"]); lp = np.asarray(x["logprob_steps"])
                p = np.vstack([[0, 0], T[i]])
                seg = np.diff(p, axis=0)
                rows.append([e.mean(), e.max(), e[5:].mean(), lp.sum(), lp.min(),
                             pair[i].sum() / (n - 1), (pair[i] < 1.0).sum() / (n - 1), pair[i, 0], float(i == 0),
                             np.linalg.norm(seg, axis=1).sum(), np.abs(np.diff(seg, n=2, axis=0)).max() if len(seg) > 2 else 0.0,
                             abs(T[i, -1, 1])])
                err = np.linalg.norm(T[i] - gt, axis=1)
                a_.append(float(not a_eval(T[i].tolist(), r["trajectory_gt"])))
                f_.append(err[-1]); d_.append(err.mean())
            X.append(rows); A.append(a_); F.append(f_); D.append(d_); logs.append(r["log"]); toks.append(r["token"])
    np.savez_compressed(out, X=np.asarray(X, np.float32), A=np.asarray(A, np.float32), FDE=np.asarray(F, np.float32),
                        ADE=np.asarray(D, np.float32), logs=np.asarray(logs), tokens=np.asarray(toks), feats=np.asarray(FEATS))
    print(f"[done] {len(toks)} scenes x {len(X[0])} candidates x {len(FEATS)} features -> {out}")


if __name__ == "__main__":
    main()
