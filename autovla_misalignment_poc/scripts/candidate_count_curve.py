#!/usr/bin/env python
"""
Candidate-count curve for rank-sum selection (CPU only; no new decoding).

The 16 T = 1.0 candidates stored per scene are i.i.d. samples, so selection with fewer candidates is
emulated by subsampling: N counts the natural plan plus N-1 random samples (N = 1 is the natural plan
alone). For each scene R random subsets are drawn (seeded) and the metrics of the rank-sum pick are
averaged over the subsets. Reported per N: A- rate, amplification rate (A- and FDE5 > 3 m), ADE5, FDE5,
and the difference vs N = 1 with a log-cluster bootstrap 95% CI.
Also writes, for each N, the deterministic "prefix subset" pick (natural + samples 1..N-1) used for PDMS.

  python scripts/candidate_count_curve.py <out_dir> <run_dir> [<run_dir> ...]
"""
from __future__ import annotations

import json
import os
import random
import sys
from collections import defaultdict

import numpy as np
from scipy.stats import rankdata

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_action_history import AMP_FDE, a_eval   # noqa: E402

NS = [1, 2, 4, 8, 12, 16, 17]
R = 5


def ranksum_pick(C, sub):
    re_ = rankdata([C[i]["ent"] for i in sub]); rl_ = rankdata([-C[i]["lp"] for i in sub])
    return sub[int(np.argmin(re_ + rl_ + 1e-6 * re_))]


def main() -> None:
    out_dir, runs = sys.argv[1], sys.argv[2:]
    os.makedirs(out_dir, exist_ok=True)
    rows, prefix_picks = [], {}
    for run in runs:
        for l in open(os.path.join(run, "records.jsonl")):
            r = json.loads(l)
            if r.get("stub_cot"):
                continue
            gt = np.asarray(r["trajectory_gt"], float)[:, :2]
            C = []
            for x in r["candidates"]:
                T = np.asarray(x["trajectory_pred"], float); e = np.linalg.norm(T - gt, axis=1)
                am = float(not a_eval(T.tolist(), gt.tolist()))
                C.append({"ent": float(np.mean(x["entropy_steps"])), "lp": float(np.sum(x["logprob_steps"])),
                          "a_minus": am, "amp": float(am == 1.0 and e[-1] > AMP_FDE), "ade": float(e.mean()), "fde": float(e[-1])})
            rng = random.Random(f"{r['token']}")
            res = {}
            for n in NS:
                picks = [ranksum_pick(C, [0] + (rng.sample(range(1, len(C)), n - 1) if n > 1 else [])) for _ in range(R)]
                res[n] = {m: float(np.mean([C[i][m] for i in picks])) for m in ("a_minus", "amp", "ade", "fde")}
                prefix_picks.setdefault(r["token"], {})[str(n)] = int(ranksum_pick(C, list(range(n))))
            rows.append({"log": r["log"], "res": res})
    by = defaultdict(list)
    for x in rows:
        by[x["log"]].append(x)
    logs = sorted(by)
    rng = random.Random(0)
    boots = [[rng.choice(logs) for _ in range(len(logs))] for _ in range(2000)]
    S = {"n_scenes": len(rows), "n_logs": len(logs), "subsets_per_scene": R, "curve": {}}
    for n in NS:
        e = {}
        for m in ("a_minus", "amp", "ade", "fde"):
            d = lambda xs: np.mean([x["res"][n][m] - x["res"][1][m] for x in xs])   # noqa: E731
            e[m] = float(np.mean([x["res"][n][m] for x in rows]))
            e[f"d_{m}"] = float(d(rows))
            e[f"d_{m}_ci95"] = [float(np.percentile([d([x for lg in b for x in by[lg]]) for b in boots], q)) for q in (2.5, 97.5)]
        S["curve"][str(n)] = e
    json.dump(S, open(os.path.join(out_dir, "summary.json"), "w"), indent=1)
    json.dump(prefix_picks, open(os.path.join(out_dir, "prefix_picks.json"), "w"))
    print(f"scenes {S['n_scenes']}  logs {S['n_logs']}  random subsets per scene {R}")
    print(f"{'N':>3s} {'A- %':>6s} {'dA- %p [CI]':>24s} {'amp %':>6s} {'ADE5':>6s} {'FDE5':>6s} {'dFDE [CI]':>24s}")
    for n in NS:
        e = S["curve"][str(n)]
        print(f"{n:3d} {100*e['a_minus']:6.2f} {100*e['d_a_minus']:+6.2f} [{100*e['d_a_minus_ci95'][0]:+.2f},{100*e['d_a_minus_ci95'][1]:+.2f}] "
              f"{100*e['amp']:6.2f} {e['ade']:6.3f} {e['fde']:6.3f} {e['d_fde']:+.3f} [{e['d_fde_ci95'][0]:+.3f},{e['d_fde_ci95'][1]:+.3f}]")


if __name__ == "__main__":
    main()
