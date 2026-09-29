#!/usr/bin/env python
"""
Direct population analysis of expanded_best_of_n runs (CPU only).

Every decoded scene counts once (no failure/normal weighting): for each selection rule of
analyze_best_of_n, the population A- rate (P/R/A 5 s), failure rate (A- and FDE5 > 3 m) and FDE5,
paired against the natural plan (row 0) with McNemar / Wilcoxon and a log-cluster bootstrap CI.
Scenes whose natural plan fails form the "natural failure" subset (rescue rate).
Scenes whose stub reasoned (CoT) are excluded, as in every natural-path experiment.

  python scripts/analyze_expanded_best_of_n.py outputs/expanded_best_of_n/<run> [<run> ...]
"""
from __future__ import annotations

import json
import os
import random
import sys
from collections import defaultdict

import numpy as np
from scipy.stats import binomtest, wilcoxon

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_action_history import unit_metrics      # noqa: E402
from analyze_best_of_n import pick                   # noqa: E402

RULES = ["normal", "max_loglik", "min_entropy", "ranksum", "medoid", "oracle"]


def main() -> None:
    runs = sys.argv[1:]
    units = []
    for run in runs:
        for l in open(os.path.join(run, "records.jsonl")):
            r = json.loads(l)
            if r.get("stub_cot"):
                continue
            r["prev_traj"] = None
            u = {"token": r["token"], "log": r["log"]}
            for rule in RULES:
                x = r["candidates"][pick(r, rule)]
                m = unit_metrics(x["action_idx"], x["trajectory_pred"], x["entropy_steps"], r["gt"], r["trajectory_gt"], 0)
                u[rule] = {"a_minus": float(not m["recovery"]), "failure": float(m["amplification"]), "fde5": m["fde5"]}
            units.append(u)
    fail = [u for u in units if u["normal"]["a_minus"] == 1.0]
    S = {"n_scenes": len(units), "n_logs": len({u["log"] for u in units}), "n_natural_failures": len(fail), "rules": {}}
    by = defaultdict(list)
    for u in units:
        by[u["log"]].append(u)
    logs = sorted(by)
    rng = random.Random(0)
    boots = [[rng.choice(logs) for _ in logs] for _ in range(2000)]
    for rule in RULES:
        out = {}
        for m in ("a_minus", "failure", "fde5"):
            mean = float(np.mean([u[rule][m] for u in units]))
            d = [u[rule][m] - u["normal"][m] for u in units]
            bs = [np.mean([u[rule][m] - u["normal"][m] for lg in b for u in by[lg]]) for b in boots]
            e = {"mean": mean, "diff": float(np.mean(d)), "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]}
            if m != "fde5":
                gain = sum(1 for u in units if u[rule][m] > u["normal"][m]); loss = sum(1 for u in units if u[rule][m] < u["normal"][m])
                e["mcnemar"] = {"rule_only": gain, "normal_only": loss, "p": 1.0 if gain + loss == 0 else float(binomtest(gain, gain + loss, 0.5).pvalue)}
            else:
                dd = np.array(d)
                e["wilcoxon_p"] = float(wilcoxon(dd).pvalue) if np.any(dd != 0) else 1.0
            out[m] = e
        out["rescued_natural_failures"] = float(np.mean([u[rule]["a_minus"] == 0.0 for u in fail])) if fail else None
        S["rules"][rule] = out
    dst = runs[0] if len(runs) == 1 else os.path.dirname(os.path.abspath(runs[0]))
    json.dump(S, open(os.path.join(dst, "summary.json" if len(runs) == 1 else "summary_pooled.json"), "w"), indent=1)
    print(f"scenes {S['n_scenes']}  logs {S['n_logs']}  natural failures {S['n_natural_failures']} "
          f"({100*S['n_natural_failures']/max(1,S['n_scenes']):.2f}%)")
    print(f"{'rule':12s} {'A- %':>7s} {'dA- %p [CI]':>26s} {'McN p':>8s} {'fail %':>7s} {'FDE5':>6s} {'dFDE [CI]':>26s} {'rescued':>8s}")
    for rule in RULES:
        x = S["rules"][rule]; a, f = x["a_minus"], x["fde5"]
        print(f"{rule:12s} {100*a['mean']:7.2f} {100*a['diff']:+7.2f} [{100*a['ci95'][0]:+.2f},{100*a['ci95'][1]:+.2f}]   "
              f"{a['mcnemar']['p']:8.2g} {100*x['failure']['mean']:7.2f} {f['mean']:6.3f} {f['diff']:+7.3f} [{f['ci95'][0]:+.3f},{f['ci95'][1]:+.3f}]   "
              f"{(100*x['rescued_natural_failures'] if x['rescued_natural_failures'] is not None else float('nan')):7.1f}%")


if __name__ == "__main__":
    main()
