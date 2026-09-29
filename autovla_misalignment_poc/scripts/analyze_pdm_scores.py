#!/usr/bin/env python
"""
Analysis of pdm_score_candidates (CPU only).

Per selection rule: mean PDM Score and sub-score failure rates (no at-fault collision, drivable-area
compliance, TTC, comfort) and mean ego progress; paired difference vs the natural plan with a log-cluster
bootstrap CI and a Wilcoxon test. Also reported on the scenes whose natural plan fails the P/R/A 5 s
rule (joined from the best-of-N records) and on the scenes where the rule changed the plan.

  python scripts/analyze_pdm_scores.py <pdm_scores.jsonl> [<best_of_n run dir> ...]
"""
from __future__ import annotations

import json
import os
import random
import sys
from collections import defaultdict

import numpy as np
from scipy.stats import wilcoxon

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_action_history import a_eval   # noqa: E402

RULES = ["normal", "max_loglik", "min_entropy", "ranksum", "oracle"]
SUBS = ["no_at_fault_collisions", "drivable_area_compliance", "time_to_collision_within_bound", "comfort"]


def main() -> None:
    path, runs = sys.argv[1], sys.argv[2:]
    rows = [json.loads(l) for l in open(path)]
    ok = [r for r in rows if "error" not in r]
    global RULES
    if ok and "normal" not in ok[0]:                          # N-curve file: ranksum_N1 is the natural plan
        RULES = sorted([k for k in ok[0] if k.startswith("ranksum_N")], key=lambda k: int(k[9:]))
        for r in ok:
            r["normal"] = r["ranksum_N1"]
    elif ok:                                                  # any picks file: every rule present, normal first
        RULES = ["normal"] + [k for k in ok[0] if k not in ("token", "log", "normal")]
    fail_tok = set()
    for run in runs:
        for l in open(os.path.join(run, "records.jsonl")):
            r = json.loads(l)
            if not a_eval(r["candidates"][0]["trajectory_pred"], r["trajectory_gt"]):
                fail_tok.add(r["token"])
    by = defaultdict(list)
    for i, r in enumerate(ok):
        by[r["log"]].append(i)
    logs = sorted(by)
    rng = random.Random(0)
    boots = [np.concatenate([by[rng.choice(logs)] for _ in logs]) for _ in range(2000)]
    S = {"n_scored": len(ok), "n_errors": len(rows) - len(ok), "n_logs": len(logs), "n_natural_failures": 0, "subsets": {}}

    def block(idx):
        idx = np.asarray(idx)
        out = {}
        base = np.array([ok[i]["normal"]["score"] for i in idx])
        for rule in RULES:
            sc = np.array([ok[i][rule]["score"] for i in idx])
            d = sc - base
            pos = {j: k for k, j in enumerate(idx)}
            bs = [np.mean([d[pos[j]] for j in b if j in pos]) for b in boots] if len(idx) < len(ok) else [d[b].mean() for b in boots]
            e = {"pdms": float(sc.mean()), "d_pdms": float(d.mean()),
                 "d_pdms_ci95": [float(np.nanpercentile(bs, 2.5)), float(np.nanpercentile(bs, 97.5))],
                 "wilcoxon_p": float(wilcoxon(d).pvalue) if np.any(d != 0) else 1.0,
                 "ego_progress": float(np.mean([ok[i][rule]["ego_progress"] for i in idx])),
                 "changed_plan": float(np.mean([ok[i][rule]["picked"] != 0 for i in idx]))}
            for s in SUBS:
                e[f"viol_{s}"] = float(np.mean([ok[i][rule][s] < 1.0 for i in idx]))
            out[rule] = e
        return out
    S["subsets"]["all"] = block(range(len(ok)))
    fidx = [i for i, r in enumerate(ok) if r["token"] in fail_tok]
    S["n_natural_failures"] = len(fidx)
    if fidx:
        S["subsets"]["natural_failures"] = block(fidx)
    json.dump(S, open(os.path.join(os.path.dirname(path), "summary.json"), "w"), indent=1)
    print(f"scored {S['n_scored']} scenes ({S['n_errors']} errors), {S['n_logs']} logs, natural failures {S['n_natural_failures']}")
    for name, blk in S["subsets"].items():
        print(f"\n== {name}")
        print(f"{'rule':12s} {'PDMS':>6s} {'dPDMS [CI]':>26s} {'p':>8s} {'coll%':>6s} {'DAC%':>6s} {'TTC%':>6s} {'comf%':>6s} {'EP':>5s} {'chg':>5s}")
        for rule, e in blk.items():
            print(f"{rule:12s} {e['pdms']:6.4f} {e['d_pdms']:+.4f} [{e['d_pdms_ci95'][0]:+.4f},{e['d_pdms_ci95'][1]:+.4f}] {e['wilcoxon_p']:8.2g} "
                  f"{100*e['viol_no_at_fault_collisions']:6.2f} {100*e['viol_drivable_area_compliance']:6.2f} "
                  f"{100*e['viol_time_to_collision_within_bound']:6.2f} {100*e['viol_comfort']:6.2f} {e['ego_progress']:5.3f} {100*e['changed_plan']:4.0f}%")


if __name__ == "__main__":
    main()
