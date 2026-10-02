#!/usr/bin/env python
"""
Navhard (NAVSIM v2 two-stage) comparison of selection rules from navhard_group_scores.py outputs.
EPDMS = mean over groups of (stage1 * weighted stage2 of the orig token + same for the prev token) / 2.
Differences vs `normal` with a log-cluster bootstrap (2000 resamples, seed 0) and a paired sign count over groups.

  python scripts/analyze_navhard.py <group_scores_dir> <out.json> rule [rule ...]
"""
import json
import os
import random
import sys
from collections import defaultdict

import numpy as np

SUBSET = os.environ.get("NAVHARD_SUBSET", "/root/VLA/autovla_misalignment_poc/outputs/navhard_eval/subset")   # other navhard subsets via env

if __name__ == "__main__":
    gdir, out, rules = sys.argv[1], sys.argv[2], sys.argv[3:]
    log_of = {g["orig"]: g["log"] for g in json.load(open(f"{SUBSET}/groups.json"))}
    S = {}
    for r in rules:
        rows = json.load(open(os.path.join(gdir, f"g_{r}.json")))
        S[r] = {x["orig_token"]: x for x in rows}
    toks = sorted(set.intersection(*(set(v) for v in S.values())))
    by = defaultdict(list)
    for t in toks:
        by[log_of[t]].append(t)
    logs = sorted(by); rng = random.Random(0)
    boots = [[rng.choice(logs) for _ in logs] for _ in range(2000)]

    def val(r, t, key):
        x = S[r][t]
        if key == "epdms":
            return x["group_score"]
        return np.mean([p[key] for p in x["parts"]])

    res = {"n_groups": len(toks), "n_logs": len(logs), "rules": {}}
    for r in rules:
        e = {}
        for key in ("epdms", "stage1", "stage2"):
            v = {t: val(r, t, key) for t in toks}; b = {t: val("normal", t, key) for t in toks}
            d = {t: v[t] - b[t] for t in toks}
            bs = sorted(np.mean([d[t] for lg in bt for t in by[lg]]) for bt in boots)
            e[key] = {"mean": float(np.mean(list(v.values()))), "diff_vs_normal": float(np.mean(list(d.values()))),
                      "ci95": [float(bs[49]), float(bs[1949])]}
        d = [S[r][t]["group_score"] - S["normal"][t]["group_score"] for t in toks]
        e["groups_better_worse_tied"] = [int(sum(x > 1e-9 for x in d)), int(sum(x < -1e-9 for x in d)), int(sum(abs(x) <= 1e-9 for x in d))]
        res["rules"][r] = e
    json.dump(res, open(out, "w"), indent=1)
    print(f"{len(toks)} groups, {len(logs)} logs")
    print(f"{'rule':12s} {'EPDMS':>7s} {'diff [95% CI]':>28s} {'stage1':>7s} {'stage2':>7s}  better/worse/tied")
    for r, e in res["rules"].items():
        x = e["epdms"]
        print(f"{r:12s} {x['mean']:7.4f} {x['diff_vs_normal']:+8.4f} [{x['ci95'][0]:+.4f}, {x['ci95'][1]:+.4f}]"
              f" {e['stage1']['mean']:7.4f} {e['stage2']['mean']:7.4f}  {e['groups_better_worse_tied']}")
