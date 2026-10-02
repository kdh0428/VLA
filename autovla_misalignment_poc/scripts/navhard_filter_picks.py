#!/usr/bin/env python
"""
Safety-filter picks for the navhard subset, with the rules fixed on dev (safety_filter_select.py: same VARIANTS,
same rank-sum, same fallback to rank-sum over all candidates when nothing survives). Uses only the current-frame
flags of navhard_safety_flags.py; no GT is read. F1_maxll = F1 survivors, then max summed log-prob
(analyze_best_of_n max_loglik); nothing survives -> max_loglik over all. Output: picks.json {token: {F1, F2, F3, F1_maxll}} for
navhard_submission.py --picks, plus filter statistics.

  python scripts/navhard_filter_picks.py <flags.jsonl> <decode_run_dir> <out_dir>
"""
from __future__ import annotations

import json
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from safety_filter_select import VARIANTS, ranksum   # noqa: E402

if __name__ == "__main__":
    flags_path, run, out = sys.argv[1:4]
    os.makedirs(out, exist_ok=True)
    flags = {r["token"]: r["flags"] for r in map(json.loads, open(flags_path)) if "error" not in r}
    picks, stat, missing = {}, defaultdict(list), 0
    for r in map(json.loads, open(os.path.join(run, "records.jsonl"))):
        C = r["candidates"]
        if r["token"] not in flags:
            missing += 1
            continue
        fl, rs, p = flags[r["token"]], ranksum(C, list(range(len(C)))), {}
        for v, keep in VARIANTS.items():
            surv = [i for i in range(len(C)) if keep(fl[i])]
            stat[f"{v}_all_removed"].append(len(surv) == 0)
            stat[f"{v}_natural_removed"].append(not keep(fl[0]))
            stat[f"{v}_n_survivors"].append(len(surv))
            p[v] = ranksum(C, surv) if surv else rs
        surv = [i for i in range(len(C)) if VARIANTS["F1"](fl[i])] or list(range(len(C)))
        p["F1_maxll"] = max(surv, key=lambda i: (sum(C[i]["logprob_steps"]), -i))   # ties -> lowest index, as np.argmax
        picks[r["token"]] = p
    json.dump(picks, open(os.path.join(out, "picks.json"), "w"))
    s = {"n_scenes": len(picks), "no_flags": missing, **{k: float(np.mean(v)) for k, v in stat.items()}}
    json.dump(s, open(os.path.join(out, "filter_stats.json"), "w"), indent=1)
    print(json.dumps(s, indent=1))
