#!/usr/bin/env python
"""
Safety-filter ablation picks for navhard (outputs/safety_filter_ablation/PROTOCOL.md). Same F1 filter
(safety_filter_select.VARIANTS["F1"]), current-frame flags only, no GT, nothing tuned.

  filter_only            candidate 0 if it passes, else the lowest-index passing candidate; none pass -> candidate 0
  filter_random_s{0,1,2} uniform passing candidate; none pass -> uniform over all (random.Random(f"{seed}:{token}"))
  filter_ranksum_natfb   rank-sum among passing; none pass -> candidate 0
  filter_maxll_natfb     max summed log-prob among passing; none pass -> candidate 0

  python scripts/navhard_ablation_picks.py <flags.jsonl> <decode_run_dir> <out_dir>
"""
from __future__ import annotations

import json
import os
import random
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from safety_filter_select import VARIANTS, ranksum   # noqa: E402

KEEP = VARIANTS["F1"]


def maxll(C, sub):
    return max(sub, key=lambda i: (sum(C[i]["logprob_steps"]), -i))   # ties -> lowest index, as np.argmax


if __name__ == "__main__":
    flags_path, run, out = sys.argv[1:4]
    os.makedirs(out, exist_ok=True)
    flags = {r["token"]: r["flags"] for r in map(json.loads, open(flags_path)) if "error" not in r}
    picks, stat = {}, {"natural_kept": [], "none_pass": []}
    for r in map(json.loads, open(os.path.join(run, "records.jsonl"))):
        t, C = r["token"], r["candidates"]
        fl = flags[t]
        surv = [i for i in range(len(C)) if KEEP(fl[i])]
        p = {"filter_only": 0 if (not surv or surv[0] == 0) else surv[0]}
        for s in (0, 1, 2):
            p[f"filter_random_s{s}"] = random.Random(f"{s}:{t}").choice(surv or list(range(len(C))))
        p["filter_ranksum_natfb"] = ranksum(C, surv) if surv else 0
        p["filter_maxll_natfb"] = maxll(C, surv) if surv else 0
        picks[t] = p
        stat["natural_kept"].append(p["filter_only"] == 0); stat["none_pass"].append(not surv)
    json.dump(picks, open(os.path.join(out, "picks.json"), "w"))
    s = {"n_scenes": len(picks), **{k: float(np.mean(v)) for k, v in stat.items()}}
    json.dump(s, open(os.path.join(out, "picks_stats.json"), "w"), indent=1)
    print(json.dumps(s))
