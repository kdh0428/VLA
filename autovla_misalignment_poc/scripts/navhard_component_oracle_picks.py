#!/usr/bin/env python
"""
Picks for P2 (safety-filter components, outputs/safety_filter_components) and P3 (candidate oracle, outputs/candidate_oracle).

P2, same frozen flags and the "filter only" selection of experiment 27 (candidate 0 if it passes, else the lowest-index
passing candidate, none pass -> candidate 0):
  comp_collision  passes = no_at_fault_collisions flag >= 1
  comp_dac        passes = drivable_area_compliance flag >= 1
  comp_both       passes = both (identical to experiment 27 filter_only; written only to check that)
P3: cand00 .. cand16 = candidate k in every scene (k = 0 natural T 0.01, 1-16 samples at T 1.0), to score every
candidate with the official two-stage pipeline.

  python scripts/navhard_component_oracle_picks.py <flags.jsonl> <decode_run_dir> <out_dir>
"""
import json
import os
import sys

if __name__ == "__main__":
    flags_path, run, out = sys.argv[1:4]
    os.makedirs(out, exist_ok=True)
    flags = {r["token"]: r["flags"] for r in map(json.loads, open(flags_path)) if "error" not in r}
    KEEP = {"comp_collision": lambda f: f[0] >= 1, "comp_dac": lambda f: f[1] >= 1, "comp_both": lambda f: f[0] >= 1 and f[1] >= 1}
    comp, cand, stat = {}, {}, {k: [0, 0] for k in KEEP}
    for r in map(json.loads, open(os.path.join(run, "records.jsonl"))):
        t, n = r["token"], len(r["candidates"])
        assert n == 17, (t, n)
        p = {}
        for name, keep in KEEP.items():
            surv = [i for i in range(n) if keep(flags[t][i])]
            p[name] = 0 if (not surv or surv[0] == 0) else surv[0]
            stat[name][0] += p[name] != 0; stat[name][1] += not surv
        comp[t] = p
        cand[t] = {f"cand{k:02d}": k for k in range(n)}
    json.dump(comp, open(os.path.join(out, "component_picks.json"), "w"))
    json.dump(cand, open(os.path.join(out, "candidate_picks.json"), "w"))
    N = len(comp)
    print(json.dumps({k: {"natural_replaced": v[0] / N, "none_pass": v[1] / N} for k, v in stat.items()}))
