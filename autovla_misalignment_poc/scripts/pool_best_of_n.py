#!/usr/bin/env python
"""
Pool several best_of_n_selection runs (seed replicates) into one population estimate (CPU only).
Units from all runs are grouped by log for the log-cluster bootstrap (seeds = replicates within a log);
population weighting 52 failing / 2695 normal scenes as in analyze_best_of_n.

  python scripts/pool_best_of_n.py outputs/best_of_n_selection_n16_T1 outputs/best_of_n_selection_n16_T1_seed1 ...
"""
import json, random, sys
from collections import defaultdict

import numpy as np

if __name__ == "__main__":
    runs = sys.argv[1:]
    by=defaultdict(lambda: defaultdict(list))
    for ri,run in enumerate(runs):
        U=[json.loads(l) for l in open(f"{run}/units.jsonl")]
        nA=sum(u["group"]=="A-" for u in U); nR=len(U)-nA
        w={"A-":52/nA,"random":(2747-52)/nR}
        for rule in ("min_entropy","max_loglik","prev_consist","medoid"):
            for u in U:
                by[rule][u["log"]].append((w[u["group"]], float(u[rule]["a_minus"])-float(u["normal"]["a_minus"]), u[rule]["fde5"]-u["normal"]["fde5"], float(u["normal"]["a_minus"]), u["normal"]["fde5"]))
    for rule,b in by.items():
        logs=sorted(b)
        def st(ls):
            v=[x for lg in ls for x in b[lg]]; W=sum(x[0] for x in v)
            return [sum(x[0]*x[i] for x in v)/W for i in (1,2,3,4)]
        rng=random.Random(0); bs=np.array([st([rng.choice(logs) for _ in logs]) for _ in range(2000)]); m=st(logs)
        print(f"{rule:13s} pooled {len(runs)} runs: population normal A- {100*m[2]:.2f}% FDE {m[3]:.3f} | dA- {100*m[0]:+.2f}%p [{100*np.percentile(bs[:,0],2.5):+.2f}, {100*np.percentile(bs[:,0],97.5):+.2f}]  dFDE {m[1]:+.3f} [{np.percentile(bs[:,1],2.5):+.3f}, {np.percentile(bs[:,1],97.5):+.3f}]")
