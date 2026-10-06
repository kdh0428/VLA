#!/usr/bin/env python
"""Exp 34 token level: share of units whose normal-row translation token at step k (k=2,3,4) differs from the
unperturbed greedy chunk R (descriptive figure quoted in RESULTS.md §3; no CI in the original). Also ensemble weights
from scripts/cross_domain_temporal/action_ensemble.py rule (temp -0.8, horizon 4)."""
import json, numpy as np
SRC = "/root/VLA/autovla_misalignment_poc/outputs/cross_domain_temporal_replication/token_level/units.jsonl"
ch = np.zeros(3); n = 0
for l in open(SRC):
    f = json.loads(l); R = np.array(f["R"])
    for u in f["units"]:
        S = np.array(u["rows"]["normal"]); ch += (S[1:, 0] != R[1:, 0]); n += 1
print("N units", n, "changed counts step2..4", ch.astype(int).tolist(), "pct", (100 * ch / n).round(2).tolist())
w = np.exp(0.8 * np.arange(4)); w /= w.sum()
# stacking order in ActionEnsembler: index 0 = oldest chunk (its step 4) ... index 3 = newest chunk (its step 1)
print("weights oldest(step4 of chunk t-3) -> newest(step1 of chunk t):", w.round(4).tolist(), "share steps2-4:", round(1 - w[-1], 4))
