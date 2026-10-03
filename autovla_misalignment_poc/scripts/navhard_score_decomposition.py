#!/usr/bin/env python
"""
Where does an EPDMS gain come from? Token-level Shapley decomposition of score differences (P2 "progress sacrifice").

Per token of the official CSV: score = M * W with M = NC * DAC * DDC * TLC (multiplicative) and
W = (5 EP + 5 TTC + 2 LK + 2 HC + 2 EC) / 16 (weighted). For condition B vs baseline A:
  Δscore = 0.5 (M_A + M_B) ΔW  +  0.5 (W_A + W_B) ΔM,   and  0.5 (M_A + M_B) ΔW = Σ_i 0.5 (M_A + M_B) (c_i / 16) Δx_i
so the gain splits into the multiplicative (safety) part and one term per weighted metric (progress, TTC, lane keeping,
history comfort, extended comfort). Reported separately for stage-1 (original) and stage-2 (synthetic) tokens as unweighted
token means (the official stage aggregates additionally weight stage-2 tokens), with a log-cluster bootstrap 95% CI.

  python scripts/navhard_score_decomposition.py <out.json> --base NAME CSV_HALF1 CSV_HALF2 --cond NAME CSV_HALF1 CSV_HALF2 [...]
"""
from __future__ import annotations

import argparse
import json
import random

import numpy as np
import pandas as pd

MULT = ["no_at_fault_collisions", "drivable_area_compliance", "driving_direction_compliance", "traffic_light_compliance"]
WEIGHTED = [("ego_progress", 5), ("time_to_collision_within_bound", 5), ("lane_keeping", 2), ("history_comfort", 2),
            ("two_frame_extended_comfort", 2)]


def load(paths):
    rows = []
    for p in paths:
        df = pd.read_csv(p)
        df = df[~df["token"].astype(str).str.startswith("extended_pdm_score")]
        for st, sfx in (("stage1", "_stage_one"), ("stage2", "_stage_two")):
            d = df[df[f"ego_progress{sfx}"].notna()]
            for _, r in d.iterrows():
                x = {m: float(r[m + sfx]) for m in MULT + [w for w, _ in WEIGHTED]}
                x.update(token=str(r["token"]), stage=st, score=float(r["score"]))
                rows.append(x)
    return {(x["stage"], x["token"]): x for x in rows}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--base", nargs=3, required=True)
    ap.add_argument("--cond", nargs=3, action="append", required=True)
    ap.add_argument("--log-map", nargs="+", required=True, help="JSON files {token: log} (stage-1 and synthetic tokens)")
    a = ap.parse_args()
    log_of = {}
    for f in a.log_map:
        log_of.update(json.load(open(f)))
    A = load(a.base[1:])
    res = {"base": a.base[0], "conds": {}}
    for name, *paths in a.cond:
        B = load(paths)
        res["conds"][name] = {}
        for st in ("stage1", "stage2"):
            keys = sorted(k for k in A if k in B and k[0] == st)
            comp = {"total": [], "multiplicative": [], **{w: [] for w, _ in WEIGHTED}, "check": []}
            for k in keys:
                x, y = A[k], B[k]
                MA, MB = np.prod([x[m] for m in MULT]), np.prod([y[m] for m in MULT])
                WA = sum(c * x[w] for w, c in WEIGHTED) / 16; WB = sum(c * y[w] for w, c in WEIGHTED) / 16
                comp["total"].append(y["score"] - x["score"]); comp["check"].append(MB * WB - MA * WA)
                comp["multiplicative"].append(0.5 * (WA + WB) * (MB - MA))
                for w, c in WEIGHTED:
                    comp[w].append(0.5 * (MA + MB) * c / 16 * (y[w] - x[w]))
            logs = [log_of.get(k[1], "?") for k in keys]
            ul = sorted(set(logs)); idx = {l: [i for i, z in enumerate(logs) if z == l] for l in ul}
            rng = random.Random(0); boots = [[rng.choice(ul) for _ in ul] for _ in range(2000)]
            out = {"n_tokens": len(keys), "n_logs": len(ul),
                   "score_vs_product_maxabs": float(np.max(np.abs(np.array(comp["total"]) - np.array(comp["check"])))) if keys else None}
            for c_, v in comp.items():
                if c_ == "check":
                    continue
                v = np.array(v); bs = sorted(np.mean(np.concatenate([v[idx[l]] for l in bt])) for bt in boots)
                out[c_] = {"mean": float(v.mean()), "ci95": [float(bs[49]), float(bs[1949])]}
            res["conds"][name][st] = out
    json.dump(res, open(a.out, "w"), indent=1)
    for name, d in res["conds"].items():
        for st, o in d.items():
            parts = " ".join(f"{k}={o[k]['mean']:+.4f}" for k in ["total", "multiplicative"] + [w for w, _ in WEIGHTED])
            print(f"{name:16s} {st}: {parts}  (score-product mismatch {o['score_vs_product_maxabs']:.2g})")
