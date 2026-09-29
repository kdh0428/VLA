#!/usr/bin/env python
"""
Safety-filtered rank-sum selection (experiment 22; CPU only).

Uses the current-frame flags of safety_filter_flags.py (constant-velocity extrapolation of the current
objects + map; no future frames). Pre-specified variants:
  F1  drop candidates with a predicted collision (no_collision < 1) or drivable-area exit (drivable_area < 1)
  F2  F1 + drop predicted oncoming-lane travel (driving_direction < 1)
  F3  F1 + drop predicted TTC violation (ttc < 1)
then rank-sum among the survivors; if nothing survives, rank-sum over all candidates (fallback).
Writes picks.json {token: {normal, ranksum, F1, F2, F3}} for pdm_score_candidates --picks-file, and the
open-loop comparison (A-, amplification, FDE5; log-cluster bootstrap vs normal).

  python scripts/safety_filter_select.py <flags.jsonl> <out_dir> <run_dir> [<run_dir> ...]
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

VARIANTS = {"F1": lambda f: f[0] >= 1 and f[1] >= 1,
            "F2": lambda f: f[0] >= 1 and f[1] >= 1 and f[2] >= 1,
            "F3": lambda f: f[0] >= 1 and f[1] >= 1 and f[3] >= 1}


def ranksum(C, sub):
    re_ = rankdata([np.mean(C[i]["entropy_steps"]) for i in sub]); rl_ = rankdata([-np.sum(C[i]["logprob_steps"]) for i in sub])
    return sub[int(np.argmin(re_ + rl_ + 1e-6 * re_))]


def main() -> None:
    flags_path, out_dir, runs = sys.argv[1], sys.argv[2], sys.argv[3:]
    os.makedirs(out_dir, exist_ok=True)
    flags = {}
    for l in open(flags_path):
        r = json.loads(l)
        if "error" not in r:
            flags[r["token"]] = r["flags"]
    picks, rows, stat = {}, [], defaultdict(list)
    for run in runs:
        for l in open(os.path.join(run, "records.jsonl")):
            r = json.loads(l)
            if r.get("stub_cot") or r["token"] not in flags:
                continue
            C, fl = r["candidates"], flags[r["token"]]
            p = {"normal": 0, "ranksum": ranksum(C, list(range(len(C))))}
            for v, keep in VARIANTS.items():
                surv = [i for i in range(len(C)) if keep(fl[i])]
                stat[f"{v}_all_removed"].append(len(surv) == 0)
                stat[f"{v}_natural_removed"].append(not keep(fl[0]))
                stat[f"{v}_n_survivors"].append(len(surv))
                p[v] = ranksum(C, surv) if surv else p["ranksum"]
            picks[r["token"]] = p
            gt = np.asarray(r["trajectory_gt"], float)[:, :2]
            m = {}
            for k, i in p.items():
                T = np.asarray(C[i]["trajectory_pred"], float); e = np.linalg.norm(T - gt, axis=1)
                am = float(not a_eval(T.tolist(), gt.tolist()))
                m[k] = {"a_minus": am, "amp": float(am == 1 and e[-1] > AMP_FDE), "fde": float(e[-1])}
            rows.append({"log": r["log"], "m": m})
    json.dump(picks, open(os.path.join(out_dir, "picks.json"), "w"))
    by = defaultdict(list)
    for x in rows:
        by[x["log"]].append(x)
    logs = sorted(by); rng = random.Random(0)
    boots = [[rng.choice(logs) for _ in logs] for _ in range(2000)]
    S = {"n_scenes": len(rows), "filter_stats": {k: float(np.mean(v)) for k, v in stat.items()}, "open_loop": {}}
    for k in ["normal", "ranksum", *VARIANTS]:
        e = {}
        for mm in ("a_minus", "amp", "fde"):
            d = lambda xs: np.mean([x["m"][k][mm] - x["m"]["normal"][mm] for x in xs])   # noqa: E731
            e[mm] = float(np.mean([x["m"][k][mm] for x in rows])); e[f"d_{mm}"] = float(d(rows))
            e[f"d_{mm}_ci95"] = [float(np.percentile([d([x for lg in b for x in by[lg]]) for b in boots], q)) for q in (2.5, 97.5)]
        S["open_loop"][k] = e
    json.dump(S, open(os.path.join(out_dir, "open_loop_summary.json"), "w"), indent=1)
    print(f"scenes {S['n_scenes']}")
    for v in VARIANTS:
        print(f"  {v}: survivors per scene {S['filter_stats'][v+'_n_survivors']:.1f}/17, all removed {100*S['filter_stats'][v+'_all_removed']:.1f}%, "
              f"natural plan flagged {100*S['filter_stats'][v+'_natural_removed']:.1f}%")
    print(f"{'rule':8s} {'A- %':>6s} {'dA- %p [CI]':>24s} {'amp %':>6s} {'FDE5':>6s} {'dFDE [CI]':>24s}")
    for k, e in S["open_loop"].items():
        print(f"{k:8s} {100*e['a_minus']:6.2f} {100*e['d_a_minus']:+6.2f} [{100*e['d_a_minus_ci95'][0]:+.2f},{100*e['d_a_minus_ci95'][1]:+.2f}] "
              f"{100*e['amp']:6.2f} {e['fde']:6.3f} {e['d_fde']:+.3f} [{e['d_fde_ci95'][0]:+.3f},{e['d_fde_ci95'][1]:+.3f}]")


if __name__ == "__main__":
    main()
