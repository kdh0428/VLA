#!/usr/bin/env python
"""
Analysis of receding_horizon_replanning (CPU only).

Per condition and group (A-, A+): position error vs the log at 2/5/8 s, ADE over 8 s, the P/R/A
A-label at 5 s (same rule as every experiment), "failure" = A- at 5 s AND FDE5 > 3 m (the
amplification definition), and the executed-vs-logged pose gap at replanning points (size of the
camera/state inconsistency). Paired contrasts vs `ol` (5 s metrics) and vs m{m}_log (8 s) with
log-cluster bootstrap CIs. Only scenes whose log covers the full 8 s enter the 8 s metrics.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_action_history import AMP_FDE, a_eval, cboot   # noqa: E402

POC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ORDER = ["ol", "m8_log", "m8_exec", "m8_exec_prev", "m4_log", "m4_exec", "m4_exec_prev", "m2_log", "m2_exec", "m2_exec_prev"]


def metrics(c, log_local, full8):
    e = np.asarray(c["err"], float)
    ex = np.asarray(c["executed_local"], float)
    out = {"e2": float(e[3]) if len(e) > 3 else None, "e5": float(e[9]) if len(e) > 9 else None,
           "e8": float(e[15]) if (full8 and len(e) > 15) else None,
           "ade8": float(e[:16].mean()) if (full8 and len(e) > 15) else None}
    if len(ex) >= 10:
        ok = a_eval(ex[:10, :2].tolist(), np.asarray(log_local)[:10].tolist())
        out["A_plus5"] = bool(ok)
        out["fail5"] = (not ok) and float(e[9]) > AMP_FDE
    else:
        out["A_plus5"] = out["fail5"] = None
    g = [x["pos_gap_m"] for x in c.get("replan_gaps", [])]
    out["max_replan_gap"] = float(max(g)) if g else 0.0
    out["mean_replan_gap"] = float(np.mean(g)) if g else 0.0
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=os.path.join(POC_DIR, "outputs/receding_horizon_replanning"))
    args = ap.parse_args()
    recs = [json.loads(l) for l in open(os.path.join(args.run, "records.jsonl"))]
    conds = [c for c in ORDER if c in recs[0]["conditions"]]
    units = []
    for r in recs:
        full8 = r["horizon_steps"] >= 16 and all(r["conditions"][c]["n_steps"] >= 16 for c in conds if c != "ol")
        u = {"token": r["token"], "log": r["log"], "group": r["group"], "full8": full8}
        for c in conds:
            u[c] = metrics(r["conditions"][c], r["log_local"], full8)
        units.append(u)
    M = ["e2", "e5", "e8", "ade8", "A_plus5", "fail5", "mean_replan_gap", "max_replan_gap"]
    S = {"n": {g: sum(u["group"] == g for u in units) for g in ("A-", "A+")},
         "n_full8": {g: sum(u["group"] == g and u["full8"] for u in units) for g in ("A-", "A+")}, "groups": {}}
    for g in ("A-", "A+"):
        us = [u for u in units if u["group"] == g]
        blk = {"conditions": {c: {m: cboot([(u["log"], u[c][m]) for u in us if u[c][m] is not None]) for m in M} for c in conds},
               "vs_ol_5s": {}, "exec_vs_log_8s": {}, "prev_vs_exec_8s": {}}
        for c in conds[1:]:
            blk["vs_ol_5s"][c] = {m: cboot([(u["log"], float(u[c][m]) - float(u["ol"][m])) for u in us
                                            if u[c][m] is not None and u["ol"][m] is not None]) for m in ("e5", "fail5", "A_plus5")}
        for m_ in (2, 4, 8):
            a, b, p = f"m{m_}_log", f"m{m_}_exec", f"m{m_}_exec_prev"
            if a in conds and b in conds:
                blk["exec_vs_log_8s"][f"m{m_}"] = {m: cboot([(u["log"], u[b][m] - u[a][m]) for u in us if u[a][m] is not None and u[b][m] is not None])
                                                   for m in ("e8", "ade8")}
            if p in conds and b in conds:
                blk["prev_vs_exec_8s"][f"m{m_}"] = {m: cboot([(u["log"], u[p][m] - u[b][m]) for u in us if u[p][m] is not None and u[b][m] is not None])
                                                    for m in ("e5", "e8", "ade8")}
        S["groups"][g] = blk
    json.dump(S, open(os.path.join(args.run, "summary.json"), "w"), indent=1)
    with open(os.path.join(args.run, "units.jsonl"), "w") as f:
        for u in units:
            f.write(json.dumps(u) + "\n")

    def f(x, pct=False):
        if not x:
            return "–"
        k = 100 if pct else 1
        return f"{k*x['mean']:.2f}" if not pct else f"{k*x['mean']:.1f}"
    L = [f"scenes {S['n']}  full 8 s {S['n_full8']}"]
    for g in ("A-", "A+"):
        b = S["groups"][g]["conditions"]
        L.append(f"\n== {g} ==")
        L.append(f"{'condition':14s} {'err2s':>6s} {'err5s':>6s} {'err8s':>6s} {'ADE8':>6s} {'fail5%':>7s} {'A+5%':>6s} {'gap':>6s}")
        for c in conds:
            x = b[c]
            L.append(f"{c:14s} {f(x['e2']):>6s} {f(x['e5']):>6s} {f(x['e8']):>6s} {f(x['ade8']):>6s} "
                     f"{f(x['fail5'], True):>7s} {f(x['A_plus5'], True):>6s} {f(x['mean_replan_gap']):>6s}")
        for k, v in S["groups"][g]["prev_vs_exec_8s"].items():
            e8 = v["e8"]
            if e8:
                L.append(f"  prev-stab - exec {k}: err8 {e8['mean']:+.2f} [{e8['ci95'][0]:+.2f}, {e8['ci95'][1]:+.2f}]")
        for k, v in S["groups"][g]["exec_vs_log_8s"].items():
            e8 = v["e8"]
            if e8:
                L.append(f"  exec - log {k}: err8 {e8['mean']:+.2f} [{e8['ci95'][0]:+.2f}, {e8['ci95'][1]:+.2f}]")
    txt = "\n".join(L)
    print(txt)
    open(os.path.join(args.run, "analysis_console.txt"), "w").write(txt + "\n")


if __name__ == "__main__":
    main()
