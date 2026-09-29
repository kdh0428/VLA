#!/usr/bin/env python
"""
Analysis of best_of_n_selection (CPU only). Selection rules over the N+1 candidates (row 0 = normal):

  normal        row 0 (T = 0.01)
  prev_consist  candidate whose 5 s trajectory is closest (ADE) to the previous-frame plan
  max_loglik    highest summed log-probability (T = 1) of its own tokens
  min_entropy   lowest mean entropy along its own decode
  medoid        candidate with the smallest summed ADE to all other candidates (self-consistency)
  oracle        candidate with the smallest FDE to GT                      (upper bound, not deployable)
  random_T      the first T > 0 candidate                                    (sampling-only control)

Metrics as in the natural experiment: A- (P/R/A 5 s), failure (A- and FDE5 > 3 m), ADE5 / FDE5;
paired vs normal (McNemar / Wilcoxon, log-cluster CI); population A- estimate (52 failing / 2695 normal).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_action_history import BINARY, cboot, unit_metrics      # noqa: E402
from analyze_prev_action_identity import paired                     # noqa: E402

BINARY.update({"a_minus", "failure"})
POC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
N_TOTAL, N_FAIL = 2747, 52
RULES = ["normal", "prev_consist", "max_loglik", "min_entropy", "medoid", "random_T", "oracle"]


def ade(a, b):
    return float(np.linalg.norm(np.asarray(a, float) - np.asarray(b, float), axis=1).mean())


def pick(r, rule):
    c = r["candidates"]
    if rule == "normal":
        return 0
    if rule == "random_T":
        return 1
    if rule == "prev_consist":
        return int(np.argmin([ade(x["trajectory_pred"], r["prev_traj"]) for x in c])) if r["prev_traj"] else 0
    if rule == "max_loglik":
        return int(np.argmax([sum(x["logprob_steps"]) for x in c]))
    if rule == "min_entropy":
        return int(np.argmin([np.mean(x["entropy_steps"]) for x in c]))
    if rule == "medoid":
        return int(np.argmin([sum(ade(x["trajectory_pred"], y["trajectory_pred"]) for y in c) for x in c]))
    if rule == "oracle":
        gt = np.asarray(r["trajectory_gt"], float)[:, :2]
        return int(np.argmin([np.linalg.norm(np.asarray(x["trajectory_pred"])[-1] - gt[-1]) for x in c]))
    raise ValueError(rule)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=os.path.join(POC_DIR, "outputs/best_of_n_selection"))
    args = ap.parse_args()
    recs = [json.loads(l) for l in open(os.path.join(args.run, "records.jsonl"))]
    units = []
    for r in recs:
        u = {"token": r["token"], "log": r["log"], "group": r["group"], "picked": {}}
        for rule in RULES:
            i = pick(r, rule)
            x = r["candidates"][i]
            m = unit_metrics(x["action_idx"], x["trajectory_pred"], x["entropy_steps"], r["gt"], r["trajectory_gt"], 0)
            m["failure"] = m.pop("amplification")
            m["a_minus"] = not m["recovery"]
            u[rule] = m
            u["picked"][rule] = i
        # spread of the candidate set (how different the model's own samples are)
        c = r["candidates"]
        u["cand_spread"] = float(np.mean([ade(a["trajectory_pred"], b["trajectory_pred"]) for a in c for b in c if a is not b]))
        units.append(u)
    M = ["a_minus", "failure", "ade5", "fde5"]
    S = {"n": {g: sum(u["group"] == g for u in units) for g in ("A-", "random")}, "groups": {}, "population_a_minus_rate": {}}
    for g in ("A-", "random"):
        us = [u for u in units if u["group"] == g]
        S["groups"][g] = {"rules": {rule: {m: cboot([(u["log"], float(u[rule][m])) for u in us]) for m in M} for rule in RULES},
                          "vs_normal": {rule: {m: paired(us, rule, "normal", m) for m in M} for rule in RULES[1:]},
                          "picked_normal_frac": {rule: float(np.mean([u["picked"][rule] == 0 for u in us])) for rule in RULES},
                          "cand_spread": float(np.mean([u["cand_spread"] for u in us]))}
    for rule in RULES:
        a = S["groups"]["A-"]["rules"][rule]["a_minus"]["mean"]
        b = S["groups"]["random"]["rules"][rule]["a_minus"]["mean"]
        S["population_a_minus_rate"][rule] = (N_FAIL * a + (N_TOTAL - N_FAIL) * b) / N_TOTAL
    json.dump(S, open(os.path.join(args.run, "summary.json"), "w"), indent=1)
    with open(os.path.join(args.run, "units.jsonl"), "w") as f:
        for u in units:
            f.write(json.dumps(u) + "\n")
    L = [f"units {S['n']}"]
    for g in ("A-", "random"):
        b = S["groups"][g]
        L.append(f"\n== {g} ==  candidate spread (mean pairwise ADE) {b['cand_spread']:.2f} m")
        L.append(f"{'rule':13s} {'A- %':>20s} {'fail %':>7s} {'FDE5':>6s} {'dA- %p (CI)':>24s} {'p':>8s} {'dFDE':>7s} {'p':>8s} {'=normal':>8s}")
        for rule in RULES:
            x = b["rules"][rule]; a = x["a_minus"]
            v = b["vs_normal"].get(rule)
            if v:
                d, f = v["a_minus"], v["fde5"]
                ds, ps = f"{100*d['mean']:+.1f} [{100*d['ci95'][0]:+.1f},{100*d['ci95'][1]:+.1f}]", f"{d['mcnemar']['p']:.2g}"
                fs, fp = f"{f['mean']:+.2f}", f"{f['wilcoxon_p']:.2g}"
            else:
                ds = ps = fs = fp = "–"
            L.append(f"{rule:13s} {100*a['mean']:6.1f} [{100*a['ci95'][0]:5.1f},{100*a['ci95'][1]:5.1f}] {100*x['failure']['mean']:7.1f} "
                     f"{x['fde5']['mean']:6.2f} {ds:>24s} {ps:>8s} {fs:>7s} {fp:>8s} {b['picked_normal_frac'][rule]:8.2f}")
    L.append("\npopulation natural A- rate estimate: " + "  ".join(f"{k} {100*v:.2f}%" for k, v in S["population_a_minus_rate"].items()))
    txt = "\n".join(L)
    print(txt)
    open(os.path.join(args.run, "analysis_console.txt"), "w").write(txt + "\n")


if __name__ == "__main__":
    main()
