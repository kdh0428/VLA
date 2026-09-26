#!/usr/bin/env python
"""
Summarise the L35 MLP intervention.

Two questions, kept separate:

  Is the component a causal lever at all?   -> the oracle arm, which writes along the
        per-sample (w_GT - w_rival) direction. It uses ground truth, so it is a CEILING,
        not a method: it only answers "can editing this component change the answer".

  Is there a usable, GT-free lever?         -> mean_swap and steer, fit on a disjoint set
        of logs, judged against random (matched norm) and null_swap (failure-group mean).

Collateral damage is measured on already-correct samples, which must not be broken.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from collections import defaultdict

import numpy as np

POC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(max(p * (1 - p) / n + z * z / (4 * n * n), 0)) / d
    return (max(0, c - h), min(1, c + h))


def mcnemar(a, b):
    from scipy.stats import binomtest
    a = np.asarray(a, bool); b = np.asarray(b, bool)
    n01 = int((~a & b).sum()); n10 = int((a & ~b).sum())
    if n01 + n10 == 0:
        return {"n01": n01, "n10": n10, "p": 1.0}
    return {"n01": n01, "n10": n10, "p": float(binomtest(n10, n01 + n10, 0.5).pvalue)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=os.path.join(POC_DIR, "outputs/mlp_intervention/mlp_intervention_raw.json"))
    ap.add_argument("--outdir", default=os.path.join(POC_DIR, "outputs/mlp_intervention"))
    args = ap.parse_args()

    D = json.load(open(args.raw))
    rows = D["rows"]
    fail = [r for r in rows if r["kind"] == "failure"]
    ctrl = [r for r in rows if r["kind"] == "control"]
    arms = list(rows[0]["arms"].keys())
    print(f"L{D['layer']} intervention, ||d||={D['d_norm']:.1f}")
    print(f"{len(fail)} failure samples, {len(ctrl)} correct controls, "
          f"{len({r['log'] for r in rows})} logs\n")

    out = {"layer": D["layer"], "d_norm": D["d_norm"],
           "n_failure": len(fail), "n_control": len(ctrl), "arms": {}}

    print("=" * 96)
    print(f"{'Arm':14s} | {'step-0 = GT':>18s} | {'downstream err':>14s} | {'ADE':>7s} | "
          f"{'coarse ok':>9s} | {'broke ctrl':>10s}")
    print("-" * 96)
    base_fix = np.array([r["arms"]["baseline"]["token0_is_gt"] for r in fail])
    base_ctrl_ok = np.array([r["arms"]["baseline"]["token0_is_gt"] for r in ctrl])
    for a in arms:
        fx = np.array([r["arms"][a]["token0_is_gt"] for r in fail])
        de = np.array([r["arms"][a]["downstream_error_rate"] for r in fail], float)
        ad = np.array([r["arms"][a]["ade"] for r in fail], float)
        cm = np.array([r["arms"][a]["coarse_match"] for r in fail])
        ck = np.array([r["arms"][a]["token0_is_gt"] for r in ctrl])
        broke = int((base_ctrl_ok & ~ck).sum())
        lo, hi = wilson(int(fx.sum()), len(fx))
        print(f"{a:14s} | {int(fx.sum()):4d}/{len(fx):<4d} {fx.mean():6.1%} "
              f"[{lo:.0%},{hi:.0%}] | {np.nanmean(de):13.1%} | {np.nanmean(ad):7.3f} | "
              f"{cm.mean():8.1%} | {broke:3d}/{len(ck):<3d} {broke/max(len(ck),1):5.0%}")
        out["arms"][a] = {
            "fix_rate": float(fx.mean()), "fix_ci95": [lo, hi],
            "n_fixed": int(fx.sum()), "n": len(fx),
            "downstream_error": float(np.nanmean(de)), "ade": float(np.nanmean(ad)),
            "coarse_match": float(cm.mean()),
            "broke_controls": broke, "n_controls": len(ck),
        }
    print("=" * 96)

    print("\nvs baseline (McNemar on step-0 = GT, failure samples):")
    for a in arms:
        if a == "baseline":
            continue
        fx = np.array([r["arms"][a]["token0_is_gt"] for r in fail])
        m = mcnemar(fx, base_fix)
        out["arms"][a]["mcnemar_vs_baseline"] = m
        print(f"  {a:14s} n01={m['n01']:3d} n10={m['n10']:3d} p={m['p']:.4g}")

    print("\nGT-free arms vs their matched random control:")
    for a in arms:
        if not a.startswith("steer"):
            continue
        alpha = a.split("_a")[1]
        r_ = f"random_a{alpha}"
        if r_ not in arms:
            continue
        fs = np.array([x["arms"][a]["token0_is_gt"] for x in fail])
        fr = np.array([x["arms"][r_]["token0_is_gt"] for x in fail])
        m = mcnemar(fs, fr)
        print(f"  steer a{alpha} vs random a{alpha}: "
              f"{fs.mean():.1%} vs {fr.mean():.1%}  n01={m['n01']} n10={m['n10']} p={m['p']:.4g}")
        out["arms"][a]["vs_random"] = {"steer": float(fs.mean()), "random": float(fr.mean()), **m}

    with open(os.path.join(args.outdir, "mlp_intervention_summary.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {args.outdir}/mlp_intervention_summary.json")


if __name__ == "__main__":
    main()
