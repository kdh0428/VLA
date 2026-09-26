#!/usr/bin/env python
"""Summarise mode-conditioned steering and build the per-mode table."""
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
    p = k / n; d = 1 + z * z / n
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
    ap.add_argument("--raw", default=os.path.join(POC_DIR, "outputs/mode_steering/mode_steering_raw.json"))
    ap.add_argument("--outdir", default=os.path.join(POC_DIR, "outputs/mode_steering"))
    args = ap.parse_args()
    D = json.load(open(args.raw))
    rows = D["rows"]
    fail = [r for r in rows if r["kind"] == "failure"]
    ctrl = [r for r in rows if r["kind"] == "control"]
    alphas = D["alphas"]
    print(f"modes {D['modes']}  ||d_global||={D['scale']:.1f}")
    print(f"cos(mode, global): { {k: round(v,3) for k,v in D['cos_mode_global'].items()} }")
    print(f"cos(mode_i, mode_j): { {k: round(v,3) for k,v in D['cos_mode_pairs'].items()} }")
    print(f"\n{len(fail)} failures, {len(ctrl)} controls, {len({r['log'] for r in rows})} logs\n")

    arms = ["baseline"] + [f"{k}_a{a}" for a in alphas
                           for k in ("global", "mode", "wrongmode", "random")]

    def col(rs, arm, key):
        return np.array([r["arms"][arm][key] for r in rs], dtype=float)

    print("=" * 90)
    print(f"{'Arm':16s} | {'first-token fix':>20s} | {'downstream':>11s} | {'ADE':>7s} | {'broke ctrl':>10s}")
    print("-" * 90)
    base_ctrl = np.array([r["arms"]["baseline"]["is_gt"] for r in ctrl], bool)
    out = {"modes": D["modes"], "cos_mode_global": D["cos_mode_global"],
           "cos_mode_pairs": D["cos_mode_pairs"], "arms": {}}
    for a in arms:
        fx = np.array([r["arms"][a]["is_gt"] for r in fail], bool)
        lo, hi = wilson(int(fx.sum()), len(fx))
        dn = np.nanmean(col(fail, a, "downstream"))
        adv = np.nanmean(col(fail, a, "ade"))
        ck = np.array([r["arms"][a]["is_gt"] for r in ctrl], bool)
        broke = int((base_ctrl & ~ck).sum())
        print(f"{a:16s} | {int(fx.sum()):3d}/{len(fx):<3d} {fx.mean():6.1%} "
              f"[{lo:.0%},{hi:.0%}] | {dn:10.1%} | {adv:7.3f} | {broke:2d}/{len(ck):<2d} "
              f"{broke/max(len(ck),1):5.0%}")
        out["arms"][a] = {"fix": float(fx.mean()), "ci95": [lo, hi], "n": len(fx),
                          "downstream": float(dn), "ade": float(adv),
                          "broke": broke, "n_ctrl": len(ck)}
    print("=" * 90)

    print("\nthe decisive contrast — own mode vs matched random, and vs wrong mode:")
    for a in alphas:
        m = np.array([r["arms"][f"mode_a{a}"]["is_gt"] for r in fail], bool)
        rd = np.array([r["arms"][f"random_a{a}"]["is_gt"] for r in fail], bool)
        wm = np.array([r["arms"][f"wrongmode_a{a}"]["is_gt"] for r in fail], bool)
        gl = np.array([r["arms"][f"global_a{a}"]["is_gt"] for r in fail], bool)
        s1, s2, s3 = mcnemar(m, rd), mcnemar(m, wm), mcnemar(m, gl)
        print(f"  a={a}: mode {m.mean():.1%} | wrong-mode {wm.mean():.1%} | "
              f"global {gl.mean():.1%} | random {rd.mean():.1%}")
        print(f"        mode vs random    n01={s1['n01']} n10={s1['n10']} p={s1['p']:.3g}")
        print(f"        mode vs wrongmode n01={s2['n01']} n10={s2['n10']} p={s2['p']:.3g}")
        out[f"tests_a{a}"] = {"mode_vs_random": s1, "mode_vs_wrongmode": s2,
                             "mode_vs_global": s3}

    print("\nper failure mode (fix rate):")
    by = defaultdict(list)
    for r in fail:
        by[r["mode"]].append(r)
    hdr = f"  {'mode':12s} {'n':>4s} | " + " ".join(f"{k:>10s}" for k in
                                                    ("global", "mode", "wrongmode", "random"))
    print(hdr)
    per = {}
    for m, rs in sorted(by.items(), key=lambda kv: -len(kv[1])):
        a = alphas[-1]
        vals = {k: float(np.mean([x["arms"][f"{k}_a{a}"]["is_gt"] for x in rs]))
                for k in ("global", "mode", "wrongmode", "random")}
        per[m] = {"n": len(rs), **vals}
        print(f"  {m:12s} {len(rs):>4d} | " + " ".join(f"{vals[k]:9.1%}" for k in
                                                       ("global", "mode", "wrongmode", "random")))
    out["per_mode"] = per
    with open(os.path.join(args.outdir, "mode_steering_summary.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {args.outdir}/mode_steering_summary.json")


if __name__ == "__main__":
    main()
