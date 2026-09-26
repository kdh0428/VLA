#!/usr/bin/env python
"""
Analyse the first-error teacher-forcing counterfactual.

Every sample contributes one value per arm from the SAME shared prefix, so the comparison
is paired and the right tests are paired ones: a bootstrap over samples (resampling whole
samples, keeping the arms together) and McNemar for the binary outcomes.

The random arm averages its seeds within a sample first, so a sample with 3 random draws
still counts once — otherwise the random arm would silently get 3x the weight.
"""
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict

import numpy as np

POC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARMS = ("original", "gt", "plausible", "random")


def agg_arm(per: list[dict], key: str):
    """Mean over the seeds of one arm within one sample; None if undefined."""
    vals = [p[key] for p in per if p.get(key) is not None]
    if not vals:
        return None
    if isinstance(vals[0], bool):
        return float(np.mean([float(v) for v in vals]))
    return float(np.mean(vals))


def boot_ci(x: np.ndarray, n_boot: int = 10000, seed: int = 0):
    x = np.asarray([v for v in x if v is not None and np.isfinite(v)], float)
    if x.size == 0:
        return (float("nan"),) * 3
    rng = np.random.default_rng(seed)
    m = rng.choice(x, size=(n_boot, x.size), replace=True).mean(1)
    return float(x.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def paired_boot_diff(a: np.ndarray, b: np.ndarray, n_boot: int = 10000, seed: int = 0):
    """Bootstrap CI of mean(a) - mean(b), resampling SAMPLES (keeps arms paired)."""
    m = np.array([(x is not None and np.isfinite(x)) and (y is not None and np.isfinite(y))
                  for x, y in zip(a, b)])
    a2, b2 = np.asarray(a, float)[m], np.asarray(b, float)[m]
    if a2.size == 0:
        return (float("nan"),) * 4
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, a2.size, size=(n_boot, a2.size))
    d = (a2[idx] - b2[idx]).mean(1)
    obs = float(a2.mean() - b2.mean())
    lo, hi = float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))
    p = 2 * min((d >= 0).mean(), (d <= 0).mean())     # two-sided bootstrap p
    return obs, lo, hi, float(min(p, 1.0))


def mcnemar(a_bin: np.ndarray, b_bin: np.ndarray):
    """Exact-ish McNemar on paired binary outcomes (a=1 means 'error')."""
    from scipy.stats import binomtest
    m = np.array([x is not None and y is not None for x, y in zip(a_bin, b_bin)])
    a2 = np.asarray([bool(round(x)) for x in np.asarray(a_bin, float)[m]])
    b2 = np.asarray([bool(round(x)) for x in np.asarray(b_bin, float)[m]])
    n01 = int((~a2 & b2).sum())
    n10 = int((a2 & ~b2).sum())
    if n01 + n10 == 0:
        return {"n01": n01, "n10": n10, "p": 1.0}
    p = binomtest(n10, n01 + n10, 0.5).pvalue
    return {"n01": n01, "n10": n10, "p": float(p)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=os.path.join(POC_DIR, "outputs/counterfactual/counterfactual_raw.json"))
    ap.add_argument("--outdir", default=os.path.join(POC_DIR, "outputs/counterfactual"))
    args = ap.parse_args()

    R = json.load(open(args.raw))
    print(f"{len(R)} samples, {len({r['log'] for r in R})} logs")

    metrics = ["next_token_error", "downstream_error_rate", "remaining_accuracy",
               "sequence_recovered", "ade", "fde", "coarse_match"]
    per_arm = {a: {m: [] for m in metrics} for a in ARMS}
    for r in R:
        for a in ARMS:
            per = r["arms"].get(a, [])
            for m in metrics:
                per_arm[a][m].append(agg_arm(per, m))

    table = {}
    for a in ARMS:
        table[a] = {}
        for m in metrics:
            mean, lo, hi = boot_ci(np.array(per_arm[a][m], dtype=object))
            table[a][m] = {"mean": mean, "ci95": [lo, hi]}

    # deltas vs the original arm (what the intervention bought)
    deltas = {}
    for a in ("gt", "plausible", "random"):
        deltas[a] = {}
        for m in metrics:
            obs, lo, hi, p = paired_boot_diff(per_arm[a][m], per_arm["original"][m])
            deltas[a][m] = {"diff_vs_original": obs, "ci95": [lo, hi], "p": p}

    # the decisive contrast: GT vs random
    gt_vs_rand = {}
    for m in metrics:
        obs, lo, hi, p = paired_boot_diff(per_arm["gt"][m], per_arm["random"][m])
        gt_vs_rand[m] = {"diff_gt_minus_random": obs, "ci95": [lo, hi], "p": p}
    gt_vs_rand["next_token_error_mcnemar"] = mcnemar(
        per_arm["gt"]["next_token_error"], per_arm["random"]["next_token_error"])
    gt_vs_rand["next_token_error_mcnemar_vs_original"] = mcnemar(
        per_arm["gt"]["next_token_error"], per_arm["original"]["next_token_error"])

    # by first-error position
    by_t = defaultdict(lambda: {a: [] for a in ARMS})
    for i, r in enumerate(R):
        for a in ARMS:
            by_t[r["t"]][a].append(per_arm[a]["downstream_error_rate"][i])
    pos = {}
    for t in sorted(by_t):
        row = {"n": len(by_t[t]["gt"])}
        for a in ARMS:
            mean, lo, hi = boot_ci(np.array(by_t[t][a], dtype=object))
            row[a] = {"mean": mean, "ci95": [lo, hi]}
        row["low_n"] = row["n"] < 20
        pos[str(t)] = row

    out = {"n_samples": len(R), "n_logs": len({r["log"] for r in R}),
           "table": table, "deltas_vs_original": deltas,
           "gt_vs_random": gt_vs_rand, "by_first_error_position": pos}
    with open(os.path.join(args.outdir, "counterfactual_summary.json"), "w") as f:
        json.dump(out, f, indent=2)

    # ---------------- report ----------------
    def pct(v):
        return "  n/a " if v is None or not np.isfinite(v) else f"{v*100:6.1f}%"

    print("\n" + "=" * 92)
    print(f"{'Condition':10s} | {'Next-token err':>15s} | {'Downstream err':>15s} | "
          f"{'Seq recovery':>13s} | {'ADE':>7s} | {'FDE':>7s}")
    print("-" * 92)
    for a in ARMS:
        t_ = table[a]
        print(f"{a:10s} | {pct(t_['next_token_error']['mean'])}          | "
              f"{pct(t_['downstream_error_rate']['mean'])}          | "
              f"{pct(t_['sequence_recovered']['mean'])}        | "
              f"{t_['ade']['mean']:7.3f} | {t_['fde']['mean']:7.3f}")
    print("=" * 92)

    print("\n95% CI (bootstrap over samples):")
    for a in ARMS:
        d = table[a]["downstream_error_rate"]
        ad = table[a]["ade"]
        print(f"  {a:9s} downstream {d['mean']*100:5.1f}% [{d['ci95'][0]*100:.1f}, {d['ci95'][1]*100:.1f}]"
              f"   ADE {ad['mean']:.3f} [{ad['ci95'][0]:.3f}, {ad['ci95'][1]:.3f}]")

    # GT vs the tighter (plausible) control
    gt_vs_plaus = {}
    for m in metrics:
        obs, lo, hi, p = paired_boot_diff(per_arm["gt"][m], per_arm["plausible"][m])
        gt_vs_plaus[m] = {"diff_gt_minus_plausible": obs, "ci95": [lo, hi], "p": p}
    gt_vs_plaus["next_token_error_mcnemar"] = mcnemar(
        per_arm["gt"]["next_token_error"], per_arm["plausible"]["next_token_error"])
    out["gt_vs_plausible"] = gt_vs_plaus
    with open(os.path.join(args.outdir, "counterfactual_summary.json"), "w") as f:
        json.dump(out, f, indent=2)

    print("\nDecisive contrast — GT vs PLAUSIBLE control (model's own runner-up):")
    for m in ("next_token_error", "downstream_error_rate", "sequence_recovered", "ade", "coarse_match"):
        g = gt_vs_plaus[m]
        print(f"  {m:24s} diff={g['diff_gt_minus_plausible']:+.4f} "
              f"CI[{g['ci95'][0]:+.4f},{g['ci95'][1]:+.4f}] p={g['p']:.4g}")
    mcp = gt_vs_plaus["next_token_error_mcnemar"]
    print(f"  McNemar GT vs plausible: n01={mcp['n01']} n10={mcp['n10']} p={mcp['p']:.4g}")

    print("\nSecondary contrast — GT vs random (paired):")
    for m in ("next_token_error", "downstream_error_rate", "sequence_recovered", "ade", "coarse_match"):
        g = gt_vs_rand[m]
        print(f"  {m:24s} diff={g['diff_gt_minus_random']:+.4f} "
              f"CI[{g['ci95'][0]:+.4f},{g['ci95'][1]:+.4f}] p={g['p']:.4g}")
    mc = gt_vs_rand["next_token_error_mcnemar"]
    print(f"  McNemar GT vs random: n01={mc['n01']} n10={mc['n10']} p={mc['p']:.4g}")
    mc2 = gt_vs_rand["next_token_error_mcnemar_vs_original"]
    print(f"  McNemar GT vs original: n01={mc2['n01']} n10={mc2['n10']} p={mc2['p']:.4g}")

    print("\nGT vs original (paired):")
    for m in ("downstream_error_rate", "ade", "sequence_recovered"):
        g = deltas["gt"][m]
        print(f"  {m:24s} diff={g['diff_vs_original']:+.4f} "
              f"CI[{g['ci95'][0]:+.4f},{g['ci95'][1]:+.4f}] p={g['p']:.4g}")

    print("\nBy first-error position (downstream error rate):")
    print(f"  {'t':>3s} {'n':>5s} | {'original':>9s} {'gt':>9s} {'random':>9s}")
    for t, row in pos.items():
        flag = "  (low n)" if row["low_n"] else ""
        print(f"  {t:>3s} {row['n']:>5d} | {row['original']['mean']*100:8.1f}% "
              f"{row['gt']['mean']*100:8.1f}% {row['random']['mean']*100:8.1f}%{flag}")

    print(f"\nwrote {args.outdir}/counterfactual_summary.json")


if __name__ == "__main__":
    main()
