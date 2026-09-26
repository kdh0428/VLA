#!/usr/bin/env python
"""
Attribute the first action-token error to a late-layer component.

For every layer we have, at the action query position,
    d_attn = margin(after attention) - margin(layer input)
    d_mlp  = margin(after MLP)        - margin(after attention)
with margin = logit(GT action) - logit(strongest competitor).

A component that pushes the model AWAY from the right answer has a negative delta. The
question is which (layer, component) separates the failure group from the correct group,
so the headline number is the between-group difference of these deltas, with a bootstrap CI
and a Mann-Whitney test, BH-corrected across the (layer x component) grid.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

POC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def boot_ci(x, n_boot=10000, seed=0):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    m = rng.choice(x, (n_boot, x.size), replace=True).mean(1)
    return float(x.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def boot_diff(a, b, n_boot=10000, seed=0):
    a = np.asarray(a, float); b = np.asarray(b, float)
    a = a[np.isfinite(a)]; b = b[np.isfinite(b)]
    if a.size == 0 or b.size == 0:
        return (float("nan"),) * 4
    rng = np.random.default_rng(seed)
    da = rng.choice(a, (n_boot, a.size), replace=True).mean(1)
    db = rng.choice(b, (n_boot, b.size), replace=True).mean(1)
    d = da - db
    p = 2 * min((d >= 0).mean(), (d <= 0).mean())
    return float(a.mean() - b.mean()), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5)), float(min(p, 1))


def bh(p, q=0.05):
    p = np.asarray(p, float); n = p.size
    o = np.argsort(p); thr = q * np.arange(1, n + 1) / n
    ok = p[o] <= thr
    k = np.max(np.nonzero(ok)[0]) + 1 if ok.any() else 0
    return (p <= p[o][k - 1]) if k else np.zeros(n, bool)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=os.path.join(POC_DIR, "outputs/decomposition/decomposition_raw.json"))
    ap.add_argument("--outdir", default=os.path.join(POC_DIR, "outputs/decomposition"))
    args = ap.parse_args()

    D = json.load(open(args.raw))
    rows, PL, nL = D["rows"], D["probe_layers"], D["n_layers"]
    ok = [r for r in rows if r["correct"]]
    bad = [r for r in rows if not r["correct"]]
    print(f"{len(rows)} samples: correct {len(ok)}, first-action-error {len(bad)}, "
          f"{len({r['log'] for r in rows})} logs")
    print(f"decomposed layers {PL} of {nL}\n")

    def get(rs, li, key):
        return np.array([next(L[key] for L in r["layers"] if L["layer"] == li) for r in rs])

    # final margins, as context
    fin_ok = get(ok, PL[-1], "margin_mlp")
    fin_bad = get(bad, PL[-1], "margin_mlp")
    print(f"final margin  correct {fin_ok.mean():+.3f}   failure {fin_bad.mean():+.3f}")
    ent_ok = get(ok, PL[0], "margin_in")
    ent_bad = get(bad, PL[0], "margin_in")
    print(f"margin entering L{PL[0]}  correct {ent_ok.mean():+.3f}   failure {ent_bad.mean():+.3f}")
    print(f"  -> the failure group already enters the late block behind by "
          f"{ent_ok.mean() - ent_bad.mean():.3f} logits\n")

    table, pv, keys = {}, [], []
    for li in PL:
        for comp, key in (("Attention", "d_attn"), ("MLP", "d_mlp")):
            a, b = get(ok, li, key), get(bad, li, key)
            ma, la, ha = boot_ci(a)
            mb, lb, hb = boot_ci(b)
            diff, dl, dh, p = boot_diff(b, a)          # failure - correct
            try:
                from scipy.stats import mannwhitneyu
                pu = float(mannwhitneyu(b, a).pvalue)
            except Exception:
                pu = float("nan")
            table[f"L{li}|{comp}"] = {
                "layer": li, "component": comp,
                "correct": [ma, la, ha], "failure": [mb, lb, hb],
                "difference": diff, "ci95": [dl, dh], "p_boot": p, "p_mwu": pu}
            pv.append(pu); keys.append(f"L{li}|{comp}")
    sig = bh(np.array(pv))
    for k, s in zip(keys, sig):
        table[k]["significant_bh05"] = bool(s)

    print("=" * 86)
    print(f"{'Layer':>6} {'Component':10s} | {'Correct dmargin':>16s} | {'Failure dmargin':>16s} | "
          f"{'Difference':>11s}")
    print("-" * 86)
    for k in keys:
        t = table[k]
        star = " *" if t["significant_bh05"] else "  "
        print(f"{'L'+str(t['layer']):>6} {t['component']:10s} | {t['correct'][0]:+16.4f} | "
              f"{t['failure'][0]:+16.4f} | {t['difference']:+9.4f}{star}")
    print("=" * 86)
    print("  * = significant at BH FDR 5%; difference = failure - correct")

    ranked = sorted(table.values(), key=lambda t: t["difference"])
    print("\nMost negative difference (component pushes failures away from GT):")
    for t in ranked[:5]:
        print(f"  L{t['layer']:<3} {t['component']:10s} diff={t['difference']:+.4f} "
              f"CI[{t['ci95'][0]:+.4f},{t['ci95'][1]:+.4f}] p={t['p_mwu']:.3g}"
              f"{'  *' if t['significant_bh05'] else ''}")

    # cumulative contribution of each component family across the last 4 layers
    last4 = PL[-4:]
    for comp, key in (("Attention", "d_attn"), ("MLP", "d_mlp")):
        a = np.sum([get(ok, li, key) for li in last4], axis=0)
        b = np.sum([get(bad, li, key) for li in last4], axis=0)
        diff, dl, dh, p = boot_diff(b, a)
        print(f"\nsummed over L{last4[0]}-L{last4[-1]}  {comp:10s} "
              f"correct {a.mean():+.4f}  failure {b.mean():+.4f}  "
              f"diff {diff:+.4f} CI[{dl:+.4f},{dh:+.4f}] p={p:.3g}")

    # how much of the final gap is opened inside the last 4 layers at all?
    a_in = get(ok, last4[0], "margin_in"); b_in = get(bad, last4[0], "margin_in")
    gap_entry = a_in.mean() - b_in.mean()
    gap_final = fin_ok.mean() - fin_bad.mean()
    print(f"\ngap correct-failure entering L{last4[0]}: {gap_entry:+.3f}")
    print(f"gap correct-failure at output        : {gap_final:+.3f}")
    print(f"opened inside the last 4 layers      : {gap_final - gap_entry:+.3f} "
          f"({(gap_final-gap_entry)/gap_final*100:.0f}% of the final gap)")

    out = {"n_correct": len(ok), "n_failure": len(bad), "probe_layers": PL,
           "table": table,
           "entry_margin": {"correct": float(ent_ok.mean()), "failure": float(ent_bad.mean())},
           "final_margin": {"correct": float(fin_ok.mean()), "failure": float(fin_bad.mean())},
           "gap_entry_last4": float(gap_entry), "gap_final": float(gap_final)}
    with open(os.path.join(args.outdir, "decomposition_summary.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {args.outdir}/decomposition_summary.json")


if __name__ == "__main__":
    main()
