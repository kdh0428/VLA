#!/usr/bin/env python
"""
Does the action position attend differently when it is about to get the token wrong?

Two comparisons, the second much better controlled than the first:

  between-sample  step 0, wrong vs correct first action token. Confounded by scene
                  difficulty -- hard scenes may both look different and be wrong.

  step-matched    wrong vs correct query positions at the SAME action step index. This is
                  the comparison the conclusion rests on.

A naive within-sample contrast (first-error step t vs step t-1 of the same sample) is NOT
usable and is reported only as a diagnostic: step t has exactly one more preceding action
token than t-1, so attention mass shifts onto the action span mechanically. Measured, that
artefact is +0.050 on the action span (p=3e-27) and drags every other span down with it --
a positional effect, not a property of erring. Matching on step index removes it entirely,
because within one step index the number of preceding action tokens is identical.

Head-level results are reported with a Benjamini-Hochberg FDR correction, because 36x16
= 576 heads are tested at once and some will look significant by chance.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

POC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPANS = ("visual", "text", "scaffold", "action")


def boot_ci(x, n_boot=10000, seed=0):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    m = rng.choice(x, (n_boot, x.size), replace=True).mean(1)
    return float(x.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def paired_boot(d, n_boot=10000, seed=0):
    d = np.asarray(d, float)
    d = d[np.isfinite(d)]
    if d.size == 0:
        return float("nan"), float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    m = rng.choice(d, (n_boot, d.size), replace=True).mean(1)
    p = 2 * min((m >= 0).mean(), (m <= 0).mean())
    return float(d.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5)), float(min(p, 1))


def bh(pvals, q=0.05):
    p = np.asarray(pvals, float)
    n = p.size
    order = np.argsort(p)
    thr = q * (np.arange(1, n + 1)) / n
    passed = p[order] <= thr
    k = np.max(np.nonzero(passed)[0]) + 1 if passed.any() else 0
    cut = p[order][k - 1] if k else -1
    return p <= cut, (float(cut) if k else float("nan")), int(k)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=os.path.join(POC_DIR, "outputs/attention"))
    args = ap.parse_args()

    rows = json.load(open(os.path.join(args.dir, "rows.json")))
    M = np.load(os.path.join(args.dir, "mass.npy"))          # (N, layers, heads, spans)
    print(f"{len(rows)} probe rows, mass {M.shape}, {len({r['log'] for r in rows})} logs")
    nL, nH = M.shape[1], M.shape[2]

    kinds = np.array([r["kind"] for r in rows])
    correct = np.array([r["correct"] for r in rows])
    tokens = np.array([r["token"] for r in rows])

    out = {"n_rows": len(rows), "n_layers": nL, "n_heads": nH}

    # ---------- between-sample: step0 wrong vs correct ----------
    s0 = kinds == "step0"
    a, b = s0 & correct, s0 & ~correct
    print(f"\n=== between-sample, step 0: correct n={a.sum()}  wrong n={b.sum()} ===")
    bet = {}
    for si, s in enumerate(SPANS):
        ca = M[a][:, :, :, si].mean((1, 2))
        cb = M[b][:, :, :, si].mean((1, 2))
        ma, la, ha = boot_ci(ca)
        mb, lb, hb = boot_ci(cb)
        bet[s] = {"correct": [ma, la, ha], "wrong": [mb, lb, hb], "diff": mb - ma}
        print(f"  {s:9s} correct {ma:.4f} [{la:.4f},{ha:.4f}]   "
              f"wrong {mb:.4f} [{lb:.4f},{hb:.4f}]   diff {mb-ma:+.4f}")
    out["between_sample_step0"] = bet

    # ---------- within-sample: first_error vs pre_error ----------
    fe = {r["token"]: i for i, r in enumerate(rows) if r["kind"] == "first_error"}
    pe = {r["token"]: i for i, r in enumerate(rows) if r["kind"] == "pre_error"}
    common = sorted(set(fe) & set(pe))
    print(f"\n=== [DIAGNOSTIC ONLY - positionally confounded] within-sample: {len(common)} pairs ===")
    wit = {}
    if common:
        ie = np.array([fe[t] for t in common])
        ip = np.array([pe[t] for t in common])
        for si, s in enumerate(SPANS):
            d = M[ie][:, :, :, si].mean((1, 2)) - M[ip][:, :, :, si].mean((1, 2))
            mean, lo, hi, p = paired_boot(d)
            try:
                from scipy.stats import wilcoxon
                w = float(wilcoxon(d).pvalue) if np.any(d != 0) else 1.0
            except Exception:
                w = float("nan")
            wit[s] = {"mean_diff": mean, "ci95": [lo, hi], "p_boot": p, "p_wilcoxon": w}
            print(f"  {s:9s} err - correct = {mean:+.5f}  CI[{lo:+.5f},{hi:+.5f}]  "
                  f"p_boot={p:.4g}  p_wilcoxon={w:.4g}")
        out["within_sample"] = wit

        # ---------- head-level, visual span, within-sample ----------
        vi = SPANS.index("visual")
        D = M[ie][:, :, :, vi] - M[ip][:, :, :, vi]          # (n_pairs, L, H)
        means = D.mean(0)
        ps = np.ones((nL, nH))
        try:
            from scipy.stats import wilcoxon
            for l in range(nL):
                for h in range(nH):
                    d = D[:, l, h]
                    ps[l, h] = wilcoxon(d).pvalue if np.any(d != 0) else 1.0
        except Exception:
            pass
        sig, cut, k = bh(ps.ravel())
        sig = sig.reshape(nL, nH)
        print(f"\n=== head-level visual-attention change (within-sample) ===")
        print(f"  heads passing BH FDR 5%: {int(sig.sum())} / {nL*nH}  (cut p<={cut:.3g})")
        flat = [(means[l, h], ps[l, h], l, h) for l in range(nL) for h in range(nH)]
        dec = sorted(flat)[:6]
        inc = sorted(flat, reverse=True)[:6]
        print("  largest DECREASE in visual attention on the error step:")
        for m_, p_, l, h in dec:
            print(f"    L{l:>2} H{h:>2}  {m_:+.5f}  p={p_:.3g}{'  *' if sig[l,h] else ''}")
        print("  largest INCREASE:")
        for m_, p_, l, h in inc:
            print(f"    L{l:>2} H{h:>2}  {m_:+.5f}  p={p_:.3g}{'  *' if sig[l,h] else ''}")
        out["head_level_visual"] = {
            "n_significant_bh05": int(sig.sum()), "bh_cut": cut,
            "top_decrease": [[float(m_), float(p_), int(l), int(h)] for m_, p_, l, h in dec],
            "top_increase": [[float(m_), float(p_), int(l), int(h)] for m_, p_, l, h in inc],
            "mean_by_layer": means.mean(1).tolist(),
        }

        # per-layer visual attention profile, both steps
        prof = {"first_error": M[ie][:, :, :, vi].mean((0, 2)).tolist(),
                "pre_error": M[ip][:, :, :, vi].mean((0, 2)).tolist()}
        out["visual_mass_by_layer"] = prof
        print("\n  visual attention by layer (mean over heads):")
        for l in range(0, nL, 6):
            print(f"    L{l:>2}  err {prof['first_error'][l]:.4f}   correct {prof['pre_error'][l]:.4f}")

    # ---------- step-matched: wrong vs correct at the SAME step index ----------
    steps = np.array([r["step"] for r in rows])
    print("\n=== step-matched, wrong vs correct at the same action step ===")
    sm = {}
    pooled = {s_: {"d": [], "w": 0, "c": 0} for s_ in SPANS}
    for st in sorted(set(steps.tolist())):
        m_w = (steps == st) & ~correct
        m_c = (steps == st) & correct
        if m_w.sum() < 5 or m_c.sum() < 5:
            continue
        row = {"n_wrong": int(m_w.sum()), "n_correct": int(m_c.sum())}
        for si, s_ in enumerate(SPANS):
            cw = M[m_w][:, :, :, si].mean((1, 2))
            cc = M[m_c][:, :, :, si].mean((1, 2))
            mw, lw, hw = boot_ci(cw)
            mc, lc, hc = boot_ci(cc)
            try:
                from scipy.stats import mannwhitneyu
                pu = float(mannwhitneyu(cw, cc).pvalue)
            except Exception:
                pu = float("nan")
            row[s_] = {"wrong": [mw, lw, hw], "correct": [mc, lc, hc],
                       "diff": mw - mc, "p_mwu": pu}
            pooled[s_]["d"].append((mw - mc, m_w.sum()))
        sm[str(st)] = row
        print(f"  step {st}: wrong n={row['n_wrong']:4d} correct n={row['n_correct']:4d} | " +
              "  ".join(f"{s_} {row[s_]['diff']:+.4f}(p={row[s_]['p_mwu']:.2g})" for s_ in SPANS))
    out["step_matched"] = sm
    print("  weighted mean diff across steps:")
    for s_ in SPANS:
        arr = pooled[s_]["d"]
        if arr:
            w = np.array([a[1] for a in arr], float)
            v = np.array([a[0] for a in arr], float)
            print(f"    {s_:9s} {float((v*w).sum()/w.sum()):+.5f}")

    with open(os.path.join(args.dir, "attention_summary.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {args.dir}/attention_summary.json")


if __name__ == "__main__":
    main()
