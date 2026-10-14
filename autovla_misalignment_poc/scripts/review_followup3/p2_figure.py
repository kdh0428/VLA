#!/usr/bin/env python
"""P2 figure (review_followup3): Delta_model - Delta_comparator (family G; M1 endpoint and M2 suffix), drive 95% CIs, per
stratum.  Generated ONLY from P2/MOTION_BASELINE_RESULTS.csv and P2/LEARNED_EXTRAPOLATOR_RESULTS.csv (if present).
Output: P2/fig_p2_motion_baselines.{pdf,png}.   Run: nice -n 19 python p2_figure.py"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

OUT = "/root/VLA/autovla_misalignment_poc/outputs/review_followup3/P2"
CLS = {"kinematic (prefix only)": "#2a78d6", "learned extrapolator (seed mean)": "#eb6834", "privileged (Geo)": "#52514e"}
MARK = {"kinematic (prefix only)": "o", "learned extrapolator (seed mean)": "s", "privileged (Geo)": "D"}


def rows():
    T = pd.read_csv(f"{OUT}/MOTION_BASELINE_RESULTS.csv")
    T = T[(T.row_type == "excess") & (T.family == "G")]
    R = [{"group": r.group, "metric": r.metric, "label": r.baseline,
          "cls": "privileged (Geo)" if r.baseline == "Geo" else "kinematic (prefix only)",
          "est": r.est_scene, "lo": r.ci95_lo, "hi": r.ci95_hi} for r in T.itertuples()]
    p = f"{OUT}/LEARNED_EXTRAPOLATOR_RESULTS.csv"
    if os.path.exists(p):
        E = pd.read_csv(p)
        E = E[(E.row_type == "excess") & (E.family == "G") & E.analysis.str.startswith("primary")]
        R += [{"group": r.group, "metric": r.metric, "label": {"E:prefix": "learned E-prefix", "E:prefix_ego": "learned E-prefix+ego"}[r.comparator],
               "cls": "learned extrapolator (seed mean)", "est": r.est_scene, "lo": r.ci95_lo, "hi": r.ci95_hi} for r in E.itertuples()]
    return pd.DataFrame(R)


def main():
    D = rows()
    order = ["CM", "CV0", "CTRV-K3", "CTRA-2", "learned E-prefix", "learned E-prefix+ego", "Geo"]
    order = [o for o in order if o in set(D.label)]
    mets = [("end", "M1 endpoint change: Δ_model − Δ_b (m)"), ("sfx", "M2 suffix local-motion change: Δ_model − Δ_b (m / 0.5 s)")]
    groups = [("A-", "A− (13 drives)"), ("A+", "A+ (19 drives)"), ("all", "all (19 drives)")]
    plt.rcParams.update({"font.size": 8.5, "axes.edgecolor": "#8a8984", "axes.linewidth": 0.6})
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 5.6), sharey=True)
    for i, (m, mlab) in enumerate(mets):
        for k, (g, glab) in enumerate(groups):
            ax = axes[i, k]
            ax.axvline(0, color="#8a8984", lw=0.8, zorder=0)
            for yi, lab in enumerate(order):
                r = D[(D.group == g) & (D.metric == m) & (D.label == lab)]
                if r.empty:
                    continue
                r = r.iloc[0]
                ax.plot([r.lo, r.hi], [yi, yi], color=CLS[r.cls], lw=2, solid_capstyle="round")
                ax.plot(r.est, yi, MARK[r.cls], color=CLS[r.cls], ms=6, mec="white", mew=1.2)
            ax.set_yticks(range(len(order))); ax.set_yticklabels(order)
            ax.invert_yaxis() if k == 0 and i == 0 else None
            ax.grid(axis="x", color="#e4e3df", lw=0.6); ax.set_axisbelow(True)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
            if i == 0:
                ax.set_title(glab, fontsize=9.5)
            if k == 1:
                ax.set_xlabel(mlab)
    handles = [plt.Line2D([], [], marker=MARK[c], color=v, lw=2, ms=6, mec="white", label=c) for c, v in CLS.items()
               if c in set(D.cls)]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False)
    fig.suptitle("AutoVLA open-loop 5 s plan, P1-A family G: model response minus comparator response (drive-cluster 95% CI) — PROVISIONAL",
                 fontsize=9.5)
    fig.tight_layout(rect=(0, 0.05, 1, 0.96))
    for ext in ("pdf", "png"):
        fig.savefig(f"{OUT}/fig_p2_motion_baselines.{ext}", dpi=200)
    print("ok")


if __name__ == "__main__":
    main()
