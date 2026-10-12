#!/usr/bin/env python
"""Figure: log -> drive re-analysis (effect sizes and 95% CIs), generated ONLY from CLUSTER_REANALYSIS.csv.
POST-HOC. Usage: nice -n 19 python a3_figure.py"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

OUT = "/root/VLA/autovla_misalignment_poc/outputs/review_followup2/A_autovla_cluster/run_20261013"
df = pd.read_csv(os.path.join(OUT, "CLUSTER_REANALYSIS.csv"))
df = df[df["metric"] == "amplification"]

PANELS = [
    ("AutoVLA PoC, A- (16 logs / 13 drives)", ["AV-P1 Recent-Normal", "AV-P2 Reverse-Full", "AV-P3 Direction-Magnitude",
                                               "AV-S GeoNN(GT)-RecentGT [equivalence]"]),
    ("Impromptu exp 33 (28 logs / 20 drives)", ["IMP Recent-Normal", "IMP GT-history-Normal", "IMP Reverse-Full",
                                                "IMP Direction-Magnitude"]),
    ("P1-C family F (41 logs / 18 drives)", ["P1C RN F", "P1C RV F", "P1C DM F"]),
    ("P1-C family P (64 logs / 24 drives)", ["P1C RN P", "P1C RV P", "P1C DM P"]),
]
SER = [  # (label, est col, lo col, hi col, color, marker)
    ("original reported (PoC: unit-weighted, log-bootstrap CI; P1-C: prereg. drive-level)", "orig_estimate", "orig_ci95_lo", "orig_ci95_hi", "#8a8984", "s"),
    ("log clusters, scene-weighted (bootstrap)", "log_estimate_scene", "log_ci95_lo", "log_ci95_hi", "#eb6834", "o"),
    ("drive clusters, scene-weighted (bootstrap)", "drive_estimate_scene", "drive_boot_ci95_lo", "drive_boot_ci95_hi", "#2a78d6", "o"),
    ("drive clusters, sign-flip inversion CI", "drive_estimate_scene", "drive_sf_ci95_lo", "drive_sf_ci95_hi", "#1baf7a", "D"),
]

fig, axes = plt.subplots(2, 2, figsize=(13, 8.5))
for ax, (title, ids) in zip(axes.ravel(), PANELS):
    sub = df.set_index("contrast_id").loc[ids]
    y0 = np.arange(len(ids))[::-1]
    for k, (lab, e, lo, hi, col, mk) in enumerate(SER):
        y = y0 + 0.27 - 0.18 * k
        est, l, h = sub[e].values * 100, sub[lo].values * 100, sub[hi].values * 100
        ok = np.isfinite(l) & np.isfinite(h)
        ax.hlines(y[ok], l[ok], h[ok], color=col, lw=2)
        ax.plot(est, y, mk, color=col, ms=6, label=lab, mec="white", mew=0.8)
    ax.axvline(0, color="#555", lw=0.8)
    lbl = []
    for cid, r in sub.iterrows():
        ptxt = f"drive p={r['drive_p_signflip']:.1g}" + (f", Holm {r['drive_holm_p']:.1g}" if np.isfinite(r["drive_holm_p"]) else "")
        lbl.append(f"{cid.replace(' [equivalence]', '*')}\n{ptxt} (log p={r['log_p_signflip']:.1g})")
    ax.set_yticks(y0); ax.set_yticklabels(lbl, fontsize=8)
    ax.set_title(title, fontsize=10, loc="left")
    ax.set_xlabel("effect on amplification rate (percentage points)", fontsize=8)
    ax.grid(axis="x", color="#e5e5e2", lw=0.6); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
h, l = axes[0, 0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=4, fontsize=8, frameon=False)
fig.suptitle("Log -> drive cluster re-analysis (POST-HOC; P1-C rows re-derive the preregistered drive-level analysis)\n"
             "* GeoNN(GT)-RecentGT: equivalence question, margin +-5 pp", fontsize=10)
fig.tight_layout(rect=(0, 0.05, 1, 0.95))
fig.savefig(os.path.join(OUT, "fig_log_to_drive_effects.png"), dpi=150)
print("saved")
