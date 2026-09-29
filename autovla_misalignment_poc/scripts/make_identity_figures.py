#!/usr/bin/env python
"""Figures for prev_action_identity_decomposition (CPU only; reads summary.json / units.jsonl)."""
from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

POC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUN = os.path.join(POC_DIR, "outputs/prev_action_identity_decomposition")
FIG = os.path.join(RUN, "figures")
# reference palette slots 1-2 + chart chrome (dataviz skill, references/palette.md)
C = {"A-": "#2a78d6", "A+": "#eb6834"}
MK = {"A-": "o", "A+": "s"}
INK, INK2, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"
ROWS = ["normal", "gt_history", "recent_gt", "geo_nn_gt", "emb_nn_gt", "geo_nn_self", "emb_nn_self",
        "random_tok", "mean_emb"]
LABEL = {"normal": "Normal AR", "gt_history": "GT-history", "recent_gt": "Recent-GT",
         "geo_nn_gt": "motion-nearest to GT", "emb_nn_gt": "embedding-nearest to GT",
         "geo_nn_self": "motion-nearest to own", "emb_nn_self": "embedding-nearest to own",
         "random_tok": "random token", "mean_emb": "mean embedding (OOD)"}


def style(ax):
    ax.set_facecolor(SURF)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def fig_amplification(S):
    fig, ax = plt.subplots(figsize=(8.2, 5.0), facecolor=SURF)
    style(ax)
    y = np.arange(len(ROWS))[::-1]
    for gi, g in enumerate(("A-", "A+")):
        off = 0.17 if g == "A-" else -0.17
        for yy, c in zip(y, ROWS):
            a = S["groups"][g]["conditions"][c]["amplification"]
            m, lo, hi = 100 * a["mean"], 100 * a["ci95"][0], 100 * a["ci95"][1]
            ax.plot([lo, hi], [yy + off] * 2, color=C[g], lw=2, solid_capstyle="round")
            ax.plot(m, yy + off, MK[g], color=C[g], ms=8, mec=SURF, mew=2,
                    label=f"{g} scenes" if c == "normal" else None)
    ax.set_yticks(y, [LABEL[c] for c in ROWS], color=INK)
    for yy in (y[2] - 0.5, y[6] - 0.5):
        ax.axhline(yy, color=GRID, lw=0.8)
    ax.set_xlabel("Amplification after first mismatch (%)  ·  A− and FDE5 > 3 m, log-cluster 95% CI", color=INK2, fontsize=9)
    ax.set_xlim(0, 65)
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=9, labelcolor=INK)
    fig.suptitle("Previous action token replaced: which substitute stops amplification",
                 x=0.01, ha="left", color=INK, fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig1_amplification_by_substitution.png"), dpi=160, facecolor=SURF)
    plt.close(fig)


def fig_dose(U):
    subs = ["geo_nn_gt", "emb_nn_gt", "geo_nn_self", "emb_nn_self"]
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.8), sharey=True, facecolor=SURF)
    for ax, key, xl in ((axes[0], "geo", "Δ codebook distance to GT (sub − own), quintile mean"),
                        (axes[1], "cos", "Δ embedding cosine to GT (sub − own), quintile mean")):
        style(ax)
        ax.grid(axis="y", color=GRID, lw=0.8)
        for g in ("A-", "A+"):
            pts = []
            for u in U:
                if u["group"] != g:
                    continue
                for c in subs:
                    s = u.get(c + "_sub")
                    if s:
                        x = (s["geo_d_sub_gt"] - s["geo_d_own_gt"]) if key == "geo" else (s["cos_sub_gt"] - s["cos_own_gt"])
                        pts.append((x, u[c]["fde5"] - u["normal"]["fde5"]))
            a = np.array(pts)
            q = np.quantile(a[:, 0], np.linspace(0, 1, 6))
            b = np.clip(np.searchsorted(q, a[:, 0], side="right") - 1, 0, 4)
            xs = [a[b == i, 0].mean() for i in range(5)]
            ys = [a[b == i, 1].mean() for i in range(5)]
            ax.plot(xs, ys, "-", color=C[g], lw=2)
            ax.plot(xs, ys, MK[g], color=C[g], ms=8, mec=SURF, mew=2)
            ax.annotate(g, (xs[-1], ys[-1]), xytext=(6, 0), textcoords="offset points", color=INK2, fontsize=9, va="center")
        ax.axhline(0, color=MUTED, lw=0.8)
        ax.set_xlabel(xl, color=INK2, fontsize=8.5)
    axes[0].set_ylabel("Δ FDE5 vs Normal (m)", color=INK2, fontsize=9)
    fig.suptitle("Outcome tracks the substitute's motion, not its embedding", x=0.01, ha="left", color=INK, fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig2_geometry_vs_embedding.png"), dpi=160, facecolor=SURF)
    plt.close(fig)


def main():
    os.makedirs(FIG, exist_ok=True)
    S = json.load(open(os.path.join(RUN, "summary.json")))
    U = [json.loads(l) for l in open(os.path.join(RUN, "units.jsonl"))]
    fig_amplification(S)
    fig_dose(U)
    print("wrote", FIG)


if __name__ == "__main__":
    main()
