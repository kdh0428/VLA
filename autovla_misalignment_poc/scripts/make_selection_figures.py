#!/usr/bin/env python
"""Figures for the selection-validation report (CPU only; reads summaries of experiments 21-23)."""
from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

POC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(POC, "outputs/selection_validation/figures")
C = {"dev": "#2a78d6", "heldout": "#eb6834"}          # reference palette slots 1-2 (dataviz skill)
MK = {"dev": "o", "heldout": "s"}
INK, INK2, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"
LABEL = {"dev": "dev (56 logs)", "heldout": "held-out (52 logs)"}


def style(ax):
    ax.set_facecolor(SURF)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def fig_entropy():
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6), facecolor=SURF)
    for ax, key, title in ((axes[0], "pre_dev_entropy", "before the first deviation"),
                           (axes[1], "post_dev_entropy", "after the first deviation")):
        style(ax)
        for j, split in enumerate(("dev", "heldout")):
            s = json.load(open(os.path.join(POC, f"outputs/mechanism_selection_link/{split}/summary.json")))["selected_vs_discarded"]["all_scenes"][key]
            x = np.array([0, 1]) + (j - 0.5) * 0.18
            ax.plot(x, [s["selected"], s["discarded"]], "-", color=C[split], lw=2)
            ax.plot(x, [s["selected"], s["discarded"]], MK[split], color=C[split], ms=8, mec=SURF, mew=2, label=LABEL[split])
        ax.set_xticks([0, 1], ["selected\n(rank-sum)", "discarded"], color=INK)
        ax.set_xlim(-0.5, 1.5); ax.set_ylim(0.3, 1.1)
        ax.set_title(f"Mean decoding entropy {title}", loc="left", color=INK, fontsize=10)
    axes[0].set_ylabel("entropy (nats)", color=INK2, fontsize=9)
    axes[1].legend(frameon=False, fontsize=9, labelcolor=INK, loc="upper left")
    fig.suptitle("Selected and discarded rollouts diverge only after the deviation", x=0.01, ha="left", color=INK, fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig1_entropy_before_after_deviation.png"), dpi=160, facecolor=SURF)
    plt.close(fig)


def fig_ncurve():
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6), facecolor=SURF)
    for split in ("dev", "heldout"):
        s = json.load(open(os.path.join(POC, f"outputs/pdm_score_best_of_n/{split}_ncurve/summary.json")))["subsets"]["all"]
        ns = sorted(int(k[9:]) for k in s)
        axes[0].plot(ns, [s[f"ranksum_N{n}"]["pdms"] for n in ns], "-", color=C[split], lw=2)
        axes[0].plot(ns, [s[f"ranksum_N{n}"]["pdms"] for n in ns], MK[split], color=C[split], ms=7, mec=SURF, mew=2, label=LABEL[split])
        c = json.load(open(os.path.join(POC, f"outputs/candidate_count_curve/{split}/summary.json")))["curve"]
        ns2 = sorted(int(k) for k in c)
        axes[1].plot(ns2, [100 * c[str(n)]["a_minus"] for n in ns2], "-", color=C[split], lw=2)
        axes[1].plot(ns2, [100 * c[str(n)]["a_minus"] for n in ns2], MK[split], color=C[split], ms=7, mec=SURF, mew=2)
    for ax, yl, t in ((axes[0], "PDM Score", "PDM Score"), (axes[1], "A- rate (%)", "open-loop failure rate")):
        style(ax)
        ax.set_xscale("log", base=2); ax.set_xticks([1, 2, 4, 8, 16], ["1", "2", "4", "8", "16"])
        ax.set_xlabel("candidates N (natural plan + N-1 samples at T 1.0)", color=INK2, fontsize=8.5)
        ax.set_ylabel(yl, color=INK2, fontsize=9)
        ax.set_title(t, loc="left", color=INK, fontsize=10)
    axes[0].legend(frameon=False, fontsize=9, labelcolor=INK, loc="lower right")
    fig.suptitle("Rank-sum selection vs candidate count (relative inference cost at N=16: 1.3x)", x=0.01, ha="left", color=INK, fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig2_candidate_count.png"), dpi=160, facecolor=SURF)
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    fig_entropy()
    fig_ncurve()
    print("wrote", OUT)
