#!/usr/bin/env python
"""Figure F — first-error teacher-forcing counterfactual."""
from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

POC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
C_BLUE, C_ORANGE, C_AQUA, C_YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8880", "#e3e2dd"

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "axes.titlecolor": INK,
    "text.color": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 10, "axes.titlesize": 12, "legend.frameon": False,
    "lines.linewidth": 2.0, "lines.markersize": 5,
})

S = json.load(open(os.path.join(POC_DIR, "outputs/counterfactual/counterfactual_summary.json")))
ARMS = ["original", "gt", "plausible", "random"]
NICE = {"original": "original\n(wrong token kept)", "gt": "GT token\nforced",
        "plausible": "plausible control\n(model's 2nd choice)", "random": "random token"}
COL = {"original": MUTED, "gt": C_AQUA, "plausible": C_BLUE, "random": C_ORANGE}

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

# --- panel 1: downstream error with CI ---
ax = axes[0]
vals = [S["table"][a]["downstream_error_rate"]["mean"] * 100 for a in ARMS]
lo = [(S["table"][a]["downstream_error_rate"]["mean"]
       - S["table"][a]["downstream_error_rate"]["ci95"][0]) * 100 for a in ARMS]
hi = [(S["table"][a]["downstream_error_rate"]["ci95"][1]
       - S["table"][a]["downstream_error_rate"]["mean"]) * 100 for a in ARMS]
bars = ax.bar(range(len(ARMS)), vals, yerr=[lo, hi], width=0.62,
              color=[COL[a] for a in ARMS],
              error_kw=dict(ecolor=INK2, lw=1.3, capsize=4))
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2, v + 2.5, f"{v:.1f}%", ha="center",
            fontsize=10, color=INK)
ax.set_xticks(range(len(ARMS)))
ax.set_xticklabels([NICE[a] for a in ARMS], fontsize=8.5)
ax.set_ylabel("downstream action-token error (%)")
ax.set_ylim(0, 108)
ax.set_title("Only the correct token recovers generation")
ax.grid(axis="x", visible=False)

# --- panel 2: by first-error position ---
ax = axes[1]
pos = S["by_first_error_position"]
ts = sorted(pos, key=int)
for a, mk in (("original", "o"), ("gt", "^"), ("plausible", "s")):
    y = [pos[t][a]["mean"] * 100 for t in ts]
    n_ok = [int(t) for t in ts]
    ax.plot(n_ok, y, color=COL[a], marker=mk, label=NICE[a].replace("\n", " "))
low = [int(t) for t in ts if pos[t]["low_n"]]
for t in low:
    ax.axvspan(t - 0.4, t + 0.4, color=GRID, alpha=0.45, zorder=0)
ax.set_xlabel("first-error step t")
ax.set_ylabel("downstream error (%)")
ax.set_ylim(0, 105)
ax.set_title("Correction helps more the later the first error")
ax.legend(fontsize=8, loc="center left")
ax.text(0.98, 0.03, "shaded: n < 20", transform=ax.transAxes, ha="right",
        fontsize=8, color=INK2, style="italic")

# --- panel 3: ADE ---
ax = axes[2]
vals = [S["table"][a]["ade"]["mean"] for a in ARMS]
lo = [S["table"][a]["ade"]["mean"] - S["table"][a]["ade"]["ci95"][0] for a in ARMS]
hi = [S["table"][a]["ade"]["ci95"][1] - S["table"][a]["ade"]["mean"] for a in ARMS]
bars = ax.bar(range(len(ARMS)), vals, yerr=[lo, hi], width=0.62,
              color=[COL[a] for a in ARMS],
              error_kw=dict(ecolor=INK2, lw=1.3, capsize=4))
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.09, f"{v:.3f}", ha="center",
            fontsize=10, color=INK)
ax.set_xticks(range(len(ARMS)))
ax.set_xticklabels([NICE[a] for a in ARMS], fontsize=8.5)
ax.set_ylabel("ADE (m)")
ax.set_title("Trajectory error after the intervention")
ax.grid(axis="x", visible=False)

fig.suptitle("Figure F — First-error teacher-forcing counterfactual (n=400, 28 logs, paired)",
             y=1.03)
fig.tight_layout()
out = os.path.join(POC_DIR, "figures", "figF_counterfactual.png")
fig.savefig(out, dpi=170, bbox_inches="tight")
print("wrote", out)
