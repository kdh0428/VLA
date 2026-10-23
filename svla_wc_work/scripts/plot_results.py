#!/usr/bin/env python
"""Figures from analysis/per_observation.csv (CPU). Usage: plot_results.py <analysis_dir>"""
import os, sys
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

A = sys.argv[1]
df = pd.read_csv(os.path.join(A, "per_observation.csv"))
TN = {"google_robot_pick_coke_can": "pick coke can", "google_robot_move_near": "move near"}
CL, CO = "#2a78d6", "#eb6834"
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#b5b4ae",
                     "axes.labelcolor": "#52514e", "xtick.color": "#52514e", "ytick.color": "#52514e"})


def save(fig, name):
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(A, f"{name}.{ext}"), dpi=170)


# 1. per task: layout-mean E_L vs E_O (paired) and layout-mean Delta; all joint-valid and matched subsets
fig, ax = plt.subplots(1, 3, figsize=(8.6, 2.7), gridspec_kw={"width_ratios": [1, 1, 1.1]})
for j, task in enumerate(["google_robot_pick_coke_can", "google_robot_move_near"]):
    d = df[df.task == task].groupby("seed")[["E_L", "E_O"]].mean()
    a = ax[j]
    for _, r in d.iterrows():
        a.plot([0, 1], [r.E_O, r.E_L], color="#c3c2b7", lw=0.8)
    a.scatter(np.zeros(len(d)), d.E_O, color=CO, s=14, zorder=3, label="O (older, a1 edited)")
    a.scatter(np.ones(len(d)), d.E_L, color=CL, s=14, zorder=3, label="L (latest, a2 edited)")
    a.set_xticks([0, 1]); a.set_xticklabels(["O", "L"]); a.set_xlim(-0.4, 1.4)
    a.set_title(f"{TN[task]} ({len(d)} layouts, {int((df.task == task).sum())} obs.)", fontsize=8)
    if j == 0: a.set_ylabel("response D(arm, R)\n(normalized a3–a4 motion)")
a = ax[2]
for k, (task, col) in enumerate((("google_robot_pick_coke_can", "#4a3aa7"), ("google_robot_move_near", "#1baf7a"))):
    d = df[df.task == task].groupby("seed").Delta.mean()
    a.scatter(np.full(len(d), k) + np.random.default_rng(0).uniform(-0.12, 0.12, len(d)), d, s=12, color=col)
    m = df[(df.task == task) & (df.matched)].groupby("seed").Delta.mean()
    a.scatter(np.full(len(m), k + 0.32), m, s=16, facecolors="none", edgecolors=col, label="matched" if k == 0 else None)
a.axhline(0, color="#8a8984", lw=0.8, ls=":")
a.set_xticks([0, 1]); a.set_xticklabels(["coke", "move near"]); a.set_ylabel("Δ = E_L − E_O (layout mean)")
a.set_title("paired Δ (filled: all; open: matched)", fontsize=8)
h, l = ax[0].get_legend_handles_labels()
fig.tight_layout(rect=(0, 0, 1, 0.86))
fig.legend(h, l, loc="upper center", ncol=2, frameon=False, fontsize=7, bbox_to_anchor=(0.5, 1.0))
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(A, f"fig_L_vs_O_by_task.{ext}"), dpi=170)

# 2. a3 vs a4 responses
fig, ax = plt.subplots(1, 2, figsize=(5.6, 2.4), sharey=True)
for j, task in enumerate(["google_robot_pick_coke_can", "google_robot_move_near"]):
    d = df[df.task == task].groupby("seed")[["E_L_a3", "E_L_a4", "E_O_a3", "E_O_a4"]].mean().mean()
    a = ax[j]
    a.bar([0, 1], [d.E_O_a3, d.E_O_a4], width=0.35, color=CO, label="O")
    a.bar([0.4, 1.4], [d.E_L_a3, d.E_L_a4], width=0.35, color=CL, label="L")
    a.set_xticks([0.2, 1.2]); a.set_xticklabels(["a3", "a4"]); a.set_title(TN[task], fontsize=8)
    if j == 0: a.set_ylabel("layout-weighted mean response"); a.legend(frameon=False, fontsize=7)
save(fig, "fig_a3_a4")

# 3. donor distance vs response
fig, ax = plt.subplots(1, 2, figsize=(5.6, 2.5), sharey=True)
for a, (dx, ey, col, lab) in zip(ax, (("d_O", "E_O", CO, "O: d_O vs E_O"), ("d_L", "E_L", CL, "L: d_L vs E_L"))):
    a.scatter(df[dx], df[ey], s=10, color=col, alpha=0.8)
    a.set_xlabel(f"donor distance {dx} (normalized)"); a.set_title(lab, fontsize=8)
ax[0].set_ylabel("response (normalized)")
save(fig, "fig_donor_distance")
print("ok")
