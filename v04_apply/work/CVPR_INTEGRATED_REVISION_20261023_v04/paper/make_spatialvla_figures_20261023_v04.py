#!/usr/bin/env python
"""SpatialVLA publication figures, 20261023 v04 (CPU only).
Run without arguments. Input paths are relative to this package. Adapted from the supplied plot_v02.py; values are unchanged, labels distinguish reset scenarios. Experiments remain separate.
"""
import os, sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
v01p = ROOT / 'paper/evidence/spatialvla_discovery_per_observation_20261022_v01.csv'
v02d = ROOT / 'CVPR_SPATIALVLA_WITHIN_CALL_RESULT_20261023_v02/analysis'
drd = ROOT / 'CVPR_SPATIALVLA_DRAWER_RESULT_20261023_v01/analysis'
out = ROOT / 'paper/figures' 
os.makedirs(out, exist_ok=True)
CL, CO = "#2a78d6", "#eb6834"
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#b5b4ae",
                     "axes.labelcolor": "#52514e", "xtick.color": "#52514e", "ytick.color": "#52514e"})
TN = {"google_robot_pick_coke_can": "coke", "google_robot_move_near": "move near", "google_robot_open_drawer": "open drawer"}
v01 = pd.read_csv(v01p)
v02 = pd.read_csv(os.path.join(v02d, "per_observation.csv")) if os.path.exists(os.path.join(v02d, "per_observation.csv")) else pd.DataFrame()
dr = pd.read_csv(os.path.join(drd, "per_observation.csv")) if drd != "NONE" and os.path.exists(os.path.join(drd, "per_observation.csv")) else pd.DataFrame()


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out, f"{name}.{ext}"), dpi=170)


# Figure A (main-text candidate): layout-mean E_O vs E_L per task in v02 (and drawer), paired lines
panels = [(v02, t) for t in ("google_robot_pick_coke_can", "google_robot_move_near")] + ([(dr, "google_robot_open_drawer")] if len(dr) else [])
fig, ax = plt.subplots(1, len(panels) + 1, figsize=(2.1 * (len(panels) + 1) + 0.6, 2.35), gridspec_kw={"width_ratios": [1] * len(panels) + [1.5]})
for j, (df, task) in enumerate(panels):
    a = ax[j]; d = df[df.task == task].groupby("seed")[["E_L", "E_O"]].mean() if len(df) else pd.DataFrame(columns=["E_L", "E_O"])
    for _, r in d.iterrows():
        a.plot([0, 1], [r.E_O, r.E_L], color="#c3c2b7", lw=0.8)
    a.scatter(np.zeros(len(d)), d.E_O, color=CO, s=12, zorder=3, label="O: a1 edited (older)")
    a.scatter(np.ones(len(d)), d.E_L, color=CL, s=12, zorder=3, label="L: a2 edited (latest)")
    a.set_xticks([0, 1]); a.set_xticklabels(["O", "L"]); a.set_xlim(-0.4, 1.4)
    nobs = int((df.task == task).sum()) if len(df) else 0
    unit = "scenarios" if task.endswith("drawer") else "layouts"
    a.set_title(f"{TN[task]}{' (supp.)' if task.endswith('drawer') else ''}\n{len(d)} {unit}, {nobs} obs.", fontsize=7.5)
    if j == 0: a.set_ylabel("response D$_S$(arm, R)")
a = ax[-1]
items = [("discovery all\n(secondary)", v01, None), ("discovery\nmatched", v01[v01.matched] if "matched" in v01 else v01.iloc[:0], None), ("follow-up\nmatched", v02, None)] + ([("drawer\nmatched", dr, None)] if len(dr) else [])
for k, (lab, df, _) in enumerate(items):
    if not len(df): continue
    lm = df.groupby(["task", "seed"]).Delta.mean().reset_index()
    for t, col in (("google_robot_pick_coke_can", "#4a3aa7"), ("google_robot_move_near", "#1baf7a"), ("google_robot_open_drawer", "#e87ba4")):
        v = lm[lm.task == t].Delta.values
        a.scatter(k + np.random.default_rng(k).uniform(-0.15, 0.15, len(v)), v, s=8, color=col, alpha=0.85, label=(TN[t] if ((k == 2 and not t.endswith("drawer")) or (k == 3 and t.endswith("drawer"))) else None))
a.axhline(0, color="#8a8984", lw=0.8, ls=":")
a.set_xticks(range(len(items))); a.set_xticklabels([i[0] for i in items], fontsize=6.5)
a.set_ylabel("unit-mean Δ = E_L − E_O"); a.set_title("separate experiments", fontsize=7.5)
a.legend(frameon=False, fontsize=6, loc="upper right")
h, l = ax[0].get_legend_handles_labels()
fig.tight_layout(rect=(0, 0, 1, 0.86)); fig.legend(h, l, loc="upper center", ncol=2, frameon=False, fontsize=7, bbox_to_anchor=(0.45, 1.0))
save(fig, "fig_svla_within_call")

# Figure B (supplement): a3/a4 and translation/rotation, v02 + drawer
for name, df in (("v02", v02), ("drawer", dr)):
    if not len(df): continue
    tasks = sorted(df.task.unique())
    fig, ax = plt.subplots(1, len(tasks), figsize=(2.9 * len(tasks), 2.3), sharey=True, squeeze=False)
    for j, t in enumerate(tasks):
        d = df[df.task == t].groupby("seed")[["E_L_a3", "E_O_a3", "E_L_a4", "E_O_a4", "E_L_trans", "E_O_trans", "E_L_rot", "E_O_rot"]].mean().mean()
        a = ax[0, j]; xs = np.arange(4)
        a.bar(xs - 0.18, [d.E_O_a3, d.E_O_a4, d.E_O_trans, d.E_O_rot], width=0.36, color=CO, label="O")
        a.bar(xs + 0.18, [d.E_L_a3, d.E_L_a4, d.E_L_trans, d.E_L_rot], width=0.36, color=CL, label="L")
        a.set_xticks(xs); a.set_xticklabels(["a3", "a4", "transl.", "rot."]); a.set_title(TN[t], fontsize=8)
        if j == 0: a.set_ylabel("scenario-weighted mean response" if name == "drawer" else "layout-weighted mean response"); a.legend(frameon=False, fontsize=7)
    fig.tight_layout(); save(fig, f"fig_svla_{name}_components")
    fig, ax = plt.subplots(1, 2, figsize=(5.6, 2.4), sharey=True)
    for a, (dx, ey, col, lab) in zip(ax, (("d_O", "E_O", CO, "O"), ("d_L", "E_L", CL, "L"))):
        a.scatter(df[dx], df[ey], s=9, color=col, alpha=0.8); a.set_xlabel(f"donor distance {dx}"); a.set_title(f"{lab} ({'follow-up' if name == 'v02' else 'drawer'})", fontsize=8)
    ax[0].set_ylabel("response"); fig.tight_layout(); save(fig, f"fig_svla_{name}_donor_distance")
print("ok")
