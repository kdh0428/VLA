#!/usr/bin/env python
"""power_simulation.png: calibrated 95% CI half-width and power vs number of eval drives (scene-weighted, k = all units)."""
import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

D = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
df = pd.read_csv(os.path.join(D, "power_simulation.csv"))
df = df[(df.weighting == "scene") & (df.units_per_scene.astype(str) == "all")]
COL = {2: "#2a78d6", 4: "#eb6834", 8: "#1baf7a", 16: "#eda100", 0: "#2a78d6"}
INK, MUTED, GRID = "#1f1f1e", "#6b6a63", "#e4e2da"
NAMES = {"rn": "Recent − Normal", "rv": "Reverse − Full (approx.)", "dm": "Direction − Magnitude"}
fig, axes = plt.subplots(4, 3, figsize=(13, 14), sharex=True)
for ci, c in enumerate(("rn", "rv", "dm")):
    for ri, (est, what) in enumerate((("F", "hw"), ("P", "hw"), ("F", "pow"), ("P", "pow"))):
        ax = axes[ri, ci]
        sub = df[(df.contrast == c) & (df.estimand == est)]
        ms = [0] if est == "F" else [2, 4, 8, 16]
        for m in ms:
            for cl, ls in (("drive", "-"), ("segment", ":")):
                s = sub[(sub.aplus_per_log == m) & (sub.cluster_model == cl)].sort_values("drives")
                if est == "F":
                    s = sub[(sub.aplus_per_log == 0) & (sub.cluster_model == cl)].sort_values("drives")
                if s.empty:
                    continue
                lab = (f"m = {m}" if est == "P" else "all A− scenes") + (" (drive model)" if cl == "drive" else " (segment model)")
                if est == "P" and cl == "segment" and m != 4:
                    continue
                if what == "hw":
                    ax.plot(s.drives, 100 * s.calibrated_halfwidth95, ls, color=COL[m], lw=2, marker="o", ms=4, label=lab)
                else:
                    if est == "P" and m != 4:
                        continue
                    ax.plot(s.drives, s.power_calibHolm_dev, ls, color="#2a78d6", lw=2, marker="o", ms=4,
                            label="at dev effect" + ("" if cl == "drive" else " (segment model)"))
                    ax.plot(s.drives, s.power_calibHolm_MME, ls, color="#eb6834", lw=2, marker="o", ms=4,
                            label="at MME" + ("" if cl == "drive" else " (segment model)"))
        mme = sub.mme.iloc[0]
        if what == "hw":
            ax.axhline(100 * 0.61 * abs(mme), color=MUTED, lw=1, ls="--")
            ax.text(48, 100 * 0.61 * abs(mme), " target 0.61·|MME|", color=MUTED, fontsize=8, va="bottom", ha="right")
            ax.set_ylabel("95% CI half-width (pp)" if ci == 0 else "")
            ax.set_ylim(bottom=0)
        else:
            ax.axhline(0.8, color=MUTED, lw=1, ls="--")
            ax.set_ylim(0, 1.02)
            ax.set_ylabel("power (Holm first step)" if ci == 0 else "")
        ax.axvspan(24, 50, color=GRID, alpha=0.5, lw=0)
        ax.axvline(24, color=MUTED, lw=1)
        ax.set_title(f"{NAMES[c]} — {'failure (F)' if est == 'F' else 'population (P)'}"
                     f"{'' if what == 'hw' else ', power'}", fontsize=10, color=INK, loc="left")
        ax.grid(axis="y", color=GRID, lw=0.8); ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(colors=MUTED, labelsize=8)
        if ci == 0:
            ax.legend(fontsize=7, frameon=False)
for ax in axes[-1]:
    ax.set_xlabel("eval drives (clusters); shaded = more than the 24 available (hypothetical)", fontsize=8, color=MUTED)
fig.suptitle("P1-C precision/power simulation (scene-weighted; MME = minimal meaningful effect; calibrated to simulated H0)",
             fontsize=11, color=INK, x=0.01, ha="left")
fig.tight_layout(rect=(0, 0, 1, 0.98))
fig.savefig(os.path.join(D, "power_simulation.png"), dpi=130)
print("saved")
