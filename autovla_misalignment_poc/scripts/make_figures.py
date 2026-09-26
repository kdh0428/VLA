#!/usr/bin/env python
"""
Figures A-E for the AutoVLA PoC.

Same visual system as the ORION PoC (validated categorical slots, fixed hue order, legend
plus marker shape so identity is never colour-alone, recessive grid, no dual axes), so the
two studies' figures read as one set.
"""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter

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


def save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path)


def fig_a(summary, outdir):
    """A- rate by GT scene condition (Plan A: perception as a scene condition)."""
    pc = summary["per_condition_A_minus"]
    keys = [k for k in pc if pc[k]["n"] >= 30]
    vals = [pc[k]["rate"] * 100 for k in keys]
    los = [(pc[k]["rate"] - pc[k]["ci95"][0]) * 100 for k in keys]
    his = [(pc[k]["ci95"][1] - pc[k]["rate"]) * 100 for k in keys]
    base = summary["A_minus"]["rate"] * 100

    fig, ax = plt.subplots(figsize=(9.5, 4.4))
    y = np.arange(len(keys))
    ax.barh(y, vals, xerr=[los, his], color=C_BLUE, height=0.6,
            error_kw=dict(ecolor=MUTED, lw=1.2, capsize=3))
    ax.axvline(base, color=C_ORANGE, linestyle="--", lw=1.5,
               label=f"overall {base:.1f}%")
    for i, (v, k) in enumerate(zip(vals, keys)):
        ax.text(v + max(his) + 0.4, i, f"{pc[k]['A-']}/{pc[k]['n']}",
                va="center", fontsize=8, color=INK2)
    ax.set_yticks(y)
    ax.set_yticklabels(keys)
    ax.invert_yaxis()
    ax.set_xlabel("A− rate (%)")
    ax.set_title("Figure A — AutoVLA action-failure rate by GT scene condition")
    ax.legend(fontsize=9)
    ax.grid(axis="y", visible=False)
    save(fig, os.path.join(outdir, "figA_condition_rates.png"))


def _curve(pr, grp, pos, label):
    res = pr["results"].get(grp, {}).get(pos, {})
    xs, ys, sh = [], [], []
    for L in sorted(res, key=int):
        e = res[L].get(label)
        if not e or not np.isfinite(e.get("auroc", np.nan)):
            continue
        xs.append(int(L)); ys.append(e["auroc"])
        s = e.get("auroc_shuffled")
        sh.append(s if s is not None and np.isfinite(s) else np.nan)
    return np.array(xs), np.array(ys), np.array(sh)


def fig_bc(pr, outdir, n_layers=36):
    """B: perception probe. C: action probe, GT vs the model's own action."""
    for name, labels, fname, title in (
        ("B", [("lead_vehicle", "Lead vehicle"), ("pedestrian", "Pedestrian"),
               ("critical_motion_moving", "Critical object moving")],
         "figB_perception_probe.png", "Figure B — Layer-wise perception probe"),
        ("C", [("action_gt", "GT action"), ("action_pred", "model's own action")],
         "figC_action_probe.png", "Figure C — Layer-wise action probe"),
    ):
        positions = ["prefill", "action_0", "action_9"]
        fig, axes = plt.subplots(1, len(positions), figsize=(4.6 * len(positions), 4.2),
                                 sharey=True)
        for ax, pos in zip(np.atleast_1d(axes), positions):
            for (lab, nice), color, mk in zip(labels, (C_BLUE, C_ORANGE, C_AQUA),
                                              ("o", "s", "^")):
                for grp, ls, alpha in (("A+", "-", 1.0), ("A-", "--", 0.85)):
                    x, y, sh = _curve(pr, grp, pos, lab)
                    if not len(x):
                        continue
                    ax.plot(x / n_layers, y, color=color, marker=mk, linestyle=ls,
                            alpha=alpha, label=f"{nice} · {grp}")
            ax.axhline(0.5, color=MUTED, ls=":", lw=1)
            ax.set_title(pos, fontsize=11)
            ax.set_xlabel("normalised depth (layer / 36)")
            ax.set_ylim(0.3, 1.02)
        np.atleast_1d(axes)[0].set_ylabel("AUROC")
        h, l = np.atleast_1d(axes)[0].get_legend_handles_labels()
        fig.legend(h, l, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.10), fontsize=9)
        fig.suptitle(title, y=1.02)
        save(fig, os.path.join(outdir, fname))


def fig_d(pr, outdir, n_layers=36):
    """Success vs failure: the gap between decoding GT action and the model's own action."""
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    pos = "action_0"
    for grp, color, mk in (("A+", C_BLUE, "o"), ("A-", C_ORANGE, "s")):
        x1, y1, _ = _curve(pr, grp, pos, "action_gt")
        x2, y2, _ = _curve(pr, grp, pos, "action_pred")
        if len(x1):
            ax.plot(x1 / n_layers, y1, color=color, marker=mk, ls="-",
                    label=f"{grp}: GT action")
        if len(x2):
            ax.plot(x2 / n_layers, y2, color=color, marker=mk, ls="--", alpha=0.75,
                    label=f"{grp}: model's own action")
    ax.axhline(0.5, color=MUTED, ls=":", lw=1)
    ax.set_xlabel("normalised depth (layer / 36)")
    ax.set_ylabel("AUROC")
    ax.set_ylim(0.3, 1.02)
    ax.set_title("Figure D — Failures encode an action confidently, just not the correct one")
    ax.legend(fontsize=9, loc="lower right")
    save(fig, os.path.join(outdir, "figD_gt_vs_own_action.png"))


def fig_e(ll, outdir):
    """Layer x action-generation-step logit-lens heatmaps, A+ vs A-."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    n_layers, n_steps = ll["n_layers"], ll["n_steps"]
    panels = [("A+", "top_is_gt", "A+ : P(argmax = GT action token)"),
              ("A-", "top_is_gt", "A− : P(argmax = GT action token)")]
    for ax, (grp, key, title) in zip(axes[:2], panels):
        M = np.asarray(ll["grids"][grp][key], dtype=float)
        im = ax.imshow(M, aspect="auto", origin="lower", cmap="viridis", vmin=0, vmax=1)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("action-generation step")
        ax.set_ylabel("decoder layer")
        ax.set_xticks(range(n_steps))
        ax.grid(False)
        fig.colorbar(im, ax=ax, fraction=0.046)

    ax = axes[2]
    prop = ll["autoregressive_propagation"]
    steps = sorted(int(k) for k in prop)
    a = [prop[str(t)]["P(err|prev all correct)"] for t in steps]
    b = [prop[str(t)]["P(err|some prev wrong)"] for t in steps]
    ax.plot(steps, a, color=C_BLUE, marker="o", label="P(error | all earlier correct)")
    ax.plot(steps, b, color=C_ORANGE, marker="s", label="P(error | an earlier step wrong)")
    ax.set_ylim(0, 1)
    ax.set_xlabel("action-generation step")
    ax.set_ylabel("P(token error)")
    ax.set_title("Autoregressive error propagation", fontsize=11)
    ax.legend(fontsize=8)
    fig.suptitle("Figure E — Action logit lens (layer × generation step)", y=1.03)
    save(fig, os.path.join(outdir, "figE_logit_lens.png"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=os.path.join(POC_DIR, "figures"))
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    summary = json.load(open(os.path.join(POC_DIR, "outputs/taxonomy/taxonomy_summary.json")))
    pr = json.load(open(os.path.join(POC_DIR, "outputs/probes/layerwise_probes.json")))
    ll = json.load(open(os.path.join(POC_DIR, "outputs/logit_lens/logit_lens.json")))

    fig_a(summary, args.outdir)
    fig_bc(pr, args.outdir)
    fig_d(pr, args.outdir)
    fig_e(ll, args.outdir)


if __name__ == "__main__":
    main()
