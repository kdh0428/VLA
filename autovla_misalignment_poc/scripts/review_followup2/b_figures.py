#!/usr/bin/env python
"""
B (review_followup2, 2026-10-13): figures from DIRECT_EFFECTS.csv and B4_TRAJECTORY_VS_SUCCESS.csv (post-hoc, CPU).
  fig_direct_effects.png  task / pooled success direct effects (delta_R, delta_C) with +-5 / +-10 pp bands;
                          interactions in a separate panel row. Two clusterings (primary, physical).
  fig_tcp_vs_success.png  same samples side by side: generated suffix change, executed command accumulation,
                          actual TCP mean error, success direct effect (pooled, primary clustering).
  nice -n 19 /root/miniforge3/envs/autovla/bin/python b_figures.py
"""
from __future__ import annotations

import os
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

OUT = "/root/VLA/autovla_misalignment_poc/outputs/review_followup2/B_spatialvla_direct/run_20261013"
COL = {"primary": "#2a78d6", "physical": "#eb6834", "episode": "#1baf7a"}
INK, MUTED = "#0b0b0b", "#52514e"
LIM = 40
EXPN = {"exp34": "exp 34", "exp35": "exp 35", "P1D": "P1-D"}


def forest(ax, sub, labels, title, show_y):
    ax.axvspan(-10, 10, color="#000000", alpha=0.06, lw=0)
    ax.axvspan(-5, 5, color="#000000", alpha=0.08, lw=0)
    ax.axvline(0, color=MUTED, lw=0.8)
    for k, lab in enumerate(labels):
        y = len(labels) - 1 - k
        for j, cl in enumerate(("primary", "physical")):
            r = sub[(sub.key == lab) & (sub.clustering == cl)]
            if r.empty:
                continue
            r = r.iloc[0]
            yy = y + (0.15 if j == 0 else -0.15)
            lo, hi = r.ci95_lo_pp, r.ci95_hi_pp
            lo_c, hi_c = max(lo, -LIM), min(hi, LIM)
            ax.plot([lo_c, hi_c], [yy, yy], color=COL[cl], lw=1.2, solid_capstyle="round")
            ax.plot([max(r.ci90_lo_pp, -LIM), min(r.ci90_hi_pp, LIM)], [yy, yy], color=COL[cl], lw=3,
                    solid_capstyle="round")
            for v, s in ((lo, -1), (hi, 1)):
                if not np.isfinite(v) or abs(v) > LIM:
                    ax.annotate("", xy=(s * LIM, yy), xytext=(s * (LIM - 4), yy),
                                arrowprops=dict(arrowstyle="->", color=COL[cl], lw=1))
            ax.plot(r.estimate_pp, yy, "o", ms=5, color=COL[cl], mec="white", mew=1)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(list(reversed(labels)) if show_y else [], fontsize=7.5, color=INK)
    ax.set_xlim(-LIM, LIM)
    ax.set_ylim(-0.7, len(labels) - 0.3)
    ax.set_title(title, fontsize=9, color=INK)
    ax.tick_params(axis="x", labelsize=7.5, colors=MUTED)
    ax.grid(axis="x", color="#e6e6e3", lw=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def fig_direct(df):
    d = df[(df.role == "core") & df.clustering.isin(["primary", "physical"])].copy()
    sym = {"R-N": "delta_R (R-N)", "C-F": "delta_C (C-F)"}
    dr = d[(d.kind == "direct") & d.contrast.isin(["R-N", "C-F"])].copy()
    dr["key"] = dr.experiment.map(EXPN) + " " + dr.config + "  " + dr.contrast.map(sym)
    order = []
    for exp, cfgs in (("exp34", ["r1e4/perp_left", "r1e4/opposite"]), ("exp35", ["r1e4", "r4e1"]), ("P1D", ["r1e4", "r4e1"])):
        for c in cfgs:
            for cn in ("R-N", "C-F"):
                order.append(f"{EXPN[exp]} {c}  {sym[cn]}")
    it = d[d.kind == "interaction"].copy()
    it["key"] = it.experiment.map(EXPN) + " " + it.config.str.replace(" (cross-axis, post-hoc)", "*", regex=False) + "  " + \
        it.contrast.str.replace(" interaction", "", regex=False)
    iorder = list(dict.fromkeys(it.key))
    fig, axs = plt.subplots(2, 3, figsize=(13, 10.5), gridspec_kw={"height_ratios": [len(order), len(iorder)]})
    fig.patch.set_facecolor("white")
    for j, t in enumerate(("coke", "move_near", "pooled")):
        forest(axs[0, j], dr[dr.task == t], order, f"Direct effects: {t if t != 'pooled' else 'task-equal pooled'}", j == 0)
        forest(axs[1, j], it[it.task == t], iorder, f"Interactions: {t if t != 'pooled' else 'task-equal pooled'}", j == 0)
        axs[1, j].set_xlabel("success difference (percentage points)", fontsize=8, color=MUTED)
    from matplotlib.lines import Line2D
    h = [Line2D([0], [0], color=COL["primary"], marker="o", lw=2, label="primary initial-state clusters"),
         Line2D([0], [0], color=COL["physical"], marker="o", lw=2, label="physical layouts (move_near 10)"),
         plt.Rectangle((0, 0), 1, 1, color="#000000", alpha=0.14, label="+-5 pp (post-hoc margin)"),
         plt.Rectangle((0, 0), 1, 1, color="#000000", alpha=0.06, label="+-10 pp (original margin)")]
    fig.legend(handles=h, loc="upper center", ncol=4, frameon=False, fontsize=8)
    fig.text(0.01, 0.003, textwrap.fill("Thin line 95% CI, thick 90% CI (sign-flip inversion; arrows = beyond +-40 pp or unbounded). "
             "delta_R = S(R)-S(N), delta_C = S(C)-S(F) (+ = correction benefit). Interaction = contrast(config 2) - "
             "contrast(config 1), same unit. * = post-hoc cross-axis. Post-hoc analysis of existing data.", 200),
             fontsize=7, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    fig.savefig(os.path.join(OUT, "fig_direct_effects.png"), dpi=150)
    plt.close(fig)


def fig_tcp(b4):
    b = b4[(b4.task == "pooled") & b4.contrast.isin(["R-N", "C-F", "CD-FD", "gen-N"]) &
           (b4.subset == "all units of the contrast")].copy()
    rows = []
    for exp, cfgs in (("exp34", ["r1e4"]), ("exp35", ["r1e4", "r4e1", "r1e2", "r1e1", "r2e2", "r2e1"]),
                      ("P1D", ["r1e4", "r4e1", "r1e1", "r2e1"])):
        for c in cfgs:
            for cn in ("R-N", "C-F", "CD-FD", "gen-N"):
                for dr in (["perp_left", "opposite", "-"] if exp == "exp34" else ["opposite"]):
                    r = b[(b.experiment == exp) & (b.config == c) & (b.contrast == cn) & (b.direction == dr)]
                    if not r.empty:
                        rr = r.iloc[0].copy()
                        rr["label"] = f"{EXPN[exp]} {c}{'/' + dr if exp == 'exp34' and dr != '-' else ''} {cn}"
                        rows.append(rr)
    t = pd.DataFrame(rows).reset_index(drop=True)
    n = len(t)
    fig, axs = plt.subplots(1, 4, figsize=(15, 0.32 * n + 1.6), sharey=True)
    specs = [("gen_suffix_tokfrac", 100, "generated suffix tokens changed (%)"),
             ("cmd_final_m", 100, "executed command accumulation |sum a_a - sum a_b| (cm)"),
             ("tcp_mean_cm", 1, "actual TCP mean distance, t=8-79 (cm)"),
             ("success_effect_pp", 1, "success direct effect (pp), 95% CI")]
    ys = np.arange(n)[::-1]
    for ax, (col, sc, lab) in zip(axs, specs):
        for i, r in t.iterrows():
            c = COL["physical"] if r.contrast == "gen-N" else COL["primary"]
            if col == "success_effect_pp":
                ax.axvspan(-10, 10, color="#000000", alpha=0.03, lw=0) if i == 0 else None
                ax.axvspan(-5, 5, color="#000000", alpha=0.04, lw=0) if i == 0 else None
                ax.plot([r.success_ci95_lo_pp, r.success_ci95_hi_pp], [ys[i]] * 2, color=c, lw=1.4)
                ax.plot(r[col], ys[i], "o", color=c, ms=5, mec="white")
            else:
                v = r[col] * sc
                if np.isfinite(v):
                    lo, hi = r.get(f"{col}_ci95_lo", np.nan) * sc, r.get(f"{col}_ci95_hi", np.nan) * sc
                    ax.barh(ys[i], v, height=0.6, color=c, alpha=0.85)
                    if np.isfinite(lo):
                        ax.plot([lo, hi], [ys[i]] * 2, color=INK, lw=0.8)
                else:
                    ax.text(0.2, ys[i], "not recorded", fontsize=6.5, va="center", color=MUTED)
        ax.set_title(lab, fontsize=8.5, color=INK)
        ax.tick_params(labelsize=7.5, colors=MUTED)
        ax.grid(axis="x", color="#e6e6e3", lw=0.6)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axs[3].axvline(0, color=MUTED, lw=0.8)
    axs[0].set_yticks(ys)
    axs[0].set_yticklabels(t.label, fontsize=7.5)
    fig.text(0.01, 0.003, textwrap.fill("Pooled (task-equal), same units per row. Blue = context contrasts (R-N, C-F, CD-FD); orange = "
             "natural re-run (gen-N, numeric-path noise). Generated suffix: P1-D shadow chunk at t=8,12,16,20; exp 34/35 "
             "a-vs-b chunk at t=8 (prefix-identical pairs). TCP not recorded in exp 34. CIs: cluster (primary) bootstrap "
             "for metrics, sign-flip inversion for success.", 200), fontsize=7, color=MUTED)
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    fig.savefig(os.path.join(OUT, "fig_tcp_vs_success.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    df = pd.read_csv(os.path.join(OUT, "DIRECT_EFFECTS.csv"))
    fig_direct(df)
    fig_tcp(pd.read_csv(os.path.join(OUT, "B4_TRAJECTORY_VS_SUCCESS.csv")))
    print("figures written")
