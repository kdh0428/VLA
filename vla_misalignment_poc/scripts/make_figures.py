#!/usr/bin/env python
"""
Figures A-D for the PoC report.

  A  failure-taxonomy distribution
  B  layer-wise perception probe, correct-action vs wrong-action groups
  C  layer-wise action/planning divergence, P+R+A+ vs P+R+A-
  C2 the headline overlay: perception decodability vs action alignment across depth
  D  representative qualitative cases

Design notes (these are print-style research figures, so they deliberately commit to a
single light look rather than being theme-aware):
  * categorical hues are assigned in fixed order from a validated palette and never
    cycled; blue = control / perception, orange = failure / action;
  * every multi-series panel carries a legend AND the two lines are also distinguished by
    marker shape, so identity is never colour-alone;
  * chance level (from the label-shuffled control) is drawn on every probe panel -- an
    AUROC curve without it is unreadable;
  * grid and axes are recessive; no dual y-axes anywhere.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

POC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, POC_DIR)

# Validated categorical slots (light mode). Slots 1-3 clear the all-pairs floors, which is
# what the line panels need; the bar panel uses the adjacent pairlist where 4 slots pass.
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


def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path)


# --------------------------------------------------------------------------------------
def figure_a(summary: dict, outdir: str) -> None:
    groups = summary.get("groups", {})
    order = ["P+R+A+", "P+R+A-", "P+R-A-", "P-R-A-", "P-R+A-", "P+R-A+", "P-R+A+", "P-R-A+"]
    keys = [k for k in order if groups.get(k)] + \
           [k for k in groups if k not in order and "?" not in k and groups[k]]
    unknown = sum(v for k, v in groups.items() if "?" in k)
    vals = [groups[k] for k in keys]
    if unknown:
        keys, vals = keys + ["undecidable"], vals + [unknown]

    # The two groups the study is about get the identity hues; the rest stay neutral so
    # the eye goes to the comparison that matters.
    colors = []
    for k in keys:
        colors.append(C_BLUE if k == "P+R+A+" else C_ORANGE if k == "P+R+A-" else
                      C_AQUA if k == "P+R-A-" else C_YELLOW if k == "P-R-A-" else MUTED)

    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    total = max(sum(vals), 1)
    bars = ax.bar(range(len(keys)), vals, color=colors, width=0.66)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v}\n{v/total:.0%}",
                ha="center", va="bottom", fontsize=9, color=INK2)
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels(keys, rotation=20, ha="right")
    ax.set_ylabel("samples")
    ax.set_title("Figure A — Failure taxonomy (P / R / A)")
    ax.set_ylim(0, max(vals) * 1.22 if vals else 1)
    ax.grid(axis="x", visible=False)
    _save(fig, os.path.join(outdir, "figA_taxonomy.png"))


# --------------------------------------------------------------------------------------
def _curve(probes: dict, group: str, label: str, metric: str = "auroc"):
    """Returns (layers, values, shuffled-null, reliable?). See `reliable` in probing.py."""
    res = probes.get("results", {}).get(group, {}).get("layers", {})
    xs, ys, ch = [], [], []
    reliable = True
    note = ""
    for L in sorted(res, key=int):
        e = res[L].get(label)
        if not e:
            continue
        v = e.get(metric)
        if v is None or not np.isfinite(v):
            continue
        xs.append(int(L))
        ys.append(v)
        s = e.get(f"{metric}_shuffled")
        ch.append(s if s is not None and np.isfinite(s) else np.nan)
        if not e.get("reliable", True):
            reliable = False
            note = e.get("reliability_note", "") or note
    return np.array(xs), np.array(ys), np.array(ch), reliable, note


def figure_b(probes: dict, outdir: str) -> None:
    """Perception decodability by layer, control vs failure group."""
    labels = [("traffic_light_present", "auroc", "Traffic light present (AUROC)"),
              ("lead_vehicle", "auroc", "Leading vehicle (AUROC)"),
              ("pedestrian", "auprc", "Pedestrian (AUPRC — rare label)"),
              ("critical_motion_moving", "auroc", "Critical object moving (AUROC)")]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.8), sharex=True)
    for ax, (lab, metric, title) in zip(axes.ravel(), labels):
        drew = False
        unreliable_note = ""
        for grp, color, marker, name in (("P+R+A+", C_BLUE, "o", "correct action (P+R+A+)"),
                                         ("P+R+A-", C_ORANGE, "s", "wrong action (P+R+A-)")):
            x, y, ch, ok, note = _curve(probes, grp, lab, metric)
            if len(x) == 0:
                continue
            # A probe whose positives sit in a handful of clips cannot show cross-clip
            # generalisation; draw it dashed and say so rather than presenting it as solid
            # evidence.
            ax.plot(x, y, color=color, marker=marker, label=name,
                    linestyle="-" if ok else "--", alpha=1.0 if ok else 0.65)
            drew = True
            if not ok:
                unreliable_note = note
            if np.isfinite(ch).any():
                ax.plot(x, ch, color=color, linestyle=":", linewidth=1.2, alpha=0.75,
                        label=f"{name} — shuffled")
        ttl = title + ("  [low clip support]" if unreliable_note else "")
        ax.set_title(ttl, fontsize=11)
        ax.set_ylabel(metric.upper())
        ax.set_ylim(0, 1.02)
        if unreliable_note:
            ax.text(0.02, 0.04, unreliable_note, transform=ax.transAxes,
                    fontsize=7.5, color=C_ORANGE, style="italic")
        if not drew:
            ax.text(0.5, 0.5, "insufficient samples", ha="center", va="center",
                    transform=ax.transAxes, color=MUTED)
    for ax in axes[-1]:
        ax.set_xlabel("decoder layer (0 = embedding)")
    h, l = axes[0, 0].get_legend_handles_labels()
    if h:
        fig.legend(h, l, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.06))
    fig.suptitle("Figure B — Layer-wise perception probe on the planning token", y=1.0)
    _save(fig, os.path.join(outdir, "figB_perception_probe.png"))


# --------------------------------------------------------------------------------------
def figure_c(div: dict, outdir: str) -> None:
    """Where the failure group's planning representation separates from the control's."""
    tags = [("PRA", "P+R+A+ vs P+R+A- (raw)"),
            ("PRA_action_matched", "GT-action matched (n=234)")]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.3))
    layers = sorted(div.get("layers", {}), key=int)

    ax = axes[0]
    for (tag, name), color, marker in zip(tags, (C_ORANGE, C_AQUA), ("s", "^")):
        x = [int(L) for L in layers if tag in div["layers"][L]]
        y = [div["layers"][L][tag].get("failure_probe_auroc", np.nan) for L in layers if tag in div["layers"][L]]
        s = [div["layers"][L][tag].get("failure_probe_auroc_shuffled", np.nan) for L in layers if tag in div["layers"][L]]
        if not x:
            continue
        ax.plot(x, y, color=color, marker=marker, label=name)
        if np.isfinite(np.array(s, dtype=float)).any():
            ax.plot(x, s, color=color, linestyle=":", linewidth=1.2, alpha=0.75,
                    label=f"{name} — shuffled")
    ax.axhline(0.5, color=MUTED, linewidth=1, linestyle="--")
    ax.set_title("Failure decodable from planning token")
    ax.set_xlabel("decoder layer"); ax.set_ylabel("AUROC"); ax.set_ylim(0.3, 1.02)
    ax.legend(fontsize=8)

    ax = axes[1]
    for (tag, name), color, marker in zip(tags, (C_ORANGE, C_AQUA), ("s", "^")):
        x = [int(L) for L in layers if tag in div["layers"][L]]
        y = [div["layers"][L][tag].get("centroid_l2_normalised", np.nan) for L in layers if tag in div["layers"][L]]
        if x:
            ax.plot(x, y, color=color, marker=marker, label=name)
    ax.set_title("Centroid distance (pooled-SD units)")
    ax.set_xlabel("decoder layer"); ax.set_ylabel("normalised L2")
    ax.legend(fontsize=8)

    # Confound panel: if this sits near |1|, the geometry is just restating ADE.
    ax = axes[2]
    for tag, color, marker, nm in (("PRA", C_ORANGE, "s", "raw"),
                                   ("PRA_action_matched", C_AQUA, "^", "action-matched")):
        x = [int(L) for L in layers if tag in div["layers"][L]]
        y = [div["layers"][L][tag].get("score_vs_ade", {}).get("spearman", np.nan)
             for L in layers if tag in div["layers"][L]]
        if x:
            ax.plot(x, y, color=color, marker=marker, label=nm)
    ax.legend(fontsize=8)
    ax.axhline(0, color=MUTED, linewidth=1)
    ax.set_ylim(-1, 1)
    ax.set_title("Confound check: score vs ADE (Spearman)")
    ax.set_xlabel("decoder layer"); ax.set_ylabel("rho")
    fig.suptitle("Figure C — Layer-wise action/planning divergence", y=1.02)
    _save(fig, os.path.join(outdir, "figC_action_divergence.png"))


def figure_c2(probes: dict, div: dict, outdir: str) -> None:
    """The headline claim, on one axis: perception stays high while action alignment falls."""
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    # Headline panel uses only probes with adequate clip support -- lead vehicle and
    # critical-object motion spread over 40+ clips, unlike traffic light / pedestrian.
    x, y, _, ok1, n1 = _curve(probes, "P+R+A-", "lead_vehicle", "auroc")
    x2, y2, _, ok2, n2 = _curve(probes, "P+R+A-", "critical_motion_moving", "auroc")
    if len(x):
        ax.plot(x, y, color=C_BLUE, marker="o", linestyle="-" if ok1 else "--",
                label="perception in failures: lead vehicle" + ("" if ok1 else " [low clip support]"))
    if len(x2):
        ax.plot(x2, y2, color=C_AQUA, marker="^", linestyle="-" if ok2 else "--",
                label="perception in failures: object motion" + ("" if ok2 else " [low clip support]"))

    layers = sorted(div.get("layers", {}), key=int)
    T = "PRA_action_matched"
    xd = [int(L) for L in layers if T in div["layers"][L]]
    yd = [div["layers"][L][T].get("failure_probe_auroc", np.nan) for L in layers if T in div["layers"][L]]
    if xd:
        ax.plot(xd, yd, color=C_ORANGE, marker="s", label="action divergence: failure AUROC (GT-action matched)")

    # Two distinct events, marked separately: the layer where failure first becomes
    # decodable at all, and the layer where the representations geometrically split.
    e = (div.get("earliest_action_divergence", {}) or {}).get("PRA_action_matched", {})
    g = (div.get("geometric_transition", {}) or {}).get("PRA_action_matched", {})
    if e.get("layer") is not None:
        ax.axvline(e["layer"], color=MUTED, linestyle=":", linewidth=1.3)
        ax.text(e["layer"], 0.30, f" decodable\n from L{e['layer']}",
                color=INK2, fontsize=8, va="bottom")
    if g.get("layer") is not None:
        ax.axvline(g["layer"], color=C_ORANGE, linestyle="--", linewidth=1.5, alpha=0.85)
        ax.text(g["layer"], 0.30, f" geometric split\n at L{g['layer']}",
                color=C_ORANGE, fontsize=9, va="bottom")
    ax.axhline(0.5, color=MUTED, linestyle="--", linewidth=1)
    ax.set_xlabel("decoder layer (0 = embedding)")
    ax.set_ylabel("AUROC")
    ax.set_ylim(0, 1.02)
    ax.set_title("Perception preservation vs action divergence across depth")
    ax.legend(fontsize=9, loc="lower left")
    _save(fig, os.path.join(outdir, "figC2_perception_vs_action.png"))


# --------------------------------------------------------------------------------------
def figure_d(rows: list[dict], probes: dict, outdir: str, n: int = 8) -> None:
    """Representative P+A- cases with their trajectories and reasoning."""
    import textwrap
    cands = [r for r in rows if r["P"] is True and r["A"] is False]
    cands.sort(key=lambda r: -(r["ADE"] if r["ADE"] == r["ADE"] else 0))
    seen, pick = set(), []
    for r in cands:                      # spread across scenarios
        if r["scenario"] in seen:
            continue
        seen.add(r["scenario"])
        pick.append(r)
        if len(pick) >= n:
            break
    for r in cands:
        if len(pick) >= n:
            break
        if r not in pick:
            pick.append(r)
    if not pick:
        print("figure D: no P+A- cases to show")
        return

    recs = {}
    mp = os.path.join(POC_DIR, "outputs", "merged", "records.jsonl")
    if os.path.exists(mp):
        with open(mp) as f:
            for line in f:
                if line.strip():
                    d = json.loads(line)
                    recs[d["sample_id"]] = d

    cols = min(4, len(pick))
    rowsn = int(np.ceil(len(pick) / cols))
    fig, axes = plt.subplots(rowsn, cols, figsize=(4.0 * cols, 4.3 * rowsn), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for i, r in enumerate(pick):
        ax = axes[i // cols][i % cols]
        ax.axis("on")
        d = recs.get(r["sample_id"], {})
        gt = np.cumsum(np.array(d.get("trajectory_gt", []), dtype=float).reshape(-1, 2), axis=0) \
            if d.get("trajectory_gt") else np.zeros((0, 2))
        pr = np.cumsum(np.array(d.get("trajectory_pred", []), dtype=float).reshape(-1, 2), axis=0) \
            if d.get("trajectory_pred") else np.zeros((0, 2))
        if len(gt):
            ax.plot(gt[:, 0], gt[:, 1], color=C_BLUE, marker="o", label="GT")
        if len(pr):
            ax.plot(pr[:, 0], pr[:, 1], color=C_ORANGE, marker="s", label="pred")
        ax.scatter([0], [0], c=MUTED, s=28, marker="*", zorder=3)
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_xlabel("lateral (m)", fontsize=8)
        ax.set_ylabel("longitudinal (m)", fontsize=8)
        reason = textwrap.shorten(r.get("reasoning_summary", "") or "", 110, placeholder="…")
        ax.set_title(f"{r['scenario']}  ADE={r['ADE']:.2f}\n"
                     f"GT {r['high_level_action_gt']} -> pred {r['high_level_action_pred']}\n"
                     f"tl={r.get('gt_traffic_light')} lead={r.get('gt_lead_vehicle')} "
                     f"ped={r.get('gt_pedestrian')}\n{textwrap.fill(reason, 46)}",
                     fontsize=7.5, loc="left")
        if i == 0:
            ax.legend(fontsize=8)
    fig.suptitle("Figure D — Representative correct-perception / wrong-action cases", y=1.0)
    _save(fig, os.path.join(outdir, "figD_cases.png"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--taxonomy-dir", default=os.path.join(POC_DIR, "outputs", "taxonomy"))
    ap.add_argument("--probes-dir", default=os.path.join(POC_DIR, "outputs", "probes"))
    ap.add_argument("--outdir", default=os.path.join(POC_DIR, "figures"))
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    def load(p, default=None):
        if os.path.exists(p):
            with open(p) as f:
                return json.load(f)
        print(f"  [skip] missing {p}")
        return default

    summary = load(os.path.join(args.taxonomy_dir, "taxonomy_summary.json"), {})
    rows = load(os.path.join(args.taxonomy_dir, "taxonomy.json"), [])
    probes = load(os.path.join(args.probes_dir, "layerwise_probes.json"), {})
    div = load(os.path.join(args.probes_dir, "layerwise_divergence.json"), {})

    if summary:
        figure_a(summary, args.outdir)
    if probes:
        figure_b(probes, args.outdir)
    if div:
        figure_c(div, args.outdir)
    if probes and div:
        figure_c2(probes, div, args.outdir)
    if rows:
        figure_d(rows, probes, args.outdir)


if __name__ == "__main__":
    main()
