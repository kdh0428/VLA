#!/usr/bin/env python
"""V2-c (schedule diagram + R9 window points) and V3 (effects for plotting, scene/drive tables). Post-hoc, CPU only.
Run: nice -n 19 /root/miniforge3/envs/autovla/bin/python ve_v2c_v3.py
"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ve_common as V  # noqa: E402

OUT = V.OUT


# ------------------------------------------------------------------ V2-c
def schedule():
    plt = V.mpl_setup()
    from matplotlib.patches import Rectangle
    ts = 1  # illustrative t* = 1; generation steps k = t*+1 .. 9, context positions j = t*..k-1
    conds = [("Normal AR", lambda k, j: False),
             ("Recent-GT (repeated, sliding)", lambda k, j: j == k - 1 and j > ts),
             ("R9 win3 (retained t*+1..t*+3)", lambda k, j: ts + 1 <= j <= ts + 3),
             ("Full GT-history", lambda k, j: j > ts)]
    fig, axs = plt.subplots(1, 4, figsize=(6.9, 2.1), sharey=True)
    for ax, (name, f) in zip(axs, conds):
        for k in range(ts + 1, 10):
            for j in range(0, k):
                if j < ts:
                    col, ec = "#ecebe8", "#ecebe8"      # pre-t* prefix (model's own, shared)
                elif j == ts:
                    col, ec = V.INK2, V.INK2           # perturbed / mismatched token at t*
                else:
                    col, ec = (V.PAL[0], V.PAL[0]) if f(k, j) else ("white", "#b9b8b3")
                ax.add_patch(Rectangle((j, k), 0.9, 0.9, facecolor=col, edgecolor=ec, lw=0.5))
        ax.set_xlim(-0.2, 9.2); ax.set_ylim(9.9, ts + 0.9)
        ax.set_title(name, loc="left", fontsize=6.5, color=V.INK)
        ax.set_xlabel("context position j")
        ax.set_xticks(np.arange(0, 10) + 0.45); ax.set_xticklabels(range(10), fontsize=5.5)
        ax.spines[["left", "bottom"]].set_visible(False); ax.tick_params(length=0)
    axs[0].set_yticks(np.arange(ts + 1, 10) + 0.45); axs[0].set_yticklabels([f"k={k}" for k in range(ts + 1, 10)], fontsize=5.5)
    axs[0].set_ylabel("generation step")
    fig.text(0.01, -0.06, "blue = context position holds the GT token while generating step k; white = model's own emitted token; "
             "dark = token at t* (illustration t* = 1). Executed actions are always the model's own tokens.",
             fontsize=5.8, color=V.INK2)
    fig.tight_layout()
    V.save(fig, "V2c_correction_schedules")
    plt.close(fig)
    # window points
    cs = pd.read_csv(f"{V.R8R9}/condition_summary.csv")
    w = cs[(cs.experiment == "R9") & (cs.metric == "amplification") &
           cs.condition.isin(["normal"] + [f"win{i}" for i in range(1, 9)] + ["gt_history"])].copy()
    u = pd.read_parquet(f"{V.R8R9}/units.parquet", columns=["experiment", "unit_id", "group", "t_star", "condition"])
    u = u[(u.experiment == "R9") & (u.condition == "normal")].drop_duplicates("unit_id")
    rows = []
    for g in ("A-", "A+"):
        ug = u[u.group == g]
        for wi in range(1, 9):
            rows.append({"group": g, "condition": f"win{wi}", "units_window_saturated_to_full": int((wi >= 8 - ug.t_star).sum()),
                         "n_units": int(len(ug))})
    sat = pd.DataFrame(rows)
    w = w.merge(sat, on=["group", "condition"], how="left")
    w["ci_status"] = "log-cluster bootstrap from R8R9 recovery; not validated (bootstrap under-covers in later checks) - not drawn"
    w["note"] = ("retained window: GT held at positions t*+1..t*+w for all later steps (never released); win w >= 8-t* equals "
                 "Full GT-history; same 365 A- / 1,043 A+ units in every row")
    w.to_csv(f"{OUT}/tables/V2c_window_results.csv", index=False)
    fig, axs = plt.subplots(1, 2, figsize=(6.9, 2.2))
    for ax, g, lab in ((axs[0], "A-", "A− (failure-selected; 52 scenes, 365 units)"), (axs[1], "A+", "A+ (156 scenes, 1,043 units)")):
        d = w[w.group == g].set_index("condition")
        order = ["normal"] + [f"win{i}" for i in range(1, 9)] + ["gt_history"]
        labels = ["Normal"] + [f"w={i}" for i in range(1, 9)] + ["Full GT"]
        y = [100 * d.loc[c, "scene_mean"] for c in order]
        cols = [V.INK2] + [V.PAL[0]] * 8 + [V.INK2]
        ax.scatter(range(len(order)), y, c=cols, s=18, zorder=3, edgecolors="white", linewidths=0.6)
        for i, c in enumerate(order):
            if c.startswith("win"):
                sat_n = d.loc[c, "units_window_saturated_to_full"]
                ax.annotate(f"{int(sat_n)}", (i, y[i]), xytext=(0, 5), textcoords="offset points", fontsize=5.2, ha="center", color=V.INK2)
        ax.set_xticks(range(len(order))); ax.set_xticklabels(labels, fontsize=5.8)
        ax.set_ylabel("severe rate (% units, scene-mean)"); ax.grid(True, axis="y"); ax.set_ylim(bottom=0)
        ax.set_title(lab, loc="left", color=V.INK)
    fig.text(0.01, -0.04, "Discrete points, no connecting curve; numbers above points = units for which window w already equals Full "
             "GT-history (w ≥ 8 − t*). Retained window: corrected positions stay GT for all later steps (no release). No interval drawn.",
             fontsize=5.6, color=V.INK2)
    fig.tight_layout()
    V.save(fig, "V2c_window_points")
    plt.close(fig)


# ------------------------------------------------------------------ V3
def effects():
    f = pd.read_csv(f"{V.AQ}/FINAL_PAPER_NUMBERS.csv")
    pr = pd.read_csv(f"{V.CVPR_NEXT}/PAPER_READY_NUMBERS.csv")
    want = {"REP.P0A.DEV.RN.Am": ("RN", "dev A− (failure-selected)", "CVPR policy: APPROXIMATE_ONLY (fixed-scene)", True),
            "REP.P0A.DEV.RV.Am": ("RV", "dev A− (failure-selected)", "CVPR policy: APPROXIMATE_ONLY (by analogy)", True),
            "REP.P0A.P1C.RN.F": ("RN", "P1-C F", "CVPR policy: APPROXIMATE_ONLY (fixed-scene)", True),
            "REP.P0A.P1C.RV.F": ("RV", "P1-C F", "CVPR policy: APPROXIMATE_ONLY (fixed-scene)", True),
            "REP.P0A.P1C.RN.P": ("RN", "P1-C weighted E", None, None), "REP.P0A.P1C.RV.P": ("RV", "P1-C weighted E", None, None),
            "B2.C_GT.F.drive": ("C_GT", "B2 F", "approximate (by analogy to the simulated C_NAT F check)", True),
            "B2.C_NAT.F.drive": ("C_NAT", "B2 F", None, None), "B2.INT.F.drive": ("INT", "B2 F", None, None),
            "B2.C_PROV.F.drive": ("C_src (=C_PROV)", "B2 F", "APPROXIMATE_ONLY (by analogy)", True),
            "B2.C_GT.P.drive": ("C_GT", "B2 weighted E4", None, None), "B2.C_NAT.P.drive": ("C_NAT", "B2 weighted E4", None, None),
            "B2.INT.P.drive": ("INT", "B2 weighted E4", None, None), "B2.C_PROV.P.drive": ("C_src (=C_PROV)", "B2 weighted E4", None, None)}
    prmap = {"REP.P0A.P1C.RN.P": "CVPR.E_RN", "REP.P0A.P1C.RV.P": "CVPR.E_RV", "B2.C_NAT.P.drive": "CVPR.E4_CNAT",
             "B2.C_GT.P.drive": "CVPR.E4_CGT", "B2.INT.P.drive": "CVPR.E4_INT", "B2.C_PROV.P.drive": "CVPR.E4_CPROV",
             "B2.INT.F.drive": "CVPR.F_INT", "B2.C_NAT.F.drive": "CVPR.F_CNAT"}
    rows = []
    for cid, (con, coh, status, draw) in want.items():
        r = f[f.claim_id == cid].iloc[0]
        lo, hi, meth, src, sh = r.ci_low, r.ci_high, r.ci_method, r.source_file, r.source_hash
        p_raw, p_adj = r.p_raw, r.p_adjusted
        if cid in prmap:
            q = pr[pr.claim_id == prmap[cid]].iloc[0]
            status = q.ci_status
            lo, hi, meth = q.ci_low, q.ci_high, q.ci_method
            p_raw, p_adj = q.p_raw, q.p_adjusted
            draw = str(status).startswith("APPROXIMATE_ONLY") or str(status).startswith("APPROXIMATE (fixed")
        rows.append({"claim_id": cid, "contrast": con, "cohort": coh, "estimand": r.estimand, "outcome": r.outcome,
                     "unit": r.unit, "sign_definition": r.contrast_definition_and_sign, "estimate": r.estimate,
                     "ci_low": lo if draw else np.nan, "ci_high": hi if draw else np.nan,
                     "ci_low_not_drawn": np.nan if draw else lo, "ci_high_not_drawn": np.nan if draw else hi,
                     "ci_status": status, "ci_drawn": bool(draw), "ci_method": meth, "cluster_axis": "drive",
                     "p_raw": p_raw, "p_adjusted": p_adj, "family": r.family, "alpha": r.alpha, "n_units": r.n_units,
                     "n_scenes": r.n_scenes, "n_drives": r.n_drives, "n_components": r.n_components, "weighting": r.weighting,
                     "row_kind": "re-visualised existing aggregate", "source_ref": V.rel(src) if str(src).startswith("/") else src,
                     "source_hash": sh})
    for cid in [c for c in f.claim_id if c.startswith("B2.") and c.endswith(".component")]:
        r = f[f.claim_id == cid].iloc[0]
        rows.append({"claim_id": cid, "contrast": cid.split(".")[1].replace("C_PROV", "C_src (=C_PROV)"),
                     "cohort": "B2 F" if ".F." in cid else "B2 weighted E4", "estimand": r.estimand, "outcome": r.outcome,
                     "unit": r.unit, "sign_definition": r.contrast_definition_and_sign, "estimate": r.estimate,
                     "ci_status": "no interval (6 day-vehicle components; exact sign-flip min p 0.03125)", "ci_drawn": False,
                     "ci_method": "none", "cluster_axis": "day-vehicle connected component (6)", "p_raw": r.p_raw,
                     "p_adjusted": r.p_adjusted, "family": r.family, "alpha": r.alpha, "n_units": r.n_units,
                     "n_scenes": r.n_scenes, "n_drives": r.n_drives, "n_components": r.n_components, "weighting": r.weighting,
                     "row_kind": "component-sensitivity (existing)", "source_ref": V.rel(r.source_file) if str(r.source_file).startswith("/") else r.source_file,
                     "source_hash": r.source_hash})
    E = pd.DataFrame(rows)
    E.to_csv(f"{OUT}/EFFECTS_FOR_PLOTTING.csv", index=False)
    # scene / drive tables (new descriptive, unit-level): B2 n_old = 1, contrasts C_GT / C_NAT / INT / C_PROV, severe & fde5
    uc = pd.read_parquet(f"{V.B2}/p1c_analysis/unit_contrasts.parquet")
    uc = uc[(uc.n_old == 1) & uc.valid]
    key = ["unit_id", "scene", "stratum", "weight", "log", "drive", "day_vehicle", "n_old", "anchor", "t_star"]
    a = uc[uc.contrast == "C_GT"].set_index(key)[["severe", "fde5"]]
    b = uc[uc.contrast == "C_NAT"].set_index(key)[["severe", "fde5"]]
    it = (a - b).dropna().reset_index().assign(contrast="INT")
    uc = pd.concat([uc[uc.contrast.isin(["C_GT", "C_NAT", "C_PROV"])], it], ignore_index=True)
    sc = uc.groupby(["contrast", "stratum", "scene", "drive", "day_vehicle", "log"]).agg(
        weight=("weight", "first"), n_units=("unit_id", "size"), severe_scene_mean=("severe", "mean"),
        fde5_scene_mean=("fde5", "mean")).reset_index()
    sc.to_csv(f"{OUT}/tables/V3_B2_scene_effects.csv", index=False)
    dr = []
    for (con, st, drv), d in sc.groupby(["contrast", "stratum", "drive"]):
        dr.append({"contrast": con, "stratum": st, "drive": drv, "n_scenes": len(d), "n_units": int(d.n_units.sum()),
                   "sum_weight": d.weight.sum(), "severe_wmean_pp": 100 * np.average(d.severe_scene_mean, weights=d.weight),
                   "fde5_wmean_m": np.average(d.fde5_scene_mean, weights=d.weight)})
    D = pd.DataFrame(dr)
    D["note"] = "scene weights kept inside a drive; drives share days/vehicles - display of heterogeneity, not independent samples"
    D.to_csv(f"{OUT}/tables/V3_B2_drive_summary.csv", index=False)
    # check: weighted scene means reproduce MATCHED_PREFIX_RESULTS theta (F scene-weighted = equal weights within F)
    mp = pd.read_csv(f"{V.B2}/p1c_analysis/MATCHED_PREFIX_RESULTS.csv")
    chk = []
    for con in ("C_GT", "C_NAT", "INT", "C_PROV"):
        for st, sub in (("F", sc[(sc.contrast == con) & (sc.stratum == "F")]), ("P", sc[sc.contrast == con])):
            mine = np.average(sub.severe_scene_mean, weights=sub.weight)
            ref = mp[(mp.n_old == 1) & (mp.metric == "severe") & (mp.contrast == con) & (mp.subset == "all") &
                     (mp.estimand == st) & (mp.cluster == "drive")]
            chk.append({"contrast": con, "estimand": st, "this_package": mine, "stored_theta": float(ref.theta.iloc[0]),
                        "abs_diff": abs(mine - float(ref.theta.iloc[0]))})
    pd.DataFrame(chk).to_csv(f"{OUT}/tables/V3_reproduction_check.csv", index=False)
    print(pd.DataFrame(chk).to_string())
    fig_effects(E, sc, D)


def fig_effects(E, sc, D):
    plt = V.mpl_setup()
    fig = plt.figure(figsize=(6.9, 3.4))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.25, 1.0], wspace=0.55)
    ax = fig.add_subplot(gs[0])
    main = E[E.row_kind == "re-visualised existing aggregate"].copy()
    order = [("RN", "dev A− (failure-selected)"), ("RN", "P1-C F"), ("RN", "P1-C weighted E"),
             ("RV", "dev A− (failure-selected)"), ("RV", "P1-C F"), ("RV", "P1-C weighted E"),
             ("C_GT", "B2 F"), ("C_GT", "B2 weighted E4"), ("C_NAT", "B2 F"), ("C_NAT", "B2 weighted E4"),
             ("INT", "B2 F"), ("INT", "B2 weighted E4"), ("C_src (=C_PROV)", "B2 F"), ("C_src (=C_PROV)", "B2 weighted E4")]
    cohort_style = {"dev A− (failure-selected)": (V.PAL[6], "D"), "P1-C F": (V.PAL[0], "o"), "B2 F": (V.PAL[0], "o"),
                    "P1-C weighted E": (V.PAL[1], "s"), "B2 weighted E4": (V.PAL[1], "s")}
    ylab = []
    for i, (con, coh) in enumerate(order):
        r = main[(main.contrast == con) & (main.cohort == coh)].iloc[0]
        col, mk = cohort_style[coh]
        if r.ci_drawn and np.isfinite(r.ci_low):
            ax.plot([r.ci_low, r.ci_high], [i, i], color=col, lw=1.4, solid_capstyle="round")
        ax.plot([r.estimate], [i], marker=mk, color=col, ms=5, mec="white", mew=0.6, ls="none")
        tag = "" if r.ci_drawn else "  (interval withheld)" if "WITHHOLD" in str(r.ci_status) or "finite" in str(r.ci_status) else "  (no interval)"
        ylab.append(f"{con} · {coh}{tag}")
    ax.axvline(0, color=V.INK2, lw=0.6)
    ax.set_yticks(range(len(order))); ax.set_yticklabels(ylab, fontsize=5.6); ax.invert_yaxis()
    ax.set_xlabel("effect on severe rate (pp; sign as defined per contrast)"); ax.grid(True, axis="x")
    ax.set_title("Aggregates (existing); bars = approximate cluster-t 95%", loc="left", color=V.INK, fontsize=6.5)
    from matplotlib.lines import Line2D
    ax.legend([Line2D([], [], marker="D", color=V.PAL[6], ls="none"), Line2D([], [], marker="o", color=V.PAL[0], ls="none"),
               Line2D([], [], marker="s", color=V.PAL[1], ls="none")],
              ["dev failure-selected", "failure stratum F", "weighted eligible frame E/E4"], frameon=False, fontsize=5.6,
              loc="upper center", bbox_to_anchor=(0.45, -0.13), ncol=3)
    ax2 = fig.add_subplot(gs[1])
    s = sc[(sc.contrast == "C_NAT")].copy()
    rng = np.random.default_rng(0)
    for xi, st, col in ((0, "F", V.PAL[0]), (1, "S", V.PAL[1])):
        d = s[s.stratum == st]
        ax2.scatter(xi + rng.uniform(-0.18, 0.18, len(d)), 100 * d.severe_scene_mean, s=6 + 4 * d.weight / d.weight.max(),
                    color=col, alpha=0.55, edgecolors="none")
        dd = D[(D.contrast == "C_NAT") & (D.stratum == st)]
        ax2.scatter(np.full(len(dd), xi + 0.36), dd.severe_wmean_pp, marker="_", s=40, color=V.INK2, linewidths=1.0)
        est = np.average(d.severe_scene_mean, weights=d.weight) * 100
        ax2.plot([xi - 0.25, xi + 0.25], [est, est], color=V.INK, lw=1.4)
    ax2.axhline(0, color=V.INK2, lw=0.6)
    ax2.set_xticks([0, 1]); ax2.set_xticklabels(["F (100 scenes)", "S (212 scenes; Hajek-weighted\nin E4)"], fontsize=5.8)
    ax2.set_ylabel("C_NAT scene mean (pp)"); ax2.grid(True, axis="y")
    ax2.set_title("B2 C_NAT per scene (dots) and per drive (ticks)", loc="left", color=V.INK, fontsize=6.5)
    fig.text(0.01, -0.12, "Black bar = stratum estimate (scene-weighted). Drive ticks keep scene weights; drives share days/vehicles "
             "and are not independent. Day–vehicle component sensitivity (6 components): min p 0.03125, Holm 0.125.",
             fontsize=5.6, color=V.INK2)
    V.save(fig, "V3_effects_composite")
    plt.close(fig)


if __name__ == "__main__":
    schedule()
    effects()
