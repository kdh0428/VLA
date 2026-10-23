#!/usr/bin/env python
"""F (post_hoc/exploratory): emitted-change vs continuation decomposition on P1-A family G (S0).
y00 = U row, y10 = O1 row (perturbed emitted token + unchanged suffix; == Geo when U = reference), y11 = OC row.
L = FDE5 / ADE5 vs the stored continuous GT (m). Conditioning-only C1 row reported as a separate check.
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ra_common as R  # noqa: E402

D = f"{R.OUT}/decomposition"


def L(traj, gt):
    e = np.linalg.norm(np.asarray(traj, float)[:, :2] - np.asarray(gt, float)[:, :2], axis=1)
    return e[-1], e.mean()


def main():
    rows = []
    for line in open(f"{R.P1A}/raw/records.jsonl"):
        r = json.loads(line)
        if r["family"] != "G":
            continue
        t = int(r["t_p"]); pi = r["perturbation_info"]
        if pi["token_p"] == r["ref_tokens"][t]:
            continue
        gt = r["trajectory_gt"]
        Rw = r["rows"]
        y00, y10, y11, c1 = (np.asarray(Rw[k]["traj_xyh"]) for k in ("U", "O1", "OC", "C1"))
        a00, a10, a11, ac1 = (Rw[k]["action_idx"] for k in ("U", "O1", "OC", "C1"))
        f00, d00 = L(y00, gt); f10, d10 = L(y10, gt); f11, d11 = L(y11, gt); fc1, dc1 = L(c1, gt)
        geo = np.asarray(r["baselines"]["Geo"]["traj_xyh"])
        rows.append({"unit_id": r["unit_id"], "scene_id": r["token"], "log_id": r["log"], "drive_id": R.drive_of(r["log"]),
                     "stratum": r["group"], "t_p": t, "sector": pi["sector"], "d0_m": pi["achieved_m"],
                     "dyaw_rad": pi["dyaw_p_minus_ref_rad"], "logp_p_given_prefix": pi["logp_p_given_prefix"],
                     "logp_ref_given_prefix": pi["logp_ref_given_prefix"], "rank_p_given_prefix": pi["rank_p_given_prefix"],
                     "U_entropy_next": Rw["U"]["ent"][0] if Rw["U"]["ent"] else np.nan,
                     "OC_entropy_next": Rw["OC"]["ent"][0] if Rw["OC"]["ent"] else np.nan,
                     "emitted_prefix_equal": a10[:t] == a00[:t] and a11[:t] == a00[:t],
                     "emitted_token_changed_at_tp": a10[t] != a00[t] and a11[t] == a10[t],
                     "conditioning_only": False, "replay_valid": a10[t + 1:] == a00[t + 1:],
                     "O1_equals_Geo": bool(np.allclose(y10, geo, atol=1e-5)), "U_equals_reference": a00 == r["ref_tokens"],
                     "edit_set": "emitted token at t_p (output + context), suffix regenerated",
                     "loss_unit": "m (FDE5 / ADE5 vs continuous GT)", "source": "P1-A raw/records.jsonl rows U, O1, OC",
                     "loss_y00": f00, "loss_y10": f10, "loss_y11": f11,
                     "delta_emitted": f10 - f00, "delta_continuation": f11 - f10, "delta_total": f11 - f00,
                     "ade_y00": d00, "ade_y10": d10, "ade_y11": d11, "ade_delta_emitted": d10 - d00,
                     "ade_delta_continuation": d11 - d10, "ade_delta_total": d11 - d00,
                     "C1_conditioning_only_delta_fde": fc1 - f00, "C1_suffix_equals_OC_suffix": ac1[t + 1:] == a11[t + 1:]})
    U = pd.DataFrame(rows)
    U.to_csv(f"{D}/F_unit_decomposition.csv", index=False)
    chk = {"n_units": len(U), "emitted_prefix_equal": int(U.emitted_prefix_equal.sum()), "replay_valid": int(U.replay_valid.sum()),
           "emitted_token_changed": int(U.emitted_token_changed_at_tp.sum()), "O1_equals_Geo": int(U.O1_equals_Geo.sum()),
           "U_equals_reference": int(U.U_equals_reference.sum()), "C1_suffix_equals_OC": int(U.C1_suffix_equals_OC_suffix.sum()),
           "identity_maxabs": float((U.delta_emitted + U.delta_continuation - U.delta_total).abs().max())}
    # aggregates (unit -> scene mean -> equal scene weight)
    def agg(df):
        s = df.groupby("scene_id")[["delta_emitted", "delta_continuation", "delta_total", "ade_delta_emitted",
                                    "ade_delta_continuation", "ade_delta_total", "C1_conditioning_only_delta_fde", "loss_y00"]].mean()
        o = s.mean().to_dict()
        o.update({"n_units": len(df), "n_scenes": len(s), "n_drives": df.drive_id.nunique(),
                  "share_cont_gt_+0.5m": float((df.delta_continuation > 0.5).mean()),
                  "share_cont_lt_-0.5m": float((df.delta_continuation < -0.5).mean()),
                  "median_delta_continuation": float(df.delta_continuation.median()),
                  "q10_delta_continuation": float(df.delta_continuation.quantile(0.1)),
                  "q90_delta_continuation": float(df.delta_continuation.quantile(0.9))})
        return o
    A = []
    for st, df in (("A-", U[U.stratum == "A-"]), ("A+", U[U.stratum == "A+"]), ("pooled 208-scene selected set", U)):
        A.append({"stratum": st, "weighting": "equal scene weight (no sampling design)", **agg(df)})
    pd.DataFrame(A).to_csv(f"{D}/F_stratum_summary.csv", index=False)
    # pre-specified condition breakdowns
    U["d0_tercile"] = pd.qcut(U.d0_m, 3, labels=["low", "mid", "high"])
    U["ent_tercile"] = pd.qcut(U.U_entropy_next, 3, labels=["low", "mid", "high"])
    br = []
    for var in ("t_p", "sector", "d0_tercile", "ent_tercile"):
        for st in ("A-", "A+"):
            for lev, df in U[U.stratum == st].groupby(var, observed=True):
                o = agg(df); br.append({"variable": var, "level": str(lev), "stratum": st, **o})
    pd.DataFrame(br).to_csv(f"{D}/F_condition_breakdown.csv", index=False)
    # exploratory prediction of sign(delta_continuation > 0)
    num = ["t_p", "d0_m", "dyaw_rad", "logp_p_given_prefix", "logp_ref_given_prefix", "U_entropy_next", "OC_entropy_next"]
    X0 = pd.concat([U[num], pd.get_dummies(U.sector, prefix="sec", dtype=float)], axis=1)
    X1 = X0.assign(stratum_Aminus=(U.stratum == "A-").astype(float))
    y = (U.delta_continuation > 0).astype(int).values
    g = U.log_id.values
    pr = []
    for name, X in (("runtime features (no stratum)", X0), ("+ GT-defined stratum (diagnostic only)", X1)):
        p = np.zeros(len(y))
        for tr, te in GroupKFold(5).split(X, y, g):
            mdl = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0)).fit(X.iloc[tr], y[tr])
            p[te] = mdl.predict_proba(X.iloc[te])[:, 1]
        pr.append({"model": name, "n_units": len(y), "n_groups(logs)": len(set(g)), "base_rate_positive": y.mean(),
                   "auroc_groupcv": roc_auc_score(y, p), "balanced_acc_groupcv": balanced_accuracy_score(y, p > 0.5),
                   "status": "post_hoc/exploratory; GroupKFold(5) by log; not a final performance claim"})
    pd.DataFrame(pr).to_csv(f"{D}/F_prediction_groupcv.csv", index=False)
    U["pred_class"] = np.where(U.delta_continuation > 0.5, "amplification", np.where(U.delta_continuation < -0.5, "compensation", "small"))
    cases = []
    for c, df in U.groupby("pred_class"):
        med = df.delta_continuation.median()
        pick = df.assign(_d=(df.delta_continuation - med).abs()).sort_values(["_d", "unit_id"]).iloc[0]
        cases.append({"class": c, "n_candidates": len(df), "class_median": med, "unit_id": pick.unit_id, "stratum": pick.stratum,
                      "t_p": pick.t_p, "delta_emitted": pick.delta_emitted, "delta_continuation": pick.delta_continuation,
                      "delta_total": pick.delta_total})
    pd.DataFrame(cases).to_csv(f"{D}/F_representative_cases.csv", index=False)
    json.dump(chk, open(f"{D}/F_checks.json", "w"), indent=1)
    pd.set_option("display.width", 250)
    print(chk); print(pd.DataFrame(A).round(3).to_string()); print(pd.DataFrame(pr).to_string()); print(pd.DataFrame(cases).to_string())
    b = pd.DataFrame(br); print(b[b.variable == "t_p"][["level", "stratum", "n_units", "delta_emitted", "delta_continuation", "delta_total"]].round(3).to_string())
    figure(U, pd.DataFrame(br))


def figure(U, B):
    plt = R.mpl()
    fig, axs = plt.subplots(1, 3, figsize=(7.4, 2.6))
    for i, (st, col) in enumerate((("A+", R.PAL[1]), ("A-", R.PAL[0]))):
        d = U[U.stratum == st]
        axs[0].scatter(d.delta_emitted, d.delta_continuation, s=3, alpha=0.35, color=col, label=f"{st} (n={len(d)})", edgecolors="none")
    axs[0].axhline(0, color=R.INK2, lw=0.6); axs[0].axvline(0, color=R.INK2, lw=0.6)
    axs[0].set_xlabel("Δ_emitted (m)"); axs[0].set_ylabel("Δ_continuation (m)")
    axs[0].legend(frameon=False, markerscale=3); axs[0].set_title("Units (P1-A family G, S0)", loc="left")
    for j, st in enumerate(("A-", "A+")):
        b = B[(B.variable == "t_p") & (B.stratum == st)].copy(); b["lv"] = b.level.astype(int); b = b.sort_values("lv")
        ax = axs[1 + j]
        x = np.arange(len(b))
        ax.bar(x - 0.2, b.delta_emitted, 0.38, color=R.PAL[2], label="Δ_emitted")
        ax.bar(x + 0.2, b.delta_continuation, 0.38, color=R.PAL[6], label="Δ_continuation")
        ax.plot(x, b.delta_total, "o", color=R.INK, ms=3, label="Δ_total")
        ax.set_xticks(x); ax.set_xticklabels([f"{v}\n{n}" for v, n in zip(b.lv, b.n_units)], fontsize=5.2)
        ax.axhline(0, color=R.INK2, lw=0.6); ax.set_xlabel("perturbed position t_p (second line: n units)"); ax.grid(True, axis="y")
        ax.set_title(f"{st}: scene-mean ΔFDE5 by t_p", loc="left")
    axs[1].set_ylabel("m"); axs[2].legend(frameon=False, fontsize=5.5)
    fig.text(0.01, -0.08, "Δ_emitted = FDE(y10) − FDE(y00); Δ_continuation = FDE(y11) − FDE(y10). Ordered decomposition (identity, not a causal split). y00 = unperturbed, y10 = perturbed emitted token + original "
             "suffix (replay; = coordinate-accumulation control), y11 = regenerated suffix. Post-hoc, descriptive.", fontsize=5.6, color=R.INK2)
    fig.tight_layout()
    R.save(fig, "F_decomposition", sub="decomposition")


if __name__ == "__main__":
    main()
