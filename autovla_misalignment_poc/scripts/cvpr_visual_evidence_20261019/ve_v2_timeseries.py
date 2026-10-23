#!/usr/bin/env python
"""V2 (post-hoc, descriptive): time-resolved response of AutoVLA vs motion baselines, P1-A family G.
D_model(k) = ||OC(k) - U(k)||; D_b(k) = ||P_b(k) - U_b(k)||; E_b = D_model - D_b. Unit -> scene mean -> equal scene weight.
Settings S0/SE/SF from review_followup5/B_NUMERIC/run (model rows only; baselines need no inference).
Run: nice -n 19 /root/miniforge3/envs/autovla/bin/python ve_v2_timeseries.py
"""
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
SCR = os.path.dirname(HERE)
sys.path.insert(0, HERE)
for d in ("review_followup3", "review_followup2", os.path.join("review_followup", "p1ab")):
    sys.path.insert(0, os.path.join(SCR, d))
import ve_common as V  # noqa: E402
import p1_common as C  # noqa: E402
import p2_kinematics as K  # noqa: E402

OUT = V.OUT
BASES = ["Geo", "CM", "CV0", "CTRV-K3", "CTRA-2"]


def drive_of(log):
    p = log.split("_"); return f"{p[0]}_{p[1]}"


def main():
    cb, disp, dyaw = C.codebook_geometry()
    bn = {}
    for line in open(f"{V.BNUM}/run/records.jsonl"):
        x = json.loads(line)
        if x["row"] in ("U", "OC"):
            bn[(x["setting"], x["unit_id"], x["row"])] = np.asarray(x["poses"], float)
    ts, units = [], []
    for line in open(f"{V.P1A}/raw/records.jsonl"):
        r = json.loads(line)
        if r["family"] != "G":
            continue
        t = int(r["t_p"]); ref = r["ref_tokens"]; p = r["perturbation_info"]["token_p"]
        if p == ref[t]:
            continue
        reft = np.asarray(r["ref_traj_xyh"], float)
        gt = np.asarray(r["trajectory_gt"], float)
        log = r["log"]; parts = log.split("_")
        meta = {"cohort": "dev mechanism set (equal-distance, 208 scenes)", "stratum": r["group"], "scene_id": r["token"],
                "unit_id": r["unit_id"], "perturbation_id": r["perturbation"], "log_id": log, "drive_id": drive_of(log),
                "day_id": parts[0][:10], "vehicle_id": parts[1], "anchor_step": t}
        pre_p = K.tokens_to_poses(disp, dyaw, list(ref[:t]) + [p])
        pre_0 = K.tokens_to_poses(disp, dyaw, list(ref[:t]) + [ref[t]])
        pairs = {"Geo": (np.asarray(r["baselines"]["Geo"]["traj_xyh"], float), reft,
                         r["baselines"]["Geo"]["action_idx"], ref),
                 "CM": (K.bl_cm(pre_p, t), K.bl_cm(pre_0, t), None, None),
                 "CV0": (K.bl_cv0(pre_p, t), K.bl_cv0(pre_0, t), None, None),
                 "CTRV-K3": (K.bl_ctrv(pre_p, t)[0], K.bl_ctrv(pre_0, t)[0], None, None),
                 "CTRA-2": (K.bl_ctra(pre_p, t)[0], K.bl_ctra(pre_0, t)[0], None, None)}
        rows = {"S0": (np.asarray(r["rows"]["OC"]["traj_xyh"], float), np.asarray(r["rows"]["U"]["traj_xyh"], float),
                       r["rows"]["OC"]["action_idx"], r["rows"]["U"]["action_idx"])}
        for s in ("SE", "SF"):
            if (s, r["unit_id"], "OC") in bn:
                rows[s] = (bn[(s, r["unit_id"], "OC")], bn[(s, r["unit_id"], "U")], None, None)
        s0b = bn.get(("S0", r["unit_id"], "OC"))
        u = dict(meta)
        u["S0_bnumeric_equal_P1A"] = bool(s0b is not None and np.array_equal(s0b, rows["S0"][0])
                                          and np.array_equal(bn[("S0", r["unit_id"], "U")], rows["S0"][1]))
        for s, (P, U, aP, aU) in rows.items():
            d = np.linalg.norm(P[:, :2] - U[:, :2], axis=1)
            u[f"D_model_{s}_k9"] = d[9]
            for k in range(10):
                for arm, X, tok in (("P_OC", P, aP), ("U", U, aU)):
                    ts.append({**meta, "setting": s, "arm": arm, "baseline": "AutoVLA", "absolute_step": k,
                               "step_after_anchor": k - t, "horizon_s": 0.5 * (k + 1), "x_m": X[k, 0], "y_m": X[k, 1],
                               "yaw_rad": X[k, 2], "token_id": (tok[k] if tok is not None else np.nan),
                               "gt_x_m": gt[k, 0], "gt_y_m": gt[k, 1],
                               "source_ref": "P1A raw/records.jsonl" if s == "S0" else "review_followup5 B_NUMERIC run/records.jsonl"})
        for b, (Pb, Ub, aPb, aUb) in pairs.items():
            db = np.linalg.norm(Pb[:, :2] - Ub[:, :2], axis=1)
            u[f"D_{b}_k9"] = db[9]
            for k in range(10):
                for arm, X, tok in (("P_b", Pb, aPb), ("U_b", Ub, aUb)):
                    ts.append({**meta, "setting": "none (no inference)", "arm": arm, "baseline": b, "absolute_step": k,
                               "step_after_anchor": k - t, "horizon_s": 0.5 * (k + 1), "x_m": X[k, 0], "y_m": X[k, 1],
                               "yaw_rad": X[k, 2], "token_id": (tok[k] if tok is not None else np.nan),
                               "gt_x_m": gt[k, 0], "gt_y_m": gt[k, 1],
                               "source_ref": "computed from P1A tokens (p2_kinematics / stored Geo row)"})
        units.append(u)
    T = pd.DataFrame(ts)
    T["component_id"] = ""
    T["scene_weight"] = 1.0
    T["missing_note"] = ("component_id empty: day-vehicle components were defined for P1-C/B2, not the dev set; "
                         "scene_weight = 1 (no sampling design); token_id empty for SE/SF rows (B_NUMERIC stores poses+own tokens "
                         "only per row) and for kinematic baselines (not token-valued)")
    cols = ["cohort", "stratum", "scene_id", "unit_id", "perturbation_id", "drive_id", "day_id", "vehicle_id", "component_id",
            "scene_weight", "setting", "arm", "baseline", "absolute_step", "anchor_step", "step_after_anchor", "horizon_s",
            "x_m", "y_m", "yaw_rad", "token_id", "gt_x_m", "gt_y_m", "source_ref", "log_id", "missing_note"]
    T = T[cols]
    T.to_parquet(f"{OUT}/AUTOVLA_TIME_SERIES.parquet", index=False)
    T.drop(columns=["missing_note"]).to_csv(f"{OUT}/AUTOVLA_TIME_SERIES.csv.gz", index=False, compression="gzip")
    Udf = pd.DataFrame(units)
    curves(T, Udf)


def wide(T, setting):
    m = T[(T.baseline == "AutoVLA") & (T.setting == setting)]
    P = m[m.arm == "P_OC"].set_index(["unit_id", "absolute_step"])
    U = m[m.arm == "U"].set_index(["unit_id", "absolute_step"])
    W = P[["stratum", "scene_id", "drive_id", "anchor_step", "step_after_anchor", "horizon_s"]].copy()
    W["D_model"] = np.hypot(P.x_m - U.x_m, P.y_m - U.y_m)
    W["err_P_gt"] = np.hypot(P.x_m - P.gt_x_m, P.y_m - P.gt_y_m)
    W["err_U_gt"] = np.hypot(U.x_m - U.gt_x_m, U.y_m - U.gt_y_m)
    for b in BASES:
        bb = T[(T.baseline == b)]
        Pb = bb[bb.arm == "P_b"].set_index(["unit_id", "absolute_step"])
        Ub = bb[bb.arm == "U_b"].set_index(["unit_id", "absolute_step"])
        W[f"D_{b}"] = np.hypot(Pb.x_m - Ub.x_m, Pb.y_m - Ub.y_m)
        W[f"E_{b}"] = W.D_model - W[f"D_{b}"]
    return W.reset_index()


def agg(W, keys, val_cols):
    sc = W.groupby(["stratum_sel", "scene_id"] + keys)[val_cols].mean().reset_index()
    out = sc.groupby(["stratum_sel"] + keys)[val_cols].mean().reset_index()
    n = W.groupby(["stratum_sel"] + keys).agg(n_units=("unit_id", "nunique"), n_scenes=("scene_id", "nunique"),
                                              n_drives=("drive_id", "nunique")).reset_index()
    return out.merge(n, on=["stratum_sel"] + keys)


def curves(T, Udf):
    vals = ["D_model", "err_P_gt", "err_U_gt"] + [f"D_{b}" for b in BASES] + [f"E_{b}" for b in BASES]
    allrows, checks = [], {}
    Ws = {}
    for s in ("S0", "SE", "SF"):
        W = wide(T, s)
        W = pd.concat([W.assign(stratum_sel=W.stratum), W.assign(stratum_sel="pooled (A- + A+)")])
        Ws[s] = W
        a1 = agg(W[W.step_after_anchor >= 0], ["step_after_anchor"], vals).assign(axis="step_after_anchor", sample="all observable", setting=s)
        cs = W[(W.anchor_step <= 4) & (W.step_after_anchor.between(0, 5))]
        a2 = agg(cs, ["step_after_anchor"], vals).assign(axis="step_after_anchor", sample="common (t_p<=4, j=0..5)", setting=s)
        a3 = agg(W, ["absolute_step"], vals).assign(axis="absolute_step", sample="all (prefix rows exactly 0 by construction)", setting=s)
        allrows += [a1, a2, a3]
    S = pd.concat(allrows, ignore_index=True)
    S["estimator"] = "unit -> scene mean -> equal scene weight; descriptive, no interval"
    S.to_csv(f"{OUT}/TIME_SERIES_SUMMARY.csv", index=False)
    # ---- endpoint checks ----
    p2u = pd.read_parquet(f"{V.P2}/p2_units.parquet")
    p2u = p2u[(p2u.source == "primary") & (p2u.family == "G")].set_index("unit_id")
    W0 = Ws["S0"][(Ws["S0"].absolute_step == 9) & (Ws["S0"].stratum_sel != "pooled (A- + A+)")].set_index("unit_id")
    chk = {"n_units_G": int(len(W0)), "n_units_p2": int(len(p2u)), "S0_bnumeric_equal_P1A_units": int(Udf.S0_bnumeric_equal_P1A.sum())}
    chk["unit_D_model_k9_vs_p2_model_end_maxabs"] = float((W0.D_model - p2u.loc[W0.index, "model_end"]).abs().max())
    for b in BASES:
        chk[f"unit_D_{b}_k9_vs_p2_{b}_end_maxabs"] = float((W0[f"D_{b}"] - p2u.loc[W0.index, f"{b}_end"]).abs().max())
    mb = pd.read_csv(f"{V.P2}/MOTION_BASELINE_RESULTS.csv")
    bnr = pd.read_csv(f"{V.BNUM}/B_NUMERIC_RESULTS.csv")
    rows = []
    for g in ("A-", "A+"):
        for b in BASES:
            mine = S[(S.setting == "S0") & (S.axis == "absolute_step") & (S.absolute_step == 9) & (S.stratum_sel == g)][f"E_{b}"].iloc[0]
            ref = mb[(mb.row_type == "excess") & (mb.family == "G") & (mb.group == g) & (mb.metric == "end") & (mb.baseline == b)]
            refv = float(ref.est_scene.iloc[0]) if len(ref) else np.nan
            rows.append({"check": "E_b endpoint (k=9) vs review_followup3 P2 MOTION_BASELINE_RESULTS est_scene", "setting": "S0",
                         "stratum": g, "baseline": b, "this_package": mine, "stored": refv, "abs_diff": abs(mine - refv)})
            for s in ("S0", "SE", "SF"):
                mine_s = S[(S.setting == s) & (S.axis == "absolute_step") & (S.absolute_step == 9) & (S.stratum_sel == g)][f"E_{b}"].iloc[0]
                r2 = bnr[(bnr.setting == s) & (bnr.group == g) & (bnr.quantity == f"E_{b}")]
                v2 = float(r2.est_scene.iloc[0]) if len(r2) else np.nan
                rows.append({"check": "E_b endpoint (k=9) vs review_followup5 B_NUMERIC_RESULTS est_scene", "setting": s,
                             "stratum": g, "baseline": b, "this_package": mine_s, "stored": v2, "abs_diff": abs(mine_s - v2)})
    # deviation_curves.csv (P1-A analysis): OC dev vs reference by pose offset, A- G
    dc = pd.read_csv(f"{V.P1A}/deviation_curves.csv")
    dc = dc[(dc.family == "G") & (dc.condition == "OC")]
    for _, rr in dc.iterrows():
        g = rr.group
        mine = S[(S.setting == "S0") & (S.axis == "step_after_anchor") & (S["sample"] == "all observable") &
                 (S.stratum_sel == g) & (S.step_after_anchor == rr.pose_offset_from_tp)]
        if len(mine):
            rows.append({"check": "D_model(j) vs P1-A deviation_curves.csv (OC, scene_mean_dev_ref)", "setting": "S0", "stratum": g,
                         "baseline": f"j={int(rr.pose_offset_from_tp)}", "this_package": float(mine.D_model.iloc[0]),
                         "stored": float(rr.scene_mean_dev_ref), "abs_diff": abs(float(mine.D_model.iloc[0]) - float(rr.scene_mean_dev_ref)),
                         "n_scenes_this": int(mine.n_scenes.iloc[0]), "n_scenes_stored": int(rr.n_scenes)})
    E = pd.DataFrame(rows)
    E.to_csv(f"{OUT}/tables/V2_endpoint_checks.csv", index=False)
    chk["max_abs_diff_by_check"] = E.groupby("check").abs_diff.max().to_dict()
    json.dump(chk, open(f"{OUT}/tables/V2_checks.json", "w"), indent=1, default=float)
    print(json.dumps(chk, indent=1, default=float))
    figure(S)


def figure(S):
    plt = V.mpl_setup()
    sel = S[(S.setting == "S0") & (S.axis == "step_after_anchor")]
    strata = [("A-", "A− (failure-selected, 52 scenes)"), ("pooled (A- + A+)", "Selected mechanism set (208 scenes)")]
    fig, axs = plt.subplots(3, 2, figsize=(6.9, 6.0), sharex=True)
    show = [("CM", V.PAL[1], "s", "constant motion (CM)"), ("Geo", V.PAL[2], "^", "coordinate accumulation (Geo, privileged)"),
            ("CTRA-2", V.PAL[4], "v", "CTRA-2")]
    for ci, (st, lab) in enumerate(strata):
        for sample, ls in (("all observable", "-"), ("common (t_p<=4, j=0..5)", (0, (2, 1.5)))):
            d = sel[(sel.stratum_sel == st) & (sel["sample"] == sample)].sort_values("step_after_anchor")
            x = d.step_after_anchor * 0.5
            a = axs[0, ci]
            a.plot(x, d.D_model, color=V.PAL[0], ls=ls, marker="o" if ls == "-" else None, ms=3,
                   label="AutoVLA D_model" if ls == "-" else "_nolegend_")
            for b, col, mk, bl in show:
                axs[1, ci].plot(x, d[f"D_{b}"], color=col, ls=ls, marker=mk if ls == "-" else None, ms=3,
                                label=bl if ls == "-" else "_nolegend_")
                axs[2, ci].plot(x, d[f"E_{b}"], color=col, ls=ls, marker=mk if ls == "-" else None, ms=3)
        d = sel[(sel.stratum_sel == st) & (sel["sample"] == "all observable")].sort_values("step_after_anchor")
        for _, rr in d.iterrows():
            if rr.step_after_anchor in (0, 3, 6, 9):
                axs[0, ci].annotate(f"n={int(rr.n_scenes)}", (rr.step_after_anchor * 0.5, rr.D_model), textcoords="offset points",
                                    xytext=(0, 5), fontsize=5.5, color=V.INK2, ha="center")
        axs[0, ci].set_title(lab, loc="left", color=V.INK)
        axs[2, ci].axhline(0, color=V.INK2, lw=0.6)
        axs[2, ci].set_xlabel("time after the perturbed token (s; 0.5 s per pose)")
        for r in range(3):
            axs[r, ci].grid(True)
    axs[0, 0].set_ylabel("D_model = ||P−U|| (m)")
    axs[1, 0].set_ylabel("D_b = ||P_b−U_b|| (m)")
    axs[2, 0].set_ylabel("E_b = D_model − D_b (m)")
    axs[0, 1].legend(frameon=False, loc="upper left")
    axs[0, 1].text(0.02, 0.62, "solid: all observable units (sample shrinks with time; n = scenes)\n"
                   "dashed: common sample (t_p ≤ 4, first 3 s)", transform=axs[0, 1].transAxes, fontsize=5.8, color=V.INK2)
    axs[1, 1].legend(frameon=False, loc="upper left", fontsize=5.8)
    fig.suptitle("P1-A family G (GT-free perturbation value; GT-informed position), setting S0, scene-mean points; "
                 "descriptive post-hoc, no intervals", fontsize=7, color=V.INK, x=0.01, ha="left")
    fig.tight_layout()
    V.save(fig, "V2_response_vs_baselines")
    plt.close(fig)
    # accuracy panel separately
    fig, axs = plt.subplots(1, 2, figsize=(6.9, 2.3), sharey=True)
    for ci, (st, lab) in enumerate(strata):
        d = sel[(sel.stratum_sel == st) & (sel["sample"] == "all observable")].sort_values("step_after_anchor")
        x = d.step_after_anchor * 0.5
        axs[ci].plot(x, d.err_U_gt, color=V.INK2, marker="o", ms=3, label="||U − GT|| (unperturbed)")
        axs[ci].plot(x, d.err_P_gt, color=V.PAL[0], marker="s", ms=3, label="||P − GT|| (perturbed, OC)")
        axs[ci].set_title(lab, loc="left", color=V.INK); axs[ci].grid(True)
        axs[ci].set_xlabel("time after the perturbed token (s)")
    axs[0].set_ylabel("distance to GT (m)")
    axs[0].legend(frameon=False)
    fig.tight_layout()
    V.save(fig, "V2_gt_error_separate")
    plt.close(fig)


if __name__ == "__main__":
    main()
