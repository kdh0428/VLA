#!/usr/bin/env python
"""
P0-C window / weight / denominator audit (review_followup3, 2026-10-14).  POST-HOC re-analysis of existing data:
CPU only, no model inference, nothing outside outputs/review_followup3/P0 is written.

R9 (recovery re-run of exp 9, AutoVLA, A-):
  1. per window length w: unit / scene / log / drive / day-vehicle / day / vehicle counts (Figure 2 caption n), and
     whether Normal, Full (= gt_history of the same run) and win_w are evaluated on the SAME unit cohort.
  2. arm weights: unit weight inside a scene (1 / n_units_s) and scene weight (1 / n_scenes) must be identical in all
     arms (they are, because every arm has every unit) -- asserted.
  3. recovery f_w = (P_N - P_w) / (P_N - P_Full):
       NEW (consistent, scene-weighted): per scene the paired unit means of N, w, Full first, scenes equally weighted,
       f = sum_s (mN_s - mw_s) / sum_s (mN_s - mF_s); drive-cluster bootstrap where numerator and denominator are
       re-computed from the SAME resample (ratio of resampled sums); denominator <= 0 / near 0 counted.
       Also: unit-weighted version of the same ratio, day-vehicle-cluster bootstrap, and the (wrong) mean of per-scene
       ratios with the number of scenes whose own denominator is 0 (shown to document why it is not used).
       OLD: exp 9 original summary.json (unit-weighted, log bootstrap) and review_followup2 c2_recovery_fractions.csv
       (scene-weighted, drive bootstrap) side by side, NOT overwritten.
  4. release horizon per w: free post-release generation steps = max(0, 8 - t* - w) (the last corrected context
     position is t*+w; generation steps t*+w+2..9 condition on own tokens only), mean / distribution and common-support
     counts (units with >= N free steps for every w = 1..4, N = 1, 2, 3); units for which win_w == gt_history by
     construction (w >= 8 - t*).
P1-B (A-, t* <= 2): counts recomputed from units.parquet (NOT inferred from R9) and the two position estimands kept
  separate: fixed TOTAL horizon (pose 9) vs fixed POST-RELEASE horizon (pose release + N), from review_followup2
  c2_position_length.csv (re-labelled, values unchanged).

Status: every number here is AutoVLA-derived => PROVISIONAL until the AutoVLA no-op parity (P1 Stage 0) is judged.
Run: nice -n 19 /root/miniforge3/envs/autovla/bin/python p0_c_window_weights.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

POC = "/root/VLA/autovla_misalignment_poc"
sys.path.insert(0, f"{POC}/scripts/review_followup2")
import stats_core as SC  # noqa: E402

RF = f"{POC}/outputs/review_followup"
R89 = f"{RF}/R8R9_raw_recovery/run_20261008/units.parquet"
P1B = f"{RF}/P1B_start_position/run_20261008/units.parquet"
EXP9 = f"{POC}/outputs/temporal_feedback_window/summary.json"
C2R = f"{POC}/outputs/review_followup2/C_baseline_window/run_20261013/c2_recovery_fractions.csv"
C2P = f"{POC}/outputs/review_followup2/C_baseline_window/run_20261013/c2_position_length.csv"
OUT = f"{POC}/outputs/review_followup3/P0"
SEED, B = 20261014, 10_000
NEAR0 = 0.05          # "near zero" denominator threshold (5 pp of amplification)
PROV = "PROVISIONAL (AutoVLA parity not judged)"


def ids(log):
    f = log.split("_")
    return f"{f[0]}_{f[1]}", f[0][:10], f[1], f"{f[0][:10]}_{f[1]}"


def add_ids(df):
    x = df.log_id.map(ids)
    df["drive"], df["day"], df["vehicle"], df["day_vehicle"] = zip(*x)
    return df


def counts(df):
    return {"n_units": df.unit_id.nunique(), "n_scenes": df.scene_id.nunique(), "n_logs": df.log_id.nunique(),
            "n_drives": df.drive.nunique(), "n_day_vehicles": df.day_vehicle.nunique(), "n_days": df.day.nunique(),
            "n_vehicles": df.vehicle.nunique()}


def recovery(W, c, full, weighting, cluster, seed):
    """f = sum_g N_g / sum_g D_g with N = mean(normal - c), D = mean(normal - full) per scene (scene weighting) or
    per-unit sums (unit weighting); joint cluster bootstrap of N and D."""
    x = W.copy()
    x["num"] = x["normal"] - x[c]
    x["den"] = x["normal"] - x[full]
    if weighting == "scene":
        s = x.groupby([cluster, "scene_id"])[["num", "den"]].mean().groupby(cluster).sum()
    else:
        s = x.groupby(cluster)[["num", "den"]].sum()
    N, D = s.num.values.astype(float), s.den.values.astype(float)
    norm = len(x.scene_id.unique()) if weighting == "scene" else len(x)
    f = N.sum() / D.sum()
    C = SC.boot_counts(len(N), B, seed)
    num_b, den_b = C @ N, C @ D
    with np.errstate(divide="ignore", invalid="ignore"):
        fb = num_b / den_b
    ok = den_b > 0
    return {"fraction": f, "ci95": SC.pct_ci(fb[ok & np.isfinite(fb)]), "G": len(N),
            "num": N.sum() / norm, "den": D.sum() / norm,
            "boot_den_min": float(den_b.min() / norm), "boot_den_le0": int((den_b <= 0).sum()),
            "boot_den_lt_near0": int((den_b / norm < NEAR0).sum()), "boot_share_f_ge_0.9": float(np.mean(fb[ok] >= 0.9)),
            "B": B, "seed": seed}


def main():
    os.makedirs(OUT, exist_ok=True)
    u = pd.read_parquet(R89)
    u = add_ids(u[(u.experiment == "R9") & (u.group == "A-")].copy())
    assert u.exclusion_reason.isna().all() and (u.parser_status == "ok").all()
    conds = ["normal", "gt_history"] + [f"win{w}" for w in range(1, 9)] + \
        ["delay2_len1", "delay3_len1", "delay4_len1", "delay2_len2", "delay3_len2"]
    W = u.pivot_table(index=["unit_id", "scene_id", "log_id", "drive", "day_vehicle"], columns="condition",
                      values="amplification", aggfunc="first").reset_index()
    tst = u.drop_duplicates("unit_id").set_index("unit_id").t_star
    W["t_star"] = W.unit_id.map(tst)
    rows, checks = [], {}
    base_units = set(u[u.condition == "normal"].unit_id)
    # ----------------------------------------------------------------------------------- 1/2 cohort, counts, weights
    for c in conds:
        q = u[u.condition == c]
        same = set(q.unit_id) == base_units == set(u[u.condition == "gt_history"].unit_id)
        nsc = q.groupby("scene_id").unit_id.nunique()
        wN = u[u.condition == "normal"].groupby("scene_id").unit_id.nunique()
        assert same and nsc.equals(wN.loc[nsc.index]), c
        r = {"row_id": f"P0C.R9.counts.{c}", "table": "R9 cohort / Figure 2 caption n", "run": "R9 (recovery re-run of exp 9)",
             "group": "A-", "condition": c, **counts(q), "same_unit_cohort_as_normal_and_full": same,
             "arm_weights": "unit 1/n_units_s within scene, scene 1/52; identical across N/Full/win (asserted)",
             "status": PROV, "source_path": R89}
        if c.startswith("win"):
            w = int(c[3:])
            post = np.maximum(0, 8 - W.t_star - w)
            r.update({"w": w, "mean_corrected_positions": float(np.minimum(w, 8 - W.t_star).mean()),
                      "n_units_identical_to_full_by_construction(w>=8-t*)": int((w >= 8 - W.t_star).sum()),
                      "post_release_free_steps_mean": float(post.mean()), "post_release_free_steps_median": float(np.median(post)),
                      "post_release_free_steps_distribution": json.dumps({int(k): int(v) for k, v in post.value_counts().sort_index().items()}),
                      "units_with_post_release>=1": int((post >= 1).sum()), "units_with_post_release>=2": int((post >= 2).sum()),
                      "units_with_post_release>=3": int((post >= 3).sum())})
        rows.append(r)
    for N_ in (1, 2, 3):                                  # common support across w = 1..4
        cs = W[W.t_star <= 4 - N_]
        rows.append({"row_id": f"P0C.R9.common_support.N{N_}", "table": "R9 release-horizon common support",
                     "run": "R9 (recovery re-run of exp 9)", "group": "A-",
                     "condition": f"units with >= {N_} free post-release steps for every w = 1..4 (t* <= {4 - N_})",
                     "n_units": len(cs), "n_scenes": cs.scene_id.nunique(), "n_logs": cs.log_id.nunique(),
                     "n_drives": cs.drive.nunique(), "n_day_vehicles": cs.day_vehicle.nunique(), "status": PROV, "source_path": R89})
    tdist_ = W.t_star.value_counts().sort_index()
    checks["t_star_distribution_units"] = {int(k): int(v) for k, v in tdist_.items()}
    # ----------------------------------------------------------------------------------- 3 recovery side by side
    e9 = json.load(open(EXP9))["subsets"]["all perturbations"]["A-"]["fraction"]["amplification"]
    c2 = pd.read_csv(C2R)
    c2 = c2[c2.run.str.startswith("R9") & c2.subset.str.startswith("all A-") & (c2.metric == "amplification")]
    rec = []
    seed = SEED
    for c in [f"win{w}" for w in range(1, 6)] + ["delay2_len1", "delay3_len1", "delay4_len1"]:
        base = {"condition": c, "metric": "amplification (A- and FDE5 > 3 m), pose 9 (total horizon)",
                "denominator_definition": "P_N - P_Full, Full = gt_history of the same R9 run, same 365 units"}
        if c in e9:
            rec.append(dict(base, row_id=f"P0C.R9.rec.{c}.OLD_exp9", estimator="OLD exp 9 original: unit-weighted ratio, log-cluster bootstrap (summary.json)",
                            fraction=e9[c]["mean"], ci95_lo=e9[c]["ci95"][0], ci95_hi=e9[c]["ci95"][1], cluster="log (16)",
                            status="original (exploratory exp 9); " + PROV, source_path=f"{EXP9} :: subsets/all perturbations/A-/fraction/amplification/{c}"))
        for wt in ("scene", "unit"):
            x = c2[(c2.condition == c) & (c2.weighting == wt)]
            if len(x):
                x = x.iloc[0]
                rec.append(dict(base, row_id=f"P0C.R9.rec.{c}.RF2_{wt}", estimator=f"review_followup2 C2: {wt}-weighted ratio of drive totals, drive bootstrap",
                                fraction=x.fraction, ci95_lo=x.ci95_lo, ci95_hi=x.ci95_hi, cluster=f"drive ({int(x.n_drives)})",
                                boot_share_f_ge_09=x["boot_share_f_ge_0.9"], status="post-hoc (review_followup2); " + PROV,
                                source_path=C2R))
        for wt, cl in (("scene", "drive"), ("unit", "drive"), ("scene", "day_vehicle")):
            r = recovery(W, c, "gt_history", wt, cl, seed); seed += 7
            rec.append(dict(base, row_id=f"P0C.R9.rec.{c}.NEW_{wt}_{cl}",
                            estimator=("NEW consistent: scene means first, scenes equal, ratio of sums" if wt == "scene"
                                       else "NEW unit-weighted ratio of sums") + f"; {cl}-cluster bootstrap, joint num/den resample",
                            fraction=r["fraction"], ci95_lo=r["ci95"][0], ci95_hi=r["ci95"][1], cluster=f"{cl} ({r['G']})",
                            P_N_minus_P_w=r["num"], P_N_minus_P_Full=r["den"], boot_den_min=r["boot_den_min"],
                            boot_den_le0=r["boot_den_le0"], boot_den_below_5pp=r["boot_den_lt_near0"],
                            boot_share_f_ge_09=r["boot_share_f_ge_0.9"], B=r["B"], seed=r["seed"],
                            status=("primary consistent estimator (post-hoc); " if (wt, cl) == ("scene", "drive") else "sensitivity (post-hoc); ") + PROV,
                            source_path=R89))
        # wrong estimator for documentation: mean of per-scene ratios
        g = W.groupby("scene_id")[["normal", c, "gt_history"]].mean()
        den = g.normal - g.gt_history
        okm = den.abs() > 1e-12
        rec.append(dict(base, row_id=f"P0C.R9.rec.{c}.mean_of_scene_ratios", estimator="NOT USED: mean over scenes of per-scene ratios (documentation only)",
                        fraction=float(((g.normal - g[c]) / den)[okm].mean()), cluster="-",
                        scenes_with_zero_denominator=int((~okm).sum()), scenes_with_negative_denominator=int((den < -1e-12).sum()),
                        status="invalid estimator shown for contrast; " + PROV, source_path=R89))
    recdf = pd.DataFrame(rec)
    # assert the NEW scene/drive estimator reproduces review_followup2's point estimate (same estimand)
    for c in ("win1", "win2", "win3", "win4"):
        a = recdf[recdf.row_id == f"P0C.R9.rec.{c}.NEW_scene_drive"].fraction.iloc[0]
        b = recdf[recdf.row_id == f"P0C.R9.rec.{c}.RF2_scene"].fraction.iloc[0]
        assert abs(a - b) < 1e-12, (c, a, b)
        a = recdf[recdf.row_id == f"P0C.R9.rec.{c}.NEW_unit_drive"].fraction.iloc[0]
        b = recdf[recdf.row_id == f"P0C.R9.rec.{c}.OLD_exp9"].fraction.iloc[0]
        assert abs(a - b) < 1e-12, (c, a, b)
    checks["NEW_scene_point_equals_RF2_scene"] = True
    checks["NEW_unit_point_equals_OLD_exp9"] = True
    # scene-level levels used in the ratio
    lv = W.groupby("scene_id")[["normal", "gt_history"] + [f"win{w}" for w in range(1, 5)]].mean().mean()
    checks["scene_weighted_levels"] = lv.to_dict()
    checks["unit_weighted_levels"] = W[["normal", "gt_history"] + [f"win{w}" for w in range(1, 5)]].mean().to_dict()
    # ----------------------------------------------------------------------------------- P1-B two estimands
    p = add_ids(pd.read_parquet(P1B))
    pa = p[(p.group == "A-") & (p.t_star <= 2)]
    pc = counts(pa)
    prow = [{"row_id": "P0C.P1B.counts.Aminus_tstar_le2", "table": "P1-B cohort (recomputed; not inferred from R9)",
             "run": "P1-B", "group": "A-, t* <= 2", "condition": "all P1-B conditions", **pc,
             "same_unit_cohort_as_normal_and_full": bool(pa.groupby("condition").unit_id.nunique().nunique() == 1),
             "status": PROV, "source_path": P1B}]
    pos = pd.read_csv(C2P)
    keep = pos[(pos.window_kind == "persistent") & (pos.contrast == "s1 - s3") &
               (((pos.analysis.str.startswith("total")) & (pos.metric == "amplification")) |
                ((pos.analysis.str.startswith("common post-release horizon N=1")) & (pos.metric.isin(["exceed3_diffindiff", "err_raw", "err_diffindiff"]))) |
                ((pos.analysis.str.startswith("common post-release horizon N=2")) & (pos.metric.isin(["exceed3_diffindiff", "err_raw"]))))]
    for _, r in keep.iterrows():
        est = "E_total: fixed TOTAL horizon (pose 9); post-release horizon differs early vs late" if r.analysis.startswith("total") \
            else "E_post: fixed POST-RELEASE horizon (pose release+N); absolute pose differs by 2 steps"
        prow.append({"row_id": f"P0C.P1B.{'Etotal' if r.analysis.startswith('total') else 'Epost'}.{r.analysis.split('N=')[1][0] if 'N=' in r.analysis else 'pose9'}.w{int(r.w)}.{r.metric}",
                     "table": "P1-B position estimands (values from review_followup2, unchanged)", "run": "P1-B", "group": "A-, t* <= 2",
                     "condition": f"persistent GT window w={int(r.w)}, start s1 - s3", "estimand": est, "metric": r.metric,
                     "eval_pose_early": r.eval_pose_early, "eval_pose_late": r.eval_pose_late,
                     "post_release_poses_early": r.post_release_poses_early, "post_release_poses_late": r.post_release_poses_late,
                     "n_units": r.n_units, "n_scenes": r.n_scenes, "n_logs": r.n_logs, "n_drives": r.n_drives,
                     "estimate": r.est_scene, "ci95_lo": r.ci95_lo, "ci95_hi": r.ci95_hi, "p_signflip": r.p_signflip,
                     "p_holm_m3": r.p_holm_m3, "tost_p_equiv_holm_m3": r.tost_p_equiv_holm_m3, "margin_status": r.margin_status,
                     "status": "predeclared estimand, drive re-analysis (post-hoc); " + PROV if r.analysis.startswith("total")
                     else "post-hoc estimand; " + PROV, "source_path": C2P})
    # ----------------------------------------------------------------------------------- write
    allrows = pd.concat([pd.DataFrame(rows), pd.DataFrame(prow), recdf], ignore_index=True)
    allrows.to_csv(f"{OUT}/WINDOW_WEIGHT_AUDIT.csv", index=False)
    json.dump(checks, open(f"{OUT}/_checks_C_window.json", "w"), indent=1, default=float)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    print(pd.DataFrame(rows)[["row_id", "n_units", "n_scenes", "n_logs", "n_drives", "n_day_vehicles", "n_days", "n_vehicles",
                              "post_release_free_steps_mean", "post_release_free_steps_distribution",
                              "n_units_identical_to_full_by_construction(w>=8-t*)"]].to_string())
    print(recdf[["row_id", "fraction", "ci95_lo", "ci95_hi", "cluster", "boot_den_min", "boot_den_le0", "boot_den_below_5pp",
                 "scenes_with_zero_denominator", "boot_share_f_ge_09"]].to_string())
    print(pd.DataFrame(prow)[["row_id", "n_units", "n_scenes", "n_drives", "n_day_vehicles", "estimate", "ci95_lo", "ci95_hi", "p_signflip"]].to_string())
    print(json.dumps(checks, indent=1, default=float))


if __name__ == "__main__":
    main()
