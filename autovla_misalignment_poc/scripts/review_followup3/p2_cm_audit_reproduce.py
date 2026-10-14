#!/usr/bin/env python
"""
P2 steps 1-2 (review_followup3, 2026-10-14; POST-HOC, CPU only, no model inference).

Step 1  exact kinematics of the existing 'repeat-p constant-motion' (CM) baseline, verified numerically on the STORED
        decoder output (records.jsonl baselines.CM.traj_xyh, produced by the real ActionTokenizer on GPU in P1-A):
          - body-frame SE(2) increments of every CM suffix segment vs the codebook increment (disp[p], dyaw[p]);
          - global-frame displacement repeat test (would hold only for a zero-turn global velocity repeat).
Step 2  exact reproduction of C1 (review_followup2/C_baseline_window/run_20261013): re-runs c_baseline_window.c1_units /
        c1_tables in-process (same seeds) and compares every numeric cell with the stored CSV / parquet.
Outputs -> outputs/review_followup3/P2/{cm_definition_check.csv, cm_definition_summary.json, c1_reproduction_check.json}
Run:  nice -n 19 /root/miniforge3/envs/autovla/bin/python p2_cm_audit_reproduce.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
SCR = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(SCR, "review_followup2"))
sys.path.insert(0, os.path.join(SCR, "review_followup", "p1ab"))
import p1_common as C  # noqa: E402
import p2_kinematics as K  # noqa: E402

POC = "/root/VLA/autovla_misalignment_poc"
P1A = f"{POC}/outputs/review_followup/P1A_gt_free_reference/run_20261008"
C1DIR = f"{POC}/outputs/review_followup2/C_baseline_window/run_20261013"
OUT = f"{POC}/outputs/review_followup3/P2"


def step1():
    cb, disp, dyaw = C.codebook_geometry()
    rows = []
    for src, path in (("primary", f"{P1A}/raw/records.jsonl"), ("fixedpos", f"{P1A}/raw_fixedpos/records.jsonl")):
        for line in open(path):
            r = json.loads(line)
            t = r["t_p"]; ref = r["ref_tokens"]; p = r["perturbation_info"]["token_p"]
            if p == ref[t]:
                continue
            cm = np.asarray(r["baselines"]["CM"]["traj_xyh"], float)           # real decoder output (rounded 1e-6)
            inc = K.body_increments(cm)
            sfx = inc[t:]                                                    # segments t..9 all use token p
            body_err = float(np.abs(sfx[:, :2] - disp[p]).max())
            yaw_err = float(np.abs(K.wrap(sfx[:, 2] - dyaw[p])).max())
            gdisp = np.diff(np.vstack([np.zeros(2) if t == 0 else cm[t - 1, :2], cm[t:, :2]]), axis=0)
            g_rep_dev = float(np.abs(gdisp - gdisp[0]).max()) if len(gdisp) > 1 else 0.0   # 0 iff global velocity repeat
            # numpy SE(2) chaining (p2_kinematics.compose) vs stored decoder output: round trip of the transform
            comp = K.tokens_to_poses(disp, dyaw, r["baselines"]["CM"]["action_idx"])
            rt = float(np.abs(np.c_[comp[:, :2] - cm[:, :2], K.wrap(comp[:, 2] - cm[:, 2])]).max())
            rows.append({"source": src, "family": r["family"], "group": r["group"], "unit_id": r["unit_id"], "t_p": t,
                         "token_p": p, "dyaw_p_rad": float(dyaw[p]), "abs_dyaw_p_deg": abs(float(np.degrees(dyaw[p]))),
                         "disp_p_fwd": float(disp[p][0]), "disp_p_left": float(disp[p][1]),
                         "chord_angle_minus_half_dyaw_deg": float(np.degrees(np.arctan2(disp[p][1], disp[p][0]) - dyaw[p] / 2)),
                         "body_increment_maxerr_m": body_err, "body_yaw_increment_maxerr_rad": yaw_err,
                         "global_displacement_repeat_maxdev_m": g_rep_dev, "compose_vs_decoder_maxerr": rt})
    D = pd.DataFrame(rows)
    D.to_csv(f"{OUT}/cm_definition_check.csv", index=False)
    turn = D.abs_dyaw_p_deg > 0.5
    summ = {
        "n_units": int(len(D)),
        "body_frame_SE2_repeat_max_err_m": float(D.body_increment_maxerr_m.max()),
        "body_frame_yaw_increment_max_err_rad": float(D.body_yaw_increment_maxerr_rad.max()),
        "compose_vs_stored_decoder_max_err": float(D.compose_vs_decoder_maxerr.max()),
        "share_units_|dyaw_p|>0.5deg": float(turn.mean()),
        "median_|dyaw_p|_deg": float(D.abs_dyaw_p_deg.median()),
        "p90_|dyaw_p|_deg": float(D.abs_dyaw_p_deg.quantile(.9)),
        "global_disp_repeat_maxdev_m_median_over_turning_units": float(D.loc[turn, "global_displacement_repeat_maxdev_m"].median()),
        "global_disp_repeat_maxdev_m_max": float(D.global_displacement_repeat_maxdev_m.max()),
        "median_|chord_angle - dyaw/2|_deg": float(D.chord_angle_minus_half_dyaw_deg.abs().median()),
        "p90_|chord_angle - dyaw/2|_deg": float(D.chord_angle_minus_half_dyaw_deg.abs().quantile(.9)),
        "verdict": "CM = repeat of the perturbed token's body-frame SE(2) increment (translation in the current heading frame "
                   "+ heading change dyaw[p]) -> constant body twist = discrete constant turn-rate & constant speed "
                   "(incl. the token's lateral/chord-angle component). NOT a global-velocity repeat.",
    }
    json.dump(summ, open(f"{OUT}/cm_definition_summary.json", "w"), indent=1)
    print(json.dumps(summ, indent=1))


def step2():
    import c_baseline_window as CB
    U, CK = CB.c1_units()
    T = CB.c1_tables(U)
    T0 = pd.read_csv(f"{C1DIR}/c1_excess_effects.csv")
    U0 = pd.read_parquet(f"{C1DIR}/c1_units_counterfactual.parquet")
    res = {"units_rows_equal": len(U) == len(U0), "table_rows_equal": len(T) == len(T0)}
    num = [c for c in U0.columns if pd.api.types.is_numeric_dtype(U0[c])]
    res["units_numeric_maxabs_diff"] = float(np.nanmax(np.abs(U[num].values.astype(float) - U0[num].values.astype(float))))
    res["units_id_order_equal"] = bool((U.unit_id.values == U0.unit_id.values).all())
    T = T.reset_index(drop=True)
    numt = [c for c in T0.columns if pd.api.types.is_numeric_dtype(T0[c]) and c in T.columns]
    diffs = {}
    for c in numt:
        a = pd.to_numeric(T[c], errors="coerce").values.astype(float); b = T0[c].values.astype(float)
        both_nan = np.isnan(a) & np.isnan(b)
        d = np.where(both_nan, 0.0, np.abs(a - b))
        diffs[c] = float(np.nanmax(d)) if len(d) else 0.0
    res["table_numeric_maxabs_diff_by_column"] = diffs
    res["table_numeric_maxabs_diff"] = max(diffs.values())
    res["key_cells"] = {}
    for grp in ("A-", "A+", "all"):
        for m in ("ex_geo_end", "ex_cm_end", "ex_cm_sfx", "ex_cm_dref"):
            x = T[(T.family == "G") & (T.group == grp) & (T.metric == m)].iloc[0]
            res["key_cells"][f"G|{grp}|{m}"] = {k: float(x[k]) for k in ("est_scene", "ci95_lo", "ci95_hi", "p_signflip")}
    res["reproduced_exactly"] = bool(res["units_rows_equal"] and res["table_rows_equal"] and res["table_numeric_maxabs_diff"] < 1e-9
                                     and res["units_numeric_maxabs_diff"] < 1e-9)
    json.dump(res, open(f"{OUT}/c1_reproduction_check.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "table_numeric_maxabs_diff_by_column"}, indent=1))


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    step1()
    step2()
