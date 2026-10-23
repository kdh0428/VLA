#!/usr/bin/env python
"""V1 (post-hoc): AutoVLA driving cases, B2 F stratum, NAT_REF (treatment) vs NAT_NSRC (reference), n_old = 1.
Rule: outputs/cvpr_visual_evidence_20261019/CASE_SELECTION_RULE.md (section A). CPU only, inputs read only.
Run: nice -n 19 /root/miniforge3/envs/autovla/bin/python ve_v1_cases.py
"""
import json
import os
import pickle
import shutil
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ve_common as V  # noqa: E402

TREAT, REF = "NAT_REF", "NAT_NSRC"
OUT = V.OUT


def load():
    cells = {}
    for line in open(f"{V.B2}/p1c_main/cells.jsonl"):
        x = json.loads(line)
        if x["n_old"] != 1:
            continue
        cells[(x["unit_id"], x["cell"])] = x
    uc = pd.read_parquet(f"{V.B2}/p1c_analysis/unit_contrasts.parquet")
    uc = uc[(uc.n_old == 1) & (uc.contrast == "C_NAT")]
    rec = {}
    for line in open(f"{V.P1C}/mech/action_history_causal/records.jsonl"):
        x = json.loads(line)
        rec[x["token"]] = {"log": x["log"], "gt_xy": np.asarray(x["trajectory_gt"], float), "gt_tokens": x["gt"],
                           "group": x["group"], "t_star": int(x["t_star"])}
    return cells, uc, rec


def frames_for(scene, log, cache):
    p = f"{V.NUPLAN}/navsim_logs/test/{log}.pkl"
    if log not in cache:
        cache[log] = {f["token"]: f for f in pickle.load(open(p, "rb"))} if os.path.exists(p) else {}
    fr = cache[log].get(scene)
    if fr is None:
        return None
    out = {"timestamp_us": int(fr["timestamp"]), "log_pkl": p}
    for cam in ("CAM_F0", "CAM_L0", "CAM_R0"):
        c = fr["cams"][cam]
        jp = f"{V.NUPLAN}/sensor_blobs/test/{c['data_path']}"
        out[cam] = {"path": jp, "exists": os.path.exists(jp), "data_path": c["data_path"],
                    "cam_intrinsic": np.asarray(c["cam_intrinsic"]).tolist(),
                    "sensor2lidar_rotation": np.asarray(c["sensor2lidar_rotation"]).tolist(),
                    "sensor2lidar_translation": np.asarray(c["sensor2lidar_translation"]).tolist(),
                    "distortion": np.asarray(c["distortion"]).tolist()}
    return out


def keep_set():
    s = set()
    d = f"{V.P1C}/selection"
    for f in os.listdir(d):
        if f.endswith(".keep_files.json"):
            s.update(json.load(open(f"{d}/{f}")))
    return s


def pick_closest(df, col, idcol):
    med = float(df[col].median())
    return df.assign(_dist=(df[col] - med).abs()).sort_values(["_dist", idcol]), med


def main():
    cells, uc, rec = load()
    keep = keep_set()
    cache = {}
    rows = []
    checks = {"units": 0, "prefix_tokens_equal": 0, "prefix_sha_equal": 0, "prefix_poses_equal": 0,
              "fde_recomputed_maxabs": 0.0, "contrast_matches_parquet_maxabs": 0.0}
    for _, u in uc.iterrows():
        t, r = cells.get((u.unit_id, TREAT)), cells.get((u.unit_id, REF))
        if t is None or r is None:
            continue
        a = int(t["anchor"])
        g = rec[u.scene]
        checks["units"] += 1
        checks["prefix_tokens_equal"] += int(t["acts"][:a] == r["acts"][:a])
        checks["prefix_sha_equal"] += int(t["prefix_ids_sha"] == r["prefix_ids_sha"])
        checks["prefix_poses_equal"] += int(np.allclose(np.asarray(t["poses"])[:a], np.asarray(r["poses"])[:a], atol=0))
        for c in (t, r):
            f = float(np.linalg.norm(np.asarray(c["poses"])[9, :2] - g["gt_xy"][9]))
            checks["fde_recomputed_maxabs"] = max(checks["fde_recomputed_maxabs"], abs(f - c["fde5"]))
        d = t["fde5"] - r["fde5"]
        checks["contrast_matches_parquet_maxabs"] = max(checks["contrast_matches_parquet_maxabs"],
                                                        abs(-d - float(u.fde5)) if u.valid else 0.0)
        fr = frames_for(u.scene, u.log, cache)
        has_img = bool(fr and fr["CAM_F0"]["exists"])
        rows.append({"unit_id": u.unit_id, "scene_id": u.scene, "perturbation_id": u.unit_id.split("|")[1],
                     "stratum": u.stratum, "log_id": u.log, "drive_id": u.drive, "day_vehicle": u.day_vehicle,
                     "scene_weight_E4": u.weight, "t_star": int(u.t_star), "anchor": a, "n_old": 1,
                     "valid": bool(u.valid), "fde5_treat_NAT_REF": t["fde5"], "fde5_ref_NAT_NSRC": r["fde5"],
                     "d_unit_fde5_m": d, "severe_treat": t["severe"], "severe_ref": r["severe"],
                     "s_unit": t["severe"] - r["severe"], "latest_ctx_token_treat": t["ctx_edit"][-1],
                     "latest_ctx_token_ref": r["ctx_edit"][-1], "latest_ctx_equal": t["ctx_edit"][-1] == r["ctx_edit"][-1],
                     "has_cam_f0": has_img,
                     "cam_f0_in_inference_keep_list": bool(fr and fr["CAM_F0"]["data_path"] in keep)})
    U = pd.DataFrame(rows)
    F = U[(U.stratum == "F")].copy()
    cand = F[F.valid].copy()
    excl = F[~F.valid].copy()
    sc = cand.groupby("scene_id").agg(d_scene=("d_unit_fde5_m", "mean"), s_scene=("s_unit", "mean"),
                                      n_units=("unit_id", "size"), drive_id=("drive_id", "first"),
                                      log_id=("log_id", "first"), day_vehicle=("day_vehicle", "first"),
                                      t_star=("t_star", "first"), scene_weight_E4=("scene_weight_E4", "first"),
                                      has_cam_f0=("has_cam_f0", "all")).reset_index()
    sc["group"] = np.where(sc.s_scene < 0, "improved", np.where(sc.s_scene > 0, "worsened", "unchanged_severe"))
    sc["d_scene_percentile"] = sc.d_scene.rank(pct=True, method="average") * 100
    sel = []
    for gname in ("improved", "unchanged_severe", "worsened"):
        gd = sc[(sc.group == gname)]
        if gd.empty:
            continue
        ranked, med = pick_closest(gd[gd.has_cam_f0], "d_scene", "scene_id")
        chosen = []
        for _, row in ranked.iterrows():
            if not chosen:
                chosen.append(row)
            elif row.drive_id != chosen[0].drive_id:
                chosen.append(row)
                break
        if len(chosen) == 1 and len(ranked) > 1:
            chosen.append(ranked.iloc[1])
        for rank, row in enumerate(chosen, 1):
            us = cand[cand.scene_id == row.scene_id]
            sign = {"improved": -1, "unchanged_severe": 0, "worsened": 1}[gname]
            pool = us[np.sign(us.s_unit) == sign]
            pool = pool if len(pool) else us
            pr, umed = pick_closest(pool, "d_unit_fde5_m", "unit_id")
            sel.append({"group": gname, "rank_in_group": rank, "main_text_suggestion": rank == 1,
                        "scene_id": row.scene_id, "unit_id": pr.iloc[0].unit_id, "group_median_d_scene": med,
                        "d_scene": row.d_scene, "s_scene": row.s_scene, "d_scene_percentile": row.d_scene_percentile,
                        "scene_unit_median_d": umed, "d_unit": pr.iloc[0].d_unit_fde5_m, "s_unit": pr.iloc[0].s_unit,
                        "drive_id": row.drive_id, "log_id": row.log_id, "t_star": row.t_star})
    S = pd.DataFrame(sel)
    sc["selected"] = sc.scene_id.isin(S.scene_id)
    os.makedirs(OUT, exist_ok=True)
    cand_out = F.merge(sc[["scene_id", "d_scene", "s_scene", "group", "d_scene_percentile", "selected"]],
                       on="scene_id", how="left")
    cand_out["exclusion_reason"] = np.where(~cand_out.valid, "C_NAT row not valid", np.where(~cand_out.has_cam_f0, "no CAM_F0 (visual)", ""))
    cand_out["selected_unit"] = cand_out.unit_id.isin(S.unit_id)
    cand_out.sort_values(["group", "scene_id", "unit_id"]).to_csv(f"{OUT}/CASE_CANDIDATES.csv", index=False)
    sc.sort_values(["group", "scene_id"]).to_csv(f"{OUT}/tables/V1_scene_candidates.csv", index=False)
    S.to_csv(f"{OUT}/SELECTED_CASES.csv", index=False)
    gsum = sc.groupby("group").agg(n_scenes=("scene_id", "size"), n_drives=("drive_id", "nunique"),
                                   median_d_scene=("d_scene", "median")).reset_index()
    gsum["share_scenes"] = gsum.n_scenes / gsum.n_scenes.sum()
    gsum.to_csv(f"{OUT}/tables/V1_group_counts.csv", index=False)
    checks.update({"F_units": int(len(F)), "F_valid_units": int(len(cand)), "F_invalid_units": int(len(excl)),
                   "F_scenes": int(sc.shape[0]), "F_drives": int(sc.drive_id.nunique()),
                   "F_units_without_cam_f0": int((~F.has_cam_f0).sum()),
                   "F_units_cam_f0_in_keep_list": int(F.cam_f0_in_inference_keep_list.sum())})
    json.dump(checks, open(f"{OUT}/tables/V1_checks.json", "w"), indent=1)
    print(json.dumps(checks, indent=1)); print(gsum.to_string()); print(S.to_string())
    write_cases(S, cells, rec, cache)


def write_cases(S, cells, rec, cache):
    plt = V.mpl_setup()
    import matplotlib.image as mpimg
    case_rows = []
    allxy = []
    data = []
    for _, s in S.iterrows():
        g = rec[s.scene_id]
        cs = {c: cells[(s.unit_id, c)] for c in ("GT_REF", "GT_NSRC", "GT_CSRC", "NAT_REF", "NAT_NSRC", "NAT_CSRC", "noop")}
        data.append((s, g, cs))
        for c in (TREAT, REF):
            allxy.append(np.asarray(cs[c]["poses"])[:, :2])
        allxy.append(g["gt_xy"])
    allxy = np.vstack(allxy + [np.zeros((1, 2))])
    # one common scale for every case: x (forward) range, y (left) range
    xmax = float(np.ceil(allxy[:, 0].max() + 2)); xmin = min(-2.0, float(np.floor(allxy[:, 0].min() - 1)))
    yabs = float(np.ceil(np.abs(allxy[:, 1]).max() + 1.5))
    for s, g, cs in data:
        cid = f"{s.group}_{s.rank_in_group}"
        cdir = f"{OUT}/cases/{cid}"
        os.makedirs(cdir, exist_ok=True)
        fr = frames_for(s.scene_id, s.log_id, cache)
        for cam in ("CAM_F0", "CAM_L0", "CAM_R0"):
            if fr and fr[cam]["exists"]:
                shutil.copyfile(fr[cam]["path"], f"{cdir}/{cam}_{os.path.basename(fr[cam]['path'])}")
        t, r = cs[TREAT], cs[REF]
        a = int(t["anchor"])
        meta = {"case_id": cid, "note": "illustrative case chosen by a post-hoc deterministic rule (CASE_SELECTION_RULE.md); "
                "not a representative sample; open-loop plan from a fixed observation (not closed-loop driving)",
                "cohort": "B2 primary cohort, stratum F (P1-C intervention sample within E4), n_old = 1, anchor a = t*+3",
                "contrast": "treatment NAT_REF (latest context position = GT token) vs reference NAT_NSRC (latest context "
                            "position = frozen Normal-source token); identical emitted prefix",
                "scene_id": s.scene_id, "unit_id": s.unit_id, "log_id": s.log_id, "drive_id": s.drive_id,
                "t_star": int(s.t_star), "anchor": a, "older_ctx_position": a - 2, "latest_ctx_position": a - 1,
                "coords": "ego frame at the observation time; x forward, y left (m); heading rad; pose k is the end of "
                          "segment k, time (k+1)*0.5 s; origin (0,0) = ego at observation",
                "timestamp_us": fr["timestamp_us"] if fr else None,
                "camera_files": {cam: (os.path.basename(fr[cam]["path"]) if fr and fr[cam]["exists"] else None)
                                 for cam in ("CAM_F0", "CAM_L0", "CAM_R0")},
                "camera_calibration": {cam: {k: fr[cam][k] for k in ("cam_intrinsic", "sensor2lidar_rotation",
                                                                      "sensor2lidar_translation", "distortion")}
                                       for cam in ("CAM_F0",)} if fr else None,
                "camera_note": "model input = 4 history frames of front / front-left / front-right cameras "
                               "(autovla/dataset_utils); only the current-time frame of each is shipped. No projection.",
                "gt_tokens": g["gt_tokens"], "gt_xy": g["gt_xy"].tolist(), "stratum_group_P1C": g["group"],
                "cells": {c: {k: v[k] for k in ("ctx_edit", "ctx_src", "ctx_edit_motion", "own", "acts", "poses",
                                                  "fde5", "severe", "A_plus", "prefix_ids_sha", "bank_sha")}
                          for c, v in cs.items()},
                "metrics": {"fde5_NAT_REF": t["fde5"], "fde5_NAT_NSRC": r["fde5"], "d_unit": t["fde5"] - r["fde5"],
                            "severe_NAT_REF": t["severe"], "severe_NAT_NSRC": r["severe"], "d_scene": s.d_scene,
                            "s_scene": s.s_scene, "d_scene_percentile_among_F_scenes": s.d_scene_percentile},
                "source": {"cells": V.rel(f"{V.B2}/p1c_main/cells.jsonl"), "cells_sha256": V.sha(f"{V.B2}/p1c_main/cells.jsonl"),
                           "gt": V.rel(f"{V.P1C}/mech/action_history_causal/records.jsonl"),
                           "frames": "autovla/dataset/nuplan/sensor_blobs/test/<data_path>"}}
        json.dump(meta, open(f"{cdir}/case.json", "w"), indent=1)
        np.savez_compressed(f"{cdir}/case_arrays.npz", gt_xy=g["gt_xy"],
                            **{f"poses_{c}": np.asarray(v["poses"]) for c, v in cs.items()},
                            **{f"acts_{c}": np.asarray(v["acts"]) for c, v in cs.items()})
        tr = []
        for c in (TREAT, REF):
            P = np.asarray(cs[c]["poses"])
            for k in range(10):
                tr.append({"case_id": cid, "cell": c, "pose_k": k, "time_s": 0.5 * (k + 1), "segment": "prefix" if k < a else "suffix",
                           "token_id": cs[c]["acts"][k], "x_m": P[k, 0], "y_m": P[k, 1], "yaw_rad": P[k, 2],
                           "gt_x_m": g["gt_xy"][k, 0], "gt_y_m": g["gt_xy"][k, 1]})
        pd.DataFrame(tr).to_csv(f"{cdir}/trajectories.csv", index=False)
        case_rows.append(meta)
        # ---- figure: camera | BEV ----
        fig = plt.figure(figsize=(7.2, 2.7))
        gs = fig.add_gridspec(1, 3, width_ratios=[1.45, 0.95, 0.85], wspace=0.42)
        ax0 = fig.add_subplot(gs[0])
        if fr and fr["CAM_F0"]["exists"]:
            ax0.imshow(mpimg.imread(fr["CAM_F0"]["path"]))
        ax0.set_axis_off()
        ax0.set_title(f"Front camera (CAM_F0) · {s.scene_id}", loc="left", color=V.INK)
        ax = fig.add_subplot(gs[1])
        ax.grid(True)
        gt = g["gt_xy"]
        # BEV with forward = up: plot (y_left -> -horizontal) so left is left
        ax.plot(-np.r_[0, gt[:, 1]], np.r_[0, gt[:, 0]], color=V.GT_COLOR, lw=1.2, ls=(0, (1, 1.2)), label="GT (5 s)")
        Pt, Pr = np.asarray(t["poses"]), np.asarray(r["poses"])
        pre = np.vstack([[0, 0, 0], Pt[:a]])
        ax.plot(-pre[:, 1], pre[:, 0], color=V.INK2, lw=2.2, label=f"shared emitted prefix (poses 0–{a-1})", solid_capstyle="round")
        for P, col, lab, mk in ((Pr, V.PAL[1], "NAT_NSRC: latest ctx = Normal-source (reference)", "s"),
                                (Pt, V.PAL[0], "NAT_REF: latest ctx = GT (treatment)", "o")):
            seg = P[a - 1:]
            ax.plot(-seg[:, 1], seg[:, 0], color=col, lw=1.4, label=lab)
            ax.plot(-P[a:, 1], P[a:, 0], ls="none", marker=mk, ms=3.2, mfc=col, mec="white", mew=0.5)
        # edited latest-context values: draw as hollow markers from pose a-2 (not executed)
        base = Pt[a - 2] if a >= 2 else np.zeros(3)
        for c, col in ((TREAT, V.PAL[0]), (REF, V.PAL[1])):
            m = np.asarray(cs[c]["ctx_edit_motion"][-1])
            hx = base[2]
            dx = m[0] * np.cos(hx) - m[1] * np.sin(hx); dy = m[0] * np.sin(hx) + m[1] * np.cos(hx)
            ax.plot([-base[1], -(base[1] + dy)], [base[0], base[0] + dx], color=col, lw=0.8, ls=(0, (2, 1.5)))
            ax.plot([-(base[1] + dy)], [base[0] + dx], marker="D", ms=4, mfc="none", mec=col, mew=0.9)
        ax.plot([0], [0], marker="^", ms=5, color=V.INK)
        pts = np.vstack([gt, Pt[:, :2], Pr[:, :2], [[0, 0]]])
        lo, hi = pts.min(0), pts.max(0)
        span = max(hi[0] - lo[0], 2 * np.abs(pts[:, 1]).max(), 8.0) + 3.0
        cx = (hi[0] + lo[0]) / 2
        ax.set_ylim(cx - span / 2, cx + span / 2); ax.set_xlim(-span / 2, span / 2); ax.set_aspect("equal")
        ax.set_xlabel("lateral (m; left is left)"); ax.set_ylabel("forward (m)")
        ax.set_title(f"BEV, open-loop plan (a={a})", loc="left", color=V.INK)
        h, l = ax.get_legend_handles_labels()
        from matplotlib.lines import Line2D
        h.append(Line2D([], [], marker="D", ls=(0, (2, 1.5)), mfc="none", mec=V.INK2, color=V.INK2, ms=4))
        l.append("latest context value shown to model (not executed)")
        fig.legend(h, l, loc="lower center", bbox_to_anchor=(0.5, -0.13), ncol=3, frameon=False)
        ax2 = fig.add_subplot(gs[2]); ax2.grid(True)
        tt = np.r_[0, 0.5 * np.arange(1, 11)]
        ax2.plot(tt, np.r_[0, gt[:, 0]], color=V.GT_COLOR, lw=1.2, ls=(0, (1, 1.2)))
        ax2.plot(tt[:a + 1], np.r_[0, Pt[:a, 0]], color=V.INK2, lw=2.2)
        for P, col, mk in ((Pr, V.PAL[1], "s"), (Pt, V.PAL[0], "o")):
            ax2.plot(tt[a:], P[a - 1:, 0], color=col, lw=1.4, marker=mk, ms=3.2, mfc=col, mec="white", mew=0.5)
        ax2.axvline(0.5 * a, color=V.INK2, lw=0.6, ls=(0, (3, 2)))
        ax2.text(0.5 * a, ax2.get_ylim()[1], " anchor", va="top", ha="left", fontsize=6, color=V.INK2)
        ax2.set_xlabel("plan time (s)"); ax2.set_ylabel("forward position (m)")
        ax2.set_title(f"FDE5 {t['fde5']:.2f} (GT ctx) / {r['fde5']:.2f} m", loc="left", color=V.INK)
        V.save(fig, f"V1_case_{cid}")
        plt.close(fig)


if __name__ == "__main__":
    main()
