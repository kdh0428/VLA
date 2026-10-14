#!/usr/bin/env python
"""
P2 step 3-4 (review_followup3, 2026-10-14; POST-HOC, CPU only, no model inference).

Distinct kinematic controls vs the AutoVLA response to a single perturbed executed token (P1-A rollouts), spec fixed in
outputs/review_followup3/P2/MOTION_BASELINE_AUDIT.md (section MOTION_BASELINE_SPEC; sha256 locked in spec_lock.json,
checked here).  Per unit and method:  Delta = phi(perturbed) - phi(unperturbed); contrast = Delta_model - Delta_baseline.
Inference = c_baseline_window.drive_row (scene-weighted, drive bootstrap B = 10,000, exact drive sign-flip).
Holm m = 4 over {CM, CV0, CTRV-K3, CTRA-2} within (perturbation family, stratum, metric).  Geo (privileged) outside.

Outputs (outputs/review_followup3/P2):
  p2_units.parquet                 per-unit Delta per method x metric
  MOTION_BASELINE_RESULTS.csv      excess (Delta_model - Delta_b) and level (Delta_method) rows with CIs / p / Holm
  baseline_fit_diagnostics.csv     clip / fallback activation (inputs only)
Run:  nice -n 19 /root/miniforge3/envs/autovla/bin/python p2_motion_baselines.py
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
SCR = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(SCR, "review_followup2"))
sys.path.insert(0, os.path.join(SCR, "review_followup", "p1ab"))
import p1_common as C  # noqa: E402
import p2_kinematics as K  # noqa: E402
import stats_core as SC  # noqa: E402
import c_baseline_window as CB  # noqa: E402  (drive_row, drive_of: same estimator as C1)

POC = "/root/VLA/autovla_misalignment_poc"
P1A = f"{POC}/outputs/review_followup/P1A_gt_free_reference/run_20261008"
C1DIR = f"{POC}/outputs/review_followup2/C_baseline_window/run_20261013"
OUT = f"{POC}/outputs/review_followup3/P2"
SEED = 20261014
KIN = ["CM", "CV0", "CTRV-K3", "CTRA-2"]
METHODS = KIN + ["Geo"]
METRICS = {"end": "M1 endpoint displacement norm ||X_p9-X_0_9|| (m)",
           "sfx": "M2 suffix local-motion change mean_k>t ||ell_k(X_p)-ell_k(X_0)|| (m/seg)",
           "dref": "M3 signed error change vs reference r at pose 9 (m)",
           "dgt": "M4 signed GT error change at pose 9 (m) [GT-relative, secondary]",
           "area": "M5 path area sum_k=t..9 ||X_p(k)-X_0(k)|| * 0.5 s (m*s)"}
T0 = time.time()
GATE = f"{POC}/outputs/review_followup3/P1/NOOP_CACHE_AUDIT.md"
STATUS = ("PROVISIONAL (AutoVLA P1 no-op/cache parity gate not judged at run time; P1-A rollouts = C1 input path, "
          "manuscript section 6.2 / Figure 1 held until the gate passes)")


def log(*a):
    print(f"[{time.time() - T0:7.1f}s]", *a, flush=True)


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def phi_pair(Xp, X0, t, ref9, gt9, lp=None, l0=None):
    """metrics for one method; lp/l0 optional exact body increments (token disp) for the suffix."""
    if lp is None:
        lp, l0 = K.body_increments(Xp)[:, :2], K.body_increments(X0)[:, :2]
    d = np.linalg.norm(Xp[:, :2] - X0[:, :2], axis=1)
    return {"end": float(d[9]),
            "sfx": float(np.mean(np.linalg.norm(lp[t + 1:] - l0[t + 1:], axis=1))),
            "dref": float(np.linalg.norm(Xp[9, :2] - ref9) - np.linalg.norm(X0[9, :2] - ref9)),
            "dgt": float(np.linalg.norm(Xp[9, :2] - gt9) - np.linalg.norm(X0[9, :2] - gt9)),
            "area": float(d[t:].sum() * K.DT)}


def build_units():
    cb, disp, dyaw = C.codebook_geometry()
    rows, diag = [], []
    for src, path in (("primary", f"{P1A}/raw/records.jsonl"), ("fixedpos", f"{P1A}/raw_fixedpos/records.jsonl")):
        for line in open(path):
            r = json.loads(line)
            t = r["t_p"]; ref = r["ref_tokens"]; p = r["perturbation_info"]["token_p"]
            if p == ref[t]:
                continue
            reft = np.asarray(r["ref_traj_xyh"], float); ref9 = reft[9, :2]
            gt9 = np.asarray(r["trajectory_gt"], float)[9, :2]
            OC = np.asarray(r["rows"]["OC"]["traj_xyh"], float); U = np.asarray(r["rows"]["U"]["traj_xyh"], float)
            aOC, aU = r["rows"]["OC"]["action_idx"], r["rows"]["U"]["action_idx"]
            rec = {"source": src, "family": r["family"], "group": r["group"], "unit_id": r["unit_id"], "scene_id": r["token"],
                   "log_id": r["log"], "drive_id": CB.drive_of(r["log"]), "t_p": t}
            res = {"model": phi_pair(OC, U, t, ref9, gt9, disp[aOC], disp[aU])}
            geo = np.asarray(r["baselines"]["Geo"]["traj_xyh"], float)
            ga = r["baselines"]["Geo"]["action_idx"]
            res["Geo"] = phi_pair(geo, reft, t, ref9, gt9, disp[ga], disp[ref])
            pre_p = K.tokens_to_poses(disp, dyaw, list(ref[:t]) + [p])
            pre_0 = K.tokens_to_poses(disp, dyaw, list(ref[:t]) + [ref[t]])
            res["CM"] = phi_pair(K.bl_cm(pre_p, t), K.bl_cm(pre_0, t), t, ref9, gt9)
            res["CV0"] = phi_pair(K.bl_cv0(pre_p, t), K.bl_cv0(pre_0, t), t, ref9, gt9)
            (vp, ip), (v0, i0) = K.bl_ctrv(pre_p, t), K.bl_ctrv(pre_0, t)
            res["CTRV-K3"] = phi_pair(vp, v0, t, ref9, gt9)
            (ap, jp), (a0, j0) = K.bl_ctra(pre_p, t), K.bl_ctra(pre_0, t)
            res["CTRA-2"] = phi_pair(ap, a0, t, ref9, gt9)
            for m, v in res.items():
                for k, x in v.items():
                    rec[f"{m}_{k}"] = x
            for b in METHODS:
                for k in METRICS:
                    rec[f"ex_{b}_{k}"] = rec[f"model_{k}"] - rec[f"{b}_{k}"]
            rows.append(rec)
            diag.append({"source": src, "family": r["family"], "group": r["group"], "unit_id": r["unit_id"], "t_p": t,
                         "ctrv_k_eff": ip["k_eff"], "ctrv_omega_clipped_p": ip["omega_clipped"], "ctrv_omega_clipped_0": i0["omega_clipped"],
                         "ctra_fallback_a0_t0": t == 0, "ctra_acc_clipped_p": jp["acc_clipped"], "ctra_acc_clipped_0": j0["acc_clipped"],
                         "ctra_omega_clipped_p": jp["omega_clipped"], "ctra_omega_clipped_0": j0["omega_clipped"],
                         "ctra_stopped_p": jp["stopped"], "ctra_stopped_0": j0["stopped"],
                         "ctra_a_raw_p": jp["a_raw"], "ctra_a_raw_0": j0["a_raw"]})
    return pd.DataFrame(rows), pd.DataFrame(diag)


def tables(U):
    c1 = pd.read_csv(f"{C1DIR}/c1_excess_effects.csv")
    c1map = {"CM": {"end": "ex_cm_end", "sfx": "ex_cm_sfx", "dref": "ex_cm_dref"},
             "Geo": {"end": "ex_geo_end", "sfx": "ex_geo_sfx", "dref": "ex_geo_dref"}}
    out = []
    seed = SEED
    for fam in ["G", "E", "GF1", "GF3"]:
        gf = U[U.family == fam]
        for grp in ("A-", "A+", "all"):
            g = gf if grp == "all" else gf[gf.group == grp]
            role = {"G": "primary (family G)", "E": "secondary (GT-anchored E)"}.get(fam, "secondary (fixed position, GT-free)")
            for met, mdesc in METRICS.items():
                block = []
                for b in METHODS:
                    r = CB.drive_row(g, f"ex_{b}_{met}", margin=None, seed=seed); seed += 11
                    c1v = np.nan
                    if b in c1map and met in c1map[b]:
                        q = c1[(c1.family == fam) & (c1.group == grp) & (c1.metric == c1map[b][met])]
                        c1v = float(q.est_scene.iloc[0]) if len(q) else np.nan
                    block.append({"row_type": "excess", "family": fam, "role": role, "group": grp, "metric": met,
                                  "metric_desc": mdesc, "baseline": b,
                                  "baseline_class": "privileged (uses model's unperturbed future r[t+1:])" if b == "Geo" else "kinematic extrapolation (prefix only)",
                                  "quantity": f"Delta_model - Delta_{b}", "c1_stored_est_scene": c1v, **r})
                kin = [x for x in block if x["baseline"] in KIN]
                adj = SC.holm([x["p_signflip"] for x in kin])
                for x, a in zip(kin, adj):
                    x["p_holm_m4"] = a
                for x in block:
                    pa = x.get("p_holm_m4", np.nan)
                    ex0 = x["ci95_lo"] > 0 or x["ci95_hi"] < 0
                    if x["baseline"] == "Geo":
                        x["judgment"] = ("CI excludes 0 (unadjusted; not in Holm family)" if ex0 else "CI includes 0")
                    elif pa < 0.05 and ex0:
                        x["judgment"] = "model > baseline" if x["est_scene"] > 0 else "model < baseline"
                    else:
                        x["judgment"] = "not detected (equivalence not tested)"
                out += block
                for m in ["model"] + METHODS:            # levels (descriptive)
                    r = CB.drive_row(g, f"{m}_{met}", margin=None, seed=seed); seed += 11
                    out.append({"row_type": "level", "family": fam, "role": role, "group": grp, "metric": met, "metric_desc": mdesc,
                                "baseline": m, "baseline_class": "model" if m == "model" else ("privileged" if m == "Geo" else "kinematic extrapolation (prefix only)"),
                                "quantity": f"Delta_{m}", "judgment": "descriptive level", **r})
            log(fam, grp, "done")
    return pd.DataFrame(out)


def main():
    # spec adopted from the stopped v1 attempt (locked 2026-10-14T10:53:20Z, before any new-baseline result existed);
    # verbatim copy of the locked file + lock record live in P2/ (MOTION_BASELINE_SPEC_LOCKED_v1.md, spec_lock_v1.json)
    lock = json.load(open(f"{OUT}/spec_lock_v1.json"))
    assert sha(f"{OUT}/MOTION_BASELINE_SPEC_LOCKED_v1.md") == lock["MOTION_BASELINE_AUDIT.md_sha256"], "spec changed after lock"
    assert sha(os.path.join(HERE, "p2_kinematics.py")) == lock["p2_kinematics.py_sha256"], "kinematics changed after lock"
    U, D = build_units()
    log("units", len(U))
    U.to_parquet(f"{OUT}/p2_units.parquet", index=False)
    dg = D.groupby(["family", "group"]).agg(n_units=("unit_id", "size"), share_t0=("ctra_fallback_a0_t0", "mean"),
                                            ctrv_keff_mean=("ctrv_k_eff", "mean"), ctrv_keff_lt3=("ctrv_k_eff", lambda s: float((s < 3).mean())),
                                            ctrv_omega_clip_p=("ctrv_omega_clipped_p", "mean"), ctrv_omega_clip_0=("ctrv_omega_clipped_0", "mean"),
                                            ctra_acc_clip_p=("ctra_acc_clipped_p", "mean"), ctra_acc_clip_0=("ctra_acc_clipped_0", "mean"),
                                            ctra_omega_clip_p=("ctra_omega_clipped_p", "mean"), ctra_stop_p=("ctra_stopped_p", "mean"),
                                            ctra_stop_0=("ctra_stopped_0", "mean")).reset_index()
    dg.to_csv(f"{OUT}/baseline_fit_diagnostics.csv", index=False)
    T = tables(U)
    T.insert(0, "status", STATUS)
    T.to_csv(f"{OUT}/MOTION_BASELINE_RESULTS.csv", index=False)
    meta = {"script": os.path.abspath(__file__), "seed_base": SEED, "bootstrap_B": CB.B, "spec_lock": lock,
            "inputs": {p: sha(p) for p in (f"{P1A}/raw/records.jsonl", f"{P1A}/raw_fixedpos/records.jsonl", C.CODEBOOK,
                                           f"{C1DIR}/c1_excess_effects.csv")},
            "n_units": int(len(U)), "status": STATUS,
            "p1_gate_file_present_at_run": os.path.exists(GATE), "elapsed_s": time.time() - T0}
    json.dump(meta, open(f"{OUT}/p2_run_meta.json", "w"), indent=1)
    log("done")


if __name__ == "__main__":
    main()
