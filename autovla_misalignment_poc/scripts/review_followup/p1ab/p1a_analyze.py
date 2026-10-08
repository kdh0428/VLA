#!/usr/bin/env python
"""
P1-A analysis (CPU only). Written before the main run (preregistered with protocol.md).

Reads <raw>/records.jsonl from p1a_gt_free_reference.py and writes into <out>:
  units.parquet           one row per (unit, condition) incl. baselines Geo/CM, plan §2 columns + metrics
  condition_summary.csv   scene-weighted level of each metric per family x group x condition (log-bootstrap CI)
  paired_effects.csv      predeclared contrasts (primary family Holm m=3; secondary unadjusted, labelled)
  trace_verification.json output-prefix lock / decoder origin / KV / U==reference checks over all units
  deviation_curves.csv    mean deviation vs reference per pose offset (i - t_p) per condition
Metric definitions (protocol.md §4):
  dev_ref[i]   L2 (m) between condition pose i and reference pose i (x, y only); heading separately (rad)
  dev9_ref     endpoint deviation vs reference at pose 9 (5 s)
  suffix_loc   mean over k > t_p of ||disp[o_k] - disp[r_k]|| (m per 0.5 s segment, token-local frame)
  suffix_yaw   mean over k > t_p of |wrap(dyaw[o_k] - dyaw[r_k])| (rad)
  suffix_mismatch  share of k > t_p with o_k != r_k
  follow       mean over k > t_p of <disp[o_k] - disp[r_k], u> / d0, u = unit vector of disp[p] - disp[r_tp]
  d0           achieved perturbation size ||disp[p] - disp[r_tp]|| (m); growth = dev9_ref / d0 if d0 >= EPS
  GT-relative: unit_metrics of analyze_action_history (amplification = A- and FDE5 > 3 m, ade5, fde5),
               plus amp2/amp4 (FDE5 > 2 / 4 m)
  reference-relative: refAm = A- label against the reference trajectory; ref_amp3 = refAm and dev9_ref > 3 m;
               severe_ref{2,3,4} = dev9_ref > {2,3,4} m
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p1_common as C  # noqa: E402
import p1_stats as ST  # noqa: E402
from analyze_action_history import unit_metrics, a_eval  # noqa: E402

EPS_D0 = 0.02
PRIMARY = [  # (id, cond_a, cond_b, metric, hypothesis, equivalence margin)
    ("P1A-1 context-only local suffix change", "C1", "U", "suffix_loc", "C1 > U", 0.02),
    ("P1A-2 endpoint deviation beyond geometric propagation", "OC", "Geo", "dev9_ref", "OC > Geo", 0.25),
    ("P1A-3 repeated GT-free reference correction", "OC_refR", "OC", "dev9_ref", "OC_refR < OC", 0.25),
]
SECONDARY = [
    ("O1 vs Geo (expected identical up to numerics)", "O1", "Geo", "dev9_ref"),
    ("OC vs constant-motion", "OC", "CM", "dev9_ref"),
    ("OC vs O1", "OC", "O1", "dev9_ref"),
    ("C1 vs U endpoint", "C1", "U", "dev9_ref"),
    ("C1 vs U local heading", "C1", "U", "suffix_yaw"),
    ("C1 vs U token mismatch", "C1", "U", "suffix_mismatch"),
    ("OC vs U local suffix change", "OC", "U", "suffix_loc"),
    ("OC follow vs C1 follow", "OC", "C1", "follow"),
    ("one-shot reference correction", "OC_ref1", "OC", "dev9_ref"),
    ("reference history", "OC_refH", "OC", "dev9_ref"),
    ("output perturbed + reference context, repeated", "O_refH", "OC", "dev9_ref"),
    ("Recent-GT (GT arm)", "OC_gtR", "OC", "dev9_ref"),
    ("Recent-GT vs Recent-reference", "OC_gtR", "OC_refR", "dev9_ref"),
    ("severe_ref3 OC vs Geo", "OC", "Geo", "severe_ref3"),
    ("severe_ref3 OC_refR vs OC", "OC_refR", "OC", "severe_ref3"),
    ("ref_amp3 OC vs Geo", "OC", "Geo", "ref_amp3"),
    ("GT amplification OC vs U", "OC", "U", "amplification"),
    ("GT amplification OC_refR vs OC", "OC_refR", "OC", "amplification"),
    ("GT amplification OC_gtR vs OC", "OC_gtR", "OC", "amplification"),
    ("GT FDE5 OC_refR vs OC", "OC_refR", "OC", "fde5"),
    ("log2 growth OC vs Geo", "OC", "Geo", "log2_growth"),
    ("log2 growth OC vs CM", "OC", "CM", "log2_growth"),
]


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    cb, disp, dyaw = C.codebook_geometry()
    recs = [json.loads(l) for l in open(os.path.join(args.raw, "records.jsonl"))]
    rows = []
    tv = defaultdict(list)
    curves = defaultdict(list)
    for r in recs:
        t = r.get("t_p", r["t_star"]); gt = r["gt"]; gtraj = r["trajectory_gt"]; ref = r["ref_tokens"]
        reft = np.asarray(r["ref_traj_xyh"], float)
        pi = r["perturbation_info"]; p = pi["token_p"]
        d0 = pi["achieved_m"]
        u = np.array([pi["vec_forward"], pi["vec_left"]]) / d0 if d0 > 0 else None
        ck = r["checks"]
        for nm in r["rows"]:
            c = ck[nm]
            tv["output_prefix_locked"].append(c["output_prefix_locked"]); tv["decoder_origin_zero"].append(c["decoder_origin_zero"])
            tv["prefix_pose_err_zero"].append(c["prefix_pose_max_abs_err_vs_reference"] == 0.0)
        tv["U_equals_reference"].append(ck["U_equals_reference_tokens"]); tv["O1_suffix_equals_U"].append(ck["O1_suffix_equals_U_suffix"])
        tv["OC_suffix_equals_C1"].append(ck.get("OC_suffix_equals_C1_suffix"))
        tv["numpy_rollout_maxabs"].append(ck["numpy_rollout_matches_decoder_maxabs"])
        if pi["prefix_logprob_matches_reference_file"] is not None:
            tv["prefix_logprob_maxabs_vs_reference_run"].append(pi["prefix_logprob_matches_reference_file"])
        excl = "no_perturbation(p==r)" if p == ref[t] else None
        allc = {**{k: (v, "model") for k, v in r["rows"].items()}, **{k: (v, "baseline") for k, v in r["baselines"].items()}}
        u_first_ent = r["rows"]["U"]["ent"][0] if r["rows"]["U"]["ent"] else None
        for nm, (x, kind) in allc.items():
            acts = x["action_idx"]; tr = np.asarray(x["traj_xyh"], float)
            m = unit_metrics(acts, tr[:, :2].tolist(), x.get("ent") or [], gt, gtraj, t)
            e_gt = np.linalg.norm(tr[:, :2] - np.asarray(gtraj, float)[:, :2], axis=1)
            dev = np.linalg.norm(tr[:, :2] - reft[:, :2], axis=1)
            hdev = np.abs(wrap(tr[:, 2] - reft[:, 2]))
            sfx = range(t + 1, C.N_ACT)
            dl = [np.linalg.norm(disp[acts[k]] - disp[ref[k]]) for k in sfx]
            dyw = [abs(wrap(dyaw[acts[k]] - dyaw[ref[k]])) for k in sfx]
            fol = [float(np.dot(disp[acts[k]] - disp[ref[k]], u)) / d0 for k in sfx] if u is not None else []
            refAm = not a_eval(tr[:, :2].tolist(), reft[:, :2].tolist())
            growth = float(dev[-1] / d0) if d0 >= EPS_D0 else np.nan
            row = {
                "model": "AutoVLA", "experiment": "P1A_gt_free_reference", "unit_id": r["unit_id"], "log_id": r["log"],
                "scene_id": r["token"], "task_id": None, "episode_id": None, "seed": r["seed"], "group": r["group"],
                "family": r["family"], "perturbation": r["perturbation"], "condition": nm, "condition_kind": kind,
                "decode_j": t, "control_t": None, "t_star": r["t_star"], "position_rule": r.get("position_rule", "t*"),
                "reference_branch": ("gt" if nm.startswith("OC_gt") else "model_reference"),
                "perturbed_span": f"[{t},{t}]", "perturbation_requested": pi["requested_m"], "perturbation_achieved": d0,
                "perturbation_direction_deg": pi["direction_deg"], "perturbation_sector": pi["sector"],
                "dyaw_p_minus_ref_rad": pi["dyaw_p_minus_ref_rad"], "logp_p_given_prefix": pi["logp_p_given_prefix"],
                "rank_p_given_prefix": pi["rank_p_given_prefix"], "freq_natural_pred_p": pi["freq_natural_pred_p"],
                "intervention_count": x.get("context_override_steps"),
                "intervention_positions_n": len(x.get("context_override_positions") or []), "intervention_count_effective": x.get("context_override_effective_steps"),
                "parser_status": "ok", "exclusion_reason": excl, "primary_outcome": float(dev[-1]),
                "dev9_ref": float(dev[-1]), "ade_ref_post": float(dev[t:].mean()), "head_dev9_ref": float(hdev[-1]),
                "suffix_loc": float(np.mean(dl)) if dl else np.nan, "suffix_yaw": float(np.mean(dyw)) if dyw else np.nan,
                "suffix_mismatch": float(np.mean([acts[k] != ref[k] for k in sfx])) if dl else np.nan,
                "follow": float(np.mean(fol)) if fol else np.nan,
                "d0": d0, "growth": growth, "log2_growth": float(np.log2(growth)) if np.isfinite(growth) and growth > 0 else np.nan,
                "refAm": float(refAm), "ref_amp3": float(refAm and dev[-1] > 3.0),
                "severe_ref2": float(dev[-1] > 2.0), "severe_ref3": float(dev[-1] > 3.0), "severe_ref4": float(dev[-1] > 4.0),
                "amplification": float(m["amplification"]), "recovery": float(m["recovery"]), "ade5": m["ade5"], "fde5": m["fde5"],
                "amp2": float((not m["recovery"]) and e_gt[-1] > 2.0), "amp4": float((not m["recovery"]) and e_gt[-1] > 4.0),
                "entropy1": m["entropy1"], "d_entropy1_vs_U": (m["entropy1"] - u_first_ent) if (m["entropy1"] is not None and u_first_ent is not None) else None,
                "dev_ref_curve": json.dumps(np.round(dev, 5).tolist()),
            }
            rows.append(row)
            if excl is None:
                for i in range(t, C.N_ACT):
                    curves[(r["family"], r["group"], nm, i - t)].append((r["log"], r["token"], float(dev[i])))
    U = pd.DataFrame(rows)
    U.to_parquet(os.path.join(args.out, "units.parquet"), index=False)

    # ---- trace verification
    tvs = {k: (float(np.mean(v)) if k not in ("numpy_rollout_maxabs", "prefix_logprob_maxabs_vs_reference_run") else float(np.max(v)))
           for k, v in tv.items() if len(v)}
    tvs["n_units"] = len(recs)
    json.dump(tvs, open(os.path.join(args.out, "trace_verification.json"), "w"), indent=1)

    # ---- OOD flag (condition level, as the original analysis): mean first-step entropy change vs U > 1 nat
    ood = U[U.condition_kind == "model"].groupby(["family", "group", "condition"])["d_entropy1_vs_U"].mean()

    # ---- condition summary
    cs = []
    V = U[U.exclusion_reason.isna()]
    for (fam, grp, cond), g in V.groupby(["family", "group", "condition"]):
        for met in ("dev9_ref", "suffix_loc", "suffix_yaw", "follow", "severe_ref3", "ref_amp3", "amplification", "fde5", "log2_growth"):
            df = g[["log_id", "scene_id", met]].rename(columns={"log_id": "log", "scene_id": "scene", met: "d"}).dropna()
            if len(df) == 0:
                continue
            res = ST.log_cluster_effects(df, with_tests=False, b=2000)
            cs.append({"family": fam, "group": grp, "condition": cond, "metric": met, "scene_mean": res["scene"]["estimate"],
                       "ci95_lo": res["scene"]["ci95"][0], "ci95_hi": res["scene"]["ci95"][1], "unit_mean": res["unit"]["estimate"],
                       "n_units": res["n_units"], "n_scenes": res["n_scenes"], "n_logs": res["n_clusters"],
                       "ood_mean_d_entropy1": float(ood.get((fam, grp, cond), np.nan)) if cond in ood.index.get_level_values(2) else None})
    pd.DataFrame(cs).to_csv(os.path.join(args.out, "condition_summary.csv"), index=False)

    # ---- paired effects
    eff, loo = [], []
    seed = ST.SEED
    for fam in ["G", "E"] + sorted(f for f in U.family.unique() if f.startswith("GF")):
        for grp in ("A-", "A+", "all"):
            mask = (U.family == fam) & U.exclusion_reason.isna() & ((U.group == grp) if grp != "all" else True)
            if not mask.any():
                continue
            is_primary = fam == "G" and grp == "A-"
            for cid, a, b, met, hyp, margin in PRIMARY:
                df = ST.paired_frame(U, a, b, met, mask)
                if len(df):
                    eff.append(ST.effect_row("P1A_primary" if is_primary else f"P1A_secondary_{fam}_{grp}", cid, grp, met, df,
                                             seed, margin=margin, label=f"{a} - {b}; H1: {hyp}") | {"perturbation_family": fam})
                    if is_primary:
                        loo += ST.loo_rows("P1A_primary", cid, grp, met, df, seed)
                seed += 7
            for cid, a, b, met in SECONDARY:
                df = ST.paired_frame(U, a, b, met, mask)
                if len(df):
                    eff.append(ST.effect_row(f"P1A_secondary_{fam}_{grp}", cid, grp, met, df, seed, label=f"{a} - {b}")
                               | {"perturbation_family": fam})
                seed += 7
    # predeclared secondary sensitivities of the primary contrasts (family G, A-)
    full8 = set()
    sp = os.path.join(args.raw, "perturbation_support.jsonl")
    if os.path.exists(sp):
        full8 = {x["token"] for x in map(json.loads, open(sp)) if x.get("n_sectors") == 8 and x.get("family", "G") == "G"}
    rank_ok = set(U.loc[(U.condition == "U") & (U.rank_p_given_prefix <= 100), "unit_id"])
    for sname, extra in (("rank_le_100", U.unit_id.isin(rank_ok)), ("full_8_sector_scenes", U.scene_id.isin(full8))):
        mask = (U.family == "G") & U.exclusion_reason.isna() & (U.group == "A-") & extra
        for cid, a, b, met, hyp, margin in PRIMARY:
            df = ST.paired_frame(U, a, b, met, mask)
            if len(df):
                eff.append(ST.effect_row(f"P1A_sensitivity_{sname}", cid, "A-", met, df, seed, margin=margin,
                                         label=f"{a} - {b}") | {"perturbation_family": "G"})
            seed += 7
    ST.add_holm(eff, "P1A_primary")
    pd.DataFrame(eff).to_csv(os.path.join(args.out, "paired_effects.csv"), index=False)
    pd.DataFrame(loo).to_csv(os.path.join(args.out, "loo_logs.csv"), index=False)

    # ---- deviation curves
    cv = []
    for (fam, grp, nm, off), items in sorted(curves.items()):
        df = pd.DataFrame(items, columns=["log", "scene", "d"])
        sm = df.groupby(["log", "scene"])["d"].mean()
        cv.append({"family": fam, "group": grp, "condition": nm, "pose_offset_from_tp": off, "scene_mean_dev_ref": float(sm.mean()),
                   "n_scenes": int(len(sm)), "n_units": int(len(df))})
    pd.DataFrame(cv).to_csv(os.path.join(args.out, "deviation_curves.csv"), index=False)
    print(json.dumps(tvs, indent=1))
    print(f"[done] units={len(recs)} rows={len(U)} effects={len(eff)}")


if __name__ == "__main__":
    main()
